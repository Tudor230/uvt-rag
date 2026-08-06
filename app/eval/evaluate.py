#!/usr/bin/env python
"""RAG retrieval evaluation harness.

Runs retrieval (fusion by default, or any registered retriever) and scores
the base and reranked stages with two independent metric families:

  A. ragas (golden-anchored, LLM judges on DeepSeek):
       context_precision, context_recall, context_entity_recall, context_relevance
  B. open-rag-eval UMBRELA (golden-free, LLM-judged relevance 0-3):
       precision@k, AP@k, MRR

Outputs a labeled JSON report plus Phoenix traces.

Run from the project root:

  python -m app.eval.evaluate --max-queries 5          # smoke test
  python -m app.eval.evaluate --label baseline-2026-08-05
  python -m app.eval.evaluate --retriever dense        # dense-only
  python -m app.eval.evaluate --retriever sparse       # sparse-only
  python -m app.eval.evaluate --retriever '{"type": "dense", "kwargs": {"top_k": 5, "rerank_on": false}}'
"""
import argparse
import asyncio
import ast
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle

from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
)

from open_rag_eval.data_classes.rag_results import RetrievalResult

from app.config import JUDGE_CONFIG, RETRIEVAL_CONFIG
from app.judges import DeepSeekJudge, ParallelUMBRELAMetric
from app.rag_pipeline import (
    RETRIEVER_BUILDERS,
    build_index,
    build_reranker,
    build_retriever,
    setup_logging,
    setup_phoenix,
)

logger = logging.getLogger("evaluate")

# k-values per stage (base = the retriever's top-k output, reranked = rerank_top_n)
BASE_K_VALUES = [1, 3, 5, 10]
RERANKED_K_VALUES = [1, 3, 5]

RAGAS_METRICS = [
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "context_relevance",
]

DEFAULT_CACHE_DIR = Path("data/eval/.ragas-cache")


class RecordingFusionRetriever(QueryFusionRetriever):
    """QueryFusionRetriever that records the generated sub-queries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_sub_queries: list[str] = []

    def _get_queries(self, original_query):
        queries = super()._get_queries(original_query)
        self.last_sub_queries = [q.query_str for q in queries]
        return queries

    async def _aget_queries(self, original_query):
        queries = await super()._aget_queries(original_query)
        self.last_sub_queries = [q.query_str for q in queries]
        return queries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/testset_single.csv"),
        help="Path to the testset CSV.",
    )
    parser.add_argument(
        "--label",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="Run label (used in the output filename).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval"),
        help="Directory for the labeled JSON output.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge the vector store table before ingesting (default off in eval).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap the number of queries processed (debug/smoke runs).",
    )
    parser.add_argument(
        "--k-values-base",
        type=int,
        nargs="+",
        default=BASE_K_VALUES,
        help="UMBRELA k values for the base (pre-rerank) stage.",
    )
    parser.add_argument(
        "--k-values-reranked",
        type=int,
        nargs="+",
        default=RERANKED_K_VALUES,
        help="UMBRELA k values for the reranked stage.",
    )
    parser.add_argument(
        "--retriever",
        default=None,
        help="Retriever to use: 'dense', 'sparse', 'fusion' (default), or a "
        "JSON spec like '{\"type\": \"dense\", \"kwargs\": {\"top_k\": 5, "
        "\"rerank_on\": false}}'. kwargs override the default retrieval "
        "config. Type must be one of the registered retriever builders.",
    )
    parser.add_argument(
        "--umbrela-workers",
        type=int,
        default=8,
        help="Concurrent judge calls per query for UMBRELA scoring.",
    )
    parser.add_argument(
        "--ragas-workers",
        type=int,
        default=8,
        help="Concurrent judge calls for the ragas metric phase.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Disk cache dir for ragas judge calls (deterministic at temp 0).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the ragas judge disk cache.",
    )
    return parser.parse_args()


def load_testset(path: Path) -> pd.DataFrame:
    """Load and tolerant-parse the testset CSV.

    Drops rows whose ``reference_contexts`` column cannot be parsed via
    ``ast.literal_eval`` into a non-empty list, logging each drop, and reports
    the final count.
    """

    def _skip_bad_line(bad_line):
        logger.warning("Dropping malformed CSV line: %r", bad_line)
        return None

    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines=_skip_bad_line,
        dtype={"user_input": str, "reference": str},
    )
    before = len(df)

    valid_rows = []
    for idx, row in df.iterrows():
        raw = row.get("reference_contexts")
        try:
            refs = ast.literal_eval(raw)
            if not isinstance(refs, list) or not refs:
                raise ValueError("reference_contexts is not a non-empty list")
        except Exception as exc:  # noqa: BLE001 - tolerant parsing by design
            logger.warning(
                "Dropping row %d (%r): %s", idx, str(row.get("user_input"))[:80], exc
            )
            continue
        row_dict = row.to_dict()
        row_dict["reference_contexts"] = refs
        valid_rows.append(row_dict)

    df = pd.DataFrame(valid_rows)
    logger.info(
        "Loaded %d rows from %s (%d dropped).", len(df), path, before - len(df)
    )
    return df


def parse_retriever_spec(value: str | None) -> dict:
    """Parse the --retriever flag into a config dict.

    Accepts a bare name ('dense'/'sparse'/'fusion') or a JSON object like
    '{"type": "dense", "kwargs": {"top_k": 5, "rerank_on": false}}'. The
    kwargs are merged into the retrieval config (overriding defaults).
    """
    if value is None:
        return {}
    if value in RETRIEVER_BUILDERS:
        return {"retriever_type": value}

    path = Path(value)
    if path.is_file():
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--retriever file {value!r} is not valid JSON: {exc}")
    else:
        try:
            spec = json.loads(value)
        except json.JSONDecodeError:
            raise SystemExit(
                f"Invalid --retriever {value!r}: expected 'dense', 'sparse', "
                "'fusion', or a JSON spec with a 'type' key."
            )

    if not isinstance(spec, dict) or "type" not in spec:
        raise SystemExit("--retriever JSON spec must be an object with a 'type' key.")

    retriever_type = spec["type"]
    if retriever_type not in RETRIEVER_BUILDERS:
        raise SystemExit(
            f"Unknown retriever type {retriever_type!r}. "
            f"Known types: {sorted(RETRIEVER_BUILDERS)}"
        )

    cfg = {"retriever_type": retriever_type}
    cfg.update(spec.get("kwargs", {}))
    return cfg


def get_index(purge: bool):
    """Return the index, reusing a populated vector store when purge is off.

    ``build_index`` handles the reuse internally: purge=True truncates and
    re-ingests; purge=False wraps the existing store when populated (no
    re-ingestion) and only ingests an empty store.
    """
    return build_index(purge=purge)


def make_judges(cache_dir: Path | None = None):
    judge = DeepSeekJudge(
        {
            "name": JUDGE_CONFIG["model"],
            "api_key": JUDGE_CONFIG["api_key"],
            "base_url": JUDGE_CONFIG["api_base"],
            "thinking": JUDGE_CONFIG.get("thinking"),
            "timeout": JUDGE_CONFIG.get("timeout", 600.0),
            "max_retries": JUDGE_CONFIG.get("max_retries", 5),
        }
    )
    cache = None
    if cache_dir is not None:
        cache = DiskCacheBackend(cache_dir=str(cache_dir))
        logger.info("Ragas judge cache enabled at %s", cache_dir)
    ragas_llm = llm_factory(
        JUDGE_CONFIG["model"],
        client=AsyncOpenAI(
            api_key=JUDGE_CONFIG["api_key"],
            base_url=JUDGE_CONFIG["api_base"],
            timeout=JUDGE_CONFIG.get("timeout", 600.0),
            max_retries=JUDGE_CONFIG.get("max_retries", 5),
        ),
        max_tokens=16000,
        temperature=JUDGE_CONFIG.get("temperature", 0.0),
        extra_body={
            "thinking": {"type": JUDGE_CONFIG.get("thinking", "disabled")}
        },
        cache=cache,
    )
    return judge, ragas_llm


def _ascore_kwargs(metric, sample: dict) -> dict:
    name = type(metric).__name__
    if name == "ContextPrecision":
        return dict(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    if name == "ContextRecall":
        return dict(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
        )
    if name == "ContextEntityRecall":
        return dict(
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    # ContextRelevance
    return dict(
        user_input=sample["user_input"],
        retrieved_contexts=sample["retrieved_contexts"],
    )


async def _score_stage(
    samples: list[dict],
    metrics: list,
    concurrency: int = 8,
) -> list[dict]:
    """Score all samples with all metrics via ``ascore``.

    A semaphore bounds the number of in-flight judge calls; failures are
    logged and recorded as NaN so one flaky sample doesn't abort the stage.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(metric, sample):
        async with sem:
            try:
                result = await metric.ascore(**_ascore_kwargs(metric, sample))
                return result.value
            except Exception as exc:  # noqa: BLE001 - tolerate judge failures
                logger.warning(
                    "%s ascore failed for %r: %s",
                    type(metric).__name__,
                    sample.get("user_input", "")[:60],
                    exc,
                )
                return float("nan")

    tasks = [one(m, s) for s in samples for m in metrics]
    values = await asyncio.gather(*tasks)

    per_sample: list[dict] = []
    idx = 0
    for _ in samples:
        row = {}
        for metric in metrics:
            row[metric.name] = values[idx]
            idx += 1
        per_sample.append(row)
    return per_sample


def run_ragas(
    samples: list[dict],
    ragas_llm,
    concurrency: int = 8,
) -> list[dict]:
    """Score one stage with the four ragas retrieval metrics."""
    if not samples:
        return []
    metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        ContextEntityRecall(llm=ragas_llm),
        ContextRelevance(llm=ragas_llm),
    ]
    return asyncio.run(_score_stage(samples, metrics, concurrency))


def run_umbrela(
    query: str,
    passages: list[str],
    umbrela_metric,
    k_values: list[int],
) -> dict | None:
    """Score one query/stage with UMBRELA -> TREC metrics.

    Returns None (logged) if the judge fails across retries for any passage,
    so a single flaky passage doesn't abort the whole run.
    """
    retrieval_result = RetrievalResult(
        query=query,
        retrieved_passages={
            str(i): passage for i, passage in enumerate(passages)
        },
    )
    try:
        return umbrela_metric.compute(retrieval_result, k_values)
    except Exception as exc:  # noqa: BLE001 - tolerate judge failures
        logger.warning("UMBRELA failed for %r: %s", query[:80], exc)
        return None


def mean(values: list[float]) -> float:
    values = [v for v in values if v is not None and v == v]  # drop None/NaN
    return round(sum(values) / len(values), 6) if values else None


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None and v == v)
    if not values:
        return None
    import math

    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return round(values[int(k)], 4)
    frac = k - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 4)


def latency_stats(latencies: list[float]) -> dict:
    return {
        "mean": mean(latencies),
        "p50": percentile(latencies, 0.5),
        "p95": percentile(latencies, 0.95),
        "p99": percentile(latencies, 0.99),
    }


def print_summary(report: dict):
    print("\n" + "=" * 78)
    print(f"RAG RETRIEVAL EVALUATION SUMMARY — {report['label']}")
    print("=" * 78)
    for stage in report["aggregates"]:
        ragas_agg = report["aggregates"][stage]["ragas"]
        umbrela_agg = report["aggregates"][stage]["umbrela"]
        print(f"\n[{stage}]")
        print(
            f"  ragas: precision={ragas_agg['context_precision']} "
            f"recall={ragas_agg['context_recall']} "
            f"entity_recall={ragas_agg['context_entity_recall']} "
            f"relevance={ragas_agg['context_relevance']}"
        )
        print(f"  umbrela: {json.dumps(umbrela_agg, indent=2)}")

    lat = report.get("latencies", {})
    if lat:
        print("\n[retrieval latency (s)]")
        for part in lat:
            stats = lat[part]
            print(
                f"  {part:7s} mean={stats.get('mean')} "
                f"p50={stats.get('p50')} p95={stats.get('p95')} p99={stats.get('p99')}"
            )


def main():
    args = parse_args()
    setup_logging()

    dataset = load_testset(args.dataset)
    if args.max_queries:
        dataset = dataset.head(args.max_queries)
        logger.info("Capped to %d queries for smoke run.", len(dataset))

    setup_phoenix()
    import nest_asyncio

    nest_asyncio.apply()

    from opentelemetry import trace

    tracer = trace.get_tracer("evaluate")

    judge, ragas_llm = make_judges(
        cache_dir=None if args.no_cache else args.cache_dir
    )
    umbrela_metric = ParallelUMBRELAMetric(
        judge, max_workers=args.umbrela_workers
    )

    retriever_spec = parse_retriever_spec(args.retriever)
    retrieval_cfg = {**RETRIEVAL_CONFIG, **retriever_spec}
    retriever_type = retrieval_cfg.get("retriever_type", "fusion")
    retriever_class = (
        RecordingFusionRetriever if retriever_type == "fusion" else None
    )
    index = get_index(args.purge)
    retriever = build_retriever(
        index, retrieval_cfg, retriever_class=retriever_class
    )
    reranker = build_reranker(retrieval_cfg)
    stage_names = ["base"] if reranker is None else ["base", "reranked"]
    logger.info(
        "Retriever: %s (%s) | stages: %s",
        retriever_type,
        type(retriever).__name__,
        stage_names,
    )

    per_query_records = []
    t0 = time.time()
    for pos, (_, row) in enumerate(dataset.iterrows(), start=1):
        query = row["user_input"]
        logger.info("[%d/%d] %s", pos, len(dataset), query)
        qb = QueryBundle(query)

        with tracer.start_as_current_span(f"eval_query_{pos}") as span:
            span.set_attribute("label", args.label)
            span.set_attribute("qa.question", query)
            span.set_attribute("eval.num_queries", len(dataset))
            span.set_attribute("eval.persona", str(row.get("persona_name", "")))
            span.set_attribute("eval.query_style", str(row.get("query_style", "")))
            span.set_attribute("eval.query_length", str(row.get("query_length", "")))

            t_base = time.perf_counter()
            base_nodes = retriever.retrieve(qb)
            base_latency = time.perf_counter() - t_base
            sub_queries = list(getattr(retriever, "last_sub_queries", []))
            base_texts = [n.node.get_content() for n in base_nodes]

            latencies = {"base": round(base_latency, 4)}
            stages = [("base", base_texts, args.k_values_base)]
            if reranker is not None:
                t_rerank = time.perf_counter()
                reranked_nodes = reranker.postprocess_nodes(base_nodes, qb)
                latencies["rerank"] = round(time.perf_counter() - t_rerank, 4)
                stages.append(
                    ("reranked",
                     [n.node.get_content() for n in reranked_nodes],
                     args.k_values_reranked)
                )

            record = {
                "user_input": query,
                "sub_queries": sub_queries,
                "latencies": latencies,
                "reference": str(row.get("reference", "")),
                "persona_name": row.get("persona_name"),
                "query_style": row.get("query_style"),
                "query_length": row.get("query_length"),
                "synthesizer_name": row.get("synthesizer_name"),
                "reference_contexts": row["reference_contexts"],
                "stages": {},
            }

            for stage_name, texts, k_vals in stages:
                umbrela = run_umbrela(query, texts, umbrela_metric, k_vals)
                if umbrela is None:
                    record["stages"][stage_name] = {
                        "retrieved_contexts": texts,
                        "umbrela_scores": {},
                        "umbrela": None,
                        "token_usage": None,
                        "ragas": None,
                    }
                    span.set_attribute(f"eval.{stage_name}.umbrela.MRR", -1)
                    span.set_attribute(f"eval.{stage_name}.num_retrieved", len(texts))
                    continue
                umbrela_scores = umbrela.pop("umbrela_scores")
                retrieval_scores = umbrela.pop("retrieval_scores")
                record["stages"][stage_name] = {
                    "retrieved_contexts": texts,
                    "umbrela_scores": {
                        int(k): v for k, v in umbrela_scores.items()
                    },
                    "umbrela": {
                        "precision@": {str(k): retrieval_scores["precision@"].get(str(k)) for k in k_vals},
                        "AP@": {str(k): retrieval_scores["AP@"].get(str(k)) for k in k_vals},
                        "MRR": retrieval_scores.get("MRR"),
                    },
                    "token_usage": umbrela,
                }
                mrr = record["stages"][stage_name]["umbrela"]["MRR"]
                span.set_attribute(
                    f"eval.{stage_name}.umbrela.MRR", -1 if mrr is None else mrr
                )
                span.set_attribute(
                    f"eval.{stage_name}.num_retrieved", len(texts)
                )

        per_query_records.append(record)

    logger.info("Retrieval done in %.1fs. Scoring ragas metrics ...", time.time() - t0)

    # ── ragas per stage ──────────────────────────────────────────────────
    ragas_per_query = {s: [] for s in stage_names}
    for stage_name in stage_names:
        samples = [
            {
                "user_input": rec["user_input"],
                "retrieved_contexts": rec["stages"][stage_name]["retrieved_contexts"],
                "reference": rec["reference"],
            }
            for rec in per_query_records
        ]
        scores = run_ragas(samples, ragas_llm, concurrency=args.ragas_workers)
        for rec, score in zip(per_query_records, scores):
            rec["stages"][stage_name]["ragas"] = score
            ragas_per_query[stage_name].append(score)

    # ── aggregate ────────────────────────────────────────────────────────
    def mean_dict_over_keys(rows: list[dict], keys: list[str]) -> dict:
        return {k: mean([r.get(k) for r in rows]) for k in keys}

    def stage_aggregates(ragas_rows: list[dict], umbrela_rows: list[dict]) -> dict:
        precision_keys = sorted(
            {k for r in umbrela_rows for k in r.get("precision@", {})}
        )
        ap_keys = sorted({k for r in umbrela_rows for k in r.get("AP@", {})})
        return {
            "ragas": mean_dict_over_keys(ragas_rows, RAGAS_METRICS),
            "umbrela": {
                "precision@": mean_dict_over_keys(
                    [r.get("precision@", {}) for r in umbrela_rows], precision_keys
                ),
                "AP@": mean_dict_over_keys(
                    [r.get("AP@", {}) for r in umbrela_rows], ap_keys
                ),
                "MRR": mean([r.get("MRR") for r in umbrela_rows]),
            },
        }

    aggregates = {
        stage: stage_aggregates(
            ragas_per_query[stage],
            [rec["stages"][stage]["umbrela"] for rec in per_query_records],
        )
        for stage in stage_names
    }

    # ── strata ───────────────────────────────────────────────────────────
    strata = {}
    for col in ("persona_name", "query_style", "query_length"):
        if col not in dataset.columns:
            continue
        grouped: dict[str, list[int]] = {}
        for idx in dataset.index:
            grouped.setdefault(str(dataset.at[idx, col]), []).append(int(idx))
        stage_strata = {}
        for stage in stage_names:
            stage_strata[stage] = {
                value: stage_aggregates(
                    [ragas_per_query[stage][i] for i in idxs],
                    [per_query_records[i]["stages"][stage]["umbrela"] for i in idxs],
                )
                for value, idxs in grouped.items()
            }
        strata[col] = stage_strata

    report = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_config": {**JUDGE_CONFIG, "api_key": "***"},
        "retriever": {
            "type": retriever_type,
            "class": type(retriever).__name__,
            "kwargs": {
                k: v for k, v in retriever_spec.items() if k != "retriever_type"
            },
        },
        "retrieval_config": {
            k: v for k, v in retrieval_cfg.items() if k not in ("retriever_kwargs",)
        },
        "k_values": {
            stage: args.k_values_base if stage == "base" else args.k_values_reranked
            for stage in stage_names
        },
        "num_queries": len(per_query_records),
        "dataset": str(args.dataset),
        "aggregates": aggregates,
        "latencies": {
            part: latency_stats([r["latencies"][part] for r in per_query_records])
            for part in ("base", "rerank")
            if part in per_query_records[0]["latencies"]
        },
        "strata": strata,
        "per_query": per_query_records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.label}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote report to %s", out_path)

    print_summary(report)


if __name__ == "__main__":
    main()
