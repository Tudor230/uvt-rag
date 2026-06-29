import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()

from ragas.embeddings import HuggingFaceEmbeddings as RagasEmbeddings
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer, MultiHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer
from ragas.testset.transforms import Parallel, apply_transforms, default_transforms_for_prechunked
from ragas.testset.transforms.extractors import EmbeddingExtractor, NERExtractor, SummaryExtractor

logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(os.path.dirname(__file__), "docs-ragas"))
OUTPUT_CSV = os.getenv("RAGAS_OUTPUT", os.path.join(os.path.dirname(__file__), "data", "testset.csv"))
TESTSET_SIZE = 50

_EMBED_DEVICE = "cuda"
_EMBED_MODEL = "BAAI/bge-m3"
_LLM_MODEL = "ibm-granite/granite-4.1-8b"
_LLM_BASE_URL = "http://localhost:8000/v1"


def load_knowledge_graph(kg_path: Path) -> KnowledgeGraph:
    logging.info("Loading knowledge graph from %s ...", kg_path)
    kg = KnowledgeGraph.load(str(kg_path))
    logging.info("Loaded %d nodes, %d relationships.", len(kg.nodes), len(kg.relationships))
    return kg


def build_chunks(docs_path: Path) -> list:
    logging.info("Loading documents from %s with Docling ...", docs_path)
    from app.utils import DoclingIngester

    nodes = DoclingIngester(docs_dir=docs_path).ingest()
    logging.info("Produced %d chunks.", len(nodes))
    return nodes


def setup_llm(llm_model: str | None = None, llm_base_url: str | None = None):
    model = llm_model or _LLM_MODEL
    url = llm_base_url or _LLM_BASE_URL
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=url,
        timeout=120.0,
        max_retries=3,
    )
    ragas_llm = llm_factory(
        model,
        client=client,
        max_tokens=4096,
    )
    logging.info("LLM ready (%s) via llm_factory.", model)
    return ragas_llm


def setup_embeddings(embed_model: str | None = None, embed_device: str | None = None) -> RagasEmbeddings:
    ragas_embeddings = RagasEmbeddings(
        model=embed_model or _EMBED_MODEL,
        device=embed_device or _EMBED_DEVICE,
    )
    logging.info("Embeddings ready (device=%s).", embed_device or _EMBED_DEVICE)
    return ragas_embeddings



def load_personas_from_config(config: dict) -> list[Persona]:
    """
    Build a flat list of Persona objects from the YAML config,
    duplicating entries according to their weight so that weighted
    sampling naturally follows the declared proportions.
    """
    flat: list[Persona] = []
    for entry in config["personas"]:
        p = Persona(name=entry["name"], role_description=entry["role_description"])
        w = entry.get("weight", 1)
        flat.extend([p] * w)
    return flat


def main():
    parser = argparse.ArgumentParser(description="Generate a RAGAS testset from PDF documents.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML config with defaults and custom personas.",
    )
    parser.add_argument(
        "--load-kg",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load a pre-built knowledge graph JSON instead of building from documents.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Number of test samples to generate (overrides config / default %d)." % TESTSET_SIZE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the output CSV (overrides config / default).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers for transforms (overrides config / default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Max concurrent query generation calls (overrides config / default).",
    )
    args_raw = parser.parse_args()

    persona_list: list[Persona] | None = None

    if args_raw.config:
        config_path = args_raw.config
        logging.info("Loading config from %s", config_path)
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        defaults = config.get("defaults", {})
        base = config_path.parent

        testset_size = args_raw.size or defaults.get("testset_size", TESTSET_SIZE)
        docs_dir = defaults.get("docs_dir", DOCS_DIR)
        output_file = args_raw.output or Path(
            defaults.get("output_csv", OUTPUT_CSV)
        )
        if not output_file.is_absolute():
            output_file = base / output_file

        embed_device = defaults.get("embed_device", _EMBED_DEVICE)
        embed_model = defaults.get("embed_model", _EMBED_MODEL)
        llm_model = defaults.get("llm_model", _LLM_MODEL)
        llm_base_url = defaults.get("llm_base_url", _LLM_BASE_URL)
        max_workers = args_raw.max_workers or defaults.get("max_workers", 32)
        batch_size = args_raw.batch_size or defaults.get("batch_size", 8)

        persona_list = load_personas_from_config(config)
        logging.info(
            "Loaded %d persona entries (weighted from %d personas).",
            len(persona_list),
            len(config.get("personas", [])),
        )
    else:
        testset_size = args_raw.size or TESTSET_SIZE
        docs_dir = DOCS_DIR
        output_file = args_raw.output or Path(OUTPUT_CSV)
        embed_device = _EMBED_DEVICE
        embed_model = _EMBED_MODEL
        llm_model = _LLM_MODEL
        llm_base_url = _LLM_BASE_URL
        max_workers = args_raw.max_workers or 32
        batch_size = args_raw.batch_size or 8

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ragas_llm = setup_llm(llm_model=llm_model, llm_base_url=llm_base_url)
    ragas_embeddings = setup_embeddings(embed_model=embed_model, embed_device=embed_device)

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )
    if persona_list:
        generator.persona_list = persona_list

    run_config = RunConfig(max_workers=max_workers)

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.5),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.3),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.2)
    ]

    # ── Generate ──────────────────────────────────────────────────────────
    if args_raw.load_kg:
        generator.knowledge_graph = load_knowledge_graph(args_raw.load_kg)
        logging.info(
            "Generating %d test samples from cached KG (max_workers=%d, batch_size=%d) ...",
            testset_size, max_workers, batch_size,
        )
        testset = generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            num_personas=len(persona_list) if persona_list else 3,
            run_config=run_config,
            batch_size=batch_size,
        )
    else:
        chunks = build_chunks(Path(docs_dir))
        nodes = []
        for chunk in chunks:
            page_content = chunk.get_content()
            metadata = chunk.metadata or {}
            if page_content is not None and page_content.strip() != "":
                node = Node(
                    type=NodeType.CHUNK,
                    properties={
                        "page_content": page_content,
                        "document_metadata": metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)
        apply_transforms(kg, default_transforms_for_prechunked(ragas_llm, ragas_embeddings), run_config=run_config)
        generator.knowledge_graph = kg

        logging.info(
            "Generating %d test samples (max_workers=%d, batch_size=%d) ...",
            testset_size, max_workers, batch_size,
        )
        testset = generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            num_personas=len(persona_list) if persona_list else 3,
            run_config=run_config,
            batch_size=batch_size,
        )

    logging.info("Generation complete. Collected %d samples.", len(testset))

    # ── Save knowledge graph ──────────────────────────────────────────────
    kg_path = output_path.with_name("knowledge_graph.json")
    generator.knowledge_graph.save(str(kg_path))
    logging.info("Saved knowledge graph to %s", kg_path)

    # ── Export to CSV ─────────────────────────────────────────────────────
    df = testset.to_pandas()
    df.to_csv(str(output_path), index=False)
    logging.info("Saved %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
