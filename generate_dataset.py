import argparse
import asyncio
import concurrent.futures
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from openai import AsyncOpenAI

load_dotenv()

from ragas.embeddings import HuggingFaceEmbeddings as RagasEmbeddings
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, NodeType
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import Parallel
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


def build_chunks(docs_path: Path) -> list[str]:
    # TODO: Add proper chunking
    logging.info("Loading documents from %s ...", docs_path)
    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    logging.info("Loaded %d documents.", len(documents))

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    chunks = [node.get_content() for node in nodes]
    logging.info("Produced %d chunks (chunk_size=512, overlap=50).", len(chunks))
    return chunks


def setup_llm():
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=_LLM_BASE_URL,
        timeout=120.0,
        max_retries=3,
    )
    ragas_llm = llm_factory(
        _LLM_MODEL,
        client=client,
        max_tokens=4096,
    )
    logging.info("Granite LLM ready (%s) via llm_factory.", _LLM_MODEL)
    return ragas_llm


def setup_embeddings() -> RagasEmbeddings:
    ragas_embeddings = RagasEmbeddings(
        model=_EMBED_MODEL,
        device=_EMBED_DEVICE,
    )
    logging.info("BGE-M3 embeddings ready (device=%s).", _EMBED_DEVICE)
    return ragas_embeddings



def build_minimal_transforms(llm, embeddings):
    """
    Only the transforms actually needed by SingleHopSpecificQuerySynthesizer.
    ThemesExtractor is skipped — it causes connection errors with Granite.
    """
    filter_chunks = lambda n: n.type == NodeType.CHUNK
    return [
        SummaryExtractor(llm=llm, filter_nodes=filter_chunks),
        Parallel(
            EmbeddingExtractor(
                embedding_model=embeddings,
                property_name="summary_embedding",
                embed_property_name="summary",
                filter_nodes=filter_chunks,
            ),
            NERExtractor(llm=llm, filter_nodes=filter_chunks),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate a RAGAS testset from PDF documents.")
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
        default=TESTSET_SIZE,
        help="Number of test samples to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_CSV),
        metavar="PATH",
        help="Path for the output CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Max parallel workers for transforms (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max concurrent query generation calls (default: %(default)s).",
    )
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ragas_llm = setup_llm()
    ragas_embeddings = setup_embeddings()

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    run_config = RunConfig(max_workers=args.max_workers)

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0)
    ]

    # ── Generate ──────────────────────────────────────────────────────────
    if args.load_kg:
        generator.knowledge_graph = load_knowledge_graph(args.load_kg)
        logging.info(
            "Generating %d test samples from cached KG (max_workers=%d, batch_size=%d) ...",
            args.size, args.max_workers, args.batch_size,
        )
        testset = generator.generate(
            testset_size=args.size,
            query_distribution=query_distribution,
            run_config=run_config,
            batch_size=args.batch_size,
        )
    else:
        chunks = build_chunks(Path(DOCS_DIR))
        logging.info(
            "Generating %d test samples (max_workers=%d) ...",
            args.size, args.max_workers,
        )
        testset = generator.generate_with_chunks(
            chunks=chunks,
            testset_size=args.size,
            query_distribution=query_distribution,
            run_config=run_config,
            transforms=build_minimal_transforms(ragas_llm, ragas_embeddings),
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
