"""Purge and ingest the docs into the PGVectorStore.

This is the only entrypoint that writes to the vector store. The RAG pipeline
(and its API workers) only reads the existing store via ``build_index()``.

Usage:
    python -m app.ingest                 # purge, then ingest docs from DOCS_DIR
    python -m app.ingest --no-purge      # append to the existing store
    python -m app.ingest --docs-dir /path/to/pdfs
"""

import argparse
import logging

from llama_index.core import StorageContext, VectorStoreIndex

from app.config import DOCS_DIR
from app.rag_pipeline import (
    build_vector_store,
    configure_models,
    setup_logging,
    truncate_vector_store,
    vector_store_row_count,
)

logger = logging.getLogger(__name__)


def ingest(docs_dir: str = DOCS_DIR, purge: bool = True):
    """Embed and write all docs in ``docs_dir`` into the vector store.

    Args:
        docs_dir: directory containing the PDFs to index.
        purge: if True, truncate the vector store table before ingesting.
    """
    configure_models()
    vector_store = build_vector_store()

    if purge:
        truncate_vector_store()
    else:
        count = vector_store_row_count()
        logger.info("Appending to existing vector store (%d rows).", count)

    from app.utils import DoclingIngester

    nodes = DoclingIngester(docs_dir=docs_dir).ingest()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("Ingested %d nodes into the vector store.", len(nodes))
    return index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Purge and ingest docs into the vector store."
    )
    parser.add_argument(
        "--docs-dir",
        default=DOCS_DIR,
        help=f"Directory containing the PDFs to index (default: {DOCS_DIR}).",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Append to the existing vector store instead of truncating it first.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    ingest(docs_dir=args.docs_dir, purge=not args.no_purge)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
