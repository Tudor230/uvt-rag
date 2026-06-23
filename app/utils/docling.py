import logging
from pathlib import Path
from typing import Optional

from docling.chunking import HybridChunker, BaseChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_TOKENS = 1024


class DoclingIngester:
    def __init__(
        self,
        docs_dir: Path,
        doc_converter: Optional[DocumentConverter] = None,
        chunker: Optional[BaseChunker] = None,
        max_tokens: int = _DEFAULT_CHUNK_TOKENS,
    ):
        self._docs_dir = Path(docs_dir)
        self._reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            doc_converter=doc_converter or DocumentConverter(),
        )

        if chunker is None:
            tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name="BAAI/bge-m3",
                max_tokens=max_tokens,
            )
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

        self._node_parser = DoclingNodeParser(chunker=chunker)

    def ingest(self) -> list[TextNode]:
        pdf_paths = sorted(self._docs_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found in %s", self._docs_dir)
            return []

        logger.info("Loading %d PDFs from %s with DoclingReader ...", len(pdf_paths), self._docs_dir)
        documents = list(self._reader.load_data(file_path=pdf_paths))
        logger.info("Loaded %d documents.", len(documents))

        logger.info("Parsing documents with DoclingNodeParser ...")
        nodes = self._node_parser.get_nodes_from_documents(documents)
        logger.info("Produced %d nodes.", len(nodes))

        return nodes


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import textwrap

    parser = argparse.ArgumentParser(description="Preview Docling chunks from a PDF directory.")
    parser.add_argument(
        "docs_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory containing PDFs (default: DOCS_DIR env var or ./docs)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10,
        help="Max chunks to print (default: 10).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_CHUNK_TOKENS,
        help="Max tokens per chunk (default: %(default)s).",
    )
    args = parser.parse_args()

    docs_dir = args.docs_dir or os.getenv("DOCS_DIR", "./docs")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

    ingester = DoclingIngester(docs_dir=docs_dir, max_tokens=args.max_tokens)
    nodes = ingester.ingest()

    limit = min(args.max_nodes, len(nodes))
    width = max(len(str(len(nodes))), 3)

    for i, node in enumerate(nodes[:limit]):
        meta = node.metadata or {}
        headings = meta.get("headings", [])
        heading_str = " > ".join(headings) if headings else "(no heading)"

        page = "?"
        for doc_item in meta.get("doc_items", []):
            for prov in doc_item.get("prov", []):
                if "page_no" in prov:
                    page = prov["page_no"]
                    break

        print(f"\n{'=' * 80}")
        print(f"Chunk {i + 1:>{width}} / {len(nodes)}  |  page {page}  |  {heading_str}")
        print(f"{'=' * 80}")
        print(textwrap.fill(node.get_content(), width=100))

    print(f"\n{'=' * 80}")
    print(f"Total chunks: {len(nodes)}, avg text len: {sum(len(n.get_content()) for n in nodes) // max(len(nodes), 1)} chars")
    print(f"Showing {limit}/{len(nodes)} chunks")
