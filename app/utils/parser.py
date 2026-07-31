import json
import logging
import warnings
from pathlib import Path
from typing import Optional

from docling.chunking import HybridChunker, BaseChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import TextNode, Document

from app.utils.postprocessing import GLOBAL_STEPS, AttachHyperlinks
from app.utils.preprocessing import (
    DEFERRED_STEPS,
    DICT_STEPS,
    ITEM_STEPS,
    PreprocessContext,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_TOKENS = 1024


class DoclingIngester:
    def __init__(
        self,
        docs_dir: Path,
        doc_converter: Optional[DocumentConverter] = None,
        chunker: Optional[BaseChunker] = None,
        max_tokens: int = _DEFAULT_CHUNK_TOKENS,
        debug: bool = False,
    ):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_formula_enrichment = True
        self.pipeline_options.ocr_options = EasyOcrOptions(lang=["ro", "en"], use_gpu=True)
        self.pipeline_options.images_scale = 2.0
        self.pipeline_options.generate_parsed_pages = True
        self._docs_dir = Path(docs_dir)
        self._doc_converter = doc_converter or DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        self._reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            doc_converter=self._doc_converter
        )
        self.debug = debug

        if chunker is None:
            tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name="BAAI/bge-m3",
                max_tokens=max_tokens,
            )
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, always_emit_headings=True)

        self._node_parser = DoclingNodeParser(chunker=chunker)

    def ingest(self) -> list[TextNode]:
        pdf_paths = sorted(self._docs_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found in %s", self._docs_dir)
            return []

        logger.info("Loading %d PDFs from %s with DoclingReader ...", len(pdf_paths), self._docs_dir)
        documents = []
        items_to_delete = []
        for path in pdf_paths:
            # 1. Parse the document natively with Docling
            result = self._doc_converter.convert(path)
            dl_doc = result.document

            if self.debug:
                inspection_data = []

                for item, level in dl_doc.iterate_items():
                    item_dict = item.model_dump(exclude_none=True, mode="json")

                    item_dict["_docling_level"] = level

                    inspection_data.append(item_dict)

                export_path = path.with_name(f"{path.stem}_inspection.json")
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(inspection_data, f, indent=2)

            # 2. Pre-process: intercept and fix misclassified layout elements
            ctx = PreprocessContext(dl_doc, items_to_delete)
            for item, level in dl_doc.iterate_items():
                for step in ITEM_STEPS:
                    if step.apply(item, ctx):
                        break
            for step in DEFERRED_STEPS:
                step.apply(ctx)

            # 3. Export and post-process the doc dict
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                doc_dict = dl_doc.export_to_dict()

            for step in DICT_STEPS:
                doc_dict = step.apply(doc_dict)

            llama_doc = Document(text=json.dumps(doc_dict))

            # 4. Chunk this single document
            nodes = self._node_parser.get_nodes_from_documents([llama_doc])

            # 5. Attach hyperlinks for this document's chunks
            nodes = AttachHyperlinks().apply(nodes, result, path.name)

            documents.append((path, nodes))

        logger.info("Chunking done. Processing %d documents...", len(documents))

        # Collect all nodes
        all_nodes = []
        for _path, nodes in documents:
            all_nodes.extend(nodes)

        logger.info("Produced %d nodes.", len(all_nodes))

        # 6. Post-process the chunk collection
        for step in GLOBAL_STEPS:
            all_nodes = step.apply(all_nodes)

        return all_nodes


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
    parser.add_argument(
        "--export-md",
        type=Path,
        default=None,
        metavar="DIR",
        help="Export Docling markdown to DIR instead of chunking.",
    )
    args = parser.parse_args()

    docs_dir = args.docs_dir or os.getenv("DOCS_DIR", "./docs")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

    if args.export_md:
        out_dir = args.export_md
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_formula_enrichment = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        pdfs = sorted(Path(docs_dir).glob("*.pdf"))
        for path in pdfs:
            result = converter.convert(path)
            md_text = result.document.export_to_markdown()
            out_path = out_dir / f"{path.stem}.md"
            out_path.write_text(md_text, encoding="utf-8")
            print(f"{path.name} -> {out_path}  ({len(md_text)} chars)")
        sys.exit(0)

    ingester = DoclingIngester(docs_dir=docs_dir, max_tokens=args.max_tokens, debug=True)
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
