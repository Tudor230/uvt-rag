import json
import logging
import warnings
from pathlib import Path
from typing import Optional
import re

from docling.chunking import HybridChunker, BaseChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import TextNode, Document
from docling_core.types.doc.labels import DocItemLabel
logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_TOKENS = 1024

_HEADING_PATTERNS = [
    re.compile(r"^Articolul\s+\d+\."),
    re.compile(r"^Programul\s+de\s+studii\s+universitare\s+de\s+licență:.+"),
    re.compile(r"^Anexa\s+\d+\."),
    re.compile(r"^CAPITOLUL\s"),
]


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

    #TODO: This is a workaround for a bug in Docling where the body children are not ordered correctly.
    #It brakes two column layouts and other complex layouts.
    @staticmethod
    def _reorder_body_children(doc_dict: dict) -> dict:
        items_by_ref = {}
        for category in ("texts", "tables", "pictures", "groups"):
            for item in doc_dict.get(category, []):
                items_by_ref[item["self_ref"]] = item

        def _sort_key(ref_dict):
            ref = ref_dict.get("$ref", "")
            item = items_by_ref.get(ref, {})
            prov = item.get("prov", [])
            if isinstance(prov, list) and prov:
                p = prov[0]
                page = p.get("page_no", 0) or 0
                y = p.get("bbox", {}).get("t", 0)
                return (page, -y)
            children = item.get("children", [])
            if children:
                first = children[0]
                first_ref = first.get("$ref", first) if isinstance(first, dict) else first
                child = items_by_ref.get(first_ref, {})
                child_prov = child.get("prov", [])
                if isinstance(child_prov, list) and child_prov:
                    cp = child_prov[0]
                    return (cp.get("page_no", 0) or 0, -(cp.get("bbox", {}).get("t", 0)))
            return (0, 0)

        if "body" in doc_dict and "children" in doc_dict["body"]:
            doc_dict["body"]["children"].sort(key=_sort_key)

        return doc_dict

    def ingest(self) -> list[TextNode]:
        pdf_paths = sorted(self._docs_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found in %s", self._docs_dir)
            return []

        logger.info("Loading %d PDFs from %s with DoclingReader ...", len(pdf_paths), self._docs_dir)
        # documents = list(self._reader.load_data(file_path=pdf_paths))
        # logger.info("Loaded %d documents.", len(documents))
        documents = []
        for path in pdf_paths:
            # 1. Parse the document natively with Docling
            result = self._doc_converter.convert(path)
            dl_doc = result.document

            if (self.debug == True):
                inspection_data = []
                
                for item, level in dl_doc.iterate_items():
                    item_dict = item.model_dump(exclude_none=True, mode="json")
                    
                    item_dict["_docling_level"] = level 
                    
                    inspection_data.append(item_dict)
                
                export_path = path.with_name(f"{path.stem}_inspection.json")
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(inspection_data, f, indent=2)
            
            # 2. Intercept and fix misclassified layout elements
            for item, level in dl_doc.iterate_items():
                if hasattr(item, "text") and isinstance(item.text, str):
                    cleaned_text = re.sub(r"(?i)pagina\s+\d+\s+din\s+\d+", "", item.text).strip()
                    item.text = cleaned_text

                text = getattr(item, "text", "").strip()
                is_heading = any(p.match(text) for p in _HEADING_PATTERNS)
                is_formula = ("=" in text) or ("unde" in text) and len(text) < 150

                if is_heading and item.label is DocItemLabel.TEXT:
                    heading_text = text
                    remaining = ""
                    if "\n" in text:
                        heading_text, _, remaining = text.partition("\n")
                        heading_text = heading_text.strip()
                        remaining = remaining.strip()
                    else:
                        for p in _HEADING_PATTERNS:
                            m = p.match(text)
                            if m and m.end() < len(text):
                                heading_text = m.group(0).strip()
                                remaining = text[m.end():].strip()
                                break
                    dl_doc.add_heading(
                        text=heading_text,
                        prov=item.prov[0] if item.prov else None,
                        parent=dl_doc.body,
                    )
                    if remaining:
                        dl_doc.add_text(
                            label=DocItemLabel.TEXT,
                            text=remaining,
                            prov=item.prov[0] if item.prov else None,
                            parent=dl_doc.body,
                        )
                    dl_doc.body.children = [
                        c for c in dl_doc.body.children
                        if getattr(c, "cref", "") != item.self_ref
                    ]
                    continue

                if is_formula and item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.FORMULA, DocItemLabel.TITLE):
                    dl_doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text,
                        prov=item.prov[0] if item.prov else None,
                        parent=dl_doc.body,
                    )
                    dl_doc.body.children = [
                        c for c in dl_doc.body.children
                        if getattr(c, "cref", "") != item.self_ref
                    ]
                    continue

                if item.label is DocItemLabel.SECTION_HEADER and not is_heading:
                    dl_doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text,
                        prov=item.prov[0] if item.prov else None,
                        parent=dl_doc.body,
                    )
                    dl_doc.body.children = [
                        c for c in dl_doc.body.children
                        if getattr(c, "cref", "") != item.self_ref
                    ]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                doc_dict = dl_doc.export_to_dict()

            refs_in_tree = {c.get("$ref", "") for c in doc_dict.get("body", {}).get("children", [])}
            for item in doc_dict.get("texts", []):
                item.pop("level", None)
                item.pop("enumerated", None)
                item.pop("marker", None)
                if item["self_ref"] not in refs_in_tree and item.get("label") == "section_header":
                    item["label"] = "text"
            for group in doc_dict.get("groups", []):
                group.pop("level", None)

            doc_dict = self._reorder_body_children(doc_dict)
            llama_doc = Document(text=json.dumps(doc_dict))
            documents.append(llama_doc)

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
