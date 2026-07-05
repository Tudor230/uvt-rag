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
    re.compile(r"^(?:\(\d+\)\s+)?Program(ul|ele)\s+de\s+studii\s+universitare\s+de\s+licență.+"),
    re.compile(r"^Anexa\s+\d+\."),
    re.compile(r"^CAPITOLUL\s"),
    re.compile(r"^[A-Z]\)\s+[A-Z\sȘȚĂÎÂ\-]+"),
    re.compile(r"^[I|V|X]+\.\s+[A-Z\sȘȚĂÎÂ]+"),
    re.compile(r"^\d+\.\s+METODOLOGIA\s+DE\s+DESFĂȘURARE.+"),
    re.compile(r"^Regulamentul\s+privind\s+")
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

    @staticmethod
    def _reorder_body_children(doc_dict: dict) -> dict:
        items_by_ref = {}
        for category in ("texts", "tables", "pictures", "groups"):
            for item in doc_dict.get(category, []):
                items_by_ref[item["self_ref"]] = item

        def _first_prov(item: dict):
            prov = item.get("prov", [])
            if isinstance(prov, list) and prov:
                return prov[0]
            queue = list(item.get("children", []))
            while queue:
                child_ref_dict = queue.pop(0)
                child_ref = child_ref_dict.get("$ref", child_ref_dict) if isinstance(child_ref_dict, dict) else child_ref_dict
                child = items_by_ref.get(child_ref, {})
                prov = child.get("prov", [])
                if isinstance(prov, list) and prov:
                    return prov[0]
                grandchildren = child.get("children", [])
                if grandchildren:
                    queue.extend(grandchildren)
            return None

        if "body" not in doc_dict or "children" not in doc_dict["body"]:
            return doc_dict

        children = doc_dict["body"]["children"]

        from collections import defaultdict

        page_items = defaultdict(list)
        child_info = []
        for idx, child in enumerate(children):
            ref = child.get("$ref", "")
            item = items_by_ref.get(ref, {})
            p = _first_prov(item)
            if p:
                page = p.get("page_no", 0) or 0
                y = p.get("bbox", {}).get("t", 0)
                x = p.get("bbox", {}).get("l", 0)
            else:
                page, y, x = 0, 0, 0
            child_info.append((child, page, y, x))
            page_items[page].append((idx, y, x))

        two_column_pages = set()
        for page, items in page_items.items():
            for i in range(len(items)):
                y1, x1 = items[i][1], items[i][2]
                for j in range(i + 1, len(items)):
                    y2, x2 = items[j][1], items[j][2]
                    if abs(y1 - y2) <= 2 and abs(x1 - x2) > 150:
                        two_column_pages.add(page)
                        break
                if page in two_column_pages:
                    break

        def _sort_key(info):
            _, page, y, x = info
            if page in two_column_pages:
                pg_items = page_items[page]
                min_x = min(it[2] for it in pg_items)
                max_x = max(it[2] for it in pg_items)
                mid_x = (min_x + max_x) / 2
                col = 0 if x < mid_x else 1
                return (page, col, -y)
            return (page, -y, x)

        child_info.sort(key=_sort_key)
        doc_dict["body"]["children"] = [info[0] for info in child_info]

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
        items_to_delete = []
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
                is_page_header = bool(re.match(r"^Regulamentul\s+(?!privind).{50,}", text))

                if is_heading and item.label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM):
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
                            label=item.label, 
                            text=remaining,
                            prov=item.prov[0] if item.prov else None,
                            parent=dl_doc.body,
                        )
                    items_to_delete.append(item)
                    continue

                if is_formula and item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.FORMULA, DocItemLabel.TITLE):
                    dl_doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text,
                        prov=item.prov[0] if item.prov else None,
                        parent=dl_doc.body,
                    )
                    items_to_delete.append(item)
                    continue

                if item.label is DocItemLabel.SECTION_HEADER and (not is_heading or is_page_header):
                    dl_doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text,
                        prov=item.prov[0] if item.prov else None,
                        parent=dl_doc.body,
                    )
                    items_to_delete.append(item)

            for target_item in reversed(items_to_delete):
                try:
                    dl_doc.delete_items_range(
                        start=target_item,
                        end=target_item,
                        start_inclusive=True,
                        end_inclusive=True
                    )
                except Exception as e:
                    logger.debug("Skipping native deletion for item due to tree structure constraints: %s", e)

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
