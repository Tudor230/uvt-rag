import json
import logging
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
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

        self._node_parser = DoclingNodeParser(chunker=chunker)

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
                if item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.FORMULA, DocItemLabel.TITLE):
                    text = getattr(item, "text", "").strip()
                    
                    if ("=" in text) or ("unde" in text) and len(text) < 150:
                        # Demote it to standard text so HybridChunker merges it
                        item.label = DocItemLabel.TEXT 

            doc_dict = dl_doc.export_to_dict()
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
    args = parser.parse_args()

    docs_dir = args.docs_dir or os.getenv("DOCS_DIR", "./docs")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

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
