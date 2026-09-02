import logging
import re

from llama_index.core.schema import TextNode

from app.utils.preprocessing import _ANNEX_PATTERN, _HEADING_PATTERNS

logger = logging.getLogger(__name__)

_HL_URL_RE = re.compile(r'(https?://(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s,.!?;)\]]*)?)', re.IGNORECASE)
_HL_WWW_RE = re.compile(r'(www\.[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s,.!?;)\]]*)?)', re.IGNORECASE)
_HL_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')


class AttachHyperlinks:
    """Extracts page hyperlinks from the parsed PDF pages and attaches them to chunks.

    Applied per document, right after chunking. Works on the converter ``result``,
    which is still available at that point.
    """

    name = "attach_hyperlinks"

    def apply(self, nodes: list[TextNode], result, doc_name: str) -> list[TextNode]:
        page_heights = {}
        page_hls = {}
        for p in result.pages:
            if p.size:
                page_heights[p.page_no] = p.size.height
            parsed = getattr(p, 'parsed_page', None)
            if parsed and parsed.hyperlinks:
                h = page_heights.get(p.page_no)
                if h:
                    links = []
                    for hl in parsed.hyperlinks:
                        if hl.uri:
                            bbox = hl.rect.to_bounding_box().to_top_left_origin(h)
                            links.append((str(hl.uri), bbox))
                    if links:
                        page_hls[p.page_no] = links
        if page_hls:
            total = sum(len(v) for v in page_hls.values())
            logger.info("Extracted %d hyperlinks from parsed pages for %s", total, doc_name)

        for node in nodes:
            content = node.get_content()
            hyperlinks = []
            seen = set()
            for m in _HL_URL_RE.finditer(content):
                url = m.group(0)
                hyperlinks.append({"text": url, "url": url})
                seen.add(url)
            for m in _HL_WWW_RE.finditer(content):
                url = m.group(0)
                if url not in seen:
                    hyperlinks.append({"text": url, "url": f"http://{url}"})
                    seen.add(url)
            for m in _HL_EMAIL_RE.finditer(content):
                email = m.group(0)
                if email not in seen:
                    hyperlinks.append({"text": email, "url": f"mailto:{email}"})
                    seen.add(email)
            if page_hls:
                for doc_item in node.metadata.get("doc_items", []):
                    for prov in doc_item.get("prov", []):
                        pno = prov.get("page_no")
                        if pno not in page_hls:
                            continue
                        for uri_str, _ in page_hls[pno]:
                            if uri_str in seen:
                                continue
                            from urllib.parse import urlparse, unquote
                            parsed = urlparse(uri_str)
                            norm = re.sub(r'\s+', ' ', content)
                            anchor = uri_str
                            if parsed.hostname:
                                host = parsed.hostname
                                if host in norm:
                                    anchor = host
                                path = unquote(parsed.path.rstrip('/'))
                                for seg in reversed(path.split('/')):
                                    if not seg:
                                        continue
                                    clean = re.sub(r'\s+', ' ', seg.replace('_', ' ').replace('-', ' ').replace('.pdf', '').strip())
                                    for end in range(len(clean.split()), 0, -1):
                                        phrase = ' '.join(clean.split()[:end])
                                        if phrase and phrase in norm:
                                            anchor = phrase
                                            break
                                    if anchor != host:
                                        break
                            hyperlinks.append({"text": anchor.strip(), "url": uri_str})
                            seen.add(uri_str)
            if hyperlinks:
                node.metadata["hyperlinks"] = hyperlinks

        return nodes


class PromoteEmptyHeadings:
    """Drops empty chunks and pushes their headings onto the next non-empty chunk."""

    name = "promote_empty_headings"

    def apply(self, nodes: list[TextNode]) -> list[TextNode]:
        _filtered = []
        pending_headings = []
        for node in nodes:
            content = node.get_content().strip()
            is_empty = len(content) < 10
            node_headings = node.metadata.get("headings", [])
            if is_empty:
                pending_headings.extend(node_headings)
            else:
                if pending_headings:
                    node.metadata["headings"] = pending_headings + node_headings
                    pending_headings = []
                _filtered.append(node)

        if pending_headings:
            logger.debug("Dropping %d trailing empty heading(s): %s", len(pending_headings), pending_headings)

        return _filtered


class NestArticlesUnderAnnex:
    """Nests article headings under their parent annex heading."""

    name = "nest_articles_under_annex"

    def apply(self, nodes: list[TextNode]) -> list[TextNode]:
        current_annex = None
        for node in nodes:
            headings = node.metadata.get("headings", [])
            if not headings:
                current_annex = None
                continue
            last = headings[-1]
            if _ANNEX_PATTERN.match(last):
                current_annex = last
            elif current_annex and any(p.match(last) for p in _HEADING_PATTERNS):
                if len(headings) == 1 or headings[0] != current_annex:
                    node.metadata["headings"] = [current_annex] + headings
                if len(headings) == 1 or headings[0] != current_annex:
                    node.metadata["headings"] = [current_annex] + headings
            else:
                current_annex = None

        return nodes


# Pipeline composition (in execution order).
# To disable a step, remove it from the list below.
GLOBAL_STEPS = [
    PromoteEmptyHeadings(),
    NestArticlesUnderAnnex(),
]
