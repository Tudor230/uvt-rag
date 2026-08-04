import logging
import re

from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel

logger = logging.getLogger(__name__)

# Matches "Articolul X", "Article X", "Art. X" — article headings
_ARTICLE_PATTERN = re.compile(
    r"^(?:Articolul|Article|Art\.?)\s+\d[\d\s]*(?:\s*\.)?(?=\s|[A-Z]|$)",
    re.IGNORECASE,
)
# Matches "Anexa Nr. X" or "Annex X" — annex headings
_ANNEX_PATTERN = re.compile(
    r"^(?:Anexa|Annex)\s+(?:Nr\s*\.\s+)?\d+\.?",
    re.IGNORECASE,
)

# Each pattern matches a heading line to promote from TEXT/LIST_ITEM to SECTION_HEADER.
_HEADING_PATTERNS = [
    _ARTICLE_PATTERN,                         # "Articolul X" / "Article X" / "Art. X"
    re.compile(                               # "(N) Program(ul/ele) de studii universitare de licență"
        r"^(?:\(\d+\)\s+)?Program(ul|ele)\s+de\s+studii\s+universitare\s+de\s+licență.+", re.IGNORECASE),
    _ANNEX_PATTERN,                           # "Anexa Nr. X" / "Annex X"
    re.compile(r"^CAPITOLUL\s", re.IGNORECASE),  # "CAPITOLUL X"
    re.compile(r"^SECȚIUNEA\s+[IVXL]+", re.IGNORECASE),  # "SECȚIUNEA I/II/III..."
    re.compile(r"^[A-Z]\)\s+[A-Z\sȘȚĂÎÂ\-]+"),          # "A) TITLE" lettered list headings
    re.compile(r"^[IVXL]+\.(?:\s*\d+\s*\.)?\s+[A-Z\sȘȚĂÎÂ\-]+"),  # "I. 1. TITLE" roman-numeral headings
    re.compile(r"^\d+\.\s+METODOLOGIA\s+DE\s+DESFĂȘURARE.+", re.IGNORECASE),  # "X. METODOLOGIA DE DESFĂȘURARE..."
]

# Deletes running page headers matching "Regulamentul Facultății de..." from items
_PAGE_HEADER_PATTERNS = [
    re.compile(r"^Regulamentul\s+Facultății\s+de\s+", re.IGNORECASE),
]

# Catches orphaned running-header continuation fragments at y > 680
# (e.g. "privind organizarea și desfășurarea" or "procesului de admitere pentru")
_RH_CONTINUATION_RE = re.compile(
    r"^(?:privind\s+)?(?:"
    r"organizarea\s+și\s+desfășurarea"
    r"|procesului\s+de\s+admitere\s+pentru"
    r"|programele\s+de\s+studii"
    r"|universitare\s+de\s+la\s+ciclul\s+de"
    r"|studii\s+de\s+licență"
    r")$",
    re.IGNORECASE,
)

# Strips "pagina X din Y" from item text (inline page number within running text)
_PAGE_NUM_STRIP_RE = re.compile(r"(?i)pagina\s+\d+\s+din\s+\d+")

# Detects footer page number variants at y > 680: "pagina X", "X din Y", or "pagina X din"
_FOOTER_PAGE_NUM_RE = re.compile(
    r"(?i)^(?:pagina(?:\s+\d+(?:\s+din)?)?|\d+\s+din\s+\d+|pagina\s+\d+\s+din\s*)$",
)

# Detects "nr. anexe" in footer area at y > 680
_FOOTER_NR_ANNEXE_RE = re.compile(r"(?i)^nr\.?\s+anexe")


class PreprocessContext:
    """Shared per-document state threaded through the preprocessing steps.

    ``items_to_delete`` is intentionally shared across all documents in a single
    ``ingest()`` call (preserves the original cross-document accumulation).
    ``heading_candidates`` is per-document.
    """

    def __init__(self, dl_doc, items_to_delete: list):
        self.dl_doc = dl_doc
        self.items_to_delete = items_to_delete
        self.heading_candidates = []


def _item_y(item):
    try:
        return item.prov[0].bbox.t
    except (IndexError, AttributeError):
        return None


class StripPageNumbers:
    """Removes inline "pagina X din Y" page numbers from every item's text."""

    name = "strip_page_numbers"

    def apply(self, item, ctx: PreprocessContext) -> None:
        if hasattr(item, "text") and isinstance(item.text, str):
            original_text = item.text
            cleaned_text = _PAGE_NUM_STRIP_RE.sub("", original_text).strip()
            item.text = cleaned_text


class RemoveRunningHeaders:
    """Deletes items whose text is a running page header ("Regulamentul Facultății de...").

    Returns ``True`` when the item was deleted so later steps skip it (the original
    single-pass ``continue``).
    """

    name = "remove_running_headers"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        if any(p.match(text) for p in _PAGE_HEADER_PATTERNS):
            ctx.items_to_delete.append(item)
            return True
        return False


class RemoveRHContinuation:
    """Deletes orphaned running-header continuation fragments located at y > 680."""

    name = "remove_rh_continuation"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y > 680 and _RH_CONTINUATION_RE.match(text):
            ctx.items_to_delete.append(item)
            return True
        return False


class RemoveFooterPageNumbers:
    """Deletes footer page number variants ("pagina X", "X din Y") located at y > 680."""

    name = "remove_footer_page_numbers"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y > 680 and _FOOTER_PAGE_NUM_RE.match(text):
            ctx.items_to_delete.append(item)
            return True
        return False


class RemoveFooterNrAnnexe:
    """Deletes "nr. anexe" references located in the footer area at y > 680."""

    name = "remove_footer_nr_annexe"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y > 680 and _FOOTER_NR_ANNEXE_RE.match(text):
            ctx.items_to_delete.append(item)
            return True
        return False


class PromoteHeadings:
    """Promotes TEXT/LIST_ITEM items matching heading patterns to SECTION_HEADER.

    Items nested inside a group are recorded as split candidates instead, so the
    heading boundary can be turned into a sibling group by :class:`SplitHeadingGroups`.
    Returns ``True`` when the item was promoted (the original single-pass ``continue``).
    """

    name = "promote_headings"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        is_heading = any(p.match(text) for p in _HEADING_PATTERNS)
        if not (is_heading and item.label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM)):
            return False

        dl_doc = ctx.dl_doc

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

        parent_cref = item.parent.cref if item.parent else None
        if parent_cref and parent_cref != "#/body":
            parent_group = next(
                (g for g in dl_doc.groups if g.self_ref == parent_cref),
                None,
            )
            if parent_group is not None:
                ctx.heading_candidates.append(
                    (item, heading_text, remaining, parent_group)
                )
                return True

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
        ctx.items_to_delete.append(item)
        return True


class DemoteSectionHeaders:
    """Demotes SECTION_HEADER items that match no heading pattern back to TEXT."""

    name = "demote_section_headers"

    def apply(self, item, ctx: PreprocessContext) -> None:
        text = getattr(item, "text", "").strip()
        is_heading = any(p.match(text) for p in _HEADING_PATTERNS)
        if item.label is DocItemLabel.SECTION_HEADER and not is_heading:
            dl_doc = ctx.dl_doc
            dl_doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                prov=item.prov[0] if item.prov else None,
                parent=dl_doc.body,
            )
            ctx.items_to_delete.append(item)


class ApplyItemDeletions:
    """Applies every item queued for deletion during the item pass."""

    name = "apply_item_deletions"

    def apply(self, ctx: PreprocessContext) -> None:
        dl_doc = ctx.dl_doc
        for target_item in reversed(ctx.items_to_delete):
            try:
                dl_doc.delete_items_range(
                    start=target_item,
                    end=target_item,
                    start_inclusive=True,
                    end_inclusive=True,
                )
            except Exception as e:
                logger.debug(
                    "Skipping native deletion for item due to tree structure constraints: %s", e
                )


class SplitHeadingGroups:
    """Splits groups at heading boundaries so promoted headings become sibling groups.

    Items removed here are appended to ``items_to_delete`` after deletions have already
    been applied, exactly like the original implementation.
    """

    name = "split_heading_groups"

    def apply(self, ctx: PreprocessContext) -> None:
        dl_doc = ctx.dl_doc
        for item, heading_text, remaining, parent_group in reversed(ctx.heading_candidates):
            item_ref = item.get_ref()
            pos = next(
                (i for i, c in enumerate(parent_group.children) if c == item_ref),
                None,
            )
            if pos is None:
                continue

            after_refs = parent_group.children[pos + 1:]

            h = dl_doc.insert_heading(
                sibling=parent_group,
                text=heading_text,
                prov=item.prov[0] if item.prov else None,
                after=True,
            )

            sib = h
            if remaining:
                sib = dl_doc.insert_text(
                    sibling=h,
                    label=DocItemLabel.TEXT,
                    text=remaining,
                    prov=item.prov[0] if item.prov else None,
                    after=True,
                )

            new_group = dl_doc.insert_list_group(sibling=sib, after=True)

            for child_ref in after_refs:
                child_item = child_ref.resolve(dl_doc)
                parent_group.children.remove(child_ref)
                new_group.children.append(child_ref)
                child_item.parent = RefItem(cref=new_group.self_ref)

            parent_group.children.remove(item_ref)
            ctx.items_to_delete.append(item)


class CleanupDocDict:
    """Cleans the exported doc dict: drops list markers and relabels orphan headers."""

    name = "cleanup_doc_dict"

    def apply(self, doc_dict: dict) -> dict:
        refs_in_tree = {c.get("$ref", "") for c in doc_dict.get("body", {}).get("children", [])}
        for item in doc_dict.get("texts", []):
            item.pop("level", None)
            item.pop("enumerated", None)
            item.pop("marker", None)
            if item["self_ref"] not in refs_in_tree and item.get("label") == "section_header":
                item["label"] = "text"
        for group in doc_dict.get("groups", []):
            group.pop("level", None)
        return doc_dict


class ReorderBodyChildren:
    """Reorders body children by reading position, grouping pages into columns."""

    name = "reorder_body_children"

    def apply(self, doc_dict: dict) -> dict:
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
        prov_items = []  # (child, idx, page, y, x)
        no_prov_children = []  # (child, idx)

        for idx, child in enumerate(children):
            ref = child.get("$ref", "")
            item = items_by_ref.get(ref, {})
            p = _first_prov(item)
            if p:
                page = p.get("page_no", 0) or 0
                y = p.get("bbox", {}).get("t", 0)
                x = p.get("bbox", {}).get("l", 0)
                prov_items.append((child, idx, page, y, x))
                page_items[page].append((idx, y, x))
            else:
                no_prov_children.append((child, idx))

        two_column_pages = set()
        for page, items in page_items.items():
            filtered = []
            for idx, y, x in items:
                ref = children[idx].get("$ref", "")
                item = items_by_ref.get(ref, {})
                label = item.get("label", "")
                text = (item.get("text", "") or "").strip()
                if label in ("picture", "table") or not text:
                    continue
                filtered.append((idx, y, x))
            for i in range(len(filtered)):
                y1, x1 = filtered[i][1], filtered[i][2]
                for j in range(i + 1, len(filtered)):
                    y2, x2 = filtered[j][1], filtered[j][2]
                    if abs(y1 - y2) <= 2 and abs(x1 - x2) > 150:
                        two_column_pages.add(page)
                        break
                if page in two_column_pages:
                    break

        def _sort_key(info):
            child, idx, page, y, x = info
            if page in two_column_pages:
                pg_items = page_items[page]
                min_x = min(it[2] for it in pg_items)
                max_x = max(it[2] for it in pg_items)
                mid_x = (min_x + max_x) / 2
                col = 0 if x < mid_x else 1
                return (page, col, -y, idx)
            return (page, -y, x, idx)

        prov_items.sort(key=_sort_key)

        result = [None] * len(children)
        for child, idx in no_prov_children:
            result[idx] = child

        prov_iter = iter(prov_items)
        for i in range(len(children)):
            if result[i] is None:
                result[i] = next(prov_iter)[0]

        doc_dict["body"]["children"] = result

        return doc_dict


# Pipeline composition (in execution order):
#   ITEM_STEPS   — single pass over doc items; a step returning True short-circuits
#                  the remaining steps for that item (mirrors the original `continue`s)
#   DEFERRED_STEPS — run once after the item pass (deletions, then group splits)
#   DICT_STEPS   — operate on the exported doc dict
#
# To disable a step, remove it from the relevant list below.
ITEM_STEPS = [
    StripPageNumbers(),
    RemoveRunningHeaders(),
    RemoveRHContinuation(),
    RemoveFooterPageNumbers(),
    RemoveFooterNrAnnexe(),
    PromoteHeadings(),
    DemoteSectionHeaders(),
]

DEFERRED_STEPS = [
    ApplyItemDeletions(),
    SplitHeadingGroups(),
]

DICT_STEPS = [
    CleanupDocDict(),
    ReorderBodyChildren(),
]
