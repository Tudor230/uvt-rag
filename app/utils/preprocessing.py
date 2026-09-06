import logging
import re

from docling_core.types.doc.document import GroupItem, ListGroup, RefItem, TextItem
from docling_core.types.doc.labels import DocItemLabel, GroupLabel

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
# Matches "X. METODOLOGIA DE DESFĂȘURARE..." — sport mastery-probe methodology headings
_METODOLOGIA_PATTERN = re.compile(
    r"^\d+\.\s+METODOLOGIA\s+DE\s+DESFĂȘURARE.+", re.IGNORECASE
)

# Each pattern matches a heading line to promote from TEXT/LIST_ITEM to SECTION_HEADER.
_HEADING_PATTERNS = [
    _ARTICLE_PATTERN,                         # "Articolul X" / "Article X" / "Art. X"
    re.compile(                               # "(N) Program(ul/ele) de studii universitare de licență"
        # Capital "Program" required: lower-case "programele de studii universitare de
        # licență și cifra de școlarizare…" is a body sentence, not a heading.
        r"^(?:\(\d+\)\s+)?Program(ul|ele)\s+de\s+studii\s+universitare\s+de\s+licență.+"),
    _ANNEX_PATTERN,                           # "Anexa Nr. X" / "Annex X"
    re.compile(r"^CAPITOLUL\s", re.IGNORECASE),  # "CAPITOLUL X"
    re.compile(r"^SECȚIUNEA\s+[IVXL]+", re.IGNORECASE),  # "SECȚIUNEA I/II/III..."
    re.compile(r"^[A-Z]\)\s+[A-Z\sȘȚĂÎÂ\-]+"),          # "A) TITLE" lettered list headings
    re.compile(r"^[IVXL]+\.(?:\s*\d+\s*\.)?\s+[A-Z\sȘȚĂÎÂ\-]+"),  # "I. 1. TITLE" roman-numeral headings
    _METODOLOGIA_PATTERN,                     # "X. METODOLOGIA DE DESFĂȘURARE..."
]

# Deletes running page headers matching "Regulamentul Facultății de..." from items.
# `de\s*` also catches the truncated header line "Regulamentul Facultății de" that
# docling emits on its own line before the faculty-name continuation.
_PAGE_HEADER_PATTERNS = [
    re.compile(r"^Regulamentul\s+Facultății\s+de\s*", re.IGNORECASE),
]

# Matches the sport→probe boundary inside a merged METODOLOGIA heading, e.g.
# "…SPORTIVĂ -ATLETISM PROBA DE 100 m plat :" or "…SPORTIVĂ FOTBAL Testul 1".
# Uses a lookahead so the leading whitespace stays on the heading side; "PROBEI"/
# "PROBELE" inside "PROBA DE MĂIESTRIE SPORTIVĂ" are never matched.
_PROBE_MARKER_RE = re.compile(
    r"\s+(?=\b(?:PROBA|Proba|Testul|Test)\b)", re.IGNORECASE
)

# The running header ends with a fixed tail ("... Facultății de X privind organizarea
# și desfășurarea procesului de admitere pentru programele de studii universitare de la
# ciclul de studii de licență"). Docling splits that tail into arbitrary contiguous
# fragments (from a single word up to the whole phrase), each its own orphaned item at
# y > 680. We generate a match for every contiguous slice (>= 3 tokens) so whichever
# split point docling picked, the fragment is still recognized.
_RH_CONTINUATION_TOKENS = (
    "organizarea", "și", "desfășurarea", "procesului", "de", "admitere",
    "pentru", "programele", "de", "studii", "universitare", "de", "la",
    "ciclul", "de", "studii", "de", "licență",
)
_RH_CONTINUATION_SLICES = "|".join(
    r"\s+".join(re.escape(t) for t in _RH_CONTINUATION_TOKENS[i:j])
    for i in range(len(_RH_CONTINUATION_TOKENS))
    for j in range(i + 3, len(_RH_CONTINUATION_TOKENS) + 1)
)
# Catches orphaned running-header continuation fragments at y > 680
# (e.g. "privind organizarea și desfășurarea" or "procesului de admitere pentru"
# or the faculty-name fragment "Educație Fizică și Sport privind")
_RH_CONTINUATION_RE = re.compile(
    r"^(?:privind\s+)?(?:" + _RH_CONTINUATION_SLICES
    # The UVT regulation's running header is the full "Regulamentul UVT privind ... licență"
    # phrase as one section_header item (or a truncated prefix of it) at y > 680.
    + r"|Regulamentul\s+UVT\s+privind\s+(?:" + _RH_CONTINUATION_SLICES + r")"
    + r"|[A-ZĂÂÎȘȚ][\wăâîșț\-()]*(?:\s+(?:și\s+)?[\wăâîșț\-()]+)*\s+privind"
    r")$",
    re.IGNORECASE,
)

# Strips "pagina X din Y" from item text (inline page number within running text)
_PAGE_NUM_STRIP_RE = re.compile(r"(?i)pagina\s+\d+\s+din\s+\d+")

# Detects page-number variants in the header band (y > 680): "pagina X", "X din Y", or "pagina X din"
_HEADER_PAGE_NUM_RE = re.compile(
    r"(?i)^(?:pagina(?:\s+\d+(?:\s+din)?)?|\d+\s+din\s+\d+|pagina\s+\d+\s+din\s*)$",
)

# Detects "nr. anexe" in the header band (y > 680)
_HEADER_NR_ANNEXE_RE = re.compile(r"(?i)^nr\.?\s+anexe")

# Item-level cleanup: removes leftover "Regulamentul Facultății de [...] privind [...]" running headers
# embedded inside longer items (whole-item headers are deleted by RemoveRunningHeaders).
_RUNNING_HEADER_RE = re.compile(
    r"(?:^|\n|(?<=[.!?])\s)Regulamentul\s+Facultății\s+de\s+"
    r"(?:.*?studii\s+de\s+licență|.*?privind(?:\s|$))",
    re.IGNORECASE | re.DOTALL,
)

# Item-level cleanup: removes fragmented "pagina\nX\ndin\nY" page numbers split across lines
_PAGE_NUM_FRAGMENTED_RE = re.compile(
    r"(?:^|\n)pagina\s*\n\d+\s*\ndin\s*\n\d+",
    re.IGNORECASE,
)

# Item-level cleanup: removes "nr. anexe <number>" references
_NR_ANNEX_RE = re.compile(
    r"\bnr\.?\s+anexe\s+\d+\b",
    re.IGNORECASE,
)


class PreprocessContext:
    """Shared per-document state threaded through the preprocessing steps.

    ``items_to_delete`` must be per-document: docling refs are positional per
    document, so accumulating them across documents makes deletions from one
    document collide with (and delete) innocent items in later documents.
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

    Only whole items located in the header band (y > 680) are deleted, so a mid-page
    body item that happens to start with the phrase is never touched.
    Returns ``True`` when the item was deleted so later steps skip it (the original
    single-pass ``continue``).
    """

    name = "remove_running_headers"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y <= 680:
            return False
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


class RemoveHeaderPageNumbers:
    """Deletes page-number variants ("pagina X", "X din Y") located in the header band (y > 680)."""

    name = "remove_header_page_numbers"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y > 680 and _HEADER_PAGE_NUM_RE.match(text):
            ctx.items_to_delete.append(item)
            return True
        return False


class RemoveHeaderNrAnnexe:
    """Deletes "nr. anexe" references located in the header band (y > 680)."""

    name = "remove_header_nr_annexe"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        y = _item_y(item)
        if y is not None and y > 680 and _HEADER_NR_ANNEXE_RE.match(text):
            ctx.items_to_delete.append(item)
            return True
        return False


class CleanItemText:
    """Scrubs leftover running headers, fragmented page numbers and "nr. anexe" from item text.

    Runs after whole-item furniture deletion so it only targets fragments embedded
    inside longer items (the doc-level equivalent of the old chunk-level CleanChunkText).
    The running-header scrub is limited to items located in the header band (y > 680):
    running headers sit at the top of a merged item, so ``prov[0]`` is still the header y.
    Page-number and "nr. anexe" scrubs stay ungated — their junk can merge at the END of a
    body item (e.g. "...devin eligibili pentru Nr. anexe 5"), where ``prov[0]`` is the body y.
    Returns ``False`` so later steps always see the item.
    """

    name = "clean_item_text"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        if hasattr(item, "text") and isinstance(item.text, str):
            new_text = item.text
            y = _item_y(item)
            if y is None or y > 680:
                new_text = _RUNNING_HEADER_RE.sub("\n", new_text)
            new_text = _PAGE_NUM_FRAGMENTED_RE.sub("\n", new_text)
            new_text = _NR_ANNEX_RE.sub("", new_text)
            if new_text != item.text:
                item.text = new_text
                item.orig = new_text
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


class SplitProbeHeadings:
    """Splits merged METODOLOGIA headings that swallowed their probe sub-heading.

    Docling sometimes merges the sport heading line with the following probe line
    into a single section_header item, e.g.
        "1. METODOLOGIA DE DESFĂȘURARE A PROBEI DE MĂIESTRIE SPORTIVĂ -ATLETISM PROBA DE 100 m plat :"
    This step truncates such headings back to the clean sport heading and re-inserts
    the probe text ("PROBA DE 100 m plat :", "Testul 1 KIHON", ...) as plain content.
    """

    name = "split_probe_headings"

    def apply(self, item, ctx: PreprocessContext) -> bool:
        text = getattr(item, "text", "").strip()
        if item.label is not DocItemLabel.SECTION_HEADER:
            return False
        if not _METODOLOGIA_PATTERN.match(text):
            return False
        m = _PROBE_MARKER_RE.search(text)
        if not m:
            return False
        heading_text = text[: m.start()].strip()
        remaining = text[m.end():].strip()
        if not heading_text or not remaining:
            return False
        item.text = heading_text
        item.orig = heading_text
        ctx.dl_doc.add_text(
            label=DocItemLabel.TEXT,
            text=remaining,
            prov=item.prov[0] if item.prov else None,
            parent=ctx.dl_doc.body,
        )
        return True


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


class SplitColumnLists:
    """Splits a list group whose items span two side-by-side columns into two lists.

    Two-column layouts (e.g. "I. Sporturi individuale" | "II. Jocuri sportive") are
    parsed by docling as a single merged list group. The chunker then attaches the whole
    merged list to one heading and orphans the other column's heading. Splitting by
    column lets each heading bind to its own list.

    Only the common two-column case is handled: the split point is the single largest
    gap between the items' x-centers, the gap must span a substantial fraction of the
    page width, and both resulting columns must hold several items arranged as a
    contiguous cut in reading order. This keeps single-column lists (with only a few
    indented items) untouched.
    """

    name = "split_column_lists"

    _GAP_FRACTION = 0.30
    _MIN_ITEMS_PER_COLUMN = 2

    def apply(self, ctx: PreprocessContext) -> None:
        dl_doc = ctx.dl_doc
        candidates = []
        stack = [ref for ref in dl_doc.body.children]
        while stack:
            ref = stack.pop()
            obj = ref.resolve(dl_doc)
            if hasattr(obj, "children"):
                stack.extend(reversed(list(obj.children)))
            if not isinstance(obj, ListGroup):
                continue
            children = list(obj.children)
            if len(children) < 2:
                continue
            centers = []
            page_no = None
            ok = True
            for c in children:
                child = c.resolve(dl_doc)
                if not child.prov or not child.prov[0].bbox:
                    ok = False
                    break
                if page_no is None:
                    page_no = child.prov[0].page_no
                elif child.prov[0].page_no != page_no:
                    ok = False
                    break
                centers.append((child.prov[0].bbox.l + child.prov[0].bbox.r) / 2)
            if not ok or len(centers) < 2:
                continue
            uniq = sorted(set(centers))
            max_gap, boundary = 0.0, None
            for a, b in zip(uniq, uniq[1:]):
                if b - a > max_gap:
                    max_gap, boundary = b - a, (a + b) / 2
            if max_gap < self._GAP_FRACTION * ctx.dl_doc.pages[page_no].size.width or boundary is None:
                continue
            left = [c for c, x in zip(children, centers) if x < boundary]
            right = [c for c, x in zip(children, centers) if x >= boundary]
            if (
                len(left) < self._MIN_ITEMS_PER_COLUMN
                or len(right) < self._MIN_ITEMS_PER_COLUMN
            ):
                continue
            split_at = len(left)
            if children[:split_at] != left or children[split_at:] != right:
                continue
            candidates.append((obj, right))

        for group, right in candidates:
            new_group = dl_doc.insert_list_group(sibling=group, after=True)
            for c in right:
                child = c.resolve(dl_doc)
                group.children.remove(c)
                new_group.children.append(c)
                child.parent = RefItem(cref=new_group.self_ref)


class FlattenGroupItemLists:
    """Promotes a wrapper ``GroupItem``'s list groups up to the document body.

    Docling sometimes nest a couple of sibling ``ListGroup``s (e.g. the two columns of a
    technique table) inside a generic ``GroupItem``. ReorderBodyChildren anchors such a
    group at its *first* list's bounding box, so the whole wrapped block is positioned as
    if it sat at the top. A body-level heading that actually falls *between* the two lists
    in reading order then ends up sorted after the entire block, mis-ordering the content
    (e.g. JUDO "TESTUL 2" appearing after its OSAE/SHIME/KANSETSU exercises).

    Flattening the wrapper makes each list a body child sorted by its own bounding box,
    so the interleaved heading keeps its reading position. Only wrappers holding several
    list groups are flattened; single-list wrappers are left untouched.
    """

    name = "flatten_group_item_lists"

    def apply(self, ctx: PreprocessContext) -> None:
        dl_doc = ctx.dl_doc
        for ref in list(dl_doc.body.children):
            obj = ref.resolve(dl_doc)
            if not isinstance(obj, GroupItem):
                continue
            kids = [c.resolve(dl_doc) for c in obj.children]
            if len(kids) < 2 or not all(isinstance(k, ListGroup) for k in kids):
                continue
            idx = dl_doc.body.children.index(ref)
            dl_doc.body.children[idx:idx + 1] = list(obj.children)
            for c in obj.children:
                child = c.resolve(dl_doc)
                child.parent = RefItem(cref=dl_doc.body.self_ref)


class CleanupDocDict:
    """Cleans the exported doc dict: drops list markers and relabels orphan headers."""

    name = "cleanup_doc_dict"

    @staticmethod
    def _collect_tree_refs(node: dict, groups: list[dict]) -> set:
        """Collect every child reference reachable from ``node`` (recursing into groups).

        A ``section_header`` nested inside a group is referenced by the group's children,
        so it must count as "in the tree"; otherwise CleanupDocDict wrongly relabels it to
        text and the chunker never emits it as a heading.
        """
        refs = set()
        for c in node.get("children", []):
            ref = c.get("$ref", c) if isinstance(c, dict) else c
            refs.add(ref)
            if str(ref).startswith("#/groups"):
                group = next((g for g in groups if g["self_ref"] == ref), None)
                if group is not None:
                    refs |= CleanupDocDict._collect_tree_refs(group, groups)
        return refs

    def apply(self, doc_dict: dict) -> dict:
        refs_in_tree = CleanupDocDict._collect_tree_refs(doc_dict.get("body", {}), doc_dict.get("groups", []))
        for item in doc_dict.get("texts", []):
            item.pop("level", None)
            item.pop("enumerated", None)
            item.pop("marker", None)
            if item["self_ref"] not in refs_in_tree and item.get("label") == "section_header":
                item["label"] = "text"
        for group in doc_dict.get("groups", []):
            group.pop("level", None)
        return doc_dict


class ReorderColumnarGroupLeaves:
    """Re-emits ``key_value_area`` group leaves in PDF reading order.

    Two-column technique tables (e.g. the JUDO METODOLOGIA: uppercased test/technique
    names in the left column, exercises in the right) are parsed by docling as a
    ``key_value_area`` group whose leaves come out jumbled (column-major instead of
    row-major). This step sorts the group's leaf texts by reading position
    ``(page_no, -bbox.t, bbox.l)`` and re-emits them as one or more flat body-level
    ``ListGroup`` blocks. The blocks are split wherever a body-level text line falls
    inside the group's vertical span (e.g. "TESTUL 2" between the SUTEMI and OSAE
    blocks), so that heading keeps its reading position after ReorderBodyChildren.
    """

    name = "reorder_columnar_group_leaves"

    def apply(self, ctx: PreprocessContext) -> None:
        dl_doc = ctx.dl_doc

        def _leaves_of(node, out: list) -> None:
            for ref in node.children:
                child = ref.resolve(dl_doc)
                if isinstance(child, GroupItem) and child.children:
                    _leaves_of(child, out)
                elif isinstance(child, TextItem) and not child.children:
                    out.append(child)

        def _pos(item) -> tuple:
            b = item.prov[0].bbox
            return (item.prov[0].page_no, -b.t, b.l)

        def _pos_info(item) -> tuple:
            return item.prov[0].page_no, item.prov[0].bbox.t

        def _subtree_groups(node, out: list) -> None:
            for ref in list(node.children):
                child = ref.resolve(dl_doc)
                if isinstance(child, GroupItem) and child is not node:
                    out.append(child)
                    _subtree_groups(child, out)

        body_texts = [
            ref.resolve(dl_doc)
            for ref in dl_doc.body.children
            if isinstance(ref.resolve(dl_doc), TextItem)
            and (ref.resolve(dl_doc).text or "").strip()
            and ref.resolve(dl_doc).prov
        ]

        for group in list(dl_doc.groups):
            if group.label != GroupLabel.KEY_VALUE_AREA:
                continue
            leaves = []
            _leaves_of(group, leaves)
            if len(leaves) < 2:
                continue
            sorted_leaves = sorted(leaves, key=_pos)
            if sorted_leaves == leaves:
                continue
            pages = {_pos_info(leaf)[0] for leaf in sorted_leaves}
            if len(pages) != 1:
                continue
            page = next(iter(pages))
            ys = [_pos_info(leaf)[1] for leaf in sorted_leaves]
            max_y, min_y = max(ys), min(ys)
            split_ys = sorted(
                (
                    item.prov[0].bbox.t
                    for item in body_texts
                    if item.prov[0].page_no == page and min_y < item.prov[0].bbox.t < max_y
                ),
                reverse=True,
            )

            blocks = []
            remaining = list(sorted_leaves)
            for hy in split_ys:
                above = [leaf for leaf in remaining if _pos_info(leaf)[1] > hy]
                below = [leaf for leaf in remaining if _pos_info(leaf)[1] < hy]
                if above:
                    blocks.append(above)
                remaining = below
            if remaining:
                blocks.append(remaining)
            blocks = [b for b in blocks if b]
            if not blocks:
                continue

            subtree = []
            _subtree_groups(group, subtree)

            parent = group.parent.resolve(dl_doc) if group.parent else None
            if parent is not None:
                parent.children = [r for r in parent.children if r.resolve(dl_doc) is not group]
            detached = [group] + subtree
            dl_doc.body.children = [r for r in dl_doc.body.children if r.resolve(dl_doc) not in detached]
            group.children = []
            for nested in subtree:
                nested.children = []

            for block in blocks:
                list_group = dl_doc.add_list_group(parent=dl_doc.body)
                for leaf in block:
                    old_parent = leaf.parent.resolve(dl_doc) if leaf.parent else None
                    if old_parent is not None and old_parent is not list_group:
                        old_parent.children = [
                            r for r in old_parent.children if r.resolve(dl_doc) is not leaf
                        ]
                    leaf.parent = RefItem(cref=list_group.self_ref)
                    list_group.children.append(RefItem(cref=leaf.self_ref))


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
    RemoveHeaderPageNumbers(),
    RemoveHeaderNrAnnexe(),
    CleanItemText(),
    PromoteHeadings(),
    DemoteSectionHeaders(),
    SplitProbeHeadings(),
]

DEFERRED_STEPS = [
    ApplyItemDeletions(),
    SplitHeadingGroups(),
    SplitColumnLists(),
    FlattenGroupItemLists(),
    ReorderColumnarGroupLeaves(),
]

DICT_STEPS = [
    CleanupDocDict(),
    ReorderBodyChildren(),
]
