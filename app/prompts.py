import re

from llama_index.core import PromptTemplate

from ragas.testset.synthesizers.single_hop.prompts import (
    QueryAnswerGenerationPrompt as SingleHopQueryAnswerGenerationPrompt,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryAnswerGenerationPrompt as MultiHopQueryAnswerGenerationPrompt,
)

QUERY_GEN_STR = """
Ești un asistent util care generează mai multe interogări de căutare pe baza unei singure interogări introduse de utilizator.
Generează 3 interogări de căutare în limba română, fiecare pe un rând separat, legate de următoarea interogare:
Interogare: {query}
Interogări:
"""

query_gen_prompt = PromptTemplate(QUERY_GEN_STR)

# ── Faculty map ────────────────────────────────────────────────────────
# Maps filename abbreviations to full Romanian faculty names.
# When an abbreviation is found in the PDF filename, the corresponding
# faculty name is prepended to the chunk context so the LLM can
# incorporate it into generated questions.

FACULTY_MAP: dict[str, str] = {
    "FFM": "Facultatea de Fizică și Matematică",
    "FI": "Facultatea de Informatică",
    "FEAA": "Facultatea de Economie și Administrare a Afacerilor",
    "FD": "Facultatea de Drept",
    "FEFS": "Facultatea de Educație Fizică și Sport",
    "FLIFT": "Facultatea de Litere, Istorie, Filosofie și Teologie ",
    "FPSE": "Facultatea de Psihologie și Științe ale Educației",
    "FAD": "Facultatea de Arte și Design",
}

_FACULTY_FALLBACK = "Universitatea de Vest din Timișoara"


def extract_faculty_name(metadata: dict) -> str | None:
    """Extract the full faculty name from node metadata.

    Reads ``origin.filename`` (set by Docling) and tries each
    abbreviation in `FACULTY_MAP`.  Returns the fallback
    (``Universitatea de Vest din Timișoara``) when no abbreviation
    matches or when there is no filename at all.
    """
    origin = metadata.get("origin", {})
    filename: str = ""
    if isinstance(origin, dict):
        filename = origin.get("filename", "") or ""
    if not filename and isinstance(origin, str):
        filename = origin

    for abbr in sorted(FACULTY_MAP, key=len, reverse=True):
        pattern = re.compile(rf"(?:^|[-_]){re.escape(abbr)}\.pdf$", re.IGNORECASE)
        if pattern.search(filename):
            return FACULTY_MAP[abbr]

    return _FACULTY_FALLBACK


# ── Extra prompt instructions ──────────────────────────────────────────
# These are appended to the original RAGAS prompt instructions.
# Edit the strings below to tweak behaviour without touching RAGAS internals.

SINGLE_HOP_EXTRA = (
    "\n"
    "4. **Content Focus — CRITICAL**: The query must ask about the TOPIC, not about document structure. "
    "Never use phrases like \"conform Articolului\", \"according to Article\", \"în secțiunea\", "
    "\"în Capitolul\", or any reference to article numbers, section numbers, or heading names.\n"
    "   - BAD: \"Conform Articolului 18, ce taxe...\"\n"
    "   - GOOD: \"Ce taxe trebuie să plătească...\"\n"
    "   - BAD: \"According to Article 5, what are...\"\n"
    "   - GOOD: \"What are the requirements for...\"\n"
    "If a faculty name appears in brackets at the start of the context (e.g., [Facultatea de ...]), "
    "naturally incorporate that faculty into the question."
)

MULTI_HOP_SPECIFIC_EXTRA = (
    "\n"
    "5. **Content Focus — CRITICAL**: The query must ask about the TOPIC, not about document structure. "
    "Never use phrases like \"conform Articolului\", \"according to Article\", \"în secțiunea\", "
    "\"în Capitolul\", or any reference to article numbers, section numbers, or heading names.\n"
    "   - BAD: \"Conform Articolului 18, ce taxe...\"\n"
    "   - GOOD: \"Ce taxe trebuie să plătească...\"\n"
    "   - BAD: \"According to Article 5, what are...\"\n"
    "   - GOOD: \"What are the requirements for...\"\n"
    "If a faculty name appears in brackets at the start of any context segment (e.g., "
    "[Facultatea de ...]), naturally incorporate that faculty into the question."
)

MULTI_HOP_ABSTRACT_EXTRA = (
    "\n"
    "5. **Content Focus — CRITICAL**: The query must ask about the TOPIC, not about document structure. "
    "Never use phrases like \"conform Articolului\", \"according to Article\", \"în secțiunea\", "
    "\"în Capitolul\", or any reference to article numbers, section numbers, or heading names.\n"
    "   - BAD: \"Conform Articolului 18, ce taxe...\"\n"
    "   - GOOD: \"Ce taxe trebuie să plătească...\"\n"
    "   - BAD: \"According to Article 5, what are...\"\n"
    "   - GOOD: \"What are the requirements for...\"\n"
    "If a faculty name appears in brackets at the start of any context segment (e.g., "
    "[Facultatea de ...]), naturally incorporate that faculty into the question."
)


# ── Prompt builders ────────────────────────────────────────────────────

def build_single_hop_prompt():
    """Return a deep-copied single-hop prompt with the extra instructions."""
    prompt = SingleHopQueryAnswerGenerationPrompt()
    prompt.instruction = prompt.instruction.rstrip("\n") + SINGLE_HOP_EXTRA
    return prompt


def build_multi_hop_specific_prompt():
    """Return a deep-copied multi-hop prompt for entity-overlap based queries."""
    prompt = MultiHopQueryAnswerGenerationPrompt()
    prompt.instruction = prompt.instruction.rstrip("\n") + MULTI_HOP_SPECIFIC_EXTRA
    return prompt


def build_multi_hop_abstract_prompt():
    """Return a deep-copied multi-hop prompt for theme-similarity based queries."""
    prompt = MultiHopQueryAnswerGenerationPrompt()
    prompt.instruction = prompt.instruction.rstrip("\n") + MULTI_HOP_ABSTRACT_EXTRA
    return prompt


def build_faculty_context(metadata: dict) -> str:
    """Build a faculty context prefix string for a chunk.

    Returns something like ``[Facultatea de Informatică]`` or
    ``[Universitatea de Vest din Timișoara]``.
    """
    faculty = extract_faculty_name(metadata)
    return f"[{faculty}]"
