"""The system prompt and the per-turn status message (specs/agent.md §2).

The system prompt states the role, the process, the rules and the source catalog (§2.1). The status
message is rebuilt every turn from the session and controller and is never stored in graph state
(§2.2). Both only read state.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_agent.controller import Controller
from rag_agent.corpus import Source
from rag_agent.session import Session
from rag_agent.tools import MAX_ANSWER_CHARS, MAX_CITATIONS

ROLE = (
    "You answer students' Python and RISC-V questions using only passages retrieved from a fixed "
    "document corpus. You act by calling tools; the application checks every call."
)

PROCESS = """Process:
1. Call search_documents with a focused query of a few key terms, not the whole question.
2. Use the `sources` or `subject` filters when the question names or implies a source.
3. If the results are off-target, search again with different terms or filters.
4. Then call finish.
Query wording: describe the content you need (for example "limitations and weaknesses the paper \
acknowledges"), not the document's own subject or title; a title-like query matches title pages and \
figure captions instead of the passage you want."""

RULES = """Rules:
- Use only retrieved passages. Never answer from memory.
- End every factual sentence with a [n] marker, where n is the citation's position in the citations \
list. List the citations in marker order, and cite only chunk ids returned by search_documents in this run.
- Paraphrase explanations freely, but keep technical details exactly as written in the passage: \
instruction mnemonics, register and ABI names, numbers, bit widths, version strings, code, and Python \
keywords and error names. Quote briefly when a precise definition matters.
- When the 2019 and 2025 RISC-V specifications differ, the 2025 manual controls; say so.
- If the passages support only part of the question, answer that part and state what is missing. If \
they support nothing central, call finish with outcome insufficient_evidence and say what you searched \
for and what is missing.
- Make exactly one tool call per turn. Never reply with plain text: the answer goes in finish.
- Keep the answer under {max_answer} characters; for a question with several parts, answer each part \
briefly.
- Cite at most {max_citations} passages.""".format(max_answer=f"{MAX_ANSWER_CHARS:,}",
                                                    max_citations=MAX_CITATIONS)


def catalog_line(source: Source) -> str:
    return f"- {source.id} ({source.subject}): {source.title}. {source.role}"


def system_prompt(registry: Iterable[Source]) -> str:
    """Role, process, rules, then the source catalog in registry order (§2.1)."""
    catalog = "\n".join(["Sources:", *(catalog_line(source) for source in registry)])
    return "\n\n".join([ROLE, PROCESS, RULES, catalog])


def status_message(session: Session, controller: Controller) -> str:
    """Steps used and the chunks available for citation (§2.2)."""
    max_steps = session.config.max_steps
    lines = [f"Step {controller.steps_used} of {max_steps}."]
    if session.retrieved:
        lines.append("Retrieved passages you may cite (chunk_id · source_id · page):")
        lines.extend(f"- {p.chunk_id} · {p.source_id} · p.{p.page}" for p in session.retrieved.values())
    else:
        lines.append("Retrieved passages you may cite: none yet. Call search_documents first.")
    if controller.steps_used == max_steps:
        lines.append("This is your last step: call finish now.")
    return "\n".join(lines)
