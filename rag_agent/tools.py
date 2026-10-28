"""The two tools the model may request: schemas, definitions and executors (specs/tools.md §2-§5).

`SearchArgs` and `FinishArgs` validate the arguments (§3); `tool_definitions()` describes both tools
to the model (§4); `execute()` runs a request the controller has already approved (§5). `execute` is
the only code that adds to `session.retrieved` (architecture §5).
"""

from __future__ import annotations

import copy
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, field_validator

from rag_agent import controller
from rag_agent.session import Session

MAX_K = 8
MAX_SOURCES = 9
MAX_CITATIONS = 12
MAX_ANSWER_CHARS = 8000   # raised from 4,000 after required run 1 (architecture D23)

Query = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=3, max_length=300)]
SourceId = Annotated[str, StringConstraints(strict=True)]
Answer = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1,
                                          max_length=MAX_ANSWER_CHARS)]
Supports = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=200)]
ChunkId = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{24}$")]


# ---------------------------------------------------------------- argument schemas (§3)

class SearchArgs(BaseModel):  # §3.1
    """Arguments of search_documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: Query = Field(description="A focused search query of a few key terms (3-300 characters).")
    sources: list[SourceId] | None = Field(
        default=None, min_length=1, max_length=MAX_SOURCES,
        description="Optional: only search these source ids from the catalog (distinct).")
    subject: Literal["python", "riscv", "pedagogy"] | None = Field(
        default=None, description="Optional: only search sources with this subject.")
    k: Annotated[StrictInt, Field(ge=1, le=MAX_K)] | None = Field(
        default=None, description=f"Optional: number of passages to return (1-{MAX_K}).")

    @field_validator("sources")
    @classmethod
    def _distinct(cls, sources: list[str] | None) -> list[str] | None:
        if sources is not None and len(set(sources)) != len(sources):
            raise ValueError("source ids must be distinct")
        return sources


class Citation(BaseModel):  # §3.2
    """One cited chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: ChunkId = Field(description="A chunk_id returned by search_documents in this run.")
    supports: Supports = Field(description="A short phrase naming the claim this chunk supports.")


class FinishArgs(BaseModel):  # §3.2
    """Arguments of finish."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["answer", "insufficient_evidence"] = Field(
        description="`answer` when the passages support an answer, else `insufficient_evidence`.")
    answer: Answer = Field(description="The answer with [n] markers, or what was searched and what is missing.")
    citations: list[Citation] = Field(
        default_factory=list, max_length=MAX_CITATIONS,
        description="The cited chunks in marker order: citation n is marker [n].")


SCHEMAS: dict[str, type[BaseModel]] = {"search_documents": SearchArgs, "finish": FinishArgs}


# ---------------------------------------------------------------- citation markers (§3.3)

_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def citation_markers(answer: str) -> list[int]:
    """Every number cited by a `[n]` or `[n, m]` marker, in order of appearance, with repeats."""
    return [int(number) for group in _MARKER.findall(answer) for number in group.split(",")]


# ---------------------------------------------------------------- definitions for the model (§4)

DESCRIPTIONS = {
    "search_documents": (
        "Search the Python and RISC-V document corpus. Returns ranked passages, each with a chunk_id "
        "you can cite, its source_id, title, page and text. Use a focused query of a few key terms, not "
        "the whole question. Filter with `sources` (source ids from the catalog) or `subject` when you "
        "know where the answer is. You may search several times with different queries or filters."
    ),
    "finish": (
        "End the run. Use outcome `answer` when retrieved passages support an answer: write the answer "
        "with [n] markers and list the citations in marker order, each with a chunk_id returned by "
        "search_documents. Use outcome `insufficient_evidence` when the passages do not support an "
        "answer: say what you searched for and what is missing."
    ),
}


def _inline(node, defs: dict):
    """Resolve local `$ref`s and drop titles, so each tool's parameters are one plain JSON schema."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]), defs)
        return {key: _inline(value, defs) for key, value in node.items() if key not in ("title", "$defs")}
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    return node


def tool_definitions() -> list[dict]:
    """Both tools in OpenAI function-tool format, generated from the schemas."""
    definitions = []
    for name, schema in SCHEMAS.items():
        raw = schema.model_json_schema()
        definitions.append({"type": "function", "function": {
            "name": name, "description": DESCRIPTIONS[name], "parameters": _inline(raw, raw.get("$defs", {})),
        }})
    return definitions


# ---------------------------------------------------------------- executors (§5)

def execute(session: Session, approval: controller.Approval, steps_remaining: int) -> controller.Observation:
    """Run an approved request. A retriever exception propagates to the controller (§5.1)."""
    if approval.tool == "search_documents":
        return _search(session, approval.args, steps_remaining)
    if approval.tool == "finish":
        return _finish(session, approval.args, steps_remaining)
    raise ValueError(f"Unknown tool {approval.tool!r}")


def _search(session: Session, args: SearchArgs, steps_remaining: int) -> controller.Observation:
    passages = session.retriever.search(args.query, args.k or session.config.default_k,
                                        sources=args.sources, subject=args.subject)
    for passage in passages:
        session.retrieved.setdefault(passage.chunk_id, passage)
    results = [{
        "rank": rank, "chunk_id": p.chunk_id, "source_id": p.source_id, "title": p.title, "page": p.page,
        "distance": p.distance, "vector_rank": p.vector_rank, "keyword_rank": p.keyword_rank, "text": p.text,
    } for rank, p in enumerate(passages, start=1)]
    if passages:
        sources = ", ".join(dict.fromkeys(p.source_id for p in passages))
        message = f"{len(passages)} passage(s) from {sources}."
    else:
        message = "No passages matched. Try a different query or remove filters."
    data = {"status": "ok" if passages else "empty", "query": args.query,
            "filters": {"sources": list(args.sources) if args.sources else None, "subject": args.subject},
            "results": results}
    return controller.Observation(tool="search_documents", accepted=True, policy=None, reason=None,
                                  message=message, data=data, steps_remaining=steps_remaining)


def _finish(session: Session, args: FinishArgs, steps_remaining: int) -> controller.Observation:
    citations = []
    for n, citation in enumerate(args.citations, start=1):
        passage = session.retrieved[citation.chunk_id]
        citations.append({"n": n, "chunk_id": passage.chunk_id, "source_id": passage.source_id,
                          "title": passage.title, "page": passage.page, "supports": citation.supports})
    return controller.Observation(tool="finish", accepted=True, policy=None, reason=None,
                                  message=f"Run finished: {args.outcome}.",
                                  data={"outcome": args.outcome, "citations": citations},
                                  steps_remaining=steps_remaining)
