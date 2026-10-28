"""Black-box tests for rag_agent.required (the nine required runs), derived from specs/agent.md §7."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fakes import JALR, REGISTRY, FakeRetriever, ScriptedModel, call, turn
from rag_agent.config import PROJECT_DIR
from rag_agent.controller import Config, TerminalReason
from rag_agent.required import RequiredQuery, parse_queries, run_all

FROZEN = PROJECT_DIR / "queries" / "required-queries.md"


def test_the_frozen_file_has_nine_queries_in_order():
    queries = parse_queries(FROZEN)
    assert [q.number for q in queries] == list(range(1, 10))
    assert all(isinstance(q, RequiredQuery) for q in queries)
    assert [q.expected for q in queries] == [TerminalReason.ANSWER] * 7 + [TerminalReason.INSUFFICIENT_EVIDENCE] * 2
    assert queries[0].title == "ISA Instruction Lookup"
    assert queries[8].title == "Off-Topic Evidence Boundary"


def test_wrapped_lines_are_joined_with_single_spaces():
    queries = parse_queries(FROZEN)
    assert queries[0].text == (
        "According to the RISC-V Unprivileged ISA manual (version 20250508), what does the JALR "
        "instruction write to its destination register, how is its target address computed, and what "
        "happens to the least-significant bit of that address?"
    )
    assert queries[8].text == "Who is the greatest quarterback in NFL history, and what statistics establish that ranking?"
    for q in queries:
        assert "\n" not in q.text and "<!--" not in q.text and "  " not in q.text


def write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_a_section_without_an_expected_outcome_is_an_error(tmp_path):
    path = write(tmp_path / "q.md", "# Queries\n\n## Q1: One\n\nWhat is ra?\n")
    with pytest.raises(ValueError):
        parse_queries(path)


def test_duplicate_numbers_are_an_error(tmp_path):
    block = "## Q1: One\n<!-- expected: ANSWER -->\n\nWhat is ra?\n\n"
    with pytest.raises(ValueError):
        parse_queries(write(tmp_path / "q.md", block + block))


def test_run_all_writes_traces_manifest_and_summary(tmp_path):
    queries = parse_queries(write(tmp_path / "q.md", (
        "# Queries\n\nIntro text is ignored.\n\n"
        "## Q1: Jump\n<!-- expected: ANSWER -->\n\nWhat does JALR\nwrite to rd?\n\n"
        "## Q2: Football\n<!-- expected: INSUFFICIENT_EVIDENCE -->\n\nWho is the best quarterback?\n"
    )))
    model = ScriptedModel([
        turn(call("search_documents", query="JALR", sources=["riscv-isa-2025"])),
        turn(call("finish", outcome="answer", answer="JALR writes pc+4 to rd [1].",
                  citations=[{"chunk_id": JALR.chunk_id, "supports": "JALR writes pc+4"}])),
        # Q2 never finishes: three searches that match nothing, so it ends at the step limit.
        turn(call("search_documents", query="quarterback", sources=["guo-2013"])),
        turn(call("search_documents", query="football statistics", sources=["guo-2013"])),
        turn(call("search_documents", query="NFL ranking", sources=["guo-2013"])),
    ])
    manifest = {"index_version": "48c92aea33b03628", "embedding_model": "Qwen3-Embedding-8B",
                "chunking": {"chunk_size": 1000, "chunk_overlap": 150}}
    out = tmp_path / "runs"
    results = run_all(queries, Config(retry_delay_seconds=0, max_steps=3), out, model=model,
                      retriever=FakeRetriever([JALR]), registry=REGISTRY, manifest=manifest)

    assert [r.terminal_reason for r in results] == [TerminalReason.ANSWER, TerminalReason.STEP_LIMIT]
    assert results[0].question == "What does JALR write to rd?"
    assert (out / "Q1.jsonl").is_file() and (out / "Q2.jsonl").is_file()
    assert json.loads((out / "manifest.json").read_text()) == manifest

    summary = (out / "summary.md").read_text()
    assert "## Recorded configuration" in summary
    assert "48c92aea33b03628" in summary and "Qwen3-Embedding-8B" in summary and "hybrid" in summary
    assert "| Query | Title | Expected | Actual | Match | Steps | Sources cited |" in summary
    # Data rows start "| Q<digit>"; the header row "| Query |" must not match (test defect found by the
    # coding agent on 2026-10-27: a plain startswith("| Q") also matched the header).
    rows = [line for line in summary.splitlines() if re.match(r"\| Q\d", line)]
    assert rows[0].split("|")[1:-1] == [" Q1 ", " Jump ", " ANSWER ", " ANSWER ", " yes ", " 2/3 ",
                                        " riscv-isa-2025 "]
    assert rows[1].split("|")[1:-1] == [" Q2 ", " Football ", " INSUFFICIENT_EVIDENCE ", " STEP_LIMIT ",
                                        " no ", " 3/3 ", " none "]
