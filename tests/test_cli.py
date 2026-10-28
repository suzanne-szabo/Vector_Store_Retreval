"""Black-box tests for the CLI (rag_agent.cli.main) and the JSONL trace, derived from specs/agent.md §5-§6."""

from __future__ import annotations

import json

import pytest

from fakes import ALL, JALR, FakeRetriever, ScriptedModel, call, turn
from rag_agent.cli import main

QUESTION = "What does JALR write to rd?"


def answering_model():
    return ScriptedModel([
        turn(call("search_documents", query="JALR", sources=["riscv-isa-2025"])),
        turn(call("finish", outcome="answer", answer="JALR writes pc+4 to rd [1].",
                  citations=[{"chunk_id": JALR.chunk_id, "supports": "JALR writes pc+4"}])),
    ])


def test_answer_exits_0_and_prints_the_answer_with_citations(capsys):
    status = main([QUESTION], model=answering_model(), retriever=FakeRetriever(ALL))
    out = capsys.readouterr().out
    assert status == 0
    assert "[step 1/8] search_documents" in out
    assert "ANSWER" in out
    assert "JALR writes pc+4 to rd [1]." in out
    assert f"[1] riscv-isa-2025, page 51 (chunk {JALR.chunk_id[:8]}" in out


def test_insufficient_evidence_exits_0():
    model = ScriptedModel([
        turn(call("search_documents", query="quarterback")),
        turn(call("finish", outcome="insufficient_evidence", answer="Not in the corpus.")),
    ])
    assert main([QUESTION], model=model, retriever=FakeRetriever([])) == 0


def test_step_limit_exits_1():
    model = ScriptedModel([turn(call("search_documents", query="JALR"))])
    assert main([QUESTION, "--max-steps", "1"], model=model, retriever=FakeRetriever(ALL)) == 1


def test_error_exits_1():
    assert main([QUESTION], model=ScriptedModel([]), retriever=FakeRetriever(ALL)) == 1   # NO_ACTION


def test_missing_store_exits_2(tmp_path, capsys):
    status = main([QUESTION, "--db", str(tmp_path / "nowhere")])
    assert status == 2
    assert "No vector store found" in capsys.readouterr().out


def test_help_lists_every_option_and_an_example(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "python -m rag_agent" in out
    for option in ("--max-steps", "--k", "--mode", "--query-instruction", "--db", "--trace-out", "--quiet"):
        assert option in out
    assert "example" in out.lower()


def test_quiet_hides_step_lines(capsys):
    main([QUESTION, "--quiet"], model=answering_model(), retriever=FakeRetriever(ALL))
    out = capsys.readouterr().out
    assert "[step" not in out and "ANSWER" in out


def test_k_and_mode_options_reach_the_run(tmp_path):
    retriever = FakeRetriever(ALL)
    main([QUESTION, "--k", "3", "--mode", "keyword", "--trace-out", str(tmp_path / "t.jsonl")],
         model=answering_model(), retriever=retriever)
    assert retriever.calls[0][1] == 3
    header = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert header["config"]["default_k"] == 3
    assert header["config"]["retrieval_mode"] == "keyword"


def test_trace_file_has_header_entries_and_result(tmp_path):
    path = tmp_path / "run.jsonl"
    main([QUESTION, "--trace-out", str(path), "--quiet"], model=answering_model(), retriever=FakeRetriever(ALL))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    header, entries, result = records[0], records[1:-1], records[-1]
    assert header["type"] == "run"
    assert header["question"] == QUESTION
    assert {"started_at", "finished_at", "config", "manifest"} <= set(header)
    assert entries and all(e["type"] == "entry" for e in entries)
    assert [e["tool"] for e in entries if e["kind"] == "tool_call"] == ["search_documents", "finish"]
    assert {"step", "kind", "tool", "args", "accepted", "policy", "reason", "message",
            "observation", "steps_remaining"} <= set(entries[0])
    assert result["type"] == "result"
    assert result["terminal_reason"] == "ANSWER" and result["error_kind"] is None
    assert result["outcome"] == "answer"
    assert result["citations"][0]["source_id"] == "riscv-isa-2025"
    assert (result["steps_used"], result["max_steps"]) == (2, 8)
