"""Black-box tests for rag_agent.tools, derived from specs/tools.md §2-§5.

Covers the assignment's "retrieval-tool argument validation and source-labeled results".
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from fakes import ALL, JALR, LOOP, RA, REGISTRY, FakeRetriever, chunk_id
from rag_agent.controller import Approval, Config, Observation
from rag_agent.session import Session
from rag_agent.tools import (
    MAX_ANSWER_CHARS, MAX_K, Citation, FinishArgs, SearchArgs, citation_markers, execute, tool_definitions,
)

FAST = Config(retry_delay_seconds=0)


def session(retriever=None, config=FAST):
    return Session.new("What does JALR do?", config, retriever or FakeRetriever(ALL), REGISTRY)


def search(**args):
    return Approval("search_documents", SearchArgs.model_validate(args))


def finish(**args):
    return Approval("finish", FinishArgs.model_validate(args))


# ---------------------------------------------------------------- SearchArgs (§3.1)

def test_minimal_search_args_use_defaults():
    args = SearchArgs.model_validate({"query": "JALR"})
    assert (args.query, args.sources, args.subject, args.k) == ("JALR", None, None, None)
    assert MAX_K == 8


def test_search_query_is_stripped():
    assert SearchArgs.model_validate({"query": "  callee-saved registers  "}).query == "callee-saved registers"


def test_full_search_args_are_accepted():
    args = SearchArgs.model_validate({"query": "ra register", "sources": ["riscv-psabi", "winans-rvalp"],
                                      "subject": "riscv", "k": 8})
    assert args.sources == ["riscv-psabi", "winans-rvalp"] and args.k == 8


@pytest.mark.parametrize("bad", [
    {},                                            # query missing
    {"query": "  "},                               # blank
    {"query": "ab"},                               # 2 characters
    {"query": "x" * 301},                          # too long
    {"query": "JALR", "k": 0},
    {"query": "JALR", "k": 9},
    {"query": "JALR", "k": "5"},                   # strict: no string-to-int coercion
    {"query": "JALR", "k": 5.0},
    {"query": "JALR", "sources": []},              # empty list
    {"query": "JALR", "sources": ["riscv-psabi", "riscv-psabi"]},   # duplicates
    {"query": "JALR", "sources": ["s"] * 10},      # more than 9
    {"query": "JALR", "sources": "riscv-psabi"},   # not a list
    {"query": "JALR", "subject": "java"},
    {"query": "JALR", "filters": {}},              # extra field
    {"query": 42},
])
def test_invalid_search_args_are_rejected(bad):
    with pytest.raises(ValidationError):
        SearchArgs.model_validate(bad)


# ---------------------------------------------------------------- FinishArgs and Citation (§3.2)

def test_valid_finish_args():
    args = FinishArgs.model_validate({"outcome": "answer", "answer": "JALR writes pc+4 to rd [1].",
                                      "citations": [{"chunk_id": chunk_id(2), "supports": "JALR writes pc+4"}]})
    assert args.outcome == "answer"
    assert isinstance(args.citations[0], Citation) and args.citations[0].chunk_id == chunk_id(2)


def test_answers_up_to_the_limit_are_accepted():
    assert MAX_ANSWER_CHARS == 8000
    assert len(FinishArgs.model_validate({"outcome": "answer", "answer": "x" * 8000}).answer) == 8000


def test_finish_citations_default_to_empty():
    args = FinishArgs.model_validate({"outcome": "insufficient_evidence", "answer": "Nothing found."})
    assert args.citations == []


@pytest.mark.parametrize("bad", [
    {"outcome": "maybe", "answer": "x"},
    {"outcome": "answer", "answer": "   "},
    {"outcome": "answer", "answer": "x" * 8001},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": "ABC", "supports": "y"}]},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": "0" * 23, "supports": "y"}]},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": "G" * 24, "supports": "y"}]},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": chunk_id(1), "supports": " "}]},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": chunk_id(1), "supports": "y" * 201}]},
    {"outcome": "answer", "answer": "x", "citations": [{"chunk_id": chunk_id(1), "supports": "y", "page": 9}]},
    {"outcome": "answer", "answer": "x",
     "citations": [{"chunk_id": chunk_id(i + 1), "supports": "y"} for i in range(13)]},
    {"outcome": "answer", "answer": "x", "confidence": 0.9},
])
def test_invalid_finish_args_are_rejected(bad):
    with pytest.raises(ValidationError):
        FinishArgs.model_validate(bad)


# ---------------------------------------------------------------- citation markers (§3.3)

@pytest.mark.parametrize("answer, expected", [
    ("JALR writes pc+4 [1].", [1]),
    ("A [2, 3] and B [1,2] and C [1].", [2, 3, 1, 2, 1]),
    ("Not markers: [a] [] [1-3] [1.5] [ ].", []),
    ("No markers at all.", []),
])
def test_citation_markers(answer, expected):
    assert citation_markers(answer) == expected


# ---------------------------------------------------------------- tool definitions (§4)

def test_tool_definitions_describe_both_tools():
    definitions = tool_definitions()
    assert [d["function"]["name"] for d in definitions] == ["search_documents", "finish"]
    for definition in definitions:
        assert definition["type"] == "function"
        assert definition["function"]["description"]
    search_params = definitions[0]["function"]["parameters"]["properties"]
    finish_params = definitions[1]["function"]["parameters"]["properties"]
    assert {"query", "sources", "subject", "k"} <= set(search_params)
    assert {"outcome", "answer", "citations"} <= set(finish_params)


# ---------------------------------------------------------------- search executor (§5.1)

def test_search_returns_source_labeled_passages():
    s = session()
    observation = execute(s, search(query="return address", sources=["riscv-psabi"]), steps_remaining=7)
    assert isinstance(observation, Observation)
    assert observation.tool == "search_documents" and observation.accepted is True
    assert observation.policy is None and observation.reason is None
    assert observation.steps_remaining == 7
    data = observation.data
    assert data["status"] == "ok"
    assert data["query"] == "return address"
    assert data["filters"] == {"sources": ["riscv-psabi"], "subject": None}
    assert data["results"] == [{
        "rank": 1, "chunk_id": RA.chunk_id, "source_id": "riscv-psabi", "title": RA.title, "page": 9,
        "distance": RA.distance, "vector_rank": RA.vector_rank, "keyword_rank": None, "text": RA.text,
    }]
    assert observation.message == "1 passage(s) from riscv-psabi."


def test_search_uses_default_k_and_passes_filters_by_keyword():
    retriever = FakeRetriever(ALL)
    s = session(retriever, Config(retry_delay_seconds=0, default_k=2))
    execute(s, search(query="registers", subject="riscv"), steps_remaining=7)
    assert retriever.calls == [("registers", 2, None, "riscv")]


def test_explicit_k_is_used():
    retriever = FakeRetriever(ALL)
    execute(session(retriever), search(query="registers", k=3), steps_remaining=7)
    assert retriever.calls[0][1] == 3


def test_search_message_lists_sources_in_rank_order_without_repeats():
    retriever = FakeRetriever([JALR, RA, replace(JALR, chunk_id=chunk_id(9))])
    observation = execute(session(retriever), search(query="jump"), steps_remaining=7)
    assert observation.message == "3 passage(s) from riscv-isa-2025, riscv-psabi."
    assert [r["rank"] for r in observation.data["results"]] == [1, 2, 3]


def test_retrieved_set_grows_in_order_without_duplicates():
    s = session()
    execute(s, search(query="jump", sources=["riscv-isa-2025"]), steps_remaining=7)
    execute(s, search(query="anything"), steps_remaining=6)
    assert list(s.retrieved) == [JALR.chunk_id, RA.chunk_id, LOOP.chunk_id]
    assert s.retrieved[JALR.chunk_id] == JALR


def test_empty_search_has_empty_status_and_hint():
    s = session(FakeRetriever(ALL))
    observation = execute(s, search(query="registers", sources=["guo-2013"]), steps_remaining=7)
    assert observation.accepted is True
    assert observation.data["status"] == "empty" and observation.data["results"] == []
    assert observation.message == "No passages matched. Try a different query or remove filters."
    assert s.retrieved == {}


def test_retriever_errors_propagate_from_the_executor():
    with pytest.raises(RuntimeError):
        execute(session(FakeRetriever(ALL, fail_times=1)), search(query="JALR"), steps_remaining=7)


def test_observation_json_is_what_the_model_sees():
    import json
    observation = execute(session(), search(query="JALR", sources=["riscv-isa-2025"]), steps_remaining=5)
    payload = json.loads(observation.to_json())
    assert set(payload) == {"tool", "accepted", "policy", "reason", "message", "data", "steps_remaining"}
    assert payload["data"]["results"][0]["chunk_id"] == JALR.chunk_id


# ---------------------------------------------------------------- finish executor (§5.2)

def test_finish_resolves_citations_with_source_and_page():
    s = session()
    execute(s, search(query="JALR", sources=["riscv-isa-2025"]), steps_remaining=7)
    observation = execute(s, finish(outcome="answer", answer="JALR writes pc+4 to rd [1].",
                                    citations=[{"chunk_id": JALR.chunk_id, "supports": "JALR writes pc+4"}]),
                          steps_remaining=6)
    assert observation.accepted is True and observation.tool == "finish"
    assert observation.message == "Run finished: answer."
    assert observation.data == {"outcome": "answer", "citations": [{
        "n": 1, "chunk_id": JALR.chunk_id, "source_id": "riscv-isa-2025", "title": JALR.title,
        "page": 51, "supports": "JALR writes pc+4",
    }]}
