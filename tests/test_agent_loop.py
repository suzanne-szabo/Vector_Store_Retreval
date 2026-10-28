"""Black-box tests for the LangGraph loop (rag_agent.agent.run), derived from specs/agent.md.

All runs use the scripted fake model and a fake retriever: no network, no API key. Covers the
assignment's "at least one supported run and one insufficient-evidence run using scripted model
actions and substitute retrieval results", the step limit, and all four terminal reasons.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from fakes import (
    ALL, JALR, RA, REGISTRY, FakeRetriever, ScriptedModel, UnbindableModel, bad_json_turn, call, turn,
)
from rag_agent.agent import RunResult, run
from rag_agent.controller import NO_ACTION_REMINDER, Config, ErrorKind, TerminalReason

FAST = Config(retry_delay_seconds=0)
QUESTION = "What does the RISC-V JALR instruction write to its destination register?"


def go(script, retriever=None, config=FAST, question=QUESTION, on_entry=None):
    model = ScriptedModel(script)
    result = run(question, config, model=model, retriever=retriever or FakeRetriever(ALL),
                 registry=REGISTRY, on_entry=on_entry)
    return result, model


def search(**args):
    return turn(call("search_documents", **args))


def finish(outcome, answer, *chunk_ids):
    citations = [{"chunk_id": c, "supports": "the claim"} for c in chunk_ids]
    return turn(call("finish", outcome=outcome, answer=answer, citations=citations))


def tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------- the two required scripted runs

def test_supported_run_ends_with_an_answer_citing_retrieved_evidence():
    result, model = go([
        search(query="JALR destination register", sources=["riscv-isa-2025"]),
        finish("answer", "JALR writes pc+4 to register rd [1].", JALR.chunk_id),
    ])
    assert isinstance(result, RunResult)
    assert result.terminal_reason is TerminalReason.ANSWER
    assert result.error_kind is None
    assert result.outcome == "answer"
    assert result.answer == "JALR writes pc+4 to register rd [1]."
    assert result.citations == ({"n": 1, "chunk_id": JALR.chunk_id, "source_id": "riscv-isa-2025",
                                 "title": JALR.title, "page": 51, "supports": "the claim"},)
    assert (result.steps_used, result.max_steps) == (2, 8)
    calls = [e for e in result.trace if e.kind == "tool_call"]
    assert [(e.tool, e.accepted) for e in calls] == [("search_documents", True), ("finish", True)]
    assert calls[0].observation["data"]["results"][0]["chunk_id"] == JALR.chunk_id
    assert result.question == QUESTION
    assert result.config == FAST.as_dict()


def test_insufficient_evidence_run():
    result, _ = go([
        search(query="CPython RISC-V code generation for loops"),
        finish("insufficient_evidence", "No retrieved passage describes CPython generating RISC-V code."),
    ], retriever=FakeRetriever([]))
    assert result.terminal_reason is TerminalReason.INSUFFICIENT_EVIDENCE
    assert result.outcome == "insufficient_evidence"
    assert result.citations == ()
    assert result.steps_used == 2


# ---------------------------------------------------------------- citations checked inside the loop

def test_an_unretrieved_citation_is_rejected_and_the_model_can_correct_it():
    result, model = go([
        search(query="JALR", sources=["riscv-isa-2025"]),
        finish("answer", "ra holds the return address [1].", RA.chunk_id),      # never retrieved
        finish("answer", "JALR writes pc+4 to rd [1].", JALR.chunk_id),
    ])
    assert result.terminal_reason is TerminalReason.ANSWER
    assert result.steps_used == 3
    rejection = json.loads(tool_messages(model.inputs[2])[-1].content)
    assert (rejection["accepted"], rejection["policy"], rejection["reason"]) == (False, "P5", "UNRETRIEVED_CITATION")


def test_finish_before_searching_is_rejected():
    result, model = go([
        finish("insufficient_evidence", "I do not know."),
        search(query="quarterback statistics"),
        finish("insufficient_evidence", "The corpus covers Python and RISC-V, not football."),
    ], retriever=FakeRetriever([]))
    assert result.terminal_reason is TerminalReason.INSUFFICIENT_EVIDENCE
    assert json.loads(tool_messages(model.inputs[1])[-1].content)["policy"] == "P4"


# ---------------------------------------------------------------- step limit and error reasons

def test_step_limit_ends_a_run_that_never_finishes():
    result, model = go([search(query=f"query number {i}") for i in range(10)],
                       config=Config(retry_delay_seconds=0, max_steps=3))
    assert result.terminal_reason is TerminalReason.STEP_LIMIT
    assert result.steps_used == 3
    assert len(model.inputs) == 3
    assert result.message == "Used all 3 steps without finishing."


def test_two_replies_without_a_tool_call_end_as_no_action():
    result, _ = go([AIMessage(content="JALR writes pc+4 to rd."), AIMessage(content="As I said.")])
    assert result.terminal_reason is TerminalReason.ERROR
    assert result.error_kind is ErrorKind.NO_ACTION
    assert result.steps_used == 2                      # the first reply got one reminder (D24)


def test_a_plain_text_answer_gets_a_reminder_and_the_model_can_then_finish():
    # Required run 1, Q3: the model wrote its cited answer as plain text instead of calling finish.
    result, model = go([
        search(query="JALR", sources=["riscv-isa-2025"]),
        AIMessage(content="JALR writes pc+4 to rd [1]."),
        finish("answer", "JALR writes pc+4 to rd [1].", JALR.chunk_id),
    ])
    assert result.terminal_reason is TerminalReason.ANSWER
    assert result.steps_used == 3
    reminders = [m for m in model.inputs[2] if isinstance(m, HumanMessage) and m.content == NO_ACTION_REMINDER]
    assert len(reminders) == 1
    assert [e.kind for e in result.trace].count("no_action") == 1


def test_a_too_long_answer_is_rejected_with_its_length_and_can_be_shortened():
    # Required run 1, Q6: two over-long answers were rejected with a generic message.
    long_answer = "JALR writes pc+4 to rd [1]. " + "Detail. " * 1100
    result, model = go([
        search(query="JALR", sources=["riscv-isa-2025"]),
        finish("answer", long_answer, JALR.chunk_id),
        finish("answer", "JALR writes pc+4 to rd [1].", JALR.chunk_id),
    ])
    assert result.terminal_reason is TerminalReason.ANSWER
    rejection = json.loads(tool_messages(model.inputs[2])[-1].content)
    assert rejection["reason"] == "BAD_ARGS" and "8,000" in rejection["message"]


def test_model_failing_twice_is_model_error():
    result, model = go([RuntimeError("gateway down"), RuntimeError("gateway down")])
    assert result.terminal_reason is TerminalReason.ERROR
    assert result.error_kind is ErrorKind.MODEL_ERROR
    assert result.steps_used == 1                      # the retry does not use a step
    assert len(model.inputs) == 2


def test_model_failing_once_is_retried():
    result, model = go([
        RuntimeError("gateway hiccup"),
        search(query="JALR", sources=["riscv-isa-2025"]),
        finish("answer", "JALR writes pc+4 to rd [1].", JALR.chunk_id),
    ])
    assert result.terminal_reason is TerminalReason.ANSWER
    assert result.steps_used == 2


def test_retrieval_failing_twice_is_retrieval_error():
    result, _ = go([search(query="JALR")], retriever=FakeRetriever(ALL, fail_times=2))
    assert result.terminal_reason is TerminalReason.ERROR
    assert result.error_kind is ErrorKind.RETRIEVAL_ERROR


def test_three_rejections_in_a_row_hit_the_rejection_limit():
    result, _ = go([turn(call("read_page", page=1)), search(query="x"), turn(call("search_documents", k=3))])
    assert result.terminal_reason is TerminalReason.ERROR
    assert result.error_kind is ErrorKind.REJECTION_LIMIT
    assert result.steps_used == 3


def test_a_model_that_cannot_be_built_is_model_error_with_no_steps():
    result = run(QUESTION, FAST, model=UnbindableModel(), retriever=FakeRetriever(ALL), registry=REGISTRY)
    assert result.terminal_reason is TerminalReason.ERROR
    assert result.error_kind is ErrorKind.MODEL_ERROR
    assert result.steps_used == 0


def test_an_empty_question_is_a_value_error():
    with pytest.raises(ValueError):
        run("   ", FAST, model=ScriptedModel([]), retriever=FakeRetriever(ALL), registry=REGISTRY)


# ---------------------------------------------------------------- model input and binding

def test_first_model_input_is_system_prompt_question_status():
    _, model = go([AIMessage(content="")])
    first = model.inputs[0]
    assert [type(m) for m in first] == [SystemMessage, HumanMessage, HumanMessage]
    assert "Sources:" in first[0].content
    assert first[1].content == QUESTION
    assert first[2].content.startswith("Step 1 of 8.")


def test_later_inputs_carry_history_and_a_fresh_status():
    _, model = go([search(query="JALR", sources=["riscv-isa-2025"]), AIMessage(content="")])
    second = model.inputs[1]
    assert isinstance(second[2], AIMessage) and isinstance(second[3], ToolMessage)
    assert second[-1].content.startswith("Step 2 of 8.")
    assert JALR.chunk_id in second[-1].content
    statuses = [m for m in second if isinstance(m, HumanMessage)]
    assert len(statuses) == 2                          # the question and the one current status


def test_tools_are_bound_with_forced_single_calls():
    _, model = go([AIMessage(content="")])
    assert model.bound_tool_choice == "required"
    assert model.bind_kwargs.get("parallel_tool_calls") is False
    assert [t["function"]["name"] for t in model.bound_tools] == ["search_documents", "finish"]


# ---------------------------------------------------------------- extra calls, bad JSON, thoughts

def test_only_the_first_of_several_calls_runs_and_every_call_is_answered():
    first, second = call("search_documents", query="JALR"), call("search_documents", query="ADDI")
    retriever = FakeRetriever(ALL)
    result, model = go([turn(first, second), AIMessage(content="")], retriever=retriever)
    assert len(retriever.calls) == 1 and retriever.calls[0][0] == "JALR"
    answered = tool_messages(model.inputs[1])
    assert [m.tool_call_id for m in answered] == [first["id"], second["id"]]
    calls = [e for e in result.trace if e.kind == "tool_call"]
    assert [(e.accepted, e.policy) for e in calls] == [(True, None), (False, "P7")]


def test_unparseable_arguments_are_rejected_not_crashed():
    result, model = go([bad_json_turn("search_documents", '{"query": "JALR"'), AIMessage(content="")])
    rejection = json.loads(tool_messages(model.inputs[1])[-1].content)
    assert (rejection["policy"], rejection["reason"]) == ("P3", "BAD_ARGS")
    assert result.error_kind is ErrorKind.NO_ACTION


def test_model_text_is_traced_as_a_thought():
    result, _ = go([turn(call("search_documents", query="JALR"), content="I will search the ISA manual."),
                    AIMessage(content="")])
    thoughts = [e for e in result.trace if e.kind == "thought"]
    assert thoughts[0].message == "I will search the ISA manual."


def test_on_entry_sees_every_trace_entry_in_order():
    seen = []
    result, _ = go([search(query="JALR", sources=["riscv-isa-2025"]),
                    finish("answer", "JALR writes pc+4 to rd [1].", JALR.chunk_id)], on_entry=seen.append)
    assert tuple(seen) == result.trace


def test_run_records_start_and_finish_times():
    result, _ = go([AIMessage(content="")])
    assert result.started_at <= result.finished_at
    assert result.started_at.endswith(("Z", "+00:00"))


def test_recursion_backstop_never_preempts_the_step_limit():
    result, _ = go([search(query=f"query number {i}") for i in range(30)],
                   config=Config(retry_delay_seconds=0, max_steps=20))
    assert result.terminal_reason is TerminalReason.STEP_LIMIT
    assert result.steps_used == 20
