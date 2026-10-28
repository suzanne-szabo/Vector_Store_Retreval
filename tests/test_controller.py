"""Black-box tests for rag_agent.controller and rag_agent.session, derived from specs/controller.md.

Covers the assignment's requirements for rejecting a citation that retrieval did not return,
enforcing the step limit, and reaching all four terminal reasons.
"""

from __future__ import annotations

import json

import pytest

from fakes import ALL, JALR, LOOP, RA, REGISTRY, FakeRetriever, chunk_id
from rag_agent.controller import (
    NO_ACTION_REMINDER, Approval, Config, Controller, ErrorKind, FinalResult, Observation, TerminalReason,
    TraceEntry,
)
from rag_agent.session import Session

FAST = Config(retry_delay_seconds=0)


def make(retriever=None, config=FAST, question="What does JALR do?"):
    retriever = retriever if retriever is not None else FakeRetriever(ALL)
    session = Session.new(question, config, retriever, REGISTRY)
    return session, Controller(session, config)


def handle(controller, tool, args):
    """One model turn: a step, then validate and (if approved) execute."""
    controller.begin_step()
    result = controller.validate(tool, args)
    if isinstance(result, Approval):
        return controller.execute(result)
    return result


def search(controller, **args):
    return handle(controller, "search_documents", {"query": "JALR", **args})


def cite(*items, outcome="answer", answer=None):
    citations = [{"chunk_id": c, "supports": "claim"} for c in items]
    if answer is None:
        answer = " ".join(f"Claim {n} [{n}]." for n in range(1, len(citations) + 1)) or "Nothing found."
    return {"outcome": outcome, "answer": answer, "citations": citations}


# ---------------------------------------------------------------- Config and Session (§1-§2)

def test_config_defaults():
    config = Config()
    assert config.max_steps == 8
    assert config.default_k == 5
    assert config.max_consecutive_rejections == 3
    assert config.temperature == 0.0
    assert config.retrieval_mode == "hybrid"
    assert config.query_instruction == "Given a question, retrieve passages that answer it"
    assert config.retry_delay_seconds == 2.0
    assert config.no_action_reminders == 1
    assert set(config.as_dict()) >= {"max_steps", "default_k", "max_consecutive_rejections", "model_name",
                                     "temperature", "retrieval_mode", "query_instruction", "retry_delay_seconds",
                                     "no_action_reminders"}


@pytest.mark.parametrize("bad", [{"max_steps": 0}, {"default_k": 0}, {"default_k": 9},
                                 {"max_consecutive_rejections": 0}, {"no_action_reminders": -1}])
def test_config_rejects_out_of_range_values(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_session_new():
    session, _ = make(question="  What does JALR do?  ")
    assert session.question == "What does JALR do?"
    assert session.retrieved == {}
    assert "riscv-psabi" in session.source_ids and len(session.source_ids) == 9


def test_session_rejects_an_empty_question():
    with pytest.raises(ValueError):
        Session.new("   ", FAST, FakeRetriever(ALL), REGISTRY)


# ---------------------------------------------------------------- steps (§4)

def test_steps_are_counted_by_begin_step():
    _, controller = make(config=Config(retry_delay_seconds=0, max_steps=5))
    assert (controller.steps_used, controller.steps_remaining) == (0, 5)
    controller.begin_step()
    assert (controller.steps_used, controller.steps_remaining) == (1, 4)


def test_observations_carry_steps_remaining():
    _, controller = make()
    observation = search(controller, sources=["riscv-isa-2025"])
    assert observation.steps_remaining == 7


# ---------------------------------------------------------------- P2, P3 (§5.1, §5.2)

def assert_rejected(observation, policy, reason):
    assert isinstance(observation, Observation)
    assert observation.accepted is False
    assert observation.policy == policy
    assert observation.reason == reason
    assert observation.data["reason"] == reason
    assert observation.message


def test_unknown_tool_is_rejected_under_p2():
    session, controller = make()
    observation = handle(controller, "read_page", {"page": 3})
    assert_rejected(observation, "P2", "UNKNOWN_TOOL")
    assert controller.consecutive_rejections == 1
    entry = controller.trace[-1]
    assert isinstance(entry, TraceEntry)
    assert (entry.kind, entry.tool, entry.args, entry.accepted, entry.policy) == \
        ("tool_call", "read_page", {"page": 3}, False, "P2")
    assert session.retrieved == {} and controller.final is None


def test_schema_violation_is_rejected_under_p3_with_field_errors():
    _, controller = make()
    observation = handle(controller, "search_documents", {"query": "JALR", "k": 99})
    assert_rejected(observation, "P3", "BAD_ARGS")
    assert observation.data["errors"]


def test_too_long_answer_rejection_states_length_and_maximum():
    _, controller = make()
    search(controller)
    observation = handle(controller, "finish", cite(JALR.chunk_id, answer="Claim [1]. " + "x" * 8205))
    assert_rejected(observation, "P3", "BAD_ARGS")
    error = next(e for e in observation.data["errors"] if e["field"] == "answer")
    assert (error["length"], error["max"]) == (8216, 8000)
    assert "8,216" in observation.message and "8,000" in observation.message


def test_unparseable_arguments_are_rejected_under_p3():
    _, controller = make()
    observation = handle(controller, "search_documents", '{"query": "JALR"')
    assert_rejected(observation, "P3", "BAD_ARGS")
    assert controller.trace[-1].args == '{"query": "JALR"'


def test_unknown_source_id_is_rejected_under_p3():
    retriever = FakeRetriever(ALL)
    _, controller = make(retriever)
    observation = search(controller, sources=["riscv-isa-2025", "food-code"])
    assert_rejected(observation, "P3", "UNKNOWN_SOURCE")
    assert observation.data["unknown"] == ["food-code"]
    assert set(observation.data["known"]) == {source.id for source in REGISTRY}
    assert retriever.calls == []               # validation never calls the retriever


# ---------------------------------------------------------------- P4, P5, P6 (§5.3)

def test_finish_before_any_search_is_rejected_under_p4():
    _, controller = make()
    observation = handle(controller, "finish", cite(outcome="insufficient_evidence"))
    assert_rejected(observation, "P4", "NO_SEARCH_YET")
    assert controller.final is None


def test_a_rejected_search_does_not_count_for_p4():
    _, controller = make()
    handle(controller, "search_documents", {"query": "x"})          # rejected under P3
    observation = handle(controller, "finish", cite(outcome="insufficient_evidence"))
    assert_rejected(observation, "P4", "NO_SEARCH_YET")


def test_citation_not_returned_by_retrieval_is_rejected_then_accepted_once_retrieved():
    session, controller = make()
    search(controller, sources=["riscv-isa-2025"])                    # returns JALR only
    observation = handle(controller, "finish", cite(RA.chunk_id))
    assert_rejected(observation, "P5", "UNRETRIEVED_CITATION")
    assert observation.data["unretrieved"] == [RA.chunk_id]
    assert controller.final is None

    search(controller, sources=["riscv-psabi"])                       # now RA is retrieved
    accepted = handle(controller, "finish", cite(RA.chunk_id))
    assert accepted.accepted is True
    assert isinstance(controller.final, FinalResult)
    assert controller.final.citations[0]["source_id"] == "riscv-psabi"
    assert controller.final.citations[0]["page"] == 9


def test_a_never_seen_but_well_formed_id_is_rejected():
    _, controller = make()
    search(controller)
    assert_rejected(handle(controller, "finish", cite(chunk_id(999))), "P5", "UNRETRIEVED_CITATION")


@pytest.mark.parametrize("args, reason", [
    (cite(), "NO_CITATIONS"),
    (cite(JALR.chunk_id, JALR.chunk_id), "DUPLICATE_CITATION"),
    (cite(JALR.chunk_id, answer="Claim [2]."), "DANGLING_MARKER"),
    (cite(JALR.chunk_id, answer="Claim [0] and [1]."), "DANGLING_MARKER"),
    (cite(JALR.chunk_id, RA.chunk_id, answer="Only one marker [1]."), "UNUSED_CITATION"),
    (cite(outcome="insufficient_evidence", answer="See [1]."), "DANGLING_MARKER"),
])
def test_marker_and_citation_rules_under_p6(args, reason):
    _, controller = make()
    search(controller)                                                # returns RA, JALR, LOOP
    observation = handle(controller, "finish", args)
    assert_rejected(observation, "P6", reason)


def test_p5_is_checked_before_p6():
    _, controller = make()
    search(controller, sources=["riscv-isa-2025"])
    observation = handle(controller, "finish", cite(RA.chunk_id, answer="No markers here."))
    assert_rejected(observation, "P5", "UNRETRIEVED_CITATION")


def test_comma_markers_resolve_to_several_citations():
    _, controller = make()
    search(controller)
    observation = handle(controller, "finish", cite(RA.chunk_id, JALR.chunk_id, answer="Both say so [1, 2]."))
    assert observation.accepted is True
    assert [c["n"] for c in controller.final.citations] == [1, 2]


def test_insufficient_evidence_may_have_no_citations():
    _, controller = make()
    search(controller, sources=["guo-2013"])                          # nothing matches
    observation = handle(controller, "finish", cite(outcome="insufficient_evidence",
                                                    answer="The corpus does not cover this."))
    assert observation.accepted is True
    assert controller.final.outcome == "insufficient_evidence"


def test_insufficient_evidence_citations_follow_the_same_rules():
    _, controller = make()
    search(controller)
    ok = handle(controller, "finish", cite(LOOP.chunk_id, outcome="insufficient_evidence",
                                           answer="Only a related passage was found [1]."))
    assert ok.accepted is True


# ---------------------------------------------------------------- P7, P1 (§5.1, §5.4)

def test_extra_calls_are_rejected_under_p7_without_using_a_step():
    _, controller = make()
    search(controller)
    steps, rejections = controller.steps_used, controller.consecutive_rejections
    observation = controller.reject_extra_call("search_documents", {"query": "again"})
    assert_rejected(observation, "P7", "EXTRA_CALL")
    assert observation.message == "Only one tool call per turn is carried out; this call was ignored."
    assert (controller.steps_used, controller.consecutive_rejections) == (steps, rejections)
    assert controller.trace[-1].policy == "P7"


def test_calls_after_the_run_ended_are_rejected_under_p1():
    _, controller = make()
    search(controller)
    handle(controller, "finish", cite(JALR.chunk_id))
    assert controller.check_terminal() is TerminalReason.ANSWER
    rejections = controller.consecutive_rejections
    observation = controller.validate("search_documents", {"query": "more"})
    assert_rejected(observation, "P1", "TERMINAL")
    assert controller.consecutive_rejections == rejections


def test_rejections_change_nothing_but_counters_and_trace():
    session, controller = make()
    search(controller, sources=["riscv-isa-2025"])
    before = dict(session.retrieved)
    handle(controller, "finish", cite(RA.chunk_id))
    handle(controller, "search_documents", {"query": "x"})
    assert session.retrieved == before and controller.final is None


def test_an_accepted_call_resets_consecutive_rejections():
    _, controller = make()
    handle(controller, "nope", {})
    handle(controller, "nope", {})
    assert controller.consecutive_rejections == 2
    search(controller)
    assert controller.consecutive_rejections == 0


# ---------------------------------------------------------------- retrieval errors (§6)

def test_one_retrieval_failure_is_retried():
    retriever = FakeRetriever(ALL, fail_times=1)
    _, controller = make(retriever)
    observation = search(controller)
    assert observation.accepted is True and len(retriever.calls) == 2
    assert controller.check_terminal() is None


def test_two_retrieval_failures_end_the_run_as_retrieval_error():
    _, controller = make(FakeRetriever(ALL, fail_times=2))
    observation = search(controller)
    assert observation.accepted is False
    assert observation.policy is None and observation.reason == "RETRIEVAL_ERROR"
    assert observation.data["status"] == "error" and "RuntimeError" in observation.data["error"]
    assert controller.consecutive_rejections == 0
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.RETRIEVAL_ERROR


# ---------------------------------------------------------------- terminal checks (§7)

def test_answer():
    _, controller = make()
    search(controller)
    handle(controller, "finish", cite(JALR.chunk_id))
    assert controller.check_terminal() is TerminalReason.ANSWER
    assert controller.error_kind is None
    assert controller.terminal_message == "Answered with 1 citation(s)."


def test_insufficient_evidence():
    _, controller = make()
    search(controller)
    handle(controller, "finish", cite(outcome="insufficient_evidence"))
    assert controller.check_terminal() is TerminalReason.INSUFFICIENT_EVIDENCE
    assert controller.terminal_message == "The retrieved passages do not support an answer."


def test_model_error():
    _, controller = make()
    controller.begin_step()
    controller.record_model_error(TimeoutError("gateway timed out"))
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.MODEL_ERROR
    assert "TimeoutError" in controller.terminal_message
    assert controller.trace[-1].kind == "model_error"


def test_no_action_gets_one_reminder_then_ends_the_run():
    _, controller = make()
    controller.begin_step()
    reminder = controller.record_no_action()
    assert reminder == NO_ACTION_REMINDER and "finish" in reminder
    assert controller.check_terminal() is None
    assert controller.consecutive_rejections == 1
    assert controller.trace[-1].kind == "no_action"

    controller.begin_step()
    assert controller.record_no_action() is None
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.NO_ACTION
    assert controller.terminal_message == "The model replied without a tool call."
    assert [e.kind for e in controller.trace] == ["no_action", "no_action"]


def test_no_action_without_reminders_ends_at_once():
    _, controller = make(config=Config(retry_delay_seconds=0, no_action_reminders=0))
    controller.begin_step()
    assert controller.record_no_action() is None
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.NO_ACTION


def test_rejection_limit():
    _, controller = make()
    for _ in range(2):
        handle(controller, "nope", {})
        assert controller.check_terminal() is None
    handle(controller, "nope", {})
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.REJECTION_LIMIT
    assert controller.terminal_message == "3 rejected requests in a row."


def test_step_limit_is_enforced_exactly():
    _, controller = make(config=Config(retry_delay_seconds=0, max_steps=3))
    for expected in (None, None, TerminalReason.STEP_LIMIT):
        search(controller)
        assert controller.check_terminal() is expected
    assert controller.steps_used == 3
    assert controller.terminal_message == "Used all 3 steps without finishing."


def test_a_valid_finish_on_the_last_step_is_an_answer():
    _, controller = make(config=Config(retry_delay_seconds=0, max_steps=3))
    search(controller)
    search(controller)
    handle(controller, "finish", cite(JALR.chunk_id))
    assert controller.check_terminal() is TerminalReason.ANSWER


def test_rejection_limit_beats_step_limit_on_the_same_step():
    _, controller = make(config=Config(retry_delay_seconds=0, max_steps=3))
    for _ in range(3):
        handle(controller, "nope", {})
    assert controller.check_terminal() is TerminalReason.ERROR
    assert controller.error_kind is ErrorKind.REJECTION_LIMIT


def test_terminal_reason_is_final():
    _, controller = make()
    search(controller)
    handle(controller, "finish", cite(JALR.chunk_id))
    controller.check_terminal()
    controller.record_no_action()
    assert controller.check_terminal() is TerminalReason.ANSWER


# ---------------------------------------------------------------- trace (§8)

def test_trace_records_calls_in_order_with_the_observation_the_model_saw():
    _, controller = make()
    first = search(controller, sources=["riscv-isa-2025"])
    second = handle(controller, "finish", cite(RA.chunk_id))
    assert [entry.kind for entry in controller.trace] == ["tool_call", "tool_call"]
    assert [entry.step for entry in controller.trace] == [1, 2]
    assert controller.trace[0].observation == json.loads(first.to_json())
    assert controller.trace[1].observation == json.loads(second.to_json())
    assert controller.trace[0].steps_remaining == 7


def test_thoughts_are_traced_only_when_not_blank():
    _, controller = make()
    controller.begin_step()
    controller.record_thought("   ")
    controller.record_thought("I should search the psABI.")
    assert [(e.kind, e.message) for e in controller.trace] == [("thought", "I should search the psABI.")]
