"""Black-box tests for rag_agent.prompts, derived from specs/agent.md §2."""

from __future__ import annotations

from fakes import ALL, JALR, RA, REGISTRY, FakeRetriever
from rag_agent.controller import Approval, Config, Controller
from rag_agent.prompts import status_message, system_prompt
from rag_agent.session import Session

FAST = Config(retry_delay_seconds=0)


def make(max_steps=8):
    config = Config(retry_delay_seconds=0, max_steps=max_steps)
    session = Session.new("What does JALR do?", config, FakeRetriever(ALL), REGISTRY)
    return session, Controller(session, config)


def search(controller, **args):
    controller.begin_step()
    approval = controller.validate("search_documents", {"query": "JALR", **args})
    assert isinstance(approval, Approval)
    return controller.execute(approval)


def test_system_prompt_lists_every_source_in_registry_order():
    prompt = system_prompt(REGISTRY)
    assert "Sources:" in prompt
    lines = [f"- {s.id} ({s.subject}): {s.title}. {s.role}" for s in REGISTRY]
    positions = [prompt.index(line) for line in lines]
    assert positions == sorted(positions)
    assert prompt.index("Sources:") < positions[0]


def test_system_prompt_states_the_rules():
    prompt = system_prompt(REGISTRY)
    for phrase in ("search_documents", "finish", "insufficient_evidence", "[n]", "2025", "2019",
                   "8,000", "plain text", "at most 12"):
        assert phrase in prompt


def test_status_before_any_search():
    session, controller = make()
    controller.begin_step()
    assert status_message(session, controller) == (
        "Step 1 of 8.\n"
        "Retrieved passages you may cite: none yet. Call search_documents first."
    )


def test_status_lists_retrieved_chunks_in_retrieval_order():
    session, controller = make()
    search(controller, sources=["riscv-isa-2025"])
    search(controller, sources=["riscv-psabi"])
    controller.begin_step()
    assert status_message(session, controller) == (
        "Step 3 of 8.\n"
        "Retrieved passages you may cite (chunk_id · source_id · page):\n"
        f"- {JALR.chunk_id} · riscv-isa-2025 · p.51\n"
        f"- {RA.chunk_id} · riscv-psabi · p.9"
    )


def test_status_on_the_last_step_tells_the_model_to_finish():
    session, controller = make(max_steps=2)
    search(controller, sources=["riscv-psabi"])
    controller.begin_step()
    assert status_message(session, controller) == (
        "Step 2 of 2.\n"
        "Retrieved passages you may cite (chunk_id · source_id · page):\n"
        f"- {RA.chunk_id} · riscv-psabi · p.9\n"
        "This is your last step: call finish now."
    )
