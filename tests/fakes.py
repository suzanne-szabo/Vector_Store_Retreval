"""Test doubles for the agent tests (specs/agent.md §8). No network, no API key.

- ScriptedModel, call, turn, bad_json_turn: carried over from HW-08. Each invoke() returns (or raises)
  the next scripted item; when the script is exhausted it returns an AIMessage with no tool calls.
- FakeRetriever and passage(): substitute retrieval results the test defines.
"""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.messages import AIMessage

from rag_agent.config import DEFAULT_REGISTRY_PATH
from rag_agent.corpus import load_registry
from rag_agent.retrieval import Passage

_ids = itertools.count(1)
REGISTRY = tuple(load_registry(DEFAULT_REGISTRY_PATH))
SUBJECT = {source.id: source.subject for source in REGISTRY}
TITLE = {source.id: source.title for source in REGISTRY}


# ---------------------------------------------------------------- scripted chat model (HW-08)

def call(name: str, **args: Any) -> dict:
    """One well-formed tool call."""
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def turn(*calls: dict, content: str = "") -> AIMessage:
    """An AIMessage carrying the given tool calls."""
    return AIMessage(content=content, tool_calls=list(calls))


def bad_json_turn(name: str, raw_args: str) -> AIMessage:
    """An AIMessage whose single tool call has unparseable JSON arguments."""
    return AIMessage(content="", invalid_tool_calls=[{
        "name": name, "args": raw_args, "id": f"call_{next(_ids)}",
        "error": "invalid JSON", "type": "invalid_tool_call",
    }])


class ScriptedModel:
    def __init__(self, responses: list[AIMessage | Exception]) -> None:
        self.responses = list(responses)
        self.inputs: list[list[Any]] = []
        self.bound_tools: list[Any] | None = None
        self.bound_tool_choice: Any = None
        self.bind_kwargs: dict = {}

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        self.bound_tools = list(tools)
        self.bound_tool_choice = tool_choice
        self.bind_kwargs = kwargs
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.inputs.append(list(messages))
        if not self.responses:
            return AIMessage(content="")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class UnbindableModel:
    """A model whose tool binding fails, as when the chat service cannot be configured."""

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        raise RuntimeError("LLM_API_KEY is not set")


# ---------------------------------------------------------------- substitute retrieval

def chunk_id(n: int) -> str:
    """A valid 24-character lowercase hex chunk id."""
    return f"{n:024x}"


def passage(n: int, source_id: str = "riscv-psabi", page: int = 9,
            text: str = "x1  ra  Return address  No", distance: float = 0.2) -> Passage:
    return Passage(chunk_id=chunk_id(n), source_id=source_id, title=TITLE[source_id], page=page,
                   text=text, distance=distance, vector_rank=n, keyword_rank=None)


class FakeRetriever:
    """Returns the first k passages matching the filters, ranked as given; records every call."""

    def __init__(self, passages=(), fail_times: int = 0) -> None:
        self.passages = list(passages)
        self.fail_times = fail_times
        self.calls: list[tuple] = []

    def search(self, query, k, sources=None, subject=None):
        self.calls.append((query, k, sources, subject))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("vector store unavailable")
        matches = [p for p in self.passages
                   if (sources is None or p.source_id in sources)
                   and (subject is None or SUBJECT[p.source_id] == subject)]
        return matches[:k]


# Three passages used across the agent tests.
RA = passage(1, "riscv-psabi", 9, "x1  ra  Return address  No")
JALR = passage(2, "riscv-isa-2025", 51, "The indirect jump instruction JALR (jump and link register) "
               "uses the I-type encoding. The address of the instruction following the jump (pc+4) "
               "is written to register rd.")
LOOP = passage(3, "python-crash-course", 90, "Forgetting to Indent: always indent the line after the "
               "for statement in a loop.")
ALL = [RA, JALR, LOOP]
