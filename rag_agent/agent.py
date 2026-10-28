"""The bounded ReAct loop as a LangGraph graph, `run()` and the JSONL trace (specs/agent.md §1, §4, §6).

    START -> llm -> controller -> tool -> continue --(terminal)--> END
              |                                |
              +--(model error)--> continue     +--(otherwise)--> llm

Graph state holds only `messages` and `pending` (§1). The session, controller and bound model reach
the nodes through `config["configurable"]`, so model output touches application state only through
controller methods (architecture §5).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from rag_agent.config import DEFAULT_DB_PATH, DEFAULT_REGISTRY_PATH
from rag_agent.controller import Approval, Config, Controller, ErrorKind, TerminalReason, TraceEntry
from rag_agent.corpus import load_registry
from rag_agent.prompts import status_message, system_prompt
from rag_agent.session import Session
from rag_agent.tools import tool_definitions

MANIFEST_KEYS = ("index_version", "embedding_model", "embedding_dimension", "chunking", "extractor",
                 "cleaning", "record_count")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    pending: Approval | None


@dataclass(frozen=True)
class RunResult:
    question: str
    terminal_reason: TerminalReason
    error_kind: ErrorKind | None
    message: str
    outcome: str | None
    answer: str | None
    citations: tuple[dict, ...]
    steps_used: int
    max_steps: int
    trace: tuple[TraceEntry, ...]
    config: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------- helpers

def _parts(config: RunnableConfig) -> tuple[Session, Controller, Any]:
    configurable = config["configurable"]
    return configurable["session"], configurable["controller"], configurable["model"]


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def _calls(reply: AIMessage) -> list[tuple[str | None, Any, str]]:
    """`(name, raw_args, call_id)`: parsed calls first, then invalid ones with their raw argument string."""
    calls = [(call["name"], call["args"], call["id"] or "") for call in reply.tool_calls]
    calls += [(call.get("name"), call.get("args"), call.get("id") or "") for call in reply.invalid_tool_calls]
    return calls


def _answer(observation, name: str | None, call_id: str) -> ToolMessage:
    return ToolMessage(content=observation.to_json(), tool_call_id=call_id, name=name or "")


def _last_reply(state: AgentState) -> AIMessage:
    return next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))


def model_input(session: Session, controller: Controller, history: list[BaseMessage]) -> list[BaseMessage]:
    """System prompt, question, this run's history, then a fresh status message (§2)."""
    return [SystemMessage(system_prompt(session.registry)), HumanMessage(session.question), *history,
            HumanMessage(status_message(session, controller))]


# ---------------------------------------------------------------- nodes (§1.1-§1.4)

def llm_node(state: AgentState, config: RunnableConfig) -> Command[Literal["controller", "continue"]]:
    session, controller, model = _parts(config)
    controller.begin_step()
    messages = model_input(session, controller, state["messages"])
    try:
        reply = model.invoke(messages)
    except Exception:
        if session.config.retry_delay_seconds > 0:
            time.sleep(session.config.retry_delay_seconds)
        try:
            reply = model.invoke(messages)
        except Exception as error:
            controller.record_model_error(error)
            return Command(goto="continue")
    reply = reply.model_copy(update={"id": f"ai-{uuid.uuid4().hex}"})
    controller.record_thought(_text(reply))
    return Command(update={"messages": [reply]}, goto="controller")


def controller_node(state: AgentState, config: RunnableConfig) -> dict:
    _, controller, _ = _parts(config)
    calls = _calls(_last_reply(state))
    if not calls:
        reminder = controller.record_no_action()
        if reminder:   # one reminder before NO_ACTION ends the run (architecture D24)
            return {"messages": [HumanMessage(reminder)], "pending": None}
        return {"pending": None}
    (name, raw_args, call_id), extras = calls[0], calls[1:]
    result = controller.validate(name, raw_args)
    if isinstance(result, Approval):
        return {"pending": result}
    answers = [_answer(result, name, call_id)]
    answers += [_answer(controller.reject_extra_call(n, a), n, i) for n, a, i in extras]
    return {"messages": answers, "pending": None}


def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    _, controller, _ = _parts(config)
    approval = state["pending"]
    if approval is None:
        return {}
    calls = _calls(_last_reply(state))
    (name, _, call_id), extras = calls[0], calls[1:]
    answers = [_answer(controller.execute(approval), name, call_id)]
    answers += [_answer(controller.reject_extra_call(n, a), n, i) for n, a, i in extras]
    return {"messages": answers, "pending": None}


def continue_node(state: AgentState, config: RunnableConfig) -> dict:
    _, controller, _ = _parts(config)
    controller.check_terminal()
    return {}


def _route(state: AgentState, config: RunnableConfig) -> str:
    _, controller, _ = _parts(config)
    return END if controller.terminal_reason is not None else "llm"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("controller", controller_node)
    graph.add_node("tool", tool_node)
    graph.add_node("continue", continue_node)
    graph.add_edge(START, "llm")
    graph.add_edge("controller", "tool")
    graph.add_edge("tool", "continue")
    graph.add_conditional_edges("continue", _route, {END: END, "llm": "llm"})
    return graph.compile()


# ---------------------------------------------------------------- run (§4)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _UnavailableRetriever:
    """Stands in for a retriever that could not be built; the run ends before any search."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def search(self, query, k, sources=None, subject=None):
        raise self.error


def _bind(model, config: Config):
    if model is None:
        from rag_agent.llm import make_model

        model = make_model(config)
    return model.bind_tools(tool_definitions(), tool_choice="required", parallel_tool_calls=False)


def run(question: str, config: Config = Config(), model=None, retriever=None, registry=None,
        on_entry: Callable[[TraceEntry], None] | None = None) -> RunResult:
    """Answer one question; every failure except an empty question becomes a terminal reason."""
    if not (question or "").strip():
        raise ValueError("The question must not be empty.")
    started_at = _now()
    registry = tuple(registry if registry is not None else load_registry(DEFAULT_REGISTRY_PATH))

    retriever_error = None
    if retriever is None:
        try:
            from rag_agent.retrieval import HybridRetriever

            retriever = HybridRetriever(DEFAULT_DB_PATH, mode=config.retrieval_mode,
                                        query_instruction=config.query_instruction)
        except Exception as error:
            retriever_error, retriever = error, _UnavailableRetriever(error)

    session = Session.new(question, config, retriever, registry)
    controller = Controller(session, config, on_entry=on_entry)
    bound = None
    if retriever_error is not None:
        controller.record_retrieval_error(retriever_error)
    else:
        try:
            bound = _bind(model, config)
        except Exception as error:
            controller.record_model_error(error)

    if controller.check_terminal() is None:
        graph = build_graph()
        graph.invoke({"messages": [], "pending": None},
                     config={"configurable": {"session": session, "controller": controller, "model": bound},
                             "recursion_limit": 4 * config.max_steps + 10})
    return _result(session, controller, config, started_at)


def _result(session: Session, controller: Controller, config: Config, started_at: str) -> RunResult:
    final = controller.final
    return RunResult(
        question=session.question, terminal_reason=controller.terminal_reason,
        error_kind=controller.error_kind, message=controller.terminal_message or "",
        outcome=final.outcome if final else None, answer=final.answer if final else None,
        citations=final.citations if final else (), steps_used=controller.steps_used,
        max_steps=config.max_steps, trace=controller.trace, config=config.as_dict(),
        started_at=started_at, finished_at=_now(),
    )


# ---------------------------------------------------------------- JSONL trace (§6)

def manifest_summary(manifest: dict | None) -> dict | None:
    if manifest is None:
        return None
    return {key: manifest.get(key) for key in MANIFEST_KEYS}


def trace_records(result: RunResult, manifest: dict | None) -> list[dict]:
    """Header, one record per trace entry, then the result."""
    header = {"type": "run", "question": result.question, "started_at": result.started_at,
              "finished_at": result.finished_at, "config": result.config,
              "manifest": manifest_summary(manifest)}
    entries = [{"type": "entry", **asdict(entry)} for entry in result.trace]
    footer = {"type": "result", "terminal_reason": result.terminal_reason.value,
              "error_kind": result.error_kind.value if result.error_kind else None,
              "message": result.message, "outcome": result.outcome, "answer": result.answer,
              "citations": list(result.citations), "steps_used": result.steps_used,
              "max_steps": result.max_steps}
    return [header, *entries, footer]


def write_trace(path: Path | str, result: RunResult, manifest: dict | None) -> None:
    """Write the run as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in trace_records(result, manifest)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
