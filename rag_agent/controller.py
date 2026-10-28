"""The controller: policies, step limit, trace and terminal reason (specs/controller.md).

The single gatekeeper between the model's requests and the application state. Every request is
validated first (§5, policies P1-P7); only an approved request is executed (§6). The controller
counts steps, keeps the append-only trace (§8), and is the only code that sets the terminal
reason (§7, architecture §6).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from rag_agent import tools
from rag_agent.config import setting
from rag_agent.retrieval import MODES
from rag_agent.session import Session

DEFAULT_QUERY_INSTRUCTION = "Given a question, retrieve passages that answer it"
EXTRA_CALL_MESSAGE = "Only one tool call per turn is carried out; this call was ignored."


# ---------------------------------------------------------------- configuration (§2)

def _default_model_name() -> str:
    return setting("LLM_MODEL", "Qwen3.8-Flash-Next")   # architecture D22


@dataclass(frozen=True)
class Config:
    max_steps: int = 8
    default_k: int = 5
    max_consecutive_rejections: int = 3
    model_name: str = field(default_factory=_default_model_name)
    temperature: float = 0.0
    retrieval_mode: str = "hybrid"
    query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    retry_delay_seconds: float = 2.0
    no_action_reminders: int = 1          # architecture D24

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, not {self.max_steps}")
        if not 1 <= self.default_k <= tools.MAX_K:
            raise ValueError(f"default_k must be 1-{tools.MAX_K}, not {self.default_k}")
        if self.max_consecutive_rejections < 1:
            raise ValueError("max_consecutive_rejections must be at least 1, "
                             f"not {self.max_consecutive_rejections}")
        if self.retrieval_mode not in MODES:
            raise ValueError(f"retrieval_mode must be one of {MODES}, not {self.retrieval_mode!r}")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        if self.no_action_reminders < 0:
            raise ValueError(f"no_action_reminders must not be negative, not {self.no_action_reminders}")

    def as_dict(self) -> dict:
        """All fields, for the trace header."""
        return asdict(self)


NO_ACTION_REMINDER = (
    "Your last reply had no tool call, so nothing was done. Every turn must be exactly one tool call. "
    "If your answer is ready, call finish with the answer text and its citations; otherwise call "
    "search_documents."
)


# ---------------------------------------------------------------- types (§3)

class TerminalReason(Enum):
    ANSWER = "ANSWER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"
    STEP_LIMIT = "STEP_LIMIT"


class ErrorKind(Enum):
    MODEL_ERROR = "MODEL_ERROR"
    NO_ACTION = "NO_ACTION"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    REJECTION_LIMIT = "REJECTION_LIMIT"


@dataclass(frozen=True)
class Observation:
    """The result of one tool call, accepted or rejected (specs/tools.md §2)."""

    tool: str
    accepted: bool
    policy: str | None
    reason: str | None
    message: str
    data: dict
    steps_remaining: int

    def to_json(self) -> str:
        """Compact JSON of all seven fields: the `ToolMessage` content the model sees."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Approval:
    """A validated request. `raw_args` keeps the arguments as received, for the trace."""

    tool: str
    args: BaseModel
    raw_args: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class FinalResult:
    outcome: str
    answer: str
    citations: tuple[dict, ...]


@dataclass(frozen=True)
class TraceEntry:
    step: int
    kind: str                      # "tool_call", "thought", "no_action" or "model_error"
    tool: str | None
    args: dict | str | None
    accepted: bool | None
    policy: str | None
    reason: str | None
    message: str
    observation: dict | None
    steps_remaining: int


class _Rejection(Exception):
    """Internal: a failed policy check, turned into a rejection observation by `validate`."""

    def __init__(self, policy: str, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.policy, self.reason, self.message, self.details = policy, reason, message, details


def describe_error(error: BaseException) -> str:
    """`"<Type>: <message>"`, the form used in traces and terminal messages."""
    return f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------- the controller (§4)

class Controller:
    def __init__(self, session: Session, config: Config,
                 on_entry: Callable[[TraceEntry], None] | None = None) -> None:
        self.session = session
        self.config = config
        self._on_entry = on_entry
        self._steps_used = 0
        self._consecutive_rejections = 0
        self._trace: list[TraceEntry] = []
        self._searched = False
        self._final: FinalResult | None = None
        self._model_error: str | None = None
        self._no_action = False
        self._reminders_left = config.no_action_reminders
        self._retrieval_error: str | None = None
        self._terminal_reason: TerminalReason | None = None
        self._error_kind: ErrorKind | None = None
        self._terminal_message: str | None = None

    # ------------------------------------------------ read-only state

    @property
    def steps_used(self) -> int:
        return self._steps_used

    @property
    def steps_remaining(self) -> int:
        return max(0, self.config.max_steps - self._steps_used)

    @property
    def consecutive_rejections(self) -> int:
        return self._consecutive_rejections

    @property
    def terminal_reason(self) -> TerminalReason | None:
        return self._terminal_reason

    @property
    def error_kind(self) -> ErrorKind | None:
        return self._error_kind

    @property
    def terminal_message(self) -> str | None:
        return self._terminal_message

    @property
    def final(self) -> FinalResult | None:
        return self._final

    @property
    def trace(self) -> tuple[TraceEntry, ...]:
        return tuple(self._trace)

    # ------------------------------------------------ steps and model events

    def begin_step(self) -> None:
        """Called once before each model call."""
        self._steps_used += 1

    def record_thought(self, text: str) -> None:
        if text and text.strip():
            self._append(kind="thought", message=text.strip())

    def record_no_action(self) -> str | None:
        """A reply without a tool call: the reminder text while reminders remain, else `None` (D24).

        A reminder counts as a rejection, so the rejection limit still bounds a model that never calls
        a tool. With no reminder left, the step is marked for `NO_ACTION`.
        """
        if self._reminders_left > 0:
            self._reminders_left -= 1
            self._consecutive_rejections += 1
            self._append(kind="no_action", message="The model replied without a tool call; it was reminded.")
            return NO_ACTION_REMINDER
        self._no_action = True
        self._append(kind="no_action", message="The model replied without a tool call.")
        return None

    def record_model_error(self, error: BaseException) -> None:
        self._model_error = describe_error(error)
        self._append(kind="model_error", message=self._model_error)

    def record_retrieval_error(self, error: BaseException) -> None:
        """Marks `RETRIEVAL_ERROR` when the retriever could not be built (specs/agent.md §4)."""
        self._retrieval_error = describe_error(error)

    # ------------------------------------------------ validation (§5)

    def validate(self, tool: str, raw_args: Any) -> Approval | Observation:
        """Return an `Approval`, or a rejection `Observation` (which is also traced)."""
        try:
            args = self._check(tool, raw_args)
        except _Rejection as rejection:
            if rejection.policy != "P1":
                self._consecutive_rejections += 1
            return self._reject(tool, raw_args, rejection)
        return Approval(tool, args, raw_args)

    def reject_extra_call(self, tool: str, raw_args: Any) -> Observation:
        """P7: every call after the first in one reply. Uses no step and no rejection count."""
        return self._reject(tool, raw_args, _Rejection("P7", "EXTRA_CALL", EXTRA_CALL_MESSAGE))

    def _check(self, tool: str, raw_args: Any) -> BaseModel:
        if self._terminal_reason is not None:
            raise _Rejection("P1", "TERMINAL", "The run has already ended; no further calls are carried out.")
        schema = tools.SCHEMAS.get(tool)
        if schema is None:
            raise _Rejection("P2", "UNKNOWN_TOOL",
                             f"Unknown tool {tool!r}. Call search_documents or finish.")
        args = self._parse(tool, schema, raw_args)
        if isinstance(args, tools.SearchArgs):
            self._check_search(args)
        else:
            self._check_finish(args)
        return args

    @staticmethod
    def _parse(tool: str, schema: type[BaseModel], raw_args: Any) -> BaseModel:
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except ValueError as error:
                raise _Rejection("P3", "BAD_ARGS", f"The arguments for {tool} are not valid JSON; "
                                 "send one JSON object with the documented fields.",
                                 errors=[{"field": None, "message": str(error)}]) from None
        if not isinstance(raw_args, dict):
            raise _Rejection("P3", "BAD_ARGS", f"The arguments for {tool} must be a JSON object.",
                             errors=[{"field": None, "message": "not a JSON object"}])
        try:
            return schema.model_validate(raw_args)
        except ValidationError as error:
            errors, too_long = [], []
            for item in error.errors(include_url=False):
                field_name = ".".join(str(part) for part in item["loc"]) or None
                entry = {"field": field_name, "message": item["msg"]}
                if item["type"] == "string_too_long" and isinstance(item.get("input"), str):
                    # State the actual length and the maximum so the model knows how much to cut (D23).
                    entry["length"] = len(item["input"].strip())
                    entry["max"] = item["ctx"]["max_length"]
                    too_long.append(f"{field_name} has {entry['length']:,} characters; "
                                    f"the maximum is {entry['max']:,}. Shorten it.")
                errors.append(entry)
            message = (f"The arguments for {tool} do not match its schema; "
                       "fix the fields listed in errors and call again.")
            if too_long:
                message += " " + " ".join(too_long)
            raise _Rejection("P3", "BAD_ARGS", message, errors=errors) from None

    def _check_search(self, args: tools.SearchArgs) -> None:
        unknown = [source for source in args.sources or [] if source not in self.session.source_ids]
        if unknown:
            raise _Rejection("P3", "UNKNOWN_SOURCE",
                             f"Unknown source id(s): {', '.join(unknown)}. Use ids from the source "
                             "catalog (listed in known), or omit sources.",
                             unknown=unknown, known=[source.id for source in self.session.registry])

    def _check_finish(self, args: tools.FinishArgs) -> None:
        if not self._searched:
            raise _Rejection("P4", "NO_SEARCH_YET",
                             "Call search_documents before finish; every result must rest on a search.")
        ids = [citation.chunk_id for citation in args.citations]
        unretrieved = [chunk_id for chunk_id in ids if chunk_id not in self.session.retrieved]
        if unretrieved:
            raise _Rejection("P5", "UNRETRIEVED_CITATION",
                             "Cite only chunk ids returned by search_documents in this run; these were "
                             f"not: {', '.join(unretrieved)}.", unretrieved=unretrieved)
        if args.outcome == "answer" and not ids:
            raise _Rejection("P6", "NO_CITATIONS",
                             "An answer needs at least one citation; add [n] markers and list the "
                             "chunk ids that support them.")
        duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
        if duplicates:
            raise _Rejection("P6", "DUPLICATE_CITATION",
                             "Each chunk may be cited once; reuse its [n] marker instead of listing it again.",
                             duplicates=duplicates)
        markers = tools.citation_markers(args.answer)
        dangling = [n for n in markers if not 1 <= n <= len(ids)]
        if dangling:
            raise _Rejection("P6", "DANGLING_MARKER",
                             f"Markers must be [1] to [{len(ids)}], one per listed citation; these have no "
                             "matching citation.", markers=dangling)
        unused = [n for n in range(1, len(ids) + 1) if n not in markers]
        if unused:
            raise _Rejection("P6", "UNUSED_CITATION",
                             "Every listed citation must be used by an [n] marker in the answer; remove "
                             "these citations or add their markers.", unused=unused)

    def _reject(self, tool: str, raw_args: Any, rejection: _Rejection) -> Observation:
        observation = Observation(tool=tool or "", accepted=False, policy=rejection.policy,
                                  reason=rejection.reason, message=rejection.message,
                                  data={"reason": rejection.reason, **rejection.details},
                                  steps_remaining=self.steps_remaining)
        self._append_call(tool, raw_args, observation)
        return observation

    # ------------------------------------------------ execution (§6)

    def execute(self, approval: Approval) -> Observation:
        raw_args = approval.raw_args if approval.raw_args is not None else approval.args.model_dump()
        try:
            observation = self._run_tool(approval)
        except Exception as error:
            if approval.tool != "search_documents":
                raise
            self._retrieval_error = describe_error(error)   # the retriever failed twice
            observation = Observation(tool=approval.tool, accepted=False, policy=None,
                                      reason="RETRIEVAL_ERROR",
                                      message="The search failed twice; the run ends with a retrieval error.",
                                      data={"status": "error", "error": self._retrieval_error},
                                      steps_remaining=self.steps_remaining)
            self._append_call(approval.tool, raw_args, observation)
            return observation
        self._consecutive_rejections = 0
        if approval.tool == "search_documents":
            self._searched = True
        else:
            self._final = FinalResult(outcome=observation.data["outcome"], answer=approval.args.answer,
                                      citations=tuple(observation.data["citations"]))
        self._append_call(approval.tool, raw_args, observation)
        return observation

    def _run_tool(self, approval: Approval) -> Observation:
        if approval.tool != "search_documents":
            return tools.execute(self.session, approval, self.steps_remaining)
        try:
            return tools.execute(self.session, approval, self.steps_remaining)
        except Exception:
            if self.config.retry_delay_seconds > 0:
                time.sleep(self.config.retry_delay_seconds)
            return tools.execute(self.session, approval, self.steps_remaining)

    # ------------------------------------------------ terminal checks (§7)

    def check_terminal(self) -> TerminalReason | None:
        if self._terminal_reason is not None:
            return self._terminal_reason
        final = self._final
        if final is not None and final.outcome == "answer":
            self._end(TerminalReason.ANSWER, None, f"Answered with {len(final.citations)} citation(s).")
        elif final is not None and final.outcome == "insufficient_evidence":
            self._end(TerminalReason.INSUFFICIENT_EVIDENCE, None,
                      "The retrieved passages do not support an answer.")
        elif self._model_error is not None:
            self._end(TerminalReason.ERROR, ErrorKind.MODEL_ERROR, self._model_error)
        elif self._no_action:
            self._end(TerminalReason.ERROR, ErrorKind.NO_ACTION, "The model replied without a tool call.")
        elif self._retrieval_error is not None:
            self._end(TerminalReason.ERROR, ErrorKind.RETRIEVAL_ERROR, self._retrieval_error)
        elif self._consecutive_rejections >= self.config.max_consecutive_rejections:
            self._end(TerminalReason.ERROR, ErrorKind.REJECTION_LIMIT,
                      f"{self._consecutive_rejections} rejected requests in a row.")
        elif self._steps_used >= self.config.max_steps:
            self._end(TerminalReason.STEP_LIMIT, None,
                      f"Used all {self.config.max_steps} steps without finishing.")
        return self._terminal_reason

    def _end(self, reason: TerminalReason, kind: ErrorKind | None, message: str) -> None:
        self._terminal_reason, self._error_kind, self._terminal_message = reason, kind, message

    # ------------------------------------------------ trace (§8)

    def _append_call(self, tool: str, raw_args: Any, observation: Observation) -> None:
        self._append(kind="tool_call", tool=tool, args=raw_args, accepted=observation.accepted,
                     policy=observation.policy, reason=observation.reason, message=observation.message,
                     observation=json.loads(observation.to_json()))

    def _append(self, *, kind: str, message: str, tool: str | None = None, args: Any = None,
                accepted: bool | None = None, policy: str | None = None, reason: str | None = None,
                observation: dict | None = None) -> None:
        entry = TraceEntry(step=self._steps_used, kind=kind, tool=tool, args=args, accepted=accepted,
                           policy=policy, reason=reason, message=message, observation=observation,
                           steps_remaining=self.steps_remaining)
        self._trace.append(entry)
        if self._on_entry is not None:
            self._on_entry(entry)
