# Controller Specification

Status: **Draft v1 for orchestrator review (2026-10-26)**
Modules: `rag_agent/controller.py`, `rag_agent/session.py`
Tests: `tests/test_controller.py`

The controller is the single gatekeeper between the model's requests and the application state. It
validates every request, counts steps, keeps the trace, and is the only code that sets the terminal
reason.

## 1. Session (`session`)

`Session` is a dataclass created by `Session.new(question, config, retriever, registry)`:

| Attribute | Type | Notes |
|---|---|---|
| `question` | `str` | stripped; an empty question raises `ValueError` |
| `config` | `Config` | read-only |
| `retriever` | `Retriever` | read-only reference |
| `registry` | `tuple[Source, ...]` | the loaded source registry |
| `source_ids` | `frozenset[str]` | ids in the registry |
| `retrieved` | `dict[str, Passage]` | insertion-ordered; starts empty; only `tools` adds to it |

## 2. Configuration

`Config` is a frozen dataclass:

| Field | Default | Rule |
|---|---|---|
| `max_steps` | `8` | at least 1 |
| `default_k` | `5` | 1–8 |
| `max_consecutive_rejections` | `3` | at least 1 |
| `model_name` | `LLM_MODEL` from the environment, else `"Qwen3.8-Flash-Next"` | changed from Qwen3.8-27B (architecture D22) |
| `temperature` | `0.0` | |
| `retrieval_mode` | `"hybrid"` | `hybrid`, `vector` or `keyword`; passed to `HybridRetriever` |
| `query_instruction` | `"Given a question, retrieve passages that answer it"` | passed to `HybridRetriever`; `None` turns it off (frozen 2026-10-27, architecture D19) |
| `retry_delay_seconds` | `2.0` | delay before the single retry of a failed model or retrieval call; tests use 0 |
| `no_action_reminders` | `1` | at least 0; replies without a tool call that get a reminder before the run ends as `NO_ACTION` (v1.2, architecture D24) |

An out-of-range value raises `ValueError` when the `Config` is created. `Config.as_dict()` returns all
fields for the trace header.

## 3. Types

- `TerminalReason` enum: `ANSWER`, `INSUFFICIENT_EVIDENCE`, `ERROR`, `STEP_LIMIT` (values equal names).
- `ErrorKind` enum: `MODEL_ERROR`, `NO_ACTION`, `RETRIEVAL_ERROR`, `REJECTION_LIMIT`.
- `Observation`: see `specs/tools.md` §2.
- `Approval`: frozen; `tool: str`, `args: BaseModel` (the validated schema instance), and optional
  `raw_args` (default `None`, excluded from equality) holding the arguments exactly as received, for the
  trace. (v1.1, 2026-10-27: added by the coding agent; the trace must record raw arguments of approved
  calls too.)
- `FinalResult`: frozen; `outcome: str`, `answer: str`, `citations: tuple[dict, ...]` (resolved as in
  tools §5.2).
- `TraceEntry`: frozen, fields below.

| Field | Type | Meaning |
|---|---|---|
| `step` | `int` | `steps_used` when the entry was recorded |
| `kind` | `str` | `"tool_call"`, `"thought"`, `"no_action"` or `"model_error"` |
| `tool` | `str \| None` | tool name for `tool_call` entries |
| `args` | `dict \| str \| None` | raw arguments exactly as received |
| `accepted` | `bool \| None` | for `tool_call` entries |
| `policy` | `str \| None` | rejecting policy |
| `reason` | `str \| None` | reason code |
| `message` | `str` | observation message, the model's text, or the error description |
| `observation` | `dict \| None` | the full observation as sent to the model |
| `steps_remaining` | `int` | `max_steps - steps_used` |

## 4. Public interface

| Member | Behavior |
|---|---|
| `Controller(session, config, on_entry=None)` | Starts with `steps_used = 0`, no trace, no terminal reason. `on_entry`, if given, is called with each trace entry as it is appended (v1.1: how `run(on_entry=...)` reaches the trace). |
| `steps_used`, `steps_remaining`, `consecutive_rejections` | Read-only counters; `steps_remaining = max(0, max_steps - steps_used)`. |
| `terminal_reason`, `error_kind`, `terminal_message` | `None` until a terminal check fires. |
| `final` | The accepted `FinalResult`, else `None`. |
| `trace` | Tuple of `TraceEntry`, in order. |
| `begin_step()` | Adds 1 to `steps_used`. Called once before each model call. |
| `record_thought(text)` | Appends a `thought` entry when `text.strip()` is non-empty. |
| `validate(tool, raw_args)` | §5. Returns an `Approval` or a rejection `Observation`. |
| `execute(approval)` | §6. Returns the observation. |
| `reject_extra_call(tool, raw_args)` | P7 rejection; §5.4. |
| `record_no_action() -> str \| None` | While reminders remain (`no_action_reminders`), uses one: adds 1 to `consecutive_rejections`, appends a `no_action` entry, and returns the reminder text (`NO_ACTION_REMINDER`), which the agent sends to the model. With none left, appends a `no_action` entry, marks the step for `NO_ACTION`, and returns `None`. (v1.2, architecture D24: in required run 1, Q3's model wrote a complete cited answer as plain text, and the run ended without it.) |
| `record_model_error(error)` | Appends a `model_error` entry (`"<Type>: <message>"`) and marks `MODEL_ERROR`. |
| `record_retrieval_error(error)` | Marks `RETRIEVAL_ERROR` when the retriever could not be built (agent spec §4). (v1.1: the gap was found by the coding agent; no method existed to record it.) |
| `check_terminal()` | §7. Returns the terminal reason or `None`. |

## 5. Validation (`validate`)

Checks run in this order; the first failure produces a rejection. A rejection returns an `Observation`
with `accepted=False`, the policy and reason, a message telling the model how to correct the request,
and `data = {"reason": <code>, ...details}`. It appends a `tool_call` trace entry, adds 1 to
`consecutive_rejections`, and changes nothing else. Validation never calls the retriever.

### 5.1 Every tool

| Policy | Reason | Condition |
|---|---|---|
| P1 | `TERMINAL` | A terminal reason is already set. Not counted as a rejection toward the limit. |
| P2 | `UNKNOWN_TOOL` | `tool` is not `search_documents` or `finish`. |
| P3 | `BAD_ARGS` | `raw_args` is not a JSON object, or fails the tool's schema. `data.errors` lists field errors. A string that decodes to a JSON object is validated as that object; any other string is rejected (v1.1). For a string that is too long, its error also has `length` (actual characters) and `max`, and the message states both, such as "answer has 8,214 characters; the maximum is 8,000. Shorten it." (v1.2, D23) |

### 5.2 `search_documents`

| Policy | Reason | Condition |
|---|---|---|
| P3 | `UNKNOWN_SOURCE` | A `sources` entry is not in `session.source_ids`. `data.unknown` lists them; `data.known` lists every id. |

### 5.3 `finish`

| Policy | Reason | Condition |
|---|---|---|
| P4 | `NO_SEARCH_YET` | No `search_documents` call has been accepted in this run. |
| P5 | `UNRETRIEVED_CITATION` | A citation's `chunk_id` is not in `session.retrieved`. `data.unretrieved` lists them. |
| P6 | `NO_CITATIONS` | `outcome` is `answer` and `citations` is empty. |
| P6 | `DUPLICATE_CITATION` | Two citations share a `chunk_id`. |
| P6 | `DANGLING_MARKER` | A marker number is below 1 or above `len(citations)` (including any marker when there are no citations). `data.markers` lists them. |
| P6 | `UNUSED_CITATION` | A citation position `n` never appears among the markers. `data.unused` lists them. |

An `insufficient_evidence` finish may have no citations; if it has any, the same P5 and P6 rules apply.

### 5.4 Extra calls (P7)

When one model reply contains several tool calls, only the first is validated and possibly executed.
Each later call gets `reject_extra_call`: an observation with policy `P7`, reason `EXTRA_CALL`, message
"Only one tool call per turn is carried out; this call was ignored.", and a `tool_call` trace entry. P7
uses no step and does not change `consecutive_rejections`.

## 6. Execution (`execute`)

- `search_documents`: calls `tools.execute`. If the retriever raises, the controller waits
  `retry_delay_seconds` and tries once more. If the second attempt also raises, it returns an
  observation with `accepted=False`, `policy=None`, `reason="RETRIEVAL_ERROR"`,
  `data={"status": "error", "error": "<Type>: <message>"}`, and marks the step for `RETRIEVAL_ERROR`.
  A retrieval error is not a policy rejection and does not count toward the rejection limit.
- `finish`: calls `tools.execute` and stores the `FinalResult`.
- An accepted execution resets `consecutive_rejections` to 0 and appends a `tool_call` entry.

## 7. Terminal checks (`check_terminal`)

Runs after every step. If a terminal reason is already set it returns it unchanged. Otherwise the first
condition that holds sets `terminal_reason`, `error_kind` and `terminal_message`:

| Order | Condition | Reason | Error kind | Message |
|---|---|---|---|---|
| 1 | `final.outcome == "answer"` | `ANSWER` | — | `"Answered with <n> citation(s)."` |
| 2 | `final.outcome == "insufficient_evidence"` | `INSUFFICIENT_EVIDENCE` | — | `"The retrieved passages do not support an answer."` |
| 3 | the model call failed after its retry | `ERROR` | `MODEL_ERROR` | the error description |
| 4 | the model's reply had no tool call and no reminder remained | `ERROR` | `NO_ACTION` | `"The model replied without a tool call."` |
| 5 | retrieval failed after its retry | `ERROR` | `RETRIEVAL_ERROR` | the error description |
| 6 | `consecutive_rejections >= max_consecutive_rejections` | `ERROR` | `REJECTION_LIMIT` | `"<n> rejected requests in a row."` |
| 7 | `steps_used >= max_steps` | `STEP_LIMIT` | — | `"Used all <max_steps> steps without finishing."` |

Once set, the terminal reason is final.

## 8. Trace guarantees

- Entries are appended in the order events happen; the trace is never edited.
- Every tool call in every model reply produces exactly one `tool_call` entry, in call order (the first
  call's entry comes before any P7 entries).
- `observation` holds the same dictionary that was serialized into the model's `ToolMessage`.

## 9. Tests derived from this spec

Use a `Session` with a fake retriever and a small registry.

- Config validation (`max_steps = 0`, `default_k = 9` rejected).
- P1–P7: each rejection has the stated policy, reason and data, appends one trace entry, leaves
  `session.retrieved` and `final` unchanged, and (except P1, P7) increments `consecutive_rejections`.
- P3 with unparseable arguments (a string) and with an unknown source id.
- **Citation provenance:** a `finish` citing a chunk id the fake retriever never returned is rejected
  under P5; after a search returns that chunk, the same `finish` is accepted.
- Each P6 case, including `[2, 3]` markers resolving correctly and an `insufficient_evidence` finish with
  no citations being accepted.
- Retrieval error: one failure then success gives an accepted observation; two failures give
  `ERROR`/`RETRIEVAL_ERROR`.
- **Step limit:** with `max_steps = 3` and only searches, `check_terminal` returns `STEP_LIMIT` exactly
  when `steps_used` reaches 3; a valid `finish` on step 3 returns `ANSWER` instead.
- **All four terminal reasons**, and each `ERROR` kind, with the order in §7 (for example, rejection
  limit and step limit reached on the same step gives `ERROR`/`REJECTION_LIMIT`).
- An accepted call resets `consecutive_rejections`; P1 after a terminal reason.
