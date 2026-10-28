# Agent Specification: Loop, Prompts, Model, CLI and Required Runs

Status: **Draft v1 for orchestrator review (2026-10-26)**
Modules: `rag_agent/agent.py`, `rag_agent/prompts.py`, `rag_agent/llm.py`, `rag_agent/cli.py`,
`rag_agent/__main__.py`, `scripts/run_required.py`, `scripts/check_llm.py`
Tests: `tests/test_agent_loop.py`, `tests/test_prompts.py`, `tests/test_cli.py`, `tests/test_required.py`

## 1. Graph

```
START -> llm -> controller -> tool -> continue? --(terminal)--> END
          |                                |
          +--(model error)--> continue?    +--(otherwise)--> llm
```

The graph state holds only `messages` (LangGraph `add_messages`) and `pending` (the current turn's
`Approval` or `None`). The `Session`, `Controller` and bound model reach the nodes through
`config["configurable"]`, never through graph state, so model output cannot touch application state
except through controller methods (as in HW-08).

### 1.1 `llm`

1. `controller.begin_step()`.
2. Build the model input (§2) and invoke the bound model.
3. On an exception: wait `retry_delay_seconds`, invoke once more with the same input. If that also
   raises, `controller.record_model_error(error)` and go to `continue`. The retry does not use a step.
4. Give the reply a fresh message id (`ai-<uuid4 hex>`) so `add_messages` always appends it.
5. `controller.record_thought(reply text)`, then append the reply and go to `controller`.

### 1.2 `controller`

- No tool calls in the reply (including `invalid_tool_calls`): call `controller.record_no_action()`. If
  it returns a reminder, append it to the conversation as a `HumanMessage`; the model sees it on the next
  turn. Go to `continue`. (v1.5, architecture D24.)
- Otherwise collect calls as `(name, raw_args, call_id)`: parsed `tool_calls` first, then
  `invalid_tool_calls`, whose raw argument string is passed on so it fails P3 instead of crashing.
- Validate the first call. If rejected, answer it and every extra call (P7) with `ToolMessage`s now;
  `pending = None`. If approved, set `pending` and answer nothing yet. Go to `tool`.

### 1.3 `tool`

If `pending` is set: execute it, then answer the first call and then each extra call (P7), in call
order. Every `tool_call_id` in the reply gets exactly one `ToolMessage` whose content is the
observation's JSON. Clear `pending`.

### 1.4 `continue`

`controller.check_terminal()`; go to END if it returns a reason, else to `llm`.

### 1.5 Backstop

`recursion_limit = 4 * max_steps + 10`, so the controller's step limit always ends a run first.

## 2. Model input (`prompts`)

Each model call receives, in order:

1. `SystemMessage(system_prompt(registry))`
2. `HumanMessage(question)`
3. every earlier `AIMessage` and `ToolMessage` of this run (no window, architecture D16)
4. `HumanMessage(status_message(session, controller))`, rebuilt every turn and never stored in the
   graph state

### 2.1 System prompt

`system_prompt(registry) -> str` must contain, in this order:

1. **Role:** answering Python and RISC-V questions using only passages retrieved from the corpus.
2. **Process:** search with focused queries; use `sources` or `subject` filters when the question names
   or implies a source; search again with different terms if results are off-target; then call
   `finish`. **Query wording:** describe the content needed (for example "limitations and weaknesses
   the paper acknowledges"), not the document's own subject or title, which matches title pages and
   captions instead of the passage (measured 2026-10-26, see `docs/process-log.md`).
3. **Rules:**
   - Use only retrieved passages; never answer from memory.
   - End every factual sentence with a `[n]` marker; list citations in marker order; cite only
     chunk ids returned by `search_documents` in this run.
   - Paraphrase explanations freely, but keep technical details exactly as written in the passage:
     instruction mnemonics, register and ABI names, numbers, bit widths, version strings, code, and
     Python keywords and error names. Quote briefly when a precise definition matters.
   - When the 2019 and 2025 RISC-V specifications differ, the 2025 manual controls; say so.
   - If the passages support only part of the question, answer that part and state what is missing.
     If they support nothing central, finish with `insufficient_evidence` and say what was searched
     and what is missing.
   - Make exactly one tool call per turn. Never reply with plain text: the answer goes in `finish`.
   - Keep the answer under 8,000 characters (`MAX_ANSWER_CHARS`, written with a thousands separator);
     for a question with several parts, answer each part briefly. (v1.5, D23)
   - Cite at most 12 passages (`MAX_CITATIONS`). (v1.6, D25)
4. **Source catalog:** the heading `Sources:` followed by one line per registry source, in registry
   order: `- <id> (<subject>): <title>. <role>`

### 2.2 Status message

`status_message(session, controller) -> str` has exactly these lines:

```
Step <steps_used> of <max_steps>.
Retrieved passages you may cite (chunk_id · source_id · page):
- <chunk_id> · <source_id> · p.<page>
...
```

- When nothing has been retrieved, the second line is
  `Retrieved passages you may cite: none yet. Call search_documents first.` and no list follows.
- The list follows `session.retrieved` order.
- When `steps_used == max_steps`, a final line is added:
  `This is your last step: call finish now.`

## 3. Chat model (`llm`)

`make_model(config) -> ChatOpenAI` with `model=config.model_name`, `base_url=LLM_BASE_URL` (default
`http://golem:4000/v1`), `api_key=require_api_key("LLM_API_KEY")`, `temperature=config.temperature`,
`max_retries=0`, `timeout=180`.

`run` binds tools with `bind_tools(tool_definitions(), tool_choice="required",
parallel_tool_calls=False)`. A test double must accept those keyword arguments.

`scripts/check_llm.py` (live; not part of the test suite) sends one question with the two tools
bound and prints, for each of `tool_choice` `auto` and `required`: whether exactly one well-formed
tool call came back, its name and arguments, and the latency. It is run once before the required runs,
and its output is saved to `docs/llm-check-<date>.txt`.

## 4. `run`

```python
run(question: str, config: Config = Config(), model=None, retriever=None, registry=None,
    on_entry: Callable[[TraceEntry], None] | None = None) -> RunResult
```

- `registry=None` loads `corpus/sources.toml`; `retriever=None` builds
  `HybridRetriever(DEFAULT_DB_PATH, mode=config.retrieval_mode,
  query_instruction=config.query_instruction)`; `model=None` calls
  `make_model(config)`.
- An empty question raises `ValueError`. Every other failure becomes a terminal reason: a failure to
  build the model gives `ERROR`/`MODEL_ERROR`, and a failure to build the retriever gives
  `ERROR`/`RETRIEVAL_ERROR`, each with no steps used.
- `on_entry` is called with each new trace entry as it is recorded (used by the CLI to print live).

`RunResult` (frozen):

| Field | Type |
|---|---|
| `question` | `str` |
| `terminal_reason` | `TerminalReason` |
| `error_kind` | `ErrorKind \| None` |
| `message` | `str` |
| `outcome` | `str \| None` (from the accepted `finish`) |
| `answer` | `str \| None` |
| `citations` | `tuple[dict, ...]` (resolved: `n`, `chunk_id`, `source_id`, `title`, `page`, `supports`) |
| `steps_used`, `max_steps` | `int` |
| `trace` | `tuple[TraceEntry, ...]` |
| `config` | `dict` (`Config.as_dict()`) |
| `started_at`, `finished_at` | `str` (ISO 8601, UTC) |

## 5. CLI (`cli`, `python -m rag_agent`)

```
uv run python -m rag_agent "QUESTION" [--max-steps N] [--k N] [--mode hybrid|vector|keyword]
                                      [--query-instruction TEXT] [--db DIR]
                                      [--trace-out FILE.jsonl] [--quiet]
```

- `cli.main(argv=None, *, model=None, retriever=None) -> int` parses `argv`, runs, prints, and returns
  the exit status; `__main__.py` calls `sys.exit(main())`. `model` and `retriever` are passed to `run`,
  so tests can drive the CLI with fakes. (v1.1: added while deriving tests.)
- `--help` (`-h`) prints usage with the program name `python -m rag_agent`, a one-line description of
  every option (including its default), and at least one example question, then exits with status 0.
  (v1.2: required at orchestrator request, 2026-10-27.)
- `--db DIR` sets the store folder (default `DEFAULT_DB_PATH`). When no `retriever` is given and the
  folder does not exist, prints `No vector store found; run: uv run python -m rag_agent.ingest
  --rebuild` and returns 2. Command-line usage errors also return 2. Because `run` takes no database
  path, the CLI builds the `HybridRetriever` itself from `--db`, `--mode` and `--query-instruction`; if
  building it fails for another reason, it prints the error and returns 1 (v1.3, coding agent).
- `--query-instruction ""` turns the query instruction off (v1.3).
- Unless `--quiet`, prints one line per trace entry as it happens:
  `[step 2/8] search_documents {"query": "JALR", "sources": ["riscv-isa-2025"]} -> ok: 5 passages from riscv-isa-2025.`
  (rejections show `REJECTED P5 UNRETRIEVED_CITATION`; thoughts show `thought: <first 200 characters>`).
- Always prints the terminal reason and message, then for an answer or insufficient-evidence result the
  answer text and a citation list: `[1] riscv-isa-2025, page 51 (chunk 4f1c9a0b…): JALR target and rd`.
- `--trace-out` writes the JSONL trace (§6).
- Exit status: 0 for `ANSWER` or `INSUFFICIENT_EVIDENCE`, 1 for `ERROR` or `STEP_LIMIT`, 2 for usage
  problems.

## 6. JSONL trace file

`rag_agent.agent.write_trace(path, result, manifest)` writes one JSON object per line (`manifest` is
the parsed `manifest.json` or `None`). `RunResult` therefore also carries `started_at` and
`finished_at` (ISO 8601 UTC strings set by `run`).

1. Header: `{"type": "run", "question", "started_at", "finished_at", "config", "manifest"}`, where
   `manifest` holds `index_version`, `embedding_model`, `embedding_dimension`, `chunking`,
   `extractor`, `cleaning` and `record_count` from `manifest.json` (or `null` when tests run without
   one).
2. One `{"type": "entry", ...}` per `TraceEntry`, with all its fields.
3. Result: `{"type": "result", "terminal_reason", "error_kind", "message", "outcome", "answer",
   "citations", "steps_used", "max_steps"}`.

## 7. Required runs (`rag_agent/required.py`, `scripts/run_required.py`)

`queries/required-queries.md` holds nine sections. Each starts with `## Q<n>: <title>`, then a line
`<!-- expected: ANSWER -->` or `<!-- expected: INSUFFICIENT_EVIDENCE -->`, then the query text up to
the next `## ` heading. Text before the first `## Q` heading is ignored.

The logic lives in `rag_agent/required.py` so tests can import it; `scripts/run_required.py` only parses
arguments and calls it. (v1.1: moved from the script while deriving tests.)

- `RequiredQuery` (frozen): `number: int`, `title: str`, `expected: TerminalReason`, `text: str`.
- `parse_queries(path) -> list[RequiredQuery]`, in file order. The query text is the section's
  non-blank, non-comment lines, each stripped, joined with single spaces (the file wraps long
  questions across lines; the wrapping is formatting, not part of the question). A section without an
  expected-outcome comment, a `## Q` heading without a number, a duplicate number, or a gap in the
  numbering (numbers must run 1 to the highest without gaps) raises `ValueError` (v1.3: "missing number"
  made precise by the coding agent).
- `run_all(queries, config, out_dir, model=None, retriever=None, registry=None, manifest=None)
  -> list[RunResult]` runs each query once with the same `config`, model and retriever, in order,
  without stopping on failures, and writes into `out_dir`:
  - `Q<n>.jsonl` per query (§6);
  - `manifest.json`, a copy of `manifest`, when it is not `None`;
  - `summary.md`: a `## Recorded configuration` section listing embedding model, chunk size and overlap,
    retrieval mode, `k`, step limit, chat model, query instruction and `index_version` (values from
    `config` and `manifest`; `unknown` when absent), then a table whose header row is exactly
    `| Query | Title | Expected | Actual | Match | Steps | Sources cited |`, one row per query, where
    Query is `Q<n>`, Actual is the terminal reason (`ERROR/<kind>` for errors), Match is `yes` or `no`,
    Steps is `<used>/<max>`, and Sources cited lists the distinct cited source ids (or `none`).

```
uv run python scripts/run_required.py [--only Q3 Q8] [--out-dir runs/<name>]
```

The default output folder is `runs/<YYYY-MM-DD_HHMM>/`; the script loads the real registry, retriever,
model and `manifest.json`. Before the first query it prints one line with the chat model, retrieval
mode, `k`, step limit and `index_version` (v1.4, 2026-10-27: nine runs once used a model left in the
shell environment, which overrides `.env`, and the mistake showed only afterward).

## 8. Test doubles (`tests/fakes.py`)

- `ScriptedModel`, `call`, `turn`, `bad_json_turn`: carried over from HW-08 unchanged in behavior.
  `call` takes a tool name and keyword arguments; the script may include exceptions to raise.
- `FakeRetriever(passages)`: holds a list of `Passage`; `search` returns the first `k` that match the
  filters, ranked as given (the subject filter uses each passage's source subject from the real
  registry); it records every call as `(query, k, sources, subject)`; `fail_times=n` makes the first
  `n` calls raise `RuntimeError`.
- `passage(chunk_id, source_id="riscv-psabi", page=9, text=..., distance=0.2)`: builds a `Passage`
  with a valid 24-character id.

## 9. Tests derived from this spec

All offline, with `ScriptedModel`, `FakeRetriever` and `retry_delay_seconds=0`:

- **Supported run:** search, then `finish(answer)` citing a returned chunk with `[1]`: result `ANSWER`,
  2 steps, citation resolved to its source and page, trace entries in order.
- **Insufficient-evidence run:** search (empty or off-topic results), then
  `finish(insufficient_evidence)`: result `INSUFFICIENT_EVIDENCE`.
- **Citation rejected, then corrected:** `finish` citing an unretrieved id is rejected (P5) and the
  model sees the rejection; a corrected `finish` then ends `ANSWER` in 3 steps.
- **Step limit:** `max_steps=3`, a script of searches only: `STEP_LIMIT` after exactly 3 model calls.
- **Each `ERROR` kind:** a reply with no tool call (`NO_ACTION`); a model that raises twice
  (`MODEL_ERROR`, and once only: retried, then succeeds); a retriever with `fail_times=2`
  (`RETRIEVAL_ERROR`); three invalid calls in a row (`REJECTION_LIMIT`).
- **Model input:** the first input is system prompt, question, status; the system prompt contains the
  rules and one catalog line per source; the status lists retrieved chunks after a search and shows the
  last-step line on the final step.
- **Binding:** `tool_choice="required"`, `parallel_tool_calls=False`, exactly the two tools.
- **Extra calls:** a reply with two calls executes only the first; both call ids are answered; trace
  order follows call order.
- **Unparseable arguments** (`bad_json_turn`) produce a P3 rejection, not a crash.
- **CLI:** exit statuses 0, 1 and 2; the trace file has a header, entries and a result line.
- **Required-run parsing:** a sample queries file yields nine queries with titles, expected outcomes and
  exact text.
