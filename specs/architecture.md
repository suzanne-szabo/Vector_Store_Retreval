# Architecture Specification: Python and RISC-V Retrieval Agent

Status: **Draft v1 for orchestrator review (2026-10-26)**
Owner (design authority): Suzanne Szabo
Implementation: ingestion by Claude with the orchestrator (built 2026-10-26, see `specs/ingestion.md`);
agent loop by a fresh coding agent working only from `specs/` and `tests/`.

This document fixes the module boundaries, owned state, public interfaces, dependencies and allowed
interactions. Each module has its own functional specification in `specs/`. Black-box tests are
derived from those specifications before the agent code is written.

---

## 1. Goals and design principles

1. **The application owns all state.** The model only *requests* typed tool calls. It never holds or
   changes the vector store, the set of retrieved chunks, the step count, the trace, or the terminal
   reason.
2. **Validate before execute.** Every request is validated first. A rejected request changes nothing
   except the step count and the trace.
3. **Only the application decides how a run ends.** A run ends with an answer only when the
   controller accepts a `finish` call whose citations all point to chunks retrieved in that run.
   Model text alone can never end a run.
4. **Evidence, not memory.** Final claims and citations are limited to retrieved passages. When the
   passages do not support an answer, the run ends with `INSUFFICIENT_EVIDENCE`.
5. **Provenance is checked by code; support is checked by people.** The controller proves each cited
   chunk came from this run's retrieval. Whether the passage supports the claim is evaluated in the
   tests (scripted cases) and in the required-query review.
6. **Ingestion and query time never mix.** Ingestion writes the store; the agent only reads it.

## 2. Conventions shared by all modules

| Item | Convention |
|---|---|
| Source id | Lowercase id from `corpus/sources.toml`, such as `riscv-psabi`. The agent filters and cites by id, never by filename. |
| Subject | One of `python`, `riscv`, `pedagogy`. |
| Page | One-based **physical** PDF page (the first page of the file is 1), not the printed page number. |
| Chunk id | 24 lowercase hex characters (see `specs/ingestion.md` §5). |
| Distance | Chroma cosine distance; lower means more similar. Reported rounded to 4 decimals. |
| Citation marker | `[n]` in answer text, where `n` is the 1-based position in the `citations` list. |
| JSON | Observations and trace records are JSON objects with the keys given in the module specs. |

## 3. Module overview

```
rag_agent/
  config.py      Paths, defaults, settings from .env                      (built)
  corpus.py      Source registry: load, select, verify SHA-256              (built)
  extract.py     Page-by-page layout-preserving PDF text extraction        (built)
  clean.py       Pure text-cleaning functions                               (built)
  chunk.py       Per-page chunking, stable chunk ids, metadata              (built)
  store.py       Embedding client and persistent Chroma collection          (built)
  ingest.py      Ingestion pipeline and CLI                                 (built)
  inspect.py     Store inspection CLI                                       (built)
  keyword.py     BM25 keyword index over the stored chunks                  (built)
  retrieval.py   Retriever interface, Passage, hybrid (vector + keyword)    (built)
  tools.py       Tool argument schemas and tool executors                   (new)
  controller.py  Config, policies, step limit, trace, terminal reasons      (new)
  session.py     Composition root for one run                               (new)
  prompts.py     System prompt, source catalog, per-turn status message     (new)
  llm.py         Chat-model factory                                         (new)
  agent.py       LangGraph loop and run()                                    (new)
  cli.py         python -m rag_agent "question"                              (new)
  required.py    Parse the required queries; run all nine; write summary    (new)
  __main__.py    Entry point for python -m rag_agent                         (new)
scripts/
  run_required.py  Thin wrapper around rag_agent.required                         (new)
  check_llm.py     Verifies forced tool calling on the course chat service         (new)
corpus/sources.toml          The nine-document registry
queries/required-queries.md  The nine frozen queries (frozen at milestone 4)
runs/                        Committed required-run traces, summary, manifest copy
tests/                       Offline tests and fakes
```

## 4. Modules: responsibility, owned state, interface, dependencies

### 4.1 Ingestion modules (`config`, `corpus`, `extract`, `clean`, `chunk`, `store`, `ingest`, `inspect`)

Specified in `specs/ingestion.md`. Summary: verify each registered PDF by SHA-256, extract text page by
page, clean it, split each page into chunks with stable ids and full metadata, embed with
Qwen3-Embedding-8B, store in one persistent Chroma collection with cosine distance, and write
`manifest.json`. Depends on nothing in the agent half.

### 4.2 `retrieval`: the read-only view of the store

- **Responsibility:** Turn a search request into ranked `Passage` objects. The only code at query time
  that touches Chroma or the embedding service.
- **Owned state:** none beyond the open store.
- **Interface:** a `Retriever` protocol with `search(query: str, k: int, sources: list[str] | None,
  subject: str | None) -> list[Passage]`; a frozen `Passage` dataclass (`chunk_id`, `source_id`,
  `title`, `page`, `text`, `distance`, `vector_rank`, `keyword_rank`); and
  `HybridRetriever(db_path, embeddings=None, mode="hybrid", query_instruction=None)`, which fuses
  vector search with BM25 keyword search by Reciprocal Rank Fusion. Tests of the agent use a fake
  retriever with the same protocol. Full detail: `specs/tools.md` §1.
- **Built** (2026-10-26) with its tests, so the retrieval probe could compare modes on live data.
- **Depends on:** `store`, `keyword`, `config`.

### 4.3 `tools`: the two actions the model may request

- **Responsibility:** Define the argument schemas (Pydantic) and execute an *already approved* request
  against the session, returning an `Observation`.
- **Owned state:** none.
- **Tools:**

| Tool | Arguments | Effect |
|---|---|---|
| `search_documents` | `query`, `sources?`, `subject?`, `k?` | Returns ranked passages with chunk id, source id, title, page, distance and text; adds each returned chunk to the session's retrieved set |
| `finish` | `outcome`, `answer`, `citations` | Records the final result; the controller then ends the run as `ANSWER` or `INSUFFICIENT_EVIDENCE` |

- **Depends on:** `session`, `retrieval` (through the session). Full detail: `specs/tools.md`.

### 4.4 `controller`: policies, step limit, trace, terminal reason

- **Responsibility:** The single gatekeeper between model requests and application state, and the
  only code that sets the terminal reason.
- **Owned state:** `steps_used`, `consecutive_rejections`, the append-only trace, the terminal reason
  with its error kind and message, and the accepted final result.
- **Interface (summary):** `begin_step()`, `validate(tool, raw_args) -> Approval | Observation`,
  `execute(approval) -> Observation`, `reject_extra_call(...)`, `record_thought(text)`,
  `record_no_action()`, `record_model_error(error)`, `check_terminal()`. Full detail:
  `specs/controller.md`.
- **Policies** (each rejection changes nothing but the step count and trace):

| Policy | Rejects |
|---|---|
| P1 | Any call after a terminal reason is set |
| P2 | An unknown tool name |
| P3 | Arguments that fail the schema, including unknown source ids |
| P4 | `finish` before any accepted `search_documents` call in this run |
| P5 | A citation whose `chunk_id` was not returned by retrieval in this run |
| P6 | An `answer` with no citations; a `[n]` marker with no matching citation; a citation the text never uses; the same chunk cited twice |
| P7 | Every tool call after the first in one model turn |

- **Depends on:** `tools`, `session`.

### 4.5 `session`: composition root for one run

- **Responsibility:** Hold one run's question, retriever, source registry, configuration and the
  retrieved set.
- **Owned state:** `retrieved`, an insertion-ordered mapping of chunk id to `Passage`. Only `tools`
  adds to it.
- **Depends on:** `retrieval`, `corpus`, `controller.Config` (type only).

### 4.6 `prompts`, `llm`, `agent`, `cli`

- `prompts`: the system prompt (task, rules, source catalog built from the registry) and the
  per-turn status message (steps remaining and the retrieved chunks available for citation).
- `llm`: `make_model(config)` returns a `ChatOpenAI` for the course chat service (`LLM_BASE_URL`,
  `LLM_MODEL`, `LLM_API_KEY`), with `temperature=0` and `max_retries=0`.
- `agent`: the LangGraph loop and `run(question, config, model=None, retriever=None) -> RunResult`.
  Passing a scripted fake model and a fake retriever makes a run deterministic for tests.
- `cli`: asks one question, prints the trace as it happens, optionally writes a JSONL trace.
- Full detail: `specs/agent.md`.

## 5. Allowed interactions

| From → To | Allowed? | Notes |
|---|---|---|
| model → anything | ✖ | The model only produces messages; the `agent` nodes interpret them. |
| `agent` → `controller` | ✔ | The only route by which a request reaches state. |
| `agent` → `session`, `retrieval` | ✖ | The graph never touches state or the store directly. It may read `session.config` and the registry to build prompts. |
| `controller` → `tools` | ✔ | Only after the policy checks pass. |
| `tools` → `session.retrieved` | ✔ | The only code that adds to the retrieved set. |
| `tools` → `retrieval` | ✔ | Through `session.retriever`. |
| `retrieval` → `store` | read-only | Never adds, updates or deletes records. |
| ingestion modules → agent modules | ✖ | Ingestion is independent of the agent. |
| tests → any public interface | ✔ | Black-box tests use only the interfaces in the specs. |

**What the model sees:** the system prompt, the question, its own earlier tool calls with their
observations, and the per-turn status message. Nothing else.

## 6. Terminal reasons (checked after every step, in this order)

`check_terminal()` runs after every step. The first condition that holds sets the terminal reason:

1. **`ANSWER`**: the step's call was an accepted `finish` with outcome `answer`.
2. **`INSUFFICIENT_EVIDENCE`**: the step's call was an accepted `finish` with outcome
   `insufficient_evidence`.
3. **`ERROR`**, with one error kind:
   - `MODEL_ERROR`: the model call raised an exception, after one retry;
   - `NO_ACTION`: the model's reply contained no tool call;
   - `RETRIEVAL_ERROR`: the retriever raised an exception, after one retry;
   - `REJECTION_LIMIT`: `consecutive_rejections >= max_consecutive_rejections` (default 3).
4. **`STEP_LIMIT`**: `steps_used >= max_steps` (default 8).

Because `ANSWER` and `INSUFFICIENT_EVIDENCE` are checked first, a valid `finish` on the last allowed
step counts. Once set, the terminal reason is final (P1) and is reported in `RunResult`.

## 7. Steps

A **step** is one model call together with the tool call it requests. `steps_used` counts model calls,
including a call that raises and a reply with no tool call. A rejected request still uses its step.
Extra calls in the same reply (P7) do not use additional steps. The default limit is **8** and is
configurable (`--max-steps`). The LangGraph `recursion_limit` is set above the worst case so the
controller's limit always ends the run first.

## 8. Trace

The controller owns an append-only trace with one entry per handled tool call and one per model event
(a thought, a reply with no tool call, a model error). Each entry records the step number, the kind,
the tool and raw arguments as received, whether it was accepted, the policy and reason for a
rejection, the full observation returned to the model, and the steps remaining. The CLI prints entries
as they happen, and `--trace-out` writes the whole run as JSONL: a header record (question,
configuration, manifest summary), the entries, and a result record (terminal reason, answer, resolved
citations). Exact keys: `specs/controller.md` §8 and `specs/agent.md` §6.

## 9. Testing strategy (summary)

- **Black-box tests are written from the specs before the agent code**, one file per module.
- Every test runs offline. Three fakes stand in for the live services: the scripted chat model from
  HW-08 (`tests/fakes.py`), a deterministic fake embedding for ingestion, and a fake retriever that
  returns passages the test defines.
- Mapping to the assignment's required tests:

| Required test | File |
|---|---|
| Ingestion metadata: source, one-based physical page, stable chunk id | `test_ingest.py` (built, passing) |
| Retrieval-tool argument validation and source-labeled results | `test_tools.py`, `test_retrieval.py` |
| Rejection of a citation that retrieval did not return | `test_controller.py` (citations) |
| Enforcement of the step limit | `test_controller.py`, `test_agent_loop.py` |
| All four terminal reasons | `test_controller.py`, `test_agent_loop.py` |
| One supported and one insufficient-evidence run with scripted model actions and substitute retrieval results | `test_agent_loop.py` |

- White-box tests written afterward carry `@pytest.mark.whitebox`. A smoke test against the real
  services carries `@pytest.mark.live` and is skipped unless `RUN_LIVE=1`.

## 10. Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | Corpus: nine PDFs on Python and RISC-V (2,017 pages), replacing the supplied FDA corpus | The orchestrator's senior research project is a personalized programming tutor; this tests the retrieval process on that subject, asking whether an agent responds well to a student's Python or assembly question. Instructor allowed a different topic if the nine queries mirror the supplied pattern. See `docs/report-notes.md`. |
| D2 | Ingestion built before its spec (orchestrator decision, 2026-10-26) | Embedding is the longest step; the spec in `specs/ingestion.md` documents the built behavior. The agent half returns to spec-first. |
| D3 | Extractor: pdfplumber layout mode, `x_tolerance=1`; two-column sources read per column | Measured: pypdf lost Python indentation; the default tolerance glued words in 5 of 9 documents. |
| D4 | Chunks never cross pages; 1,000 characters with 150 overlap to start | One page per chunk keeps page citations exact. Size compared with 800/120 and 1,400/200 at milestone 4 and then frozen. |
| D5 | Two tools: `search_documents` and `finish` (orchestrator, 2026-10-26) | Find evidence, then answer. A neighbor-reading tool is added only if the probe shows split code listings. |
| D6 | Citations are `[n]` markers plus a citation list (orchestrator) | Readable answers; markers can be checked mechanically. |
| D7 | `finish` requires at least one prior search (P4) (orchestrator) | An insufficient-evidence result must rest on an actual search. |
| D8 | Step limit 8, `k` default 5 and maximum 8 (orchestrator) | Typical runs need 2-4 steps; cross-document questions 4-5. |
| D9 | Four terminal reasons; `REJECTION_LIMIT`, `NO_ACTION`, `MODEL_ERROR` and `RETRIEVAL_ERROR` are kinds of `ERROR` | The assignment names exactly four outcomes. HW-08's five statuses map onto them. |
| D10 | Three consecutive rejections end the run | With only 8 steps, a confused model should stop early rather than waste the budget. |
| D11 | Chat model Qwen3.8-27B from the assignment's service file, `temperature=0`, `max_retries=0` (orchestrator) | Matches the recorded configuration. `max_retries=0` avoids the hidden-retry delay found in HW-08. |
| D12 | Qwen3 query instruction measured on and off at milestone 4 before choosing (orchestrator) | Qwen3-Embedding is trained with query instructions, but the gain must be measured, not assumed. Outcome: on (D19). |
| D13 | `uv` with a lockfile (orchestrator) | One reproducible install command for the grader. |
| D14 | The agent code is written by a fresh coding agent from specs and tests (orchestrator) | Repeats the HW-08 process; an outside implementer exposes underspecified interfaces. |
| D15 | The model keeps technical details exactly as written in the passage (orchestrator, 2026-10-26) | Paraphrasing mnemonics, register names, numbers and code is where errors creep in. |
| D16 | The full conversation history is sent every turn (no sliding window) | Runs are at most 8 steps; HW-08's 12-turn window is not needed. |
| D17 | Hybrid retrieval: vector and BM25 keyword rankings fused by Reciprocal Rank Fusion (k = 60, 30 candidates per list); `vector` and `keyword` modes kept for comparison (orchestrator, 2026-10-26) | First probe: vector search alone ranked Guo's title and captions above the limitations section, and the Qwen3 instruction did not change the order. Keyword matching rewards exact terms. Query-time only: no re-embedding. |
| D18 | BM25 written in the project (about 60 lines) rather than a library | Transparent for the report, fully tested, and its tokenizer keeps technical tokens (`callee-saved`, `x10`, `__init__`) whole. |
| D19 | Configuration frozen: hybrid retrieval with the query instruction "Given a question, retrieve passages that answer it" (orchestrator, 2026-10-27) | Probe of 24 cases (`runs/probe-2026-10-27_0738.md`): gold passage at rank 1 in 16 cases (next best 15), in the top 5 in 21 (tied best), worst rank 8 (others 11 to not found). |
| D20 | Chunk size not varied; stays 1,000/150 (orchestrator, 2026-10-27) | Every focused query already found its gold passage at rank 1-2; each comparison would re-embed the corpus (about 22 minutes) for little expected gain. Recorded as a limitation. |
| D21 | Required queries frozen; SHA-256 of `queries/required-queries.md` recorded in `docs/process-log.md` (orchestrator, 2026-10-27) | The assignment requires every query to be run exactly as written. |
| D22 | Chat model changed from Qwen3.8-27B (D11) to Qwen3.8-Flash-Next, before any required run (orchestrator, 2026-10-27) | The gateway returns HTTP 403 `team_model_access_denied` for Qwen3.8-27B with the course key; it offers Qwen3.8-Flash-Next and Gemma-4-26B-A4B. On the same JALR question, Flash-Next answered in 2 steps with 4 valid citations (`runs/try-flash.jsonl`); Gemma wrote chunk ids instead of `[n]` markers, was rejected under P6, then replied without a tool call (`ERROR/NO_ACTION`, `runs/try-gemma.jsonl`). Flash-Next is also the Qwen family the assignment names and the Lecture 170 example's model. |
| D23 | Answer limit raised from 4,000 to 8,000 characters; a too-long rejection states the actual length and the maximum; the prompt states the limit (orchestrator, 2026-10-27, after required run 1) | Run 1 Q6 retrieved all the evidence, then wrote 5,865 and 4,308 characters; both were rejected with a generic message, and the run ended at the step limit. The 4,000 limit had been chosen without evidence. |
| D24 | One reminder for a reply without a tool call, counted as a rejection, before `ERROR/NO_ACTION` (orchestrator, 2026-10-27, after required run 1) | Run 1 Q3 retrieved the right passage, then wrote a complete cited answer as plain text; ending at once (a rule carried over from HW-08) discarded it. Counting the reminder as a rejection keeps the rejection limit as the bound on loops. Both changes are general, not specific to Q3 or Q6, and all nine queries are rerun with them (run 2). |
| D25 | The system prompt also states the citation limit (at most 12), so it now states every schema limit the model can hit in `finish` (orchestrator, 2026-10-27, before run 2) | In an uncounted check of the D23/D24 fixes (`runs/not-required/check-fixes/`), Q6 listed 19 citations, was rejected under P3, and finished correctly only on its last step. Same principle as D23: tell the model the limits up front instead of teaching them by rejection. |

## 11. Recorded configuration (frozen 2026-10-27; used for all required runs)

| Setting | Value |
|---|---|
| Embedding model | Qwen3-Embedding-8B, 4,096 dimensions, cosine distance |
| Vector store | Chroma collection `python_riscv_corpus`, 5,655 chunks, `index_version` 48c92aea33b03628 |
| Extraction and cleaning | pdfplumber layout mode, `x_tolerance=1`; cleaning version 2026-10-26.2 |
| Chunk size and overlap | 1,000 and 150 characters; chunks never cross pages |
| Retrieval | hybrid: vector and BM25 keyword lists of 30 candidates each, fused by Reciprocal Rank Fusion (k = 60) |
| Query instruction | "Given a question, retrieve passages that answer it" |
| Results per search (`k`) | 5 by default, at most 8 |
| Step limit | 8 model/tool steps |
| Rejection limit | 3 consecutive rejected requests |
| Answer length limit | 8,000 characters (raised from 4,000 after run 1, D23) |
| Reply without a tool call | 1 reminder, then `ERROR/NO_ACTION` (D24) |
| Chat model | Qwen3.8-Flash-Next, temperature 0 (changed from Qwen3.8-27B, D22) |
