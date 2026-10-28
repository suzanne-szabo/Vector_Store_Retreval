# Development Process Log

A running record of decisions, findings from tests, and changes to the plan.

## 2026-10-25: Corpus selection

- Topic changed from the supplied FDA corpus to Python and RISC-V (allowed by the instructor, provided
  the nine queries mirror the supplied pattern). Nine documents, 2,017 pages; see `corpus/sources.toml`.
- Rejected: a Safari print of Borin's HTML book (139 of 167 pages empty, every line repeated about
  8 times), *Automate the Boring Stuff* (overlap with the other Python books), and a 909-page
  intermediate build of the ISA manual (replaced by the numbered 20250508 release).

## 2026-10-26: Ingestion built before the agent

- Orchestrator decision: start with embedding, since it is the longest step, beginning with the
  6-page Guo paper before the larger sources. The plan's spec-first order is kept for the agent loop.
- Extraction check on the real corpus:
  - pypdf flattened Python indentation in Heinold; pdfplumber's layout mode keeps it.
  - pdfplumber's default `x_tolerance=3` glued words on justified lines (for example
    "programvisualizerthatcanbeseamlesslyembeddedwithin" in Guo; 290 sampled cases in Heinold,
    368 in Winans). `x_tolerance=1` fixed them without splitting real words.
  - Guo is a two-column ACM paper; layout mode would interleave the columns, so each column is
    read separately (`columns = 2` in the registry).
- Test fixture finding: a first fixture drawn in 10-point Courier extracted with 3-space
  indentation and an extra blank line. Cause: layout mode maps positions to a character grid of
  about 7.25 x 13 points. The fixture now uses 12-point Courier on 13-point lines, like a book page.
- Defect found by tests: `--rebuild` deleted `chroma_db/` while Chroma held it open in the same
  process, leaving a read-only handle. Rebuild now resets the collection through Chroma.
- Dry run on the real corpus (no embedding) found two more problems, both fixed:
  1. Splitting Guo's pages down the middle cut the title and author block ("Online Python Tutor: Em").
     A full-width block at the top of a page (lines crossing the column gutter) is now read whole
     before the two columns; Figure 1's full-width caption on page 2 is handled the same way.
  2. Table-of-contents dot leaders (". . . . . 57") filled chunks in *Python Crash Course*; they are
     shortened to "...". `CLEANING_VERSION` is now recorded in the manifest so stores built with
     different cleaning rules cannot be mixed.
- Header/footer removal confirmed per document in the dry-run report: the ISA manual's title line
  (706 pages), Winans' build stamps, Dagstuhl's running title and authors, chapter headers in
  Heinold and *Python Crash Course*, and page numbers. Known small loss: *Python Crash Course*'s
  "Try It Yourself" heading is removed on the 19 pages where it sits at the top.
- Spot checks for the draft queries: the JALR definition (Q1) is one chunk on page 51 of the 2025
  manual; *Python Crash Course* has the indentation-errors section (Q2); Guo has a "Limitations and
  ongoing work" section (Q5); the 2025 manual's first page states its version and ratified state (Q7).
- Result: 31 offline tests pass, on the cloud workspace and on the Mac.

## 2026-10-26: First live embed attempt and design decisions

- First `--only guo-2013` run: extraction worked, but the embedding request timed out connecting to
  `golem.cs.utahtech.edu:4000`. With `http://golem:4000/v1` it failed at once (the name did not
  resolve over Tailscale). `curl` to the gateway's Tailscale address `100.64.0.1:4000` returned 401,
  meaning reachable. The address came from the orchestrator's Lecture 170 `.env`. `.env.example` now
  documents it, and connection failures print a short hint instead of a traceback.
- Orchestrator decisions on the ten open design questions (all as recommended): two tools; `[n]`
  citation markers; search required before `finish`; 8 steps and `k` = 5; `uv`; pdfplumber kept, with
  chunk size and the Qwen3 query instruction compared at milestone 4; Qwen3.8-27B; a fresh coding agent
  implements the agent from specs and tests.
- Specs written: `architecture.md`, `ingestion.md` (as built), `tools.md`, `controller.md`, `agent.md`.

## 2026-10-26: First live embed and first retrieval probe

- Guo embedded: 44 chunks in 13.8 s (about 0.3 s per chunk; about 30 minutes projected for all 5,666),
  4,096 dimensions, cosine distance.
- Probe, search restricted to Guo, looking for section 6, "Limitations and ongoing work":

| Run | Query | Section 6 rank | Top distance |
|---|---|---|---|
| A | "limitations of Online Python Tutor" | not in top 10 (a second limitations passage was 9th) | 0.1515 |
| B | A plus the Qwen3 query instruction | not in top 3; order unchanged | 0.0951 |
| C | "what limitations and weaknesses does the paper acknowledge" | 3rd (ACM copyright notice 1st) | 0.5258 |

- Diagnosis: in A and B every top-10 result contained the tool's name. Short chunks (title block,
  figure captions) are dominated by those words, so a query containing the name matches them closely.
  The query instruction lowered every distance but did not change the order. Query wording mattered
  most. Action: the agent prompt now tells the model to describe the content it needs rather than
  repeat the document's subject when a source filter already selects it (agent spec §2.1).
- Second suspect: Guo's narrow columns break many words across lines ("ben-" / "efited"), giving the
  embedding model fragments. A first rule that dropped every line-end hyphen would also have broken
  compounds such as `callee-saved` (the psABI term Q4 depends on), `floating-point` and
  `general-purpose`. Final rule: the document's own vocabulary decides (ingestion spec §4, step 6),
  with fragments excluded from that vocabulary after a check found `pop-ular` treated as two words.
  About 99% of 1,385 cases are rejoined correctly. `CLEANING_VERSION` is now `2026-10-26.2`.
- Guo re-embedded with the new cleaning (`index_version` 14464057d5f363e8) and runs A and C repeated:
  run A unchanged (other limitations passage 10th instead of 9th); run C slightly worse (section 6 fell
  out of the top 3). Rejoining broken words is neutral for ranking; it is kept because the model reads
  and quotes cleaner text.

## 2026-10-26: Hybrid retrieval (architecture D17, D18)

- Orchestrator decision: add keyword search and fuse it with vector search. Pure vector search ranked
  short chunks dense in the query's subject words above the passage that answers the question, and
  neither the Qwen3 instruction nor rejoining broken words fixed the order.
- Built `keyword.py` (BM25, about 60 lines, with a tokenizer that keeps `callee-saved`, `x10-x17` and
  `__init__` whole) and `retrieval.py` (`HybridRetriever` with `hybrid`, `vector` and `keyword` modes,
  Reciprocal Rank Fusion with k = 60 over 30 candidates per list). Every passage records its vector rank,
  keyword rank and distance, so traces show why it was chosen. Query-time only: no re-embedding.
- `inspect --search` now takes `--mode`. On the test fixture, hybrid mode ranked the "return address"
  chunk 1st although vector search alone placed it 3rd.
- Tests: 57 pass (26 new for keyword and retrieval).

## 2026-10-27: Full corpus embedded and verified

- All nine sources embedded in two runs (the first with `--rebuild`): **5,655 chunks** in about
  22 minutes of embedding (riscv-isa-2025 alone: 1,880 chunks in 512 s). `index_version`
  48c92aea33b03628. The count is 11 lower than the earlier dry run because rejoining broken words
  shortened some pages.
- Verification on a copy of the store: 5,655 unique ids; every chunk has all nine metadata fields, a
  page within its document's page count, and at most 1,000 characters; per-source counts match the
  manifest.
- Keyword-only probe (offline, `scripts/probe.py --keyword-only`): the gold passage ranked 1st in 12 of
  24 cases and within the top 5 in 21. The Guo limitations section, which vector search missed from the
  top 10, ranks 1st by keyword. Weak spots: the 2025 manual's version statement (rank 11, too generic
  for keywords) and, without a source filter, the JALR definition, where the 2019 specification ranks
  first (the Q7 version question in miniature).

## 2026-10-27: Purpose recorded

- The subject and corpus were chosen to test this process on material for the orchestrator's senior
  research project, a personalized programming tutor: can such an agent respond well when a student
  asks about a Python or assembly assignment? Recorded in architecture D1 and drafted for the report
  in `docs/report-notes.md`.

## 2026-10-27: Queries refocused on Python and RISC-V

- Orchestrator decision: the queries target Python and RISC-V assembly and lean on the large sources.
  The Dagstuhl and Guo papers stay in the corpus (already embedded) but no query depends on them.
- Replacements, each with its answering passage confirmed in the store: Q3 (several facts from one
  entry) is now `__init__()` and `self` in *Python Crash Course* p. 200; Q4 (table comparison) is
  Heinold's operator table, p. 30; Q5 (a claim plus its stated limits) is the 2025 manual's decision
  not to add overflow-check instructions, with the branch sequences and their cost, p. 48.
- Draft written to `queries/required-queries.md`; keyword-only probe of the new set: gold passage at
  rank 1 in 14 of 24 cases, top 5 in 19. Weakest: Q2 and Q4 asked word for word, and Q6's return-address
  passage without a source filter.

## 2026-10-27: Retrieval probe on the full corpus; configuration and queries frozen (milestone 4)

- `scripts/probe.py` on the full store, 24 cases (7 focused queries and 5 draft questions as written,
  each with and without a source filter), gold passage rank in the top 30
  (`runs/probe-2026-10-27_0738.md`, 505 s):

| Configuration | Rank 1 | Top 5 (k = 5) | Worst rank |
|---|---:|---:|---:|
| vector | 14 | 20 | 20 |
| vector + instruction | 15 | 21 | 11 |
| keyword | 14 | 19 | not found (2 cases) |
| hybrid | 14 | 21 | 15 |
| hybrid + instruction | **16** | **21** | **8** |

- The two rankings cover each other: the 2025 manual's version statement ranked 5-12 by vector and 1 by
  keyword; Q2 asked word for word was missed by keyword and found 1st by vector. Remaining weak spots for
  hybrid + instruction: Q2 asked word for word (7th) and Q6's return-address passage without a source
  filter (8th; 1st with the psABI filter, supporting the prompt rule to filter by source).
- The query instruction helped on the full corpus, although the 44-chunk Guo test had shown no change in
  order: a small test was misleading.
- Orchestrator decisions: freeze hybrid + instruction (D19); keep 1,000/150 chunks without a size
  comparison (D20); freeze the nine queries as written (D21). Recorded configuration: architecture §11.
- Frozen queries file `queries/required-queries.md`, SHA-256
  `7678abca9a1e94eeb291dc50cb2b31354fc07c4bf4e2982e502c297e34a41451`.

## 2026-10-27: Black-box tests for the agent (milestone 3)

- Tests written from the specs before any agent code: `test_tools.py`, `test_controller.py`,
  `test_prompts.py`, `test_agent_loop.py`, `test_cli.py`, `test_required.py`, with the test doubles in
  `tests/fakes.py` (HW-08's scripted model plus a fake retriever over three fixed passages).
- As in HW-08, deriving tests exposed underspecified interfaces. The specs were amended before the tests
  were finalized:
  1. `tools.execute` needs `steps_remaining` as a parameter (tools §5).
  2. `write_trace` had no home; it is `rag_agent.agent.write_trace`, and `RunResult` gains `started_at`
     and `finished_at` for the trace header (agent §4, §6).
  3. The CLI could not be tested without live services: `cli.main(argv, *, model, retriever) -> int`
     and a `--db` option (agent §5).
  4. The required-run logic moved from the script into `rag_agent/required.py` so it can be imported; the
     parser joins wrapped question lines with single spaces, and the summary table header is fixed
     (agent §7).
  5. The fake retriever's subject filter needed a defined source of subjects (the real registry).
- A private reference implementation was considered for checking the tests and rejected: it would mean
  writing the agent before the coding agent does (D14). The tests were checked by compiling them,
  confirming they fail only on the missing modules, and mapping each spec test bullet to a test.
- State: 57 ingestion and retrieval tests pass; the 6 new files fail at import until the agent exists.
- Orchestrator request: the question-asking CLI must offer `--help` listing every option and an example
  (agent spec §5, v1.2, with a test). The existing `ingest` and `inspect` help texts were completed.
- Orchestrator review of the tests: approved.

## 2026-10-27: Coding-agent handoff (architecture D14)

- A fresh coding agent with no access to the design conversation received an isolated copy containing
  only `specs/`, `tests/`, the built ingestion and retrieval modules, `corpus/`, `queries/`, the README
  and the project files (`docs/` was excluded because it records the design conversation). SHA-256
  hashes of the 19 spec and test files were recorded before the handoff to verify afterward that none
  was changed.
- Instructions: implement controller, session, tools, prompts, llm, agent, cli, required and the two
  scripts; do not edit specs, tests or built modules; report any spec defect instead of working around
  it, leaving the affected test failing.

## 2026-10-27: Coding-agent result and review

- Result: 11 new files (about 1,400 lines); **185 passed, 1 failed** in about 8 minutes of work. It
  edited no spec, test or built module (all 19 handoff hashes match; the 10 built modules are
  byte-identical).
- The failure was a **test defect**, which the coding agent reported rather than working around:
  `test_required.py` picked summary rows with `startswith("| Q")`, which also matches the specified
  header `| Query | ...`. The reviewer (test author) fixed the filter to `^\| Q\d`; the code was not
  changed. Full suite: **186 passed**.
- Gaps it found and filled additively; each is now written into the specs:
  1. no way to record a retriever that cannot be built: `Controller.record_retrieval_error`
     (controller §4);
  2. no path for `run(on_entry=...)` to reach the trace: `Controller(..., on_entry=None)`;
  3. raw arguments of approved calls were not traceable: `Approval.raw_args`;
  4. a JSON string that decodes to an object is validated as that object (P3);
  5. the CLI builds the retriever from `--db`, since `run` takes no database path; `--query-instruction
     ""` turns the instruction off;
  6. "missing number" in the queries parser made precise (numbers must run 1..n without gaps).
- A built-module issue it noted, now fixed: `inspect --search` did not use the frozen query instruction
  (D19), so its rankings could differ from the agent's. It now defaults to the same instruction
  (`retrieval.DEFAULT_QUERY_INSTRUCTION`), and `--instruction ""` turns it off.
- Reviewer checks of the architecture boundaries (architecture §5) by searching the code: only `tools`
  writes to `session.retrieved`; only the controller sets the terminal reason; graph state holds only
  `messages` and `pending`, with `Session` and `Controller` passed through `config["configurable"]`;
  the chat model uses `max_retries=0`, `tool_choice="required"`, `parallel_tool_calls=False`, and the
  recursion backstop is `4 * max_steps + 10`.
- Comparison with HW-08: there the coding agent found two spec defects; here it found one test defect
  and six underspecified interfaces, after the test-derivation phase had already fixed five.

## 2026-10-27: First live runs and the chat-model change (architecture D22)

- `scripts/check_llm.py` with the `.env` copied from the Lecture 170 example ran Qwen3.8-Flash-Next by
  mistake (`LLM_MODEL` in that file). Corrected to the recorded Qwen3.8-27B: the gateway returned
  HTTP 403 `team_model_access_denied`. The agent itself handled the same failure correctly: one retry,
  then `ERROR/MODEL_ERROR` with the gateway's message (a live check of that terminal reason).
- Models available to the course key: Qwen3.8-Flash-Next, Gemma-4-26B-A4B (chat); Qwen3-Embedding-8B,
  -4B, -0.6B, EmbeddingGemma-2 (embedding); Clef and Clef-Flash (unknown, not considered).
- Both chat models on "What does the RISC-V JALR instruction write to its destination register?":
  - **Flash-Next** (`runs/try-flash.jsonl`): one search (subject filter `riscv`), then `finish`;
    `ANSWER` in 2 steps with 4 citations, all retrieved in the run (2025 p. 51, 2025 p. 187, 2019 p. 39,
    2019 p. 158). Support check by reading each cited passage: every claim supported except one clause,
    "so the target is always even, which matters with RVC", which is not in the cited passage (model
    knowledge). The opening sentence carries no `[n]` marker, against the prompt rule. It applied the
    2019-vs-2025 rule unprompted.
  - **Gemma** (`runs/try-gemma.jsonl`): wrote chunk ids inside brackets instead of `[n]` markers;
    rejected under P6 (`UNUSED_CITATION`); then replied without a tool call although tool choice is
    forced, ending `ERROR/NO_ACTION` after 3 steps.
- Orchestrator decision D22: Qwen3.8-Flash-Next is the recorded chat model. No required run had been
  made, so every required run uses one configuration.
- Finding for the report: citation checks prove provenance (all four citations came from the run's
  retrieval) but not support; the unsupported "always even" clause was caught only by reading the
  passage, which is why the assignment asks for claim-by-claim evaluation.
- First attempt at the required runs (19:10) used Qwen3.8-27B and ended `ERROR/MODEL_ERROR` on all nine
  (403). Cause: an earlier `set -a; source .env` had exported `LLM_MODEL=Qwen3.8-27B` into the shell,
  and settings in the environment override `.env` by design. The folder was moved to
  `runs/not-required/2026-10-27_1910-wrong-model-403/` and is not part of the results. Fix: unset the
  variables. Safeguard: `run_required.py` now prints the configuration before the first query (agent
  spec §7, v1.4).

## 2026-10-27: Required run 1 (`runs/2026-10-27_1914/`): 7 of 9 as expected

| Query | Expected | Run 1 | Steps |
|---|---|---|---|
| Q1 | ANSWER | ANSWER | 2/8 |
| Q2 | ANSWER | ANSWER | 4/8 |
| Q3 | ANSWER | **ERROR/NO_ACTION** | 3/8 |
| Q4 | ANSWER | ANSWER | 4/8 |
| Q5 | ANSWER | ANSWER | 3/8 |
| Q6 | ANSWER | **STEP_LIMIT** | 8/8 |
| Q7 | ANSWER | ANSWER | 4/8 |
| Q8 | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | 6/8 |
| Q9 | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | 4/8 |

- **Q6 diagnosis (answer stage, then step budget):** retrieval succeeded. Six searches found the
  *Python Crash Course* pages on arguments and return values (pp. 168-174), the psABI register table
  (p. 9), Winans' function material and the JALR definition (2025 p. 51). The model then called `finish`
  with a 5,865-character answer, over the 4,000 limit (P3); the retry on the last step was still
  4,308. The rejection message did not say how long the answer was or how much to cut. One search was
  also repeated after a P7 rejection on step 1 (two searches in one reply).
- **Q3 diagnosis (tool-calling compliance):** retrieval succeeded (the `__init__` passage, p. 200, ranked
  1st). On step 3 the model wrote a complete cited answer (3,262 characters, with `[n]` markers) as plain
  text instead of calling `finish`. Forced tool choice was not enforced, and the run ended at once
  (`NO_ACTION`), discarding the answer. Gemma showed the same behavior in the model comparison.
- Both failures came from limits set in the design (an answer limit chosen without evidence; immediate
  `NO_ACTION`, carried over from HW-08), not from retrieval.

## 2026-10-27: Changes after run 1 (architecture D23, D24); all nine rerun

- Orchestrator decision: fix both general limits and rerun **all nine** queries, so the final results
  come from one configuration. Run 1 stays in the report with its diagnoses. Nothing was tuned to Q3 or
  Q6 specifically.
- D23: answer limit 4,000 → 8,000 characters (`MAX_ANSWER_CHARS`); a too-long rejection now states the
  actual length and the maximum; the system prompt states the limit and asks for brief answers to
  multi-part questions.
- D24: a reply without a tool call gets one reminder ("call finish with the answer text and its
  citations"), counted as a rejection, before the run ends as `ERROR/NO_ACTION`; the prompt adds "Never
  reply with plain text: the answer goes in finish."
- Specs amended first (tools §3.2, controller §2/§4/§5.1/§7, agent §1.2/§2.1, architecture §11), then
  tests (6 new or changed, written and failing before the code), then the code. Full suite: 192 passed.
  The code change was made by the reviewer, since it was small and targeted.
- Uncounted check of the fixes (`runs/not-required/check-fixes/`, Q3 and Q6 only): Q3 `ANSWER` in 3
  steps (it called `finish` directly; no reminder was needed this time). Q6 `ANSWER` on step 8 of 8:
  six filtered searches; step 7's 5,596-character answer was now within the limit but listed 19
  citations (maximum 12) and was rejected; step 8 cut to 12 and was accepted, citing all four expected
  sources.
- D25 (orchestrator): the system prompt also states the citation limit ("Cite at most 12 passages"), so
  it states every `finish` limit up front. Spec, then failing test, then code; full suite passing.
