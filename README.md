# Python and RISC-V Retrieval Agent (U0-HW-09)

A bounded ReAct agent that answers Python and RISC-V assembly questions from a fixed corpus of nine
PDFs. It searches a Chroma vector store with hybrid (vector + keyword) retrieval, answers only from the
passages it retrieved, cites each one by chunk id, source and page, and reports when the corpus does not
support an answer. The application, not the model, validates every tool call and citation and decides
how each run ends.

**Author:** Suzanne Szabo, Utah Tech University, AI (U0-HW-09), Fall 2026

## How it works

```
Ingestion (once):  9 PDFs -> verify SHA-256 -> extract pages -> clean -> chunk -> embed -> Chroma
Query time:        START -> llm -> controller -> tool -> continue? --(terminal)--> END
                               ^                            |
                               +-------(otherwise)----------+
```

- **Tools:** `search_documents(query, sources?, subject?, k?)` returns ranked passages, each with its
  chunk id, source id, title, one-based PDF page and text; `finish(outcome, answer, citations)` ends the
  run with `answer` or `insufficient_evidence`.
- **Controller:** validates every call (policies P1-P7), accepts a citation only if its chunk was
  retrieved in that run, counts steps, records the trace, and ends each run with one of four terminal
  reasons: `ANSWER`, `INSUFFICIENT_EVIDENCE`, `ERROR` (with a kind) or `STEP_LIMIT`.
- **Retrieval:** Qwen3-Embedding-8B vectors and a BM25 keyword index, fused by Reciprocal Rank Fusion.

The design, decisions and their evidence are in `specs/` (start with `specs/architecture.md`) and
`docs/process-log.md`.

## Requirements

- **Python 3.11 or newer** and [`uv`](https://docs.astral.sh/uv/) (`brew install uv` on macOS). `uv`
  installs a suitable Python if needed.
- The course embedding and chat services, reachable on the course network or over Tailscale. They are
  needed only to rebuild the store and to ask questions, never for the tests.
- The nine corpus PDFs (not in this repository). `corpus/sources.toml` lists each file's name, download
  URL and SHA-256 hash.

## Setup

```sh
uv sync --locked                 # install the exact tested dependencies
cp .env.example .env             # then set EMBEDDING_API_KEY and LLM_API_KEY
```

`.env` is ignored by Git and must never be committed. Set `DOCUMENTS_DIR` in `.env` to the folder holding
the nine PDFs (default `../U0-HW-09-materials/documents/personal_documents`). If a service host name does
not resolve over Tailscale, use the gateway address `http://100.64.0.1:4000/v1` for that service.

## Rebuild the vector store

```sh
uv run python -m rag_agent.ingest --rebuild
```

This verifies each PDF's hash, extracts, cleans and chunks every page, embeds the 5,655 chunks
(about 22 minutes), and writes `chroma_db/` and `manifest.json` (both ignored by Git). Useful options:
`--dry-run` (no embedding service; prints chunk counts), `--only SOURCE_ID ...` (embed some sources),
`--help`.

## Ask a question

```sh
uv run python -m rag_agent "What does the RISC-V JALR instruction write to its destination register?"
uv run python -m rag_agent "How does Python's // operator differ from /?" --trace-out runs/floor.jsonl
uv run python -m rag_agent --help
```

Each step prints as it happens, then the terminal reason, the answer and its citations
(`[1] riscv-isa-2025, page 51 (chunk d00bab35…)`). Options: `--max-steps`, `--k`, `--mode`,
`--query-instruction`, `--trace-out`, `--quiet`.

To look at retrieval alone: `uv run python -m rag_agent.inspect --search "callee-saved registers"`.

## Run the nine required queries

```sh
uv run python scripts/run_required.py
```

Runs every query in `queries/required-queries.md` exactly as written with one configuration, and writes
a JSONL action-observation trace per query, a copy of the manifest, and `summary.md` to
`runs/<date_time>/`.

## Run the tests

```sh
uv run pytest -q
```

192 tests, all offline: a scripted fake chat model, a fake retriever and fake embeddings stand in for the
course services.

## Recorded configuration (all required runs)

| Setting | Value |
|---|---|
| Embedding model | Qwen3-Embedding-8B, 4,096 dimensions, cosine distance |
| Vector store | Chroma, 5,655 chunks, `index_version` 48c92aea33b03628 |
| Chunk size and overlap | 1,000 and 150 characters; chunks never cross pages |
| Retrieval | hybrid: vector + BM25, 30 candidates each, Reciprocal Rank Fusion (k = 60) |
| Query instruction | "Given a question, retrieve passages that answer it" |
| Results per search (`k`) | 5 |
| Model/tool step limit | 8 |
| Chat model | Qwen3.8-Flash-Next, temperature 0 |

## Results

| Run | Folder | Outcome |
|---|---|---|
| Run 1 | `runs/2026-10-27_1914/` | 7 of 9 as expected (Q3 `ERROR/NO_ACTION`, Q6 `STEP_LIMIT`) |
| Run 2 (final) | `runs/2026-10-27_2054/` | 8 of 9 as expected (Q6 `STEP_LIMIT`) |

Run 1's two misses led to two general changes (architecture D23-D25) before all nine were rerun. Runs in
`runs/not-required/` were diagnostic and are not part of the results. The claim-by-claim evaluation is
in `docs/claim-evaluation.md`.

## Repository layout

| Path | Contents |
|---|---|
| `rag_agent/` | Ingestion (`corpus`, `extract`, `clean`, `chunk`, `store`, `ingest`, `inspect`), retrieval (`keyword`, `retrieval`) and the agent (`tools`, `controller`, `session`, `prompts`, `llm`, `agent`, `cli`, `required`) |
| `specs/` | Architecture and module specifications, with the decision log |
| `tests/` | Offline tests and test doubles |
| `corpus/sources.toml` | The nine source documents: ids, roles, licenses, URLs, SHA-256 |
| `queries/required-queries.md` | The nine frozen required queries |
| `runs/` | Required-run traces and summaries, the retrieval probe, model-comparison traces |
| `scripts/` | `run_required.py`, `probe.py` (retrieval probe), `check_llm.py` (tool-calling check) |
| `docs/` | Process log, claim evaluation, report notes, model check output |
