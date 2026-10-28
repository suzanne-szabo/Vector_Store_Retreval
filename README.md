# Python and RISC-V Retrieval Agent (U0-HW-09)

A bounded ReAct agent that answers Python and RISC-V questions from a fixed corpus of nine
PDFs, citing the chunks it retrieved, or reporting that the corpus does not support an answer.

**Author:** Suzanne Szabo, Utah Tech University, AI (U0-HW-09), Fall 2026

> Status: ingestion is built and tested. The agent loop comes next (see `docs/process-log.md`).

## Requirements

- Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) (`uv` installs the right Python).
- The course embedding and chat services, reachable only on the course network or over
  Tailscale. They are needed only to build the store and to ask questions, not for tests.

## Setup

```sh
uv sync --locked
cp .env.example .env      # then set both API keys and, if needed, DOCUMENTS_DIR
```

`.env` is ignored by Git and must never be committed. `DOCUMENTS_DIR` points at the folder
holding the corpus PDFs, which also stay out of Git (default:
`../U0-HW-09-materials/documents/personal_documents`).

## The corpus

`corpus/sources.toml` lists the nine documents with their ids, roles, licenses, download URLs
and SHA-256 hashes. Ingestion refuses any file whose hash does not match.

## Build the vector store

```sh
uv run python -m rag_agent.ingest --dry-run                 # no services: extract, clean, chunk, report
uv run python -m rag_agent.ingest --only guo-2013           # embed one source
uv run python -m rag_agent.ingest --rebuild                 # wipe and embed all nine sources
```

Without `--rebuild`, each selected source's chunks are replaced and other sources are kept, so the
corpus can be embedded a few sources at a time. Mixing chunk or extraction settings in one store is
refused. Defaults: 1,000-character chunks with 150 overlap, cosine distance, Qwen3-Embedding-8B.

## Inspect it

```sh
uv run python -m rag_agent.inspect --manifest
uv run python -m rag_agent.inspect --records --source guo-2013 --page 2
uv run python -m rag_agent.inspect --search "how many people use the tool" --source guo-2013
```

## Tests

No network or API key is needed.

```sh
uv run pytest -q
```
