# Ingestion Specification

Status: **v1, written after implementation (architecture D2); for orchestrator review 2026-10-26**
Modules: `config`, `corpus`, `extract`, `clean`, `chunk`, `store`, `ingest`, `inspect`
Tests: `tests/test_corpus.py`, `tests/test_clean.py`, `tests/test_ingest.py` (31 passing)

Ingestion turns the nine registered PDFs into one persistent Chroma collection. It runs as six stages,
each a separate module, so a retrieval problem can be traced to the stage that caused it.

## 1. Configuration (`config`)

| Name | Value |
|---|---|
| `DEFAULT_DB_PATH` | `<project>/chroma_db` |
| `DEFAULT_MANIFEST_PATH` | `<project>/manifest.json` |
| `DEFAULT_REGISTRY_PATH` | `<project>/corpus/sources.toml` |
| `COLLECTION_NAME` | `python_riscv_corpus` |
| `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` | 1000, 150 |
| `EMBED_BATCH_SIZE` | 32 texts per embedding request |
| `STORE_BATCH_SIZE` | 64 chunks per Chroma add and per progress line |

Settings come from `<project>/.env`; variables already set in the environment win. `require_api_key`
rejects an empty key and the placeholder `student-should-use-their-own`. `documents_dir()` reads
`DOCUMENTS_DIR` (default `../U0-HW-09-materials/documents/personal_documents`), resolving relative
paths against the project folder.

## 2. Source registry (`corpus`)

`corpus/sources.toml` holds one `[[source]]` table per document with: `id`, `file`, `title`, `subject`,
`version`, `role`, `license`, `url`, `pages`, `sha256`, and optional `columns` (1 or 2, default 1).

- `load_registry(path) -> list[Source]` validates: at least one source; ids match
  `^[a-z0-9][a-z0-9-]{1,40}$` and are unique; `subject` is `python`, `riscv` or `pedagogy`; `sha256` is
  64 lowercase hex characters; `columns` is 1 or 2. Any violation raises `CorpusError`.
- `select(sources, only)` keeps the named ids in registry order; an unknown id raises `CorpusError`
  listing the known ids.
- `resolve(sources, documents_dir)` returns `(Source, Path)` pairs. A missing file or a SHA-256 mismatch
  raises one `CorpusError` listing every problem. Nothing is extracted until every file checks out.

## 3. Extraction (`extract`)

`extract_pages(path, columns=1) -> list[str]` returns one raw string per physical page; index 0 is page 1.
Every page is returned, including blank ones, so page numbers stay physical.

- pdfplumber `extract_text(layout=True, x_tolerance=1)`. Layout mode keeps horizontal positions as
  spaces, which preserves code indentation. `x_tolerance=1` keeps justified words apart.
- For `columns = 2`: lines at the top of a page that cross the column gutter (a word overlapping
  the band of ±8 points around the page middle) form a full-width block, read first across the whole
  width. The left column and then the right column below it are read separately, each with its own
  left margin removed.
- `EXTRACTOR` names the library, version and settings; it is recorded in the manifest.

## 4. Cleaning (`clean`)

`clean_document(raw_pages, removed=None) -> list[str]` returns one cleaned string per physical page,
in order. Steps, in order:

1. `normalize`: Unicode NFKC (ligatures such as "ﬁ" become "fi"), then remove private-use characters
   U+E000–U+F8FF and U+FFFD.
2. `tidy_line` on every line: expand tabs to 4 spaces, strip trailing space, keep leading indentation,
   shrink any run of 3 or more spaces between non-space characters to 2, and shorten table-of-contents
   dot leaders (4 or more dots, with any spacing) to ` ... `.
3. Running headers and footers: for documents of at least 4 pages, a line among the first or last two
   non-blank lines of a page is a running header or footer when its signature appears in that
   position on at least `max(3, min(5, pages // 2))` pages. The signature replaces digit runs with `#`,
   collapses whitespace and lowercases. Such lines are removed from the top and bottom of every page.
   If `removed` is given, it counts removed lines by signature (reported by `--dry-run`).
4. `dedent`: remove the left margin shared by all non-blank lines; relative indentation is kept.
5. `collapse_blank_lines`: at most one blank line in a row, none at the start or end.
6. `join_broken_words`: a word of two or more letters ending a line with a hyphen, followed by a
   lowercase word on the next line, is rejoined and the rest of that line moves up. The document's
   own vocabulary (`document_vocabulary`, built with the broken fragments left out) decides the hyphen;
   the first rule that applies wins:
   1. the hyphenated form appears within a line elsewhere in the document: keep the hyphen
      (`callee-\nsaved` → `callee-saved`);
   2. the joined form appears as a word: drop it (`pro-\ngram` → `program`);
   3. both halves appear as words: keep it (`upper-\nimmediate` → `upper-immediate`);
   4. otherwise drop it (`ben-\nefited` → `benefited`).
7. A page with fewer than 40 characters left becomes `""` and is skipped by chunking.

`CLEANING_VERSION` (currently `2026-10-26.2`) must change whenever a rule changes.

Measured on the corpus: 1,385 words broken across lines, about 99% rejoined correctly. Known misses
include `opensource`, `sideeffect` and `correctnesspreserving` (hyphen wrongly dropped) and
`forgot-ten` and `Water-loo` (hyphen wrongly kept).

## 5. Chunking (`chunk`)

`chunk_document(source, pages, splitter) -> list[Chunk]` splits each non-empty page separately with
LangChain's `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, add_start_index=True)`, so every
chunk lies on exactly one physical page and is at most `chunk_size` characters. `make_splitter` rejects
`chunk_size <= 0`, a negative overlap, or `overlap >= chunk_size` (`ValueError`).

**Chunk id:** `make_chunk_id(source_sha256, page, start_index, text)` = the first 24 hex characters of
SHA-256 over `"source_sha256|page|start_index|text"`. The same file, cleaning and chunk settings always
give the same ids; any change to the text gives new ids.

**Metadata on every chunk** (`METADATA_FIELDS`):

| Field | Example | Meaning |
|---|---|---|
| `chunk_id` | `9f2c1a7e0b3d4c5e6f7a8b9c` | Stable id; also the Chroma record id |
| `source_id` | `riscv-psabi` | Registry id |
| `source_file` | `riscv-elf-psabi-v1.0.pdf` | Source filename |
| `title` | `RISC-V ABIs Specification (version 1.0)` | Registry title |
| `subject` | `riscv` | Registry subject |
| `version` | `1.0` | Registry version |
| `page` | `9` | One-based physical PDF page |
| `start_index` | `0` | Character offset of the chunk in the cleaned page |
| `source_sha256` | `a8d06bdc…` | SHA-256 of the source file |

## 6. Store (`store`)

- `build_embeddings()`: `OpenAIEmbeddings` for `EMBEDDING_MODEL` (default `Qwen3-Embedding-8B`) at
  `EMBEDDING_BASE_URL`, with `check_embedding_ctx_length=False`, 32 texts per request, `max_retries=2`,
  and a 180-second timeout. Documents are embedded without any instruction prefix.
- `open_store(db_path, embeddings=None)`: the persistent Chroma collection `python_riscv_corpus` with
  `hnsw:space = cosine`.

## 7. Pipeline and CLI (`ingest`)

```
uv run python -m rag_agent.ingest [--only ID ...] [--rebuild] [--dry-run]
       [--chunk-size N] [--chunk-overlap N] [--dump FILE.jsonl] [--db DIR] [--manifest FILE]
```

1. Load and select the registry, then `resolve` (hash check).
2. Unless `--rebuild` or `--dry-run`: if `manifest.json` exists, its settings (`embedding_model`,
   `chunking`, `extractor`, `cleaning`) must equal the requested ones, or `IngestError` asks for
   `--rebuild`. One store never mixes settings.
3. For each source: extract, clean, chunk; print pages kept, chunk count, mean and maximum chunk size,
   time, and the five most frequent removed header/footer lines.
4. `--dump FILE` writes every chunk as one JSON line `{"metadata": …, "text": …}`.
5. `--dry-run` stops here; no store is opened and nothing is embedded.
6. Otherwise: open the store; `--rebuild` resets the collection through Chroma (deleting the folder
   while Chroma holds it open leaves a read-only handle). For each source, delete its existing
   records, then add its chunks in batches of 64, printing progress.
7. Write `manifest.json`: `created_at`, `updated_at`, `collection`, `distance`, the four settings
   above, `embedding_dimension`, `record_count`, `index_version` (first 16 hex characters of SHA-256
   over the sorted chunk ids joined by newlines), and per source `file`, `sha256`, `pages_total`,
   `pages_kept`, `chunks`, `embedded_at`, `embed_seconds`.

Without `--rebuild`, sources not selected stay in the store, so the corpus can be embedded a few sources
at a time. Embedding one source at a time gives the same `index_version` as one full build.

**Errors:** registry, hash, settings and argument problems exit with `Error: <message>`. Embedding
service failures exit with a short message: a timeout or connection error names the URL and suggests
checking the network and using the gateway's Tailscale address in `EMBEDDING_BASE_URL`; a rejected key names `EMBEDDING_API_KEY`; another HTTP
error gives its status. Re-running the same command is always safe.

## 8. Inspection (`inspect`)

```
uv run python -m rag_agent.inspect --manifest
uv run python -m rag_agent.inspect --records [--source ID] [--page N] [--limit N]
uv run python -m rag_agent.inspect --search "QUERY" [--source ID] [--limit N]
```

`--records` reads stored chunks without the embedding service. `--search` embeds the query and prints
rank, distance, source id, page, chunk id and the first 600 characters of each result.

## 9. Measured results (dry run, 2026-10-26)

| Source id | Pages kept | Chunks |
|---|---:|---:|
| `riscv-isa-2025` | 724 / 727 | 1,880 |
| `riscv-isa-2019` | 220 / 238 | 706 |
| `riscv-psabi` | 53 / 56 | 114 |
| `winans-rvalp` | 91 / 91 | 413 |
| `riscv-asm-manual` | 50 / 50 | 75 |
| `heinold-python` | 247 / 263 | 612 |
| `python-crash-course` | 543 / 562 | 1,728 |
| `dagstuhl-19281` | 24 / 24 | 94 |
| `guo-2013` | 6 / 6 | 44 |
| **Total** | **1,958 / 2,017** | **5,666** |

Extraction, cleaning and chunking of all nine documents take about 2 minutes on the orchestrator's Mac.
