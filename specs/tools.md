# Tools and Retrieval Specification

Status: **v2 for orchestrator review (2026-10-26).** v2: retrieval is hybrid (architecture D17);
`retrieval` and `keyword` are built and tested (`tests/test_retrieval.py`, `tests/test_keyword.py`,
26 tests). `tools` is still to be implemented.
Modules: `rag_agent/keyword.py`, `rag_agent/retrieval.py`, `rag_agent/tools.py`
Tests: `tests/test_keyword.py`, `tests/test_retrieval.py`, `tests/test_tools.py`

The model may request exactly two tools. `tools` defines their argument schemas and executes requests
that the controller has already approved. `retrieval` is the read-only interface to the vector store.

## 1. Retrieval (`keyword`, `retrieval`)

Retrieval combines two rankings of the same chunks:

- **Vector search** (Chroma, cosine distance on Qwen3-Embedding-8B vectors) matches meaning.
- **Keyword search** (BM25 over the chunk texts) rewards the query's exact words, especially rare
  ones such as `JALR`, `callee-saved` or a section heading.

The first retrieval probe (2026-10-26, `docs/process-log.md`) showed vector search alone ranking a
paper's title block and figure captions above its "Limitations and ongoing work" section; keyword
matching is the standard remedy.

### 1.1 Keyword index (`keyword`)

- `tokenize(text) -> list[str]`: lowercase tokens matching `[a-z0-9_]+(?:[-.][a-z0-9_]+)*`, so
  `callee-saved`, `x10-x17`, `__init__`, `2.5` and `rv32i` stay whole. A token containing `-` or `.`
  also contributes its parts (`callee-saved` → `callee`, `saved`). A short list of English stopwords
  (`STOPWORDS`, such as "the", "of", "what", "does") is dropped; programming words such as "if",
  "for", "in", "is", "not", "and" and "return" are kept.
- `BM25Index(records, k1=1.5, b=0.75)` over `(chunk_id, text, metadata)` records. Each unique query
  term `t` adds `idf(t) · f · (k1 + 1) / (f + k1 · (1 − b + b · len / avg_len))` to a chunk's score,
  where `f` is the term's count in the chunk, `len` the chunk's token count, and
  `idf(t) = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))` with `N` chunks of which `n_t` contain `t`.
- `search(query, k, sources=None, subject=None) -> list[KeywordHit(chunk_id, score)]`: the `k`
  highest-scoring chunks that contain at least one query term and pass the filters, best first; ties
  keep index order.

### 1.2 `Passage`

A frozen dataclass:

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | `str` | chunk metadata |
| `source_id` | `str` | chunk metadata |
| `title` | `str` | chunk metadata |
| `page` | `int` | one-based physical page |
| `text` | `str` | the chunk text |
| `distance` | `float \| None` | cosine distance to the query, rounded to 4 decimals; `None` in keyword mode |
| `vector_rank` | `int \| None` | 1-based rank in the vector list; `None` if not in it |
| `keyword_rank` | `int \| None` | 1-based rank in the keyword list; `None` if not in it |

### 1.3 `Retriever` protocol

```python
class Retriever(Protocol):
    def search(self, query: str, k: int, sources: list[str] | None = None,
               subject: str | None = None) -> list[Passage]: ...
```

- Returns at most `k` passages, best first.
- `sources` keeps only chunks whose `source_id` is in the list; `subject` keeps only chunks with that
  subject; both together keep chunks matching both; neither means the whole collection.
- Returns `[]` when nothing matches. Raises an exception when the store or embedding service fails;
  it never returns partial results.

### 1.4 Helpers

- `make_where(sources, subject) -> dict | None`, the Chroma filter:

| Arguments | Result |
|---|---|
| neither | `None` |
| `sources` only | `{"source_id": {"$in": sources}}` |
| `subject` only | `{"subject": subject}` |
| both | `{"$and": [{"source_id": {"$in": sources}}, {"subject": subject}]}` |

- `reciprocal_rank_fusion(rankings, k=RRF_K) -> list[(id, score)]`: each id scores the sum of
  `1 / (RRF_K + rank)` over the lists it appears in (`RRF_K = 60`); best first; ties go to the better
  best rank, then to the earlier list.
- `with_instruction(query, instruction)`: `f"Instruct: {instruction}\nQuery: {query}"` when an
  instruction is set (the Qwen3-Embedding query format), else the query unchanged.

### 1.5 `HybridRetriever(db_path, embeddings=None, mode="hybrid", query_instruction=None, pool=30)`

- `mode` is `hybrid`, `vector` or `keyword` (`MODES`); anything else raises `ValueError`. A missing
  `db_path` raises `FileNotFoundError` naming the rebuild command.
- On opening, reads every stored chunk (text and metadata) once and builds the BM25 index in memory.
  Opening and searching never embed or change documents.
- `search` in each mode:
  - `vector`: embeds `with_instruction(query, query_instruction)` and returns Chroma's `k` nearest
    chunks under `make_where(...)`.
  - `keyword`: returns the BM25 index's `k` best chunks under the same filters.
  - `hybrid`: takes `pool` (30) candidates from each list, fuses them with
    `reciprocal_rank_fusion([vector, keyword])`, and returns the first `k`. A chunk found only by
    keyword search gets its distance computed from its stored vector, so every hybrid passage has a
    distance.
- Every passage records its `vector_rank` and `keyword_rank` within the candidate lists. Documents are
  never prefixed with the instruction.

## 2. Observation

Every tool call, accepted or rejected, produces one frozen `Observation` (defined in `controller`,
reused here):

| Field | Type | Meaning |
|---|---|---|
| `tool` | `str` | tool name as requested |
| `accepted` | `bool` | whether the call was carried out |
| `policy` | `str \| None` | rejecting policy (`P1`–`P7`), else `None` |
| `reason` | `str \| None` | machine-readable code, such as `BAD_ARGS` or `RETRIEVAL_ERROR` |
| `message` | `str` | one or two plain sentences for the model |
| `data` | `dict` | tool-specific payload (below) |
| `steps_remaining` | `int` | `max_steps - steps_used` at the time of the call |

`Observation.to_json()` returns compact JSON of all seven fields; this string is the `ToolMessage`
content the model sees.

## 3. Argument schemas

All schemas are Pydantic models with `extra="forbid"` and strict types (no string-to-number
coercion). Strings are stripped of surrounding whitespace before length checks.

### 3.1 `SearchArgs` (`search_documents`)

| Field | Type and limits | Default |
|---|---|---|
| `query` | `str`, 3–300 characters after stripping | required |
| `sources` | `list[str]` of 1–9 **distinct** strings, or `null` | `null` |
| `subject` | `"python"`, `"riscv"` or `"pedagogy"`, or `null` | `null` |
| `k` | `int`, 1–8 (`MAX_K = 8`) | `null`, meaning `config.default_k` |

Whether each `sources` entry is a registered source id is checked by the controller (P3,
`UNKNOWN_SOURCE`), because the registry is run data, not part of the schema.

### 3.2 `FinishArgs` (`finish`)

| Field | Type and limits | Default |
|---|---|---|
| `outcome` | `"answer"` or `"insufficient_evidence"` | required |
| `answer` | `str`, 1–8,000 characters after stripping (`MAX_ANSWER_CHARS`) | required |
| `citations` | `list[Citation]`, at most 12 | `[]` |

`Citation`: `chunk_id` (`str` matching `^[0-9a-f]{24}$`) and `supports` (`str`, 1–200 characters after
stripping: a short phrase naming the claim the chunk supports). Extra fields are forbidden.

v2.2 (2026-10-27, architecture D23): the answer limit rose from 4,000 to 8,000 characters. In required
run 1, Q6's cross-document answer (5,865, then 4,308 characters) was rejected twice and the run hit the
step limit.

### 3.3 Citation markers

`citation_markers(answer) -> list[int]` returns every number cited in the answer, in order of
appearance, with repeats. A marker is a pair of square brackets holding one or more integers separated
by commas: `[1]`, `[2, 3]` and `[1,2]` are markers; `[a]`, `[]`, `[1-3]` and `[1.5]` are not. The
controller applies the marker rules (P6) using this function.

## 4. Tool definitions for the model

`tool_definitions() -> list[dict]` returns the two tools in OpenAI function-tool format, generated from
the schemas, with these descriptions:

- `search_documents`: "Search the Python and RISC-V document corpus. Returns ranked passages, each with
  a chunk_id you can cite, its source_id, title, page and text. Use a focused query of a few key terms,
  not the whole question. Filter with `sources` (source ids from the catalog) or `subject` when you
  know where the answer is. You may search several times with different queries or filters."
- `finish`: "End the run. Use outcome `answer` when retrieved passages support an answer: write the
  answer with [n] markers and list the citations in marker order, each with a chunk_id returned by
  search_documents. Use outcome `insufficient_evidence` when the passages do not support an answer:
  say what you searched for and what is missing."

## 5. Executors

`execute(session, approval, steps_remaining) -> Observation` runs an approved request. It is the only
code that adds to `session.retrieved`. The controller passes `steps_remaining`, which is copied into
the observation unchanged. (v2.1: the parameter was implicit in v2; found while deriving tests.)

### 5.1 `search_documents`

1. Calls `session.retriever.search(query, k or session.config.default_k, sources, subject)` with
   `sources` and `subject` passed by keyword.
2. Adds each returned passage to `session.retrieved` under its `chunk_id`. A chunk already present
   keeps its original position (the mapping is insertion-ordered).
3. Returns an accepted observation:
   - `data = {"status": "ok" | "empty", "query": <query>, "filters": {"sources": <list or null>,
     "subject": <str or null>}, "results": [...]}` where each result is
     `{"rank", "chunk_id", "source_id", "title", "page", "distance", "vector_rank", "keyword_rank",
     "text"}` and `rank` starts at 1;
   - `status` is `"empty"` when no passages came back;
   - `message`: `"<n> passage(s) from <source ids, comma-separated, in rank order without repeats>."`
     or `"No passages matched. Try a different query or remove filters."`.

A retriever exception propagates to the controller, which handles the retry (controller §6).

### 5.2 `finish`

Returns an accepted observation with `data = {"outcome": <outcome>, "citations": [...]}` where each
citation is resolved from `session.retrieved` as
`{"n", "chunk_id", "source_id", "title", "page", "supports"}` (`n` starting at 1), and
`message = "Run finished: <outcome>."`. The controller records the result and ends the run.

## 6. Tests derived from this spec

- Keyword and retrieval (built, passing): tokenizer keeps technical tokens and their parts; BM25 returns
  only matching chunks, favors rare terms and shorter chunks, honors filters and `k`; `make_where` for
  all four combinations; RRF scores and tie-breaking; `HybridRetriever` on a store built with fake
  embeddings in all three modes: labels on every passage, ranks and distances per mode, filters and `k`,
  the instruction applied to queries only, missing store and bad mode rejected.
- Schemas: each invalid case is rejected (blank or 2-character query, 301-character query, `k` = 0, 9 or
  `"5"`, duplicate sources, empty `sources` list, unknown subject, extra field, bad `chunk_id` pattern,
  13 citations, blank answer); valid minimal and full requests are accepted.
- `citation_markers` on the examples in §3.3.
- `search_documents` executor: every result has rank, chunk id, source id, title, page, distance and
  text; the retrieved set grows in order without duplicates; `empty` status and message.
- `finish` executor: citations resolved with source id, title and page.
