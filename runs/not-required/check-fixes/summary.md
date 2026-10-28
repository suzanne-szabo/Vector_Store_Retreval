# Required runs

## Recorded configuration

- Embedding model: Qwen3-Embedding-8B
- Chunk size and overlap: 1000 / 150
- Retrieval mode: hybrid
- Results per search (k): 5
- Step limit: 8
- Chat model: Qwen3.8-Flash-Next
- Query instruction: Given a question, retrieve passages that answer it
- index_version: 48c92aea33b03628

## Results

| Query | Title | Expected | Actual | Match | Steps | Sources cited |
|---|---|---|---|---|---|---|
| Q3 | Python Class Initialization | ANSWER | ANSWER | yes | 3/8 | python-crash-course |
| Q6 | Cross-Document Function Call | ANSWER | ANSWER | yes | 8/8 | python-crash-course, riscv-psabi, riscv-isa-2025, winans-rvalp |
