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
| Q1 | ISA Instruction Lookup | ANSWER | ANSWER | yes | 2/8 | riscv-isa-2025 |
| Q2 | Python Indentation Errors | ANSWER | ANSWER | yes | 4/8 | python-crash-course |
| Q3 | Python Class Initialization | ANSWER | ANSWER | yes | 3/8 | python-crash-course |
| Q4 | Python Operator Table | ANSWER | ANSWER | yes | 5/8 | heinold-python |
| Q5 | ISA Design Decision and Its Cost | ANSWER | ANSWER | yes | 3/8 | riscv-isa-2025 |
| Q6 | Cross-Document Function Call | ANSWER | STEP_LIMIT | no | 8/8 | none |
| Q7 | Source Authority and Version | ANSWER | ANSWER | yes | 4/8 | riscv-isa-2025, riscv-isa-2019 |
| Q8 | Insufficient Evidence, Plausible Question | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | yes | 7/8 | heinold-python, python-crash-course, winans-rvalp, riscv-isa-2019, riscv-isa-2025 |
| Q9 | Off-Topic Evidence Boundary | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | yes | 4/8 | heinold-python, python-crash-course |
