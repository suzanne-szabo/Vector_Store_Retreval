# Required runs

## Recorded configuration

- Embedding model: Qwen3-Embedding-8B
- Chunk size and overlap: 1000 / 150
- Retrieval mode: hybrid
- Results per search (k): 5
- Step limit: 8
- Chat model: Qwen3.8-27B
- Query instruction: Given a question, retrieve passages that answer it
- index_version: 48c92aea33b03628

## Results

| Query | Title | Expected | Actual | Match | Steps | Sources cited |
|---|---|---|---|---|---|---|
| Q1 | ISA Instruction Lookup | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q2 | Python Indentation Errors | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q3 | Python Class Initialization | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q4 | Python Operator Table | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q5 | ISA Design Decision and Its Cost | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q6 | Cross-Document Function Call | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q7 | Source Authority and Version | ANSWER | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q8 | Insufficient Evidence, Plausible Question | INSUFFICIENT_EVIDENCE | ERROR/MODEL_ERROR | no | 1/8 | none |
| Q9 | Off-Topic Evidence Boundary | INSUFFICIENT_EVIDENCE | ERROR/MODEL_ERROR | no | 1/8 | none |
