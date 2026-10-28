# Report Notes

Material collected during the project for the final report (`szabo-suzanne-vector-retrieval.pdf`).
Each section names where it belongs in the report.

## Why this subject and corpus (report: task description, near the start)

Draft paragraph, in the author's voice:

> I chose Python and RISC-V assembly, instead of the supplied FDA food-safety corpus, because I plan
> to build a personalized programming tutor for my senior research project. Building this assignment
> on that subject let me test the whole retrieval process (ingestion, hybrid search, a bounded ReAct
> agent and citation checks) on the kind of material the tutor will need, and see whether such an
> agent responds well when a student asks about a Python or assembly assignment. The instructor
> approved a different topic provided the nine required queries mirror the supplied pattern, and they
> do: each query keeps the same question type as its counterpart in the supplied set.

Points to keep with it:

- The corpus pairs introductory Python textbooks (Heinold; *Python Crash Course*) with RISC-V sources
  at several levels: the authoritative ISA specifications (2019 and 2025 editions), the calling
  convention (psABI), a beginner tutorial (Winans), and an assembler reference.
- Two short papers on teaching with notional machines (the Dagstuhl 19281 report and Guo's Online
  Python Tutor) were added to test whether pedagogy material could be included. They stay in the
  corpus, but the required queries focus on Python and assembly, since those are what a student would
  ask the tutor about.
- What this project showed for the tutor (fill in after the required runs): which kinds of student
  question worked, which failed, and what the tutor would need beyond this corpus. Candidates already
  observed: the specifications answer "what is the rule" well but are not tutorials; vector search
  alone missed passages that keyword search found.

## Other material for the report (collected so far)

- Retrieval investigation (vector-only misses, three probe runs, broken-word rejoining, hybrid search):
  `docs/process-log.md`, 2026-10-26 entries, with the results table.
- Probe tables: `runs/probe-*.md`.
- Limitations already known: the line-break rejoining misses (`opensource`, `forgot-ten`); *Python
  Crash Course*'s "Try It Yourself" heading removed on 19 pages; the Winans draft's margin line numbers
  remain in its chunks; the specifications are reference texts, not tutorials.
