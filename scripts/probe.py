"""Retrieval probe (milestone 4): where does the passage that answers each draft query rank?

For each case, the "gold" passage is identified by its source, page and a phrase it contains.
The probe searches with every configuration and reports the gold passage's rank (top 30), so
retrieval choices are made from numbers rather than from single examples.

    uv run python scripts/probe.py                     # all configurations (needs the embedding service)
    uv run python scripts/probe.py --keyword-only      # offline: keyword mode only

Writes a Markdown report to runs/probe-<date>.md (or --out).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.config import DEFAULT_DB_PATH, PROJECT_DIR  # noqa: E402
from rag_agent.retrieval import HybridRetriever  # noqa: E402

DEPTH = 30
DEFAULT_INSTRUCTION = "Given a question, retrieve passages that answer it"


@dataclass(frozen=True)
class Gold:
    source_id: str
    contains: tuple[str, ...]          # every phrase must appear in the chunk text
    pages: tuple[int, ...] | None = None

    def matches(self, passage) -> bool:
        return (passage.source_id == self.source_id
                and (self.pages is None or passage.page in self.pages)
                and all(phrase in passage.text for phrase in self.contains))


@dataclass(frozen=True)
class Case:
    label: str
    query: str
    source: str
    gold: Gold


JALR = Gold("riscv-isa-2025", ("indirect jump instruction JALR",))
INDENT = Gold("python-crash-course", ("Forgetting to Indent",), pages=tuple(range(80, 100)))
INIT = Gold("python-crash-course", ("__init__()", "make, model, and year"))
OPERATORS = Gold("heinold-python", ("Operator", "integer division", "modulo"))
OVERFLOW = Gold("riscv-isa-2025", ("overflow checks", "bltu"))
RA = Gold("riscv-psabi", ("Return address",))
VERSION = Gold("riscv-isa-2025", ("ratified",), pages=(1,))

# Draft queries Q1-Q7 (Q8 and Q9 should find nothing and have no gold passage).
Q = {
    1: "According to the RISC-V Unprivileged ISA manual (version 20250508), what does the JALR instruction "
       "write to its destination register, how is its target address computed, and what happens to the "
       "least-significant bit of that address?",
    2: "A student's for loop raises an IndentationError, and another student's loop runs but prints only "
       "the last item. Using Python Crash Course, what indentation mistakes cause each problem, and how "
       "should each be fixed?",
    3: "According to Python Crash Course, what does a class's __init__() method do, why is self its first "
       "parameter, how are values such as make, model, and year stored on an instance, and how is an "
       "instance of the Car class created?",
    4: "Using the operator table in Heinold's A Practical Introduction to Python Programming, compare the "
       "/, //, %, and ** operators: what does each compute, and what example results does the book give?",
    5: "According to the RISC-V Unprivileged ISA manual (version 20250508), does the base integer "
       "instruction set include special support for detecting integer overflow? Give the stated rationale, "
       "how software checks for overflow in unsigned addition and in signed addition, and how many extra "
       "instructions each check needs.",
}

CASES = [
    # focused queries, the kind the agent is told to write
    Case("Q1 focused", "JALR target address least-significant bit", "riscv-isa-2025", JALR),
    Case("Q2 focused", "forgetting to indent for loop IndentationError", "python-crash-course", INDENT),
    Case("Q3 focused", "__init__ method self attributes make model year", "python-crash-course", INIT),
    Case("Q4 focused", "operators integer division modulo exponentiation", "heinold-python", OPERATORS),
    Case("Q5 focused", "overflow checks integer arithmetic branches", "riscv-isa-2025", OVERFLOW),
    Case("Q6 focused", "return address register ra function call", "riscv-psabi", RA),
    Case("Q7 focused", "document version and ratified state", "riscv-isa-2025", VERSION),
    # the draft questions as written, as a baseline
    Case("Q1 as asked", Q[1], "riscv-isa-2025", JALR),
    Case("Q2 as asked", Q[2], "python-crash-course", INDENT),
    Case("Q3 as asked", Q[3], "python-crash-course", INIT),
    Case("Q4 as asked", Q[4], "heinold-python", OPERATORS),
    Case("Q5 as asked", Q[5], "riscv-isa-2025", OVERFLOW),
]


def gold_rank(retriever: HybridRetriever, case: Case, filtered: bool) -> int | None:
    passages = retriever.search(case.query, k=DEPTH, sources=[case.source] if filtered else None)
    return next((rank for rank, p in enumerate(passages, start=1) if case.gold.matches(p)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyword-only", action="store_true", help="offline: keyword mode only")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="query instruction to test")
    parser.add_argument("--out", type=Path, help="report path (default runs/probe-<date>.md)")
    args = parser.parse_args()

    configs = [("keyword", "keyword", None)]
    if not args.keyword_only:
        configs = [("vector", "vector", None), ("vector+instr", "vector", args.instruction),
                   ("keyword", "keyword", None),
                   ("hybrid", "hybrid", None), ("hybrid+instr", "hybrid", args.instruction)]

    names = [name for name, _, _ in configs]
    rows: list[tuple[str, str, list[int | None]]] = []
    started = time.monotonic()
    retrievers = [(name, HybridRetriever(DEFAULT_DB_PATH, mode=mode, query_instruction=instr))
                  for name, mode, instr in configs]
    for case in CASES:
        for filtered in (True, False):
            ranks = [gold_rank(retriever, case, filtered) for _, retriever in retrievers]
            rows.append((case.label, "filtered" if filtered else "all docs", ranks))
            shown = "  ".join(f"{name}={rank or '>30'}" for name, rank in zip(names, ranks))
            print(f"{case.label:12s} {rows[-1][1]:9s} {shown}", flush=True)

    def cell(rank):
        return "**1**" if rank == 1 else (str(rank) if rank else ">30")

    lines = [
        f"# Retrieval probe, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Rank of the gold passage among the top {DEPTH} results (lower is better; >30 = not found).",
        f"Query instruction tested: `{args.instruction}`." if not args.keyword_only else "Keyword mode only.",
        "",
        "| Case | Filter | " + " | ".join(names) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    lines += [f"| {label} | {flt} | " + " | ".join(cell(r) for r in ranks) + " |" for label, flt, ranks in rows]
    lines += ["", "| Summary | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
    for title, test in (("found at rank 1", lambda r: r == 1), ("found in top 5 (k = 5)", lambda r: r and r <= 5),
                        ("found in top 30", lambda r: r is not None)):
        counts = [sum(1 for _, _, ranks in rows if test(ranks[i])) for i in range(len(names))]
        lines.append(f"| {title} (of {len(rows)}) | " + " | ".join(str(c) for c in counts) + " |")
    lines += ["", f"Probe time: {time.monotonic() - started:.0f} s."]

    out = args.out or PROJECT_DIR / "runs" / f"probe-{datetime.now():%Y-%m-%d_%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
