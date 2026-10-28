# Required Queries

Status: **frozen 2026-10-27 by the orchestrator (milestone 4).** The query text is never edited.
Every query is run exactly as written with the one recorded configuration in
`specs/architecture.md` §11.

The nine queries mirror the supplied U0-HW-09 pattern: document-specific lookups, a table comparison,
a claim with its stated limits, a cross-document scenario, a source-authority question, and two
insufficient-evidence cases. They target the Python and RISC-V sources; the two pedagogy papers stay
in the corpus but no query depends on them.

## Q1: ISA Instruction Lookup
<!-- expected: ANSWER -->

According to the RISC-V Unprivileged ISA manual (version 20250508), what does the JALR instruction
write to its destination register, how is its target address computed, and what happens to the
least-significant bit of that address?

## Q2: Python Indentation Errors
<!-- expected: ANSWER -->

A student's for loop raises an IndentationError, and another student's loop runs but prints only the
last item. Using Python Crash Course, what indentation mistakes cause each problem, and how should
each be fixed?

## Q3: Python Class Initialization
<!-- expected: ANSWER -->

According to Python Crash Course, what does a class's __init__() method do, why is self its first
parameter, how are values such as make, model, and year stored on an instance, and how is an
instance of the Car class created?

## Q4: Python Operator Table
<!-- expected: ANSWER -->

Using the operator table in Heinold's A Practical Introduction to Python Programming, compare the /,
//, %, and ** operators: what does each compute, and what example results does the book give?

## Q5: ISA Design Decision and Its Cost
<!-- expected: ANSWER -->

According to the RISC-V Unprivileged ISA manual (version 20250508), does the base integer instruction
set include special support for detecting integer overflow? Give the stated rationale, how software
checks for overflow in unsigned addition and in signed addition, and how many extra instructions each
check needs.

## Q6: Cross-Document Function Call
<!-- expected: ANSWER -->

A student writes def add(a, b): return a + b and calls add(2, 3). Using Python Crash Course for what
happens at the Python level, and the RISC-V psABI and Winans's RISC-V Assembly Language Programming
for how a function call works in RISC-V assembly, explain how arguments, the return value, and the
return address are handled at each level. Treat these as two separate models; the scenario does not
claim that Python compiles to RISC-V.

## Q7: Source Authority and Version
<!-- expected: ANSWER -->

The corpus contains both the 2019 ratified RISC-V specification (version 20191213) and the 2025
Unprivileged ISA manual (version 20250508). Which should control an answer about the current
definition of a base integer instruction such as JALR, and how may the 2019 edition still be used?

## Q8: Insufficient Evidence, Plausible Question
<!-- expected: INSUFFICIENT_EVIDENCE -->

What RISC-V instructions does CPython 3.12 generate for a Python for i in range(10) loop, and how
many clock cycles does that loop take on a RISC-V board?

## Q9: Off-Topic Evidence Boundary
<!-- expected: INSUFFICIENT_EVIDENCE -->

Who is the greatest quarterback in NFL history, and what statistics establish that ranking?
