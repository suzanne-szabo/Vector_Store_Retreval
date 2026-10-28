# Claim-by-Claim Evaluation of Required Run 2

Run: `runs/2026-10-27_2054/` (final; configuration in `specs/architecture.md` §11).

Status: **draft verdicts by Claude (2026-10-27), to be verified by the author.** Each verdict was made by
reading the claim in the run's final answer next to the full text of the passage it cites, taken from the
same run's trace. Tick each query's box once you have checked its rows.

## Method

- A **claim** is one factual statement in the final answer. Long sentences with several facts are split.
- **Provenance** was already enforced by the controller: every cited chunk was returned by
  `search_documents` in the same run (P5). This evaluation checks **support**: does the cited passage
  say what the claim says?
- Verdicts:
  - **Supported:** the cited passage states it, word for word or as a faithful paraphrase.
  - **Partly supported:** the passage supports the core, but the claim adds an inference or detail the
    passage does not state.
  - **Not supported:** the cited passage does not say this.
  - **Uncited:** the claim has no `[n]` marker; the note says where it came from.

## Summary

| Query | Outcome | Claims | Supported | Partly | Not supported | Uncited |
|---|---|---:|---:|---:|---:|---:|
| Q1 | ANSWER | 9 | 9 | 0 | 0 | 0 |
| Q2 | ANSWER | 15 | 11 | 3 | 0 | 1 |
| Q3 | ANSWER | 14 | 12 | 1 | 1 | 0 |
| Q4 | ANSWER | 16 | 15 | 0 | 1 | 0 |
| Q5 | ANSWER | 9 | 7 | 2 | 0 | 0 |
| Q6 | STEP_LIMIT | — | — | — | — | — |
| Q7 | ANSWER | 14 | 11 | 1 | 0 | 2 |
| Q8 | INSUFFICIENT_EVIDENCE | 10 | 10 | 0 | 0 | 0 |
| Q9 | INSUFFICIENT_EVIDENCE | 4 | 3 | 0 | 0 | 1 |
| **Total** | | **91** | **78 (86%)** | **7 (8%)** | **2 (2%)** | **4 (4%)** |

No claim contradicts its passage. The weaknesses follow one pattern: the model adds a small inference or
a detail from its own knowledge next to a well-supported quotation. The citation check cannot catch this,
because the cited chunk did come from retrieval; only reading the passage does.

## Q1: ISA Instruction Lookup (ANSWER, 2 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | JALR writes "the address of the instruction following the jump (pc+4)" into rd | [1] 2025 p. 51 | Supported | Quoted exactly |
| 2 | x0 can be the destination when the link result is not needed | [1] | Supported | |
| 3 | JR is a JALR with rd=x0; RET is a JALR with rd=x0, rs1=x1, imm=0 | [1] | Supported | |
| 4 | I-type encoding; target = sign-extended 12-bit I-immediate + rs1 | [1] | Supported | Quoted exactly |
| 5 | Unlike conditional branches, the immediate is not treated as multiples of 2 bytes | [2] 2025 p. 51 | Supported | |
| 6 | LUI or AUIPC followed by JALR reaches any address in a 32-bit absolute or pc-relative range | [3] 2025 p. 51 | Supported | |
| 7 | The least-significant bit of the result is set to zero | [1] | Supported | |
| 8 | Clearing it "simplifies the hardware slightly" and frees the low bit of function pointers; slight loss of error checking | [2] | Supported | Quoted exactly |
| 9 | The definition is in Section 2.5, Control Transfer Instructions | [1] | Supported | The section heading is in the passage |

## Q2: Python Indentation Errors (ANSWER, 4 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | Python uses indentation to decide whether a line is connected to the line above | [8] PCC p. 90 | Supported | |
| 2 | Every indented line after `for magician in magicians:` is inside the loop and runs once per value | [9] PCC p. 88 | Supported | |
| 3 | People "indent blocks of code that don't need to be indented or forget to indent blocks that need to be indented" | [8] | Supported | Quoted exactly |
| 4 | "Always indent the line after the for statement in a loop." | [1] PCC p. 90 | Supported | Quoted exactly |
| 5 | Forgetting gives `IndentationError: expected an indented block`, naming the line | [2] PCC p. 91 | Supported | Error text matches |
| 6 | Fix: indent the line or lines immediately after the `for` statement | [2] | Supported | |
| 7 | The colon at the end of `for` tells Python the next line starts the loop | [7] PCC p. 93 | Supported | |
| 8 | Indenting a line that doesn't need it gives `IndentationError: unexpected indent` | [3] PCC p. 92 | Supported | |
| 9 | Fix: unindent that line; "the only lines you should indent are the actions you want to repeat…" | [3] | Partly supported | The quotation is in the passage; "unindent that line" is inferred (the passage says to indent only for a reason) |
| 10 | "Forgetting to Indent Additional Lines": the loop runs without errors but gives the wrong result (example code) | [4] PCC p. 91 | Supported | |
| 11 | The unindented line is "printed only once", after the loop, using only the final value (Carolina) | [4][5] PCC p. 90 | Partly supported | "Printed only once" is quoted; "after the loop, only Carolina" is not stated in the cited text, and the extracted output in [5] is jumbled by the page layout |
| 12 | Fix: indent that line to the same level as the other loop lines | [4][9] | Partly supported | A correct inference, but neither passage states the fix |
| 13 | Indenting code that belongs after the loop repeats it per item; often only a logical error | [6] PCC p. 92 | Supported | |
| 14 | "If an action is repeated many times when it should be executed only once, determine whether you just need to unindent the code" | [7] | Supported | Quoted exactly |
| 15 | Heinold covers the same mistake (an over-indented `print` running thousands of times) | none | Uncited | A Heinold passage (p. 50) was retrieved but not cited; the claim has no marker |

## Q3: Python Class Initialization (ANSWER, 3 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | A function in a class is a method; `__init__()` is a special method Python runs automatically when a new instance is created; double underscores avoid name conflicts | [1] PCC p. 196 | Supported | Quoted exactly |
| 2 | Calling the class calls `__init__()` with the arguments, which creates the instance and sets its attributes; no explicit `return`, Python returns the instance | [5] PCC p. 197 | Supported | Stated for `Dog`; applies the same way to `Car` |
| 3 | `self` is required, "must come first before the other parameters", and is passed automatically | [1] | Supported | Quoted exactly |
| 4 | `self` "is a reference to the instance itself; it gives the individual instance access to the attributes and methods" | [2] PCC p. 196 | Supported | Quoted exactly |
| 5 | You pass only name and age when making a `Dog`; `self` is passed for you | [2] | Supported | |
| 6 | "The double-underscore naming convention is the same reason given above" | [1] | **Not supported** | Garbled: [1] explains the underscores as avoiding name conflicts, which has nothing to do with why `self` comes first |
| 7 | `Car.__init__()` takes `self` first, then `make`, `model`, `year`, and "stores them in the attributes…" | [3] PCC p. 200 | Supported | Quoted exactly |
| 8 | The `Car` class code (`self.make = make`, etc.) | [4] PCC p. 200 | Supported | Code matches |
| 9 | Variables prefixed with `self` are available to every method; `self.name = name` attaches the value to the instance | [2] | Supported | Quoted exactly |
| 10 | Methods use the values as `self.make`, `self.model`, `self.year` | [3] | Supported | |
| 11 | A default attribute such as `self.odometer_reading = 0` is set each time an instance is made | [7] PCC p. 201 | Supported | |
| 12 | Create an instance by calling the class like a function, passing every `__init__()` parameter except `self`: `my_new_car = Car('audi', 'a4', 2016)` | [4] | Partly supported | [4] shows the call; "except `self`" is supported by [2], not [4]; "like a function" is not stated |
| 13 | Python calls `__init__()` and returns the instance; a capitalized name is a class, a lowercase name an instance | [5] | Supported | |
| 14 | Dot notation reaches an instance's attributes and methods | [6][3] PCC pp. 198, 200 | Supported | |

## Q4: Python Operator Table (ANSWER, 5 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | The table lists `/` division, `**` exponentiation, `//` integer division, `%` modulo (remainder) | [1] Heinold p. 30 | Supported | Table matches |
| 2 | The table is in Section 3.2, "Math Operators" | [1] | **Not supported** | The passage shows only "CHAPTER 3. NUMBERS"; the section number and title are not in it |
| 3 | The table has no numeric examples; the examples are in the paragraphs after it | [1] | Supported | |
| 4 | `/` is ordinary division; 8/5 is 1.6 | [1] | Supported | |
| 5 | In other languages and older Python, `/` did integer division on integers | [1] | Supported | |
| 6 | `//` drops the decimal part for positive numbers; 8//5 is 1 | [1] | Supported | |
| 7 | Exercise hint: `//` for minutes and `%` for seconds; 200 seconds is 3 minutes 20 seconds | [5] Heinold p. 34 | Supported | |
| 8 | `%` returns the remainder; 18%7 is 4 | [2] Heinold p. 30 | Supported | |
| 9 | Divisibility tests: `n%2` for even, `n%3` for divisible by 3 | [2] | Supported | |
| 10 | Clock wrap-around: (8+6)%12 is 2 | [2][3] Heinold p. 30 | Supported | |
| 11 | `player = player%5+1` sets player back to 1 after 5 | [3] | Supported | |
| 12 | A number ends in 4 when it leaves remainder 4 mod 10 | [6] Heinold p. 44 | Supported | |
| 13 | `**` is exponentiation; the caret `^` "is used for something else" | [1] | Supported | Quoted exactly |
| 14 | No worked `**` example near the table; the shell example `23**2` gives 529 | [4] Heinold p. 33 | Supported | |
| 15 | The series loop uses `1/n**2` | [4] | Supported | |
| 16 | Summary: `/` 1.6, `//` 1, `%` 4, `**` 529 | [1][2][4] | Supported | |

## Q5: ISA Design Decision and Its Cost (ANSWER, 3 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | No special overflow-check support in the base instruction set | [1] 2025 p. 48 | Supported | Quoted exactly |
| 2 | Rationale: many checks can be "cheaply implemented using RISC-V branches" | [1] | Supported | Quoted exactly |
| 3 | Unsigned addition: one branch after the add, `add t0, t1, t2; bltu t0, t1, overflow` | [1] | Supported | Code matches |
| 4 | "If the sum is less than an operand, the addition wrapped" | [1] | Partly supported | Correct reasoning, but the passage gives only the code, not this explanation |
| 5 | Extra instructions for unsigned: 1 | [1] | Supported | |
| 6 | Signed addition with one operand's sign known: one branch, `addi t0, t1, +imm; blt t0, t1, overflow`; the common immediate case | [1] | Supported | |
| 7 | General signed addition: three additional instructions, using "the sum should be less than one of the operands if and only if the other operand is negative" | [1] | Supported | Quoted exactly |
| 8 | "Integer operations in general simply wrap around on overflow", for example vector instructions | [2] 2025 p. 356 | Partly supported | The passage states this for vector integer arithmetic only; "in general" overgeneralizes |
| 9 | Vector fixed-point instructions saturate and set `vxsat` | [3] 2025 p. 369 | Supported | |

## Q6: Cross-Document Function Call (STEP_LIMIT, 8 steps)

- [ ] Verified by the author

No answer was accepted in run 2, so it has no claims to evaluate. Retrieval found the evidence the
question needs (psABI p. 9, Winans pp. 43-66, *Python Crash Course* pp. 168-175). Both `finish` attempts
were rejected under P3: an extra `source_id` field in each citation, then 13 citations (maximum 12). See
`docs/process-log.md`, run 2. Run 1 also ended `STEP_LIMIT` (answers of 5,865 and 4,308 characters,
over the 4,000 limit then in force).

### Supplementary: Q6 in the uncounted check run (`runs/not-required/check-fixes/`)

- [ ] Verified by the author

This run is **not** part of the counted results. It ran only Q3 and Q6, after D23 and D24 but before
D25 (the prompt did not yet state the 12-citation limit). Q6 ended `ANSWER` on step 8 of 8: step 7's
5,596-character answer listed 19 citations and was rejected; step 8 cut to 12 citations and 4,622
characters and was accepted. It is evaluated here because it is the only accepted answer to Q6 and
shows what the system produces for the cross-document question.

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | In `def add(a, b):`, `a` and `b` are parameters; the values in `add(2, 3)` are arguments; the value goes in parentheses at the call | [1] PCC p. 168 | Supported | Definitions quoted exactly, applied to the example |
| 2 | Python matches arguments to parameters by order, so `2` binds to `a` and `3` to `b` | [2] PCC p. 169 | Supported | Quoted exactly |
| 3 | Nothing at the Python level names a register or an address | none | Uncited | A fair observation about the cited passages, but not stated by any |
| 4 | "The return statement takes a value from inside a function and sends it back to the line that called the function" | [3] PCC p. 174 | Supported | Quoted exactly |
| 5 | The caller must provide a variable where the return value can be stored (`musician = …`) | [4] PCC p. 175 | Supported | Quoted exactly |
| 6 | `add(2, 3)` yields `5`; if not assigned, the value is simply unused | none | Uncited | Correct, but from the model's own knowledge |
| 7 | Eight argument registers a0-a7, the first two also used to return values | [5] psABI p. 11 | Supported | Quoted exactly |
| 8 | A scalar at most XLEN bits wide is passed in one argument register, or on the stack | [6] psABI p. 11 | Supported | Quoted exactly |
| 9 | The first value goes in `a0` (x10) and the second in `a1` (x11) | [5][6][11] | Partly supported | The registers are supported; the passages do not state the order of assignment |
| 10 | Winans' table: x10-11 a0-1 arguments/return value, x12-17 a2-7 arguments, x1 ra return address | [11] Winans p. 66 | Supported | Table matches |
| 11 | The callee reads `a0`/`a1` and leaves the sum in `a0` | [5] (paragraph) | Partly supported | Follows from [5], but no passage describes the callee doing this |
| 12 | The result is left in `a0` (and `a1` for a second value) | [5] | Supported | |
| 13 | psABI: a0-a7 not preserved across calls; s0-s11 callee-saved | [7] psABI p. 9 | Supported | Table matches |
| 14 | Registers not specified as return or callee-saved are "unspecified upon exit" | [5] | Supported | Quoted exactly |
| 15 | "So the caller must move the value out of `a0` right away" | [4] (analogy) | **Not supported** | No passage says this; the rule only means `a0` is not preserved across a *later* call |
| 16 | No dedicated link register or stack pointer in the base ISA; x1 holds the return address, x5 is an alternate, x2 is the stack pointer | [8] 2025 p. 44 | Supported | Quoted exactly |
| 17 | `jal rd, offset` sets rd to the jal address + 4 and jumps to pc + imm | [9] Winans p. 57 | Supported | Quoted exactly |
| 18 | `jal label` → `jal x1, label`; `ret` → `jalr x0, 0(x1)` | [10] Winans p. 46 | Supported | Table matches |
| 19 | Because `ra` is not preserved, a function that calls another must save it in its prologue and restore it in its epilogue | [7][12] | Supported | [7] gives "not preserved"; [12] shows a non-leaf prologue and epilogue saving x1 |
| 20 | "The 2025 manual shows exactly that shape": `addi sp,sp,-8`, `sd x1,(sp)` … `ld x1,(sp)`, `addi sp,sp,8`, `ret` | [12] 2025 p. 614 | Partly supported | The instructions are in the passage, but it is a **shadow-stack** (Zicfiss) example; the answer drops the `sspush` and `sspopchk` lines without saying so |
| 21 | A leaf routine like `add` needs no save: it reads `a0`/`a1`, writes `a0` and executes `ret` | [10] | Partly supported | [10] gives only the `ret` expansion; "a leaf needs no save" is inferred from [12]'s "non-leaf" |
| 22 | Python integers are objects, so the correspondence is an analogy only | none | Uncited | Correct, but from the model's own knowledge |
| 23 | The 2019 manual lists the same register roles with a Saver column (`x1 ra` → Caller) | none | Uncited | A 2019 passage (p. 32) was retrieved in step 6, but its citation was dropped when the model cut 19 citations to 12 |

Supplementary totals: 23 claims; 14 supported, 4 partly supported, 1 not supported, 4 uncited. The answer
uses all three required sources and keeps the two models separate, as the question asks.

## Q7: Source Authority and Version (ANSWER, 4 steps)

- [ ] Verified by the author

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | The 2025 manual controls the current definition; the 2019 edition is for history or comparison | none | Uncited | Comes from the source-role statement in the system prompt, not from a passage; a correct use of the roles |
| 2 | The 2025 manual includes "all ratified extensions through February 2025" | [2] 2025 p. 25 | Supported | Quoted exactly |
| 3 | It removed the draft Zam extension (replaced by a misaligned atomicity granule PMA) and the RV128I sketch | [2] | Supported | |
| 4 | The 2025 JALR definition (quoted in full) | [1] 2025 p. 51 | Supported | Quoted exactly |
| 5 | JR and RET encodings | [1] | Supported | |
| 6 | The 2019 manual gives the same JALR definition word for word | [3] 2019 p. 39 | Supported | Text is identical |
| 7 | 2019 rationale: LUI/AUIPC + JALR for a 32-bit range; immediate not multiples of 2 bytes | [4] 2019 p. 39 | Supported | |
| 8 | Clearing the low bit simplifies hardware and frees the low bit of function pointers | [5] 2019 p. 39 | Supported | |
| 9 | JAL and JALR raise instruction-address-misaligned if the target is not four-byte aligned | [5] | Supported | Stated in the 2019 passage |
| 10 | The base integer ISA moved to version 2.1; the document was renamed "unprivileged" | [6] 2025 p. 27 | Supported | Quoted exactly |
| 11 | Comparing module status tables across editions is a legitimate historical use | none | Uncited | General advice |
| 12 | 2019 wording: C.JALR "does not expand exactly to a base RVI instruction…" | [7] 2019 p. 123 | Supported | Quoted exactly |
| 13 | 2025 call / return / indirect-jump definitions used by Zicfiss and Zicfilp | [8] 2025 p. 605 | Supported | |
| 14 | The 2019 edition "cannot answer" this Zicfiss/Zicfilp material | [8] | Partly supported | Plausible (the material is from 2025), but no cited passage shows the 2019 edition lacks it |

## Q8: Insufficient Evidence, Plausible Question (INSUFFICIENT_EVIDENCE, 7 steps)

- [ ] Verified by the author

Outcome correct. Six searches covered CPython internals, bytecode, generated code and cycle measurement.

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | The corpus has no CPython internals, bytecode or compiler output | search record | Supported | Consistent with all six searches; a statement about what retrieval found |
| 2 | `range(10)` yields 0 to 9 | [1] Heinold p. 24 | Supported | |
| 3 | Python retrieves one value per pass of a `for` loop | [2] PCC p. 87 | Supported | |
| 4 | No passage gives a cycle count or names a specific board | search record | Supported | Consistent with the searches |
| 5 | A counted loop in RISC-V: `addi x6, x6, 1` then `blt x6, x5, -4` | [3] Winans p. 68 | Supported | Trace matches |
| 6 | Branches use the B-type format, 12-bit offsets in multiples of 2 bytes, ±4 KiB | [4] 2019 p. 40 | Supported | |
| 7 | `RDCYCLE` reads the cycle CSR; its rate "will depend on the implementation and operating environment" | [5] 2019 p. 77 | Supported | Quoted exactly |
| 8 | "Clock cycle" is hard to define; cycles divided by instructions retired gives CPI | [6] 2025 p. 74 | Supported | |
| 9 | `RDTIME` with a known counter period | [7] 2019 p. 78 | Supported | |
| 10 | Some environments prohibit counter access | [5] | Supported | |

## Q9: Off-Topic Evidence Boundary (INSUFFICIENT_EVIDENCE, 4 steps)

- [ ] Verified by the author

Outcome correct.

| # | Claim | Cited | Verdict | Note |
|---|---|---|---|---|
| 1 | No retrieved passage concerns the NFL, quarterbacks or football statistics | search record | Supported | Consistent with both searches |
| 2 | The corpus holds Python textbooks, RISC-V documents and two pedagogy reports | none | Uncited | From the source catalog in the system prompt |
| 3 | The only sports content is a baseball.txt and NCAA basketball programming exercise | [1][2] Heinold pp. 124-125 | Supported | |
| 4 | "Statistics" elsewhere means a Pygame `GameStats` class | [3] PCC p. 318 | Supported | |

## Findings for the report

1. **Support is high but not perfect:** 78 of 91 claims (86%) are fully supported, and none contradicts its
   passage. The 9 partly or not supported claims are additions, not errors of fact.
2. **The failure pattern is "quote plus embellishment".** The model quotes the passage accurately, then
   adds an explanation (Q5 #4), a generalization (Q5 #8), a detail such as a section title (Q4 #2), or a
   sentence that does not make sense (Q3 #6). Provenance checks cannot detect this; only reading the
   passage can.
3. **Uncited claims come from the prompt, not memory:** three of the four uncited claims restate the
   source roles or source catalog given in the system prompt. Only Q2 #15 cites nothing from a retrieved
   passage it could have cited.
4. **The refusals are well grounded:** Q8 and Q9 cite the closest passages, describe what was searched,
   and state exactly what is missing, which is the behavior the assignment asks for.
5. **Q6 can be answered, but not reliably within 8 steps.** Over three runs the same question produced
   three different `finish` failures (too long in run 1; 19 citations in the check run, fixed on the
   last step; an extra field, then 13 citations, in run 2). Retrieval succeeded every time. The one
   accepted answer (supplementary, uncounted) is the weakest of the evaluated answers (61% supported),
   and cutting citations to meet the limit left one claim (Q6s #23) without its source: enforcing a
   citation limit can trade away support.
