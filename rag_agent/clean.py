"""Text cleaning for extracted PDF pages. Every function is pure and tested on its own.

The order matters (see `clean_document`):
1. normalize characters (NFKC, drop private-use glyphs);
2. remove running headers and footers found across the document's pages;
3. per page: tidy whitespace without disturbing indentation, then drop the shared left margin.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Bump whenever a cleaning rule changes: it is recorded in the manifest, and a store built
# with a different version must be rebuilt rather than mixed.
CLEANING_VERSION = "2026-10-26.2"

# Private-use glyphs (note-box icons, code callout markers) and the replacement character.
_PRIVATE_USE = re.compile("[-�]")
_INTERNAL_GAP = re.compile(r"(?<=\S) {3,}(?=\S)")
_DOT_LEADER = re.compile(r" *(?:\. *){4,}")   # table-of-contents leaders such as ".  .  .  ."
# A word broken across lines: a word of at least two letters, a hyphen at the end of the line,
# and a lowercase word starting the next line ("ben-\nefited", "callee-\nsaved").
_LINE_BREAK_HYPHEN = re.compile(r"\b([A-Za-z]{2,})-\n[ \t]*([a-z][A-Za-z]*)")
_HYPHENATED_WORD = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")
_WORD = re.compile(r"[A-Za-z]+")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")

EDGE_DEPTH = 2              # how many lines at the top and bottom of a page can be header/footer
EDGE_REPEAT_PAGES = 5       # a top/bottom line repeated on this many pages is a running header/footer
EDGE_MIN_PAGES = 4          # documents shorter than this are too small to detect repeats reliably
MIN_PAGE_CHARS = 40         # a cleaned page with less text than this is skipped (blank or divider pages)


def normalize(text: str) -> str:
    """NFKC-normalize (ligatures such as 'ﬁ' become 'fi') and drop private-use glyphs."""
    return _PRIVATE_USE.sub("", unicodedata.normalize("NFKC", text))


def tidy_line(line: str) -> str:
    """Strip trailing space and shrink wide gaps inside a line, keeping its leading indentation.

    Layout extraction pads justified text and table columns with long runs of spaces;
    shrinking a run of 3 or more to 2 keeps columns readable while saving space.
    """
    line = line.expandtabs(4).rstrip()
    stripped = line.lstrip(" ")
    indent = line[: len(line) - len(stripped)]
    stripped = _DOT_LEADER.sub(" ... ", _INTERNAL_GAP.sub("  ", stripped)).rstrip()
    return indent + stripped


def edge_signature(line: str) -> str:
    """A comparison key for header/footer detection: page numbers become '#', spacing and case ignored."""
    return _SPACES.sub(" ", _DIGITS.sub("#", line.strip())).lower()


def _edge_lines(lines: list[str]) -> list[str]:
    """The first and last EDGE_DEPTH non-blank lines of a page."""
    nonblank = [line for line in lines if line.strip()]
    return nonblank[:EDGE_DEPTH] + nonblank[-EDGE_DEPTH:]


def edge_threshold(page_count: int) -> int:
    """Pages a top/bottom line must repeat on: 5, or half the pages for a short document (at least 3)."""
    return max(3, min(EDGE_REPEAT_PAGES, page_count // 2))


def find_running_edges(pages: list[list[str]]) -> set[str]:
    """Signatures of lines that appear at the top or bottom of several pages of one document.

    Catches whole-book headers (the ISA manual's title line), chapter headers that repeat
    within a chapter, and bare page numbers (signature '#').
    """
    if len(pages) < EDGE_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({edge_signature(line) for line in _edge_lines(lines)})
    threshold = edge_threshold(len(pages))
    return {signature for signature, count in counts.items() if count >= threshold and signature}


def strip_edges(lines: list[str], running: set[str]) -> list[str]:
    """Remove running header/footer lines from the top and bottom of one page."""
    lines = list(lines)
    for direction in (0, -1):
        checked = 0
        index = 0 if direction == 0 else len(lines) - 1
        while 0 <= index < len(lines) and checked < EDGE_DEPTH:
            if not lines[index].strip():
                index += 1 if direction == 0 else -1
                continue
            checked += 1
            if edge_signature(lines[index]) in running:
                del lines[index]
                if direction == -1:
                    index -= 1
            else:
                index += 1 if direction == 0 else -1
    return lines


def dedent(lines: list[str]) -> list[str]:
    """Remove the margin shared by all non-blank lines; relative indentation is kept."""
    margins = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    if not margins:
        return []
    margin = min(margins)
    return [line[margin:] if line.strip() else "" for line in lines]


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """At most one blank line in a row, and none at the start or end."""
    result: list[str] = []
    for line in lines:
        if not line.strip():
            if result and result[-1] != "":
                result.append("")
        else:
            result.append(line)
    while result and result[-1] == "":
        result.pop()
    return result


@dataclass(frozen=True)
class Vocabulary:
    """Words seen in one document, used to decide how to rejoin a word broken across lines."""

    hyphenated: frozenset[str] = frozenset()   # lowercased compounds written within a line
    words: frozenset[str] = frozenset()        # lowercased plain words


def document_vocabulary(text: str) -> Vocabulary:
    """Collect the document's words, leaving out the fragments of words broken across lines.

    Without that exclusion, 'pop-\\nular' would count 'ular' as a word of the document and the
    split would look like a compound of two real words.
    """
    unbroken = _LINE_BREAK_HYPHEN.sub(" ", text)
    return Vocabulary(
        hyphenated=frozenset(word.lower() for word in _HYPHENATED_WORD.findall(unbroken)),
        words=frozenset(word.lower() for word in _WORD.findall(unbroken)),
    )


def join_broken_words(text: str, vocabulary: Vocabulary = Vocabulary()) -> str:
    """Rejoin words hyphenated across a line break, moving the rest of the next line up.

    For 'left-\\nright', the first rule that applies decides:
    1. 'left-right' is written within a line elsewhere in the document: keep the hyphen
       ('callee-\\nsaved' -> 'callee-saved').
    2. 'leftright' appears as a word in the document: drop the hyphen ('pro-\\ngram' -> 'program').
    3. 'left' and 'right' are both words in the document: keep the hyphen
       ('upper-\\nimmediate' -> 'upper-immediate').
    4. Otherwise it is a syllable split: drop the hyphen ('ben-\\nefited' -> 'benefited').
    """
    def rejoin(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        hyphenated, joined = f"{left}-{right}", left + right
        if hyphenated.lower() in vocabulary.hyphenated:
            return hyphenated
        if joined.lower() in vocabulary.words:
            return joined
        if left.lower() in vocabulary.words and right.lower() in vocabulary.words:
            return hyphenated
        return joined

    return _LINE_BREAK_HYPHEN.sub(rejoin, text)


def clean_document(raw_pages: list[str], removed: Counter | None = None) -> list[str]:
    """Clean every page of one document; returns one string per physical page, in order.

    A page whose cleaned text is shorter than MIN_PAGE_CHARS comes back as "" so the
    caller can skip it while page numbers stay physical. If `removed` is given, it counts
    each header/footer line removed (by its text), for the dry-run report.
    """
    pages = [[tidy_line(line) for line in normalize(text).splitlines()] for text in raw_pages]
    running = find_running_edges(pages)
    texts: list[str] = []
    for lines in pages:
        kept = strip_edges(lines, running)
        if removed is not None:
            dropped = Counter(lines) - Counter(kept)
            removed.update(edge_signature(line) for line in dropped.elements())
        texts.append("\n".join(collapse_blank_lines(dedent(kept))))
    vocabulary = document_vocabulary("\n".join(texts))
    cleaned: list[str] = []
    for text in texts:
        text = join_broken_words(text, vocabulary)
        cleaned.append(text if len(text.strip()) >= MIN_PAGE_CHARS else "")
    return cleaned
