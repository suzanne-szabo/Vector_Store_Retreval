"""Page-by-page PDF text extraction that preserves layout (and so Python indentation).

Settings chosen by the extraction check on 2026-10-26:
- pdfplumber layout mode keeps code indentation (pypdf flattened it);
- x_tolerance=1 stops justified lines from gluing words together
  (the default of 3 glued words in Heinold, Winans, Guo, Dagstuhl and the 2019 spec);
- two-column sources are read one column at a time, so columns are not interleaved.
  A full-width block at the top of a page (a paper's title and authors) is read whole
  first, so it is not cut down the middle.
"""

from __future__ import annotations

import textwrap
from importlib.metadata import version
from pathlib import Path

import pdfplumber

X_TOLERANCE = 1
GUTTER_HALF_WIDTH = 8   # points either side of the page middle that body text never crosses
EXTRACTOR = f"pdfplumber {version('pdfplumber')} (layout=True, x_tolerance={X_TOLERANCE})"


def _region_text(page, box) -> str:
    """Layout text of one region, with that region's own left margin removed."""
    text = page.crop(box).extract_text(layout=True, x_tolerance=X_TOLERANCE) or ""
    return textwrap.dedent(text)


def full_width_header_bottom(words: list[dict], middle: float) -> float | None:
    """Bottom of the run of lines at the top of a page that cross the column gutter.

    A line crosses the gutter when one of its words overlaps the band around the page
    middle; two-column body text never does. Returns None when the first line of the
    page is ordinary column text.
    """
    rows: dict[int, list[dict]] = {}
    for word in words:
        rows.setdefault(round(word["top"]), []).append(word)
    bottom = None
    for top in sorted(rows):
        crosses = any(w["x0"] < middle + GUTTER_HALF_WIDTH and w["x1"] > middle - GUTTER_HALF_WIDTH
                      for w in rows[top])
        if not crosses:
            break
        bottom = max(w["bottom"] for w in rows[top])
    return bottom


def _page_text(page, columns: int) -> str:
    if columns == 1:
        return page.extract_text(layout=True, x_tolerance=X_TOLERANCE) or ""
    middle, width, height = page.width / 2, page.width, page.height
    header = full_width_header_bottom(page.extract_words(x_tolerance=X_TOLERANCE), middle)
    parts = []
    body_top = 0
    if header is not None:
        body_top = min(header + 1, height)
        parts.append(_region_text(page, (0, 0, width, body_top)))
    parts.append(_region_text(page, (0, body_top, middle, height)))
    parts.append(_region_text(page, (middle, body_top, width, height)))
    return "\n\n".join(parts)


def extract_pages(path: Path, columns: int = 1) -> list[str]:
    """Return the raw text of every page; index 0 is physical page 1."""
    with pdfplumber.open(path) as pdf:
        return [_page_text(page, columns) for page in pdf.pages]
