"""A minimal PDF writer for test fixtures (no extra dependencies, no committed binaries).

Each page is a list of (x, y, text) placements in points, drawn in Courier 12 so that
leading spaces in `text` become real horizontal offsets, like indented code. Courier 12
is 7.2 points per character and lines are 13 points apart, matching the character grid
pdfplumber's layout mode assumes (about 7.25 x 13 points), as real book pages do.
"""

from __future__ import annotations

from pathlib import Path

PAGE_W, PAGE_H = 612, 792


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> Path:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")            # filled in once the page tree number is known
    tree = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    kids = []
    for placements in pages:
        ops = ["BT", "/F1 12 Tf"]
        for x, y, text in placements:
            ops.append(f"1 0 0 1 {x} {y} Tm ({_escape(text)}) Tj")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(add(
            f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()
        ))
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {tree} 0 R >>".encode()
    objects[tree - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                         f"/Count {len(kids)} >>").encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def lines(*texts: str, top: float = 740, step: float = 13, x: float = 72) -> list[tuple[float, float, str]]:
    """Place lines one under another, starting near the top of the page."""
    return [(x, top - i * step, text) for i, text in enumerate(texts)]


def manual_pages() -> list[list[tuple[float, float, str]]]:
    """Five pages with a running header and page-number footer; page 3 has no body text."""
    header = "Fixture Manual | Test Corpus"
    body = {
        1: ["Chapter 1. Registers", "",
            "The register x1 holds the return address and is named ra in the ABI.",
            "The register x2 is the stack pointer, named sp."],
        2: ["Chapter 2. A Python loop", "",
            "The loop below counts numbers greater than ten:", "",
            "count = 0",
            "for i in range(10):",
            "    num = int(input('Enter a number: '))",
            "    if num > 10:",
            "        count = count + 1",
            "print('There are', count, 'numbers greater than 10.')"],
        3: [],
        4: ["Chapter 3. Calls", "",
            "Arguments are passed in registers a0 through a7.",
            "The return value comes back in a0."],
        5: ["Chapter 4. Summary", "",
            "Registers s0 through s11 are preserved across calls."],
    }
    pages = []
    for number in range(1, 6):
        placed = [(72, 760, header)] + lines(*body[number], top=720)
        placed.append((300, 40, str(number)))
        pages.append(placed)
    return pages


def two_column_page() -> list[list[tuple[float, float, str]]]:
    """Page 1: a centered full-width title over two columns. Page 2: two columns only."""
    title = [(160, 760, "A Centered Title Across Both Columns")]
    left = lines("LEFT one alpha", "LEFT two beta", "LEFT three gamma", top=720, x=40)
    right = lines("RIGHT one delta", "RIGHT two eps", "RIGHT three zeta", top=720, x=330)
    return [title + left + right, left + right]
