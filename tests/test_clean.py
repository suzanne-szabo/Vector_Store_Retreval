"""Black-box tests for rag_agent.clean."""

from collections import Counter

from rag_agent.clean import (
    MIN_PAGE_CHARS, clean_document, collapse_blank_lines, dedent, edge_signature,
    document_vocabulary, find_running_edges, join_broken_words, normalize, strip_edges, tidy_line,
)


def test_normalize_expands_ligatures():
    assert normalize("ﬁnd the ﬂag") == "find the flag"


def test_normalize_drops_private_use_and_replacement_characters():
    assert normalize("Note: x0 is zero�") == "Note: x0 is zero"


def test_tidy_line_keeps_indentation_and_shrinks_inner_gaps():
    assert tidy_line("        count = count + 1     ") == "        count = count + 1"
    assert tidy_line("  x10 - x17        a0 - a7        Argument registers") == "  x10 - x17  a0 - a7  Argument registers"


def test_tidy_line_shortens_dot_leaders_but_not_ellipses_or_code():
    assert tidy_line("     Forgetting to Indent  .  .  .  .  .  .  .  .  . 57") == "     Forgetting to Indent ... 57"
    assert tidy_line("Chapter 2 ........................ 13") == "Chapter 2 ... 13"
    assert tidy_line("    print('wait...')") == "    print('wait...')"
    assert tidy_line("x = obj.attr.method()") == "x = obj.attr.method()"


def test_join_broken_words_rejoins_hyphenated_line_breaks():
    assert join_broken_words("who have ben-\nefited from it") == "who have benefited from it"
    assert join_broken_words("people used On-\n line Python Tutor") == "people used Online Python Tutor"


def test_join_broken_words_keeps_real_compounds_used_elsewhere():
    vocab = document_vocabulary("Registers s0-s11 are callee-saved. Use floating-point here.")
    assert "callee-saved" in vocab.hyphenated
    assert join_broken_words("these are callee-\nsaved registers", vocab) == "these are callee-saved registers"
    assert join_broken_words("the Floating-\npoint unit", vocab) == "the Floating-point unit"
    assert join_broken_words("who have ben-\nefited", vocab) == "who have benefited"


def test_join_broken_words_uses_whole_words_to_decide():
    vocab = document_vocabulary("A program uses the upper bits of an immediate value. Something else.")
    assert join_broken_words("this pro-\ngram runs", vocab) == "this program runs"           # 'program' seen
    assert join_broken_words("some-\nthing odd", vocab) == "something odd"                    # 'something' seen
    assert join_broken_words("the upper-\nimmediate form", vocab) == "the upper-immediate form"  # both halves seen
    assert join_broken_words("Water-\nloo campus", vocab) == "Waterloo campus"                # neither rule: syllable split


def test_fragments_of_broken_words_are_not_counted_as_words():
    vocab = document_vocabulary("it is pop-\nular with the pop crowd")
    assert "ular" not in vocab.words
    assert join_broken_words("it is pop-\nular", vocab) == "it is popular"


def test_clean_document_uses_the_whole_document_for_compounds():
    filler = "Enough ordinary words to keep this page above the minimum length."
    raw = [f"{filler}\nRegisters s0 to s11 are callee-saved.",
           f"{filler}\nThese callee-\nsaved registers must be restored."]
    assert "callee-saved registers must be restored" in clean_document(raw)[1]


def test_join_broken_words_leaves_other_hyphens_alone():
    assert join_broken_words("a 12-\nbit field") == "a 12-\nbit field"          # digit before the hyphen
    assert join_broken_words("RISC-\nV manual") == "RISC-\nV manual"            # capital after the break
    assert join_broken_words("x = a -\n    b") == "x = a -\n    b"              # code: space before the hyphen
    assert join_broken_words("web-based tool") == "web-based tool"              # no line break


def test_edge_signature_ignores_page_numbers_and_spacing():
    assert edge_signature("  Page 12   of  300 ") == edge_signature("Page 7 of 300")


def test_running_header_and_page_numbers_are_removed():
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    pages = [["Manual Title | Header", "", f"Body about {word}", "", str(i)] for i, word in enumerate(words, 1)]
    running = find_running_edges(pages)
    assert running == {edge_signature("Manual Title | Header"), "#"}
    assert strip_edges(pages[0], running) == ["", "Body about alpha", ""]


def test_lines_that_do_not_repeat_are_kept():
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    pages = [[f"Heading on {word}", f"Body about {word}"] for word in words]
    assert find_running_edges(pages) == set()


def test_short_documents_skip_header_detection():
    pages = [["Same line", "body"], ["Same line", "body"]]
    assert find_running_edges(pages) == set()


def test_dedent_keeps_relative_indentation():
    lines = ["      for i in range(3):", "          print(i)", "", "      done()"]
    assert dedent(lines) == ["for i in range(3):", "    print(i)", "", "done()"]


def test_collapse_blank_lines():
    assert collapse_blank_lines(["", "a", "", "", "b", "", ""]) == ["a", "", "b"]


def test_clean_document_blanks_short_pages_and_reports_removed_lines():
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    body = "This page has enough words to be kept as a real page about"
    raw = [f"Header Line\n\n{body} {word}.\n\n{i}" for i, word in enumerate(words, 1)]
    raw[2] = "Header Line\n\n\n3"
    removed = Counter()
    cleaned = clean_document(raw, removed)
    assert len(cleaned) == 6
    assert cleaned[2] == ""
    assert cleaned[0] == f"{body} alpha."
    assert all(len(page) >= MIN_PAGE_CHARS for page in cleaned if page)
    assert removed[edge_signature("Header Line")] == 6
