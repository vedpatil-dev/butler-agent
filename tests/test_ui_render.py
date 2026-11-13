from agent.main import friendly
from agent.ui import MdStream


def test_bold_split_across_chunks_still_renders():
    m = MdStream()
    out = m.feed("a **bo") + m.feed("ld** b\n") + m.flush()
    assert "\033[1mbold\033[0m" in out and "**" not in out


def test_fenced_code_is_not_markdown_processed():
    m = MdStream()
    out = m.feed("```py\nx = a ** b ** c\n```\nafter **x**\n")
    assert "a ** b ** c" in out and "\033[1mx\033[0m" in out


def test_flush_returns_unfinished_line():
    m = MdStream()
    assert m.feed("no newline") == "" and "no newline" in m.flush()


def test_friendly_adds_hint_for_connection_errors():
    assert "ollama serve" in friendly("ConnectError: [Errno 111] Connection refused")
    assert friendly("something else") == "something else"
