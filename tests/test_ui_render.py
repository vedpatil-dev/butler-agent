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


def test_mdstream_headings_bullets_and_quotes():
    m = MdStream()
    out = m.feed("## Title\n- item *one*\n> quote\n---\n") + m.flush()
    assert "\033[1;36mTitle\033[0m" in out
    assert "•" in out and "\033[3mone\033[0m" in out
    assert "│ quote" in out
    assert "─" in out


def test_mdstream_preserves_globs_math_and_inline_code():
    m = MdStream()
    out = m.feed("match *.py and *.md or a * b * c and `x * y * z`\n") + m.flush()
    assert "*.py and *.md" in out
    assert "a * b * c" in out
    assert "\033[36mx * y * z\033[0m" in out


def test_confirm_preview_supports_multi_edit(monkeypatch, tmp_path):
    from agent.confirm import _preview
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_CWD", str(tmp_path))
    from tools.filesystem import set_cwd
    set_cwd(tmp_path)
    diff = _preview("multi_edit", {"edits": [
        {"path": "a.py", "old": "x = 1", "new": "x = 9"},
        {"path": "a.py", "old": "x = 9", "new": "x = 42"},
    ]})
    assert "-x = 1" in diff and "+x = 9" in diff and "-x = 9" in diff and "+x = 42" in diff
