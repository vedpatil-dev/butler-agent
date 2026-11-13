"""Drives agent.ui.pick_from_list through a real pty with real key bytes - the only way to test an
interactive widget honestly (piped stdin bypasses the terminal handling entirely)."""
import os
import re
import select
import sys
import time

import pytest

pty = pytest.importorskip("pty")

ROWS = [(f"model-{i:02d}", "[claude]" if i % 2 else "[antigravity]", f"detail {i}") for i in range(30)]
ROWS[3] = ("claude-opus-5-5", "[claude]", "Opus 5.5")
ROWS[9] = ("gemini-opus-ish", "[antigravity]", "second opus match")

UP, DOWN, PGDN, ENTER, ESC, BACKSPACE = b"\x1b[A", b"\x1b[B", b"\x1b[6~", b"\r", b"\x1b", b"\x7f"

DRIVER = f"""
import json, sys
sys.path.insert(0, {os.getcwd()!r})
from agent.ui import pick_from_list
rows = {ROWS!r}
r = pick_from_list(rows, "Select a model", current=int(sys.argv[1]) if len(sys.argv) > 1 else None)
print("RESULT=" + json.dumps(r), flush=True)
"""


def drive(keys: list[bytes], current: int | None = None, timeout: float = 8.0) -> str:
    """Run the picker, send each key (with a beat between them so the app has processed the last
    one), return the RESULT=... payload."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", DRIVER] + ([str(current)] if current is not None else []))
    out, end = "", time.time() + timeout

    def pump(secs):
        nonlocal out
        stop = min(time.time() + secs, end)
        while time.time() < stop:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try: out += os.read(fd, 65536).decode(errors="replace")
                except OSError: return

    while "Esc cancel" not in out and time.time() < end: pump(0.05)    # wait until the widget has actually drawn
    for k in keys:
        os.write(fd, k)
        pump(0.1 if k != ESC else 0.35)         # a lone Esc is held back ~0.1s by the app to rule out an escape sequence
    while "RESULT=" not in out and time.time() < end: pump(0.2)
    try: os.kill(pid, 9)
    except ProcessLookupError: pass
    os.waitpid(pid, 0)
    m = re.search(r"RESULT=(\S+)", out)
    assert m, f"picker never returned; output tail: {out[-300:]!r}"
    return m.group(1)


def test_down_down_enter_picks_the_third_row():
    assert drive([DOWN, DOWN, ENTER]) == "2"


def test_up_from_the_top_wraps_to_the_last_row():
    assert drive([UP, ENTER]) == str(len(ROWS) - 1)


def test_starts_on_the_current_row():
    assert drive([ENTER], current=7) == "7"


def test_typing_filters_live_and_enter_takes_the_first_match():
    assert drive([b"o", b"p", b"u", b"s", ENTER]) == "3"                   # claude-opus-5-5 (first of two "opus" rows)
    assert drive([b"o", b"p", b"u", b"s", DOWN, ENTER]) == "9"             # arrows move within the filtered view


def test_filter_matches_the_provider_tag_too():
    assert drive([b"a", b"n", b"t", b"i", b"g", ENTER]) == "0"             # row 0 is the first [antigravity] row


def test_enter_does_nothing_while_nothing_matches_then_backspace_recovers():
    assert drive([b"z", b"z", b"z", ENTER, BACKSPACE, BACKSPACE, BACKSPACE, ENTER]) == "0"


def test_page_down_jumps_a_window():
    assert drive([PGDN, ENTER]) == "12"


def test_escape_cancels():
    assert drive([ESC]) == "null"


def test_filter_rows_and_scroll_window_pure_helpers():
    from agent.ui import filter_rows, scroll_start
    assert filter_rows(ROWS, "OPUS") == [3, 9]
    assert filter_rows(ROWS, "") == list(range(30))
    assert filter_rows(ROWS, "nope") == []
    assert scroll_start(0, 0, 12, 30) == 0
    assert scroll_start(0, 12, 12, 30) == 1       # cursor just below the window scrolls it by one
    assert scroll_start(5, 2, 12, 30) == 2        # cursor above the window scrolls up to it
    assert scroll_start(0, 29, 12, 30) == 18      # last row: window pinned to the end, never past it
    assert scroll_start(3, 1, 12, 5) == 0         # fewer rows than the window: no scrolling
