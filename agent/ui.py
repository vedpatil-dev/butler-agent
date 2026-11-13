"""Bordered, multi-line, tab-completing input box for the interactive chat loop - closer to the
claude/antigravity/codex CLI feel than a plain readline prompt. Verified via a real pty test (see
commit history): border draws, typed text appears inside, Alt+Enter inserts a newline and grows
the box, plain Enter submits. Enter submits (not Shift+Enter): most terminals don't reliably
distinguish Shift+Enter from Enter without extended keyboard protocol support, so these CLIs
converge on Alt+Enter (or similar) for a literal newline - same choice made here.
"""
import re
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Dimension, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import Frame, TextArea


class _SlashCompleter(Completer):
    """Wraps a plain complete_command(text, provider) -> list[str] function (agent/main.py) -
    only completes when the whole line so far is a bare /command token, like the old readline setup."""

    def __init__(self, complete_fn, get_provider):
        self.complete_fn, self.get_provider = complete_fn, get_provider

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = text.rsplit(None, 1)[-1] if text.split() and not text[-1].isspace() else ""
        if word.startswith("@"):  # @file mention anywhere in the line
            for rel in at_completions(word[1:]): yield Completion("@" + rel, start_position=-len(word))
            return
        if not text.startswith("/") or " " in text: return
        for c in self.complete_fn(text, self.get_provider()):
            yield Completion(c, start_position=-len(text))


def at_completions(prefix: str, root: Path | None = None, limit: int = 30) -> list[str]:
    """Project files (relative paths) containing `prefix`, for @file mentions. Skips ignored dirs and secret files."""
    import os
    from tools.filesystem import IGNORE, cwd, is_secret
    root, out, seen = root or cwd(), [], 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in IGNORE and not is_secret(d))
        for fn in sorted(fns):
            rel = os.path.relpath(os.path.join(dp, fn), root)
            seen += 1
            if not is_secret(fn) and prefix.lower() in rel.lower(): out.append(rel)
            if len(out) >= limit or seen > 5000: return out
    return out


def history_file_for(history_path: Path, provider: str | None) -> Path:
    """Each provider gets its own input-history file (chat_history_claude, chat_history_npu, ...) so Up-arrow
    recalls prompts for the active provider rather than mixing them all together."""
    return history_path.with_name(f"{history_path.name}_{provider}") if provider else history_path


def make_box_prompt(get_title, complete_fn, get_provider, history_path: Path):
    """Returns prompt() -> str | None (None on Ctrl-D with an empty box, meaning "quit"). Each
    call draws a fresh box (not a persistent full-screen app), so it composes with the existing
    plain print()-based _render() - the box disappears once Enter submits, same as a normal prompt."""
    histories: dict[Path, FileHistory] = {}

    def _history() -> FileHistory:
        p = history_file_for(history_path, get_provider() if get_provider else None)
        if p not in histories:
            p.parent.mkdir(parents=True, exist_ok=True)
            histories[p] = FileHistory(str(p))
        return histories[p]

    completer = _SlashCompleter(complete_fn, get_provider)
    kb = KeyBindings()

    # Explicit, deterministic completion handling - without these, Tab/Enter fall back to
    # TextArea's undocumented internal defaults, which behaved inconsistently in testing.
    # buf.start_completion() is asynchronous (scheduled via create_background_task, confirmed by
    # instrumenting it directly: complete_state is still None immediately after the call returns)
    # - checking its result synchronously right after is a guaranteed-to-lose race, which is what
    # the first version of this code did. insert_common_part=True is the built-in fix: for a
    # single match it inserts the whole thing once the background task resolves, same as
    # bash/readline Tab behavior, with no manual result-checking needed.
    @kb.add("tab")
    def _tab(event):
        buf = event.app.current_buffer
        if buf.complete_state: buf.complete_next()
        else: buf.start_completion(insert_common_part=True)

    @kb.add("enter", filter=completion_is_selected)
    def _accept_completion(event):
        buf = event.app.current_buffer
        buf.apply_completion(buf.complete_state.current_completion)

    @kb.add("enter", filter=~completion_is_selected)
    def _submit(event):
        event.app.exit(result=event.app.current_buffer.text)

    @kb.add("escape", "enter")  # Alt+Enter: literal newline, not a submit
    def _newline(event):
        event.app.current_buffer.insert_text("\n")

    @kb.add("c-v")
    @kb.add(Keys.BracketedPaste)  # an image-only clipboard arrives as an empty paste
    def _paste(event):
        from . import attach
        text = (event.data or "").replace("\r\n", "\n").replace("\r", "\n") if event.key_sequence[0].key == Keys.BracketedPaste else ""
        if not text.strip() and (img := attach.clipboard_image()): text = f"@{img} "
        if not text and event.key_sequence[0].key != Keys.BracketedPaste: text = event.app.clipboard.get_data().text  # ctrl-v of plain text still works
        event.current_buffer.insert_text(text)

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    @kb.add("c-d")
    def _eof_or_delete(event):
        buf = event.app.current_buffer
        if buf.text: buf.delete(1)        # normal "delete forward" with text present
        else: event.app.exit(result=None)  # empty box: Ctrl-D quits, like a bare EOF would

    def prompt() -> str | None:
        # No height limit made the empty box fill every terminal row below the cursor (21 rows on a 40-row
        # terminal). Exact height = lines typed (1 when empty), up to MAX_INPUT_LINES, then the box scrolls.
        def input_height():
            return Dimension.exact(input_lines(area.text, get_app().output.get_size().columns - 4))  # minus the frame's borders

        hist = _history()
        area = TextArea(multiline=True, wrap_lines=True, completer=completer,
                        complete_while_typing=False, history=hist, height=input_height)
        # Without this Float, Tab still cycles/applies candidates (the keybindings above work),
        # but silently - there's nothing on screen showing what the choices are or which one is
        # selected, which is exactly the bug this fixes: candidates were being applied one at a
        # time with each Tab press and no visible list to choose from.
        root = FloatContainer(content=Frame(area, title=get_title()),
                              floats=[Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=8))])
        app = Application(layout=Layout(root, focused_element=area),
                          key_bindings=kb, full_screen=False, mouse_support=False)
        try:
            result = app.run()
        except (KeyboardInterrupt, EOFError):
            return None
        if result:
            hist.append_string(result)
        return result
    return prompt


# ---------------------------------------------------------------------------------------------
# Arrow-key list picker (used by /model and /effort)
# ---------------------------------------------------------------------------------------------
MAX_VISIBLE = 12
MAX_INPUT_LINES = 12


def input_lines(text: str, cols: int) -> int:
    """Rows the input box should be: wrapped lines of `text`, at least 1, at most MAX_INPUT_LINES. The height must
    be EXACT - prompt_toolkit's HSplit otherwise grows the box up to its max (the whole terminal when unbounded)."""
    cols = max(10, cols)
    return min(MAX_INPUT_LINES, sum(max(1, -(-len(l) // cols)) for l in text.split("\n")))


def filter_rows(rows: list[tuple[str, str, str]], text: str) -> list[int]:
    """Indices of the rows containing `text` (case-insensitive) in ANY column - id, [provider] tag or detail."""
    t = text.lower()
    return [i for i, r in enumerate(rows) if t in " ".join(r).lower()]


def scroll_start(start: int, cursor: int, visible: int, total: int) -> int:
    """First row shown, so that `cursor` stays inside a `visible`-row window over `total` rows."""
    if cursor < start: start = cursor
    elif cursor >= start + visible: start = cursor - visible + 1
    return max(0, min(start, max(0, total - visible)))


def pick_from_list(rows: list[tuple[str, str, str]], title: str, current: int | None = None) -> int | None:
    """Interactive list: Up/Down move (wrapping), PgUp/PgDn/Home/End jump, typing filters live,
    Enter selects, Esc / Ctrl-C / Ctrl-D cancel. rows = (id, tag, detail) per line; `current` is
    the index of the row to start on and mark. Returns the chosen index into `rows`, or None.
    The widget erases itself when done, so the caller just prints its own confirmation line."""
    st = {"text": "", "view": list(range(len(rows))), "cursor": current or 0, "start": 0}
    w, tw = max((len(r[0]) for r in rows), default=0), max((len(r[1]) for r in rows), default=0)

    def n_visible() -> int: return max(1, min(MAX_VISIBLE, len(st["view"])))

    def refilter():
        st["view"], st["cursor"], st["start"] = filter_rows(rows, st["text"]), 0, 0

    def list_text():
        view, cur, n = st["view"], st["cursor"], n_visible()
        if not view: return [("fg:ansired", "  (nothing matches)")]
        st["start"] = scroll_start(st["start"], cur, n, len(view))
        out = []
        for pos in range(st["start"], min(len(view), st["start"] + n)):
            i = view[pos]
            primary, tag, detail = rows[i]
            sel = pos == cur
            hl = "reverse " if sel else ""  # the selected row is one solid highlighted bar
            out += [(hl + "bold", "▶ " if sel else "  "), (hl, f"{primary:<{w}}  "),
                    (hl + ("" if sel else "fg:ansicyan"), f"{tag:<{tw}}  " if tw else ""),
                    (hl + ("" if sel else "fg:ansibrightblack"), detail)]
            if i == current: out.append((hl + ("" if sel else "fg:ansigreen"), "  \u25c0 current"))
            out.append(("", "\n"))
        return out[:-1]

    filter_line = lambda: [("bold", "filter: "), ("", st["text"]), ("reverse", " "),
                           ("fg:ansibrightblack", f"   {len(st['view'])}/{len(rows)}")]
    kb = KeyBindings()

    @kb.add("<any>")  # registered FIRST: when several bindings match a key the later ones win, so every specific key below beats it
    def _typed(event):
        if len(event.data) == 1 and event.data.isprintable():
            st["text"] += event.data
            refilter()

    @kb.add("backspace")
    @kb.add("c-h")
    def _back(event):
        st["text"] = st["text"][:-1]
        refilter()

    @kb.add("c-u")
    def _clear(event):
        st["text"] = ""
        refilter()

    def move(to):  # to(cursor, n_rows) -> new cursor
        if st["view"]: st["cursor"] = to(st["cursor"], len(st["view"]))
    kb.add("up")(lambda e: move(lambda c, n: (c - 1) % n))
    kb.add("down")(lambda e: move(lambda c, n: (c + 1) % n))
    kb.add("pageup")(lambda e: move(lambda c, n: max(0, c - n_visible())))
    kb.add("pagedown")(lambda e: move(lambda c, n: min(n - 1, c + n_visible())))
    kb.add("home")(lambda e: move(lambda c, n: 0))
    kb.add("end")(lambda e: move(lambda c, n: n - 1))

    @kb.add("enter")
    def _pick(event):
        if st["view"]: event.app.exit(result=st["view"][st["cursor"]])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(event):
        event.app.exit(result=None)

    hint = [("fg:ansibrightblack", "\u2191\u2193 move \u00b7 PgUp/PgDn \u00b7 type to filter \u00b7 Enter select \u00b7 Esc cancel")]
    root = Frame(HSplit([Window(FormattedTextControl(filter_line), height=1),
                         Window(FormattedTextControl(list_text), wrap_lines=False,
                                height=lambda: Dimension.exact(n_visible())),
                         Window(FormattedTextControl(hint), height=1)]), title=title)
    app = Application(layout=Layout(root), key_bindings=kb, full_screen=False, mouse_support=False,
                      erase_when_done=True)
    app.ttimeoutlen = 0.1  # a lone Esc is held back this long to rule out an escape sequence; the default (0.5s) feels laggy
    try: return app.run()
    except (KeyboardInterrupt, EOFError): return None


# ---------------------------------------------------------------------------------------------
# Streaming markdown -> ANSI
# ---------------------------------------------------------------------------------------------
_B, _C = re.compile(r"\*\*(.+?)\*\*"), re.compile(r"`([^`\n]+?)`")


class MdStream:
    """Renders streamed model text one complete line at a time, so **bold** / `code` split across chunks still
    render, and ``` fenced blocks are shown dim-cyan. feed() returns what to print now; flush() the unfinished tail."""

    def __init__(self):
        self.buf, self.fence = "", False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")
        return "".join(self._line(l) + "\n" for l in lines)

    def flush(self) -> str:
        out, self.buf = self._line(self.buf) if self.buf else "", ""
        return out

    def _line(self, line: str) -> str:
        if line.lstrip().startswith("```"):
            self.fence = not self.fence
            return f"\033[2m{line}\033[0m"
        if self.fence: return f"\033[36m{line}\033[0m"
        return _C.sub("\033[36m\\1\033[0m", _B.sub("\033[1m\\1\033[0m", line))
