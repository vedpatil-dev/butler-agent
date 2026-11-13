"""Everything that asks the user for permission lives here: the local agent's per-action confirm, the diff/command
preview it shows, and the prompt for tool requests relayed from `claude -p` (agent/permissions.py). One place for the
wording, the y/N/a/p semantics and the colour handling, instead of three copies spread over main.py and render.py."""
import json
import sys

from .render import BOLD, DIM, GREEN, RED, RESET


def inv(code: str) -> str:
    """Wrap a color code for use INSIDE an input() prompt (never in print()). Once `readline` is
    imported (needed for tab-completion - agent/main.py's _setup_tab_completion), input()'s prompt
    is rendered by libedit/GNU readline instead of going straight to the terminal, and it strips
    raw escape bytes it isn't told are zero-width, showing literal "[1m" instead of bold text.
    \\001/\\002 (RL_PROMPT_START/END_IGNORE) mark a span as invisible so it passes through intact.
    Only wrap when readline is actually active (e.g. not `agent run`'s confirm prompt, which never
    imports it): those two bytes would otherwise be written raw to a terminal that never asked to
    filter them."""
    return f"\001{code}\002" if "readline" in sys.modules else code


def _diff_lines(old: list[str], new: list[str], path: str) -> list[str]:
    import difflib
    diff = list(difflib.unified_diff(old, new, path, path, lineterm="", n=2))[2:]
    out = [(GREEN if l[0] == "+" else RED if l[0] == "-" else DIM) + l + RESET if l[:1] in "+-@" else DIM + l + RESET
           for l in diff[:60]]
    if len(diff) > 60:
        out.append(f"{DIM}... (+{len(diff) - 60} more lines){RESET}")
    return out


def _preview(name: str, args: dict) -> str:
    """What the user is approving: the command, or a diff of the file change (not 160 chars of JSON)."""
    from tools.filesystem import safe_path
    try:
        if name == "edit_file":
            old_txt = safe_path(args["path"]).read_text()
            return "\n".join(_diff_lines(old_txt.splitlines(), old_txt.replace(args["old"], args["new"], 1).splitlines(), args["path"]))
        if name == "create_file":
            return "\n".join(_diff_lines([], str(args["content"]).splitlines(), args["path"]))
        if name == "multi_edit":
            out, virtual = [], {}
            for ed in args.get("edits") or []:
                try:
                    p = ed["path"]
                    old_txt = virtual[p] if p in virtual else safe_path(p).read_text()
                    new_txt = old_txt.replace(ed["old"], ed["new"], 1)
                    virtual[p] = new_txt
                    out.extend(_diff_lines(old_txt.splitlines(), new_txt.splitlines(), p))
                except (OSError, KeyError, ValueError, TypeError) as e:
                    out.append(f"{RED}[preview error for {ed.get('path', '?')}: {e}]{RESET}")
            if len(out) > 80:
                return "\n".join(out[:80] + [f"{DIM}... (+{len(out) - 80} more lines){RESET}"])
            return "\n".join(out)
    except (OSError, KeyError, ValueError, TypeError):
        return ""
    return ""


def cli_confirm(state):
    def confirm(name, args, level):
        if level == "write" and state["all"]: return True
        colour = inv(RED) if level == "dangerous" else ""
        shell = name == "execute_command" and level == "write"
        opts = "[y/N]" if level in ("dangerous", "read") else "[y/N/a=allow edits+commands this session" + ("/p=always allow this command" if shell else "") + "]"
        detail = args.get("command") or " ".join(  # the target (path/url) first, long values (file content) clipped
            f"{k}={json.dumps(args[k], ensure_ascii=False)[:160]}" for k in sorted(args, key=lambda k: k not in ("path", "paths", "url", "dir")))
        if diff := _preview(name, args): detail += "\n" + diff
        ans = input(f"{colour}{inv(BOLD)}? {level.upper()}: {name}{inv(RESET)}{colour} {detail}\n  allow {opts} {inv(RESET)}").strip().lower()
        if ans == "a" and level == "write": state["all"] = True
        if ans == "p" and shell: return "always"
        return ans in ("y", "a") if level == "write" else ans == "y"
    return confirm


def permission_prompt(e: dict) -> None:
    """Ask the user about one `claude -p` tool request (a {"type": "permission"} event) and answer it.
    Why: moved out of render.py so every approval prompt shares this module. Impact: unchanged behaviour -
    y allows once, a allows this tool for the rest of the run, anything else (or EOF/Ctrl-C) denies."""
    inp = e["input"]
    detail = str(inp.get("command") or inp.get("file_path") or inp.get("url") or json.dumps(inp, ensure_ascii=False))[:300]
    try: ans = input(f"\n{BOLD}? claude wants to run {e['tool']}:{RESET} {detail}\n  allow [y/N/a=always this session] ").strip().lower()
    except (EOFError, KeyboardInterrupt): ans = ""
    e["answer"](ans in ("y", "a"), ans == "a")
