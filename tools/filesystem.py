"""Sandboxed filesystem tools. Everything is confined to AGENT_WORKSPACE (default ~/Projects);
credential-looking files are refused. To approve more of the disk, change AGENT_WORKSPACE."""
import os
import re
import tempfile
from contextvars import ContextVar
from fnmatch import fnmatch
from pathlib import Path

from . import READ, WRITE, tool

_SECRET = re.compile(
    r"^(\.env(\.(?!example$|sample$|template$)[^/]*)?|\.ssh|\.aws|\.gnupg|\.netrc|\.npmrc|\.pgpass"
    r"|\.envrc|\.git-credentials|\.pypirc|id_(rsa|ed25519|ecdsa|dsa)(\.pub)?|.*\.(pem|key|p12|pfx|kdbx|tfvars)|credentials(\.json)?|secrets?(\.\w+)?)$", re.I)
IGNORE = {".git", "node_modules", ".venv", "__pycache__", "target", "build", "dist", ".next", ".idea", ".pytest_cache"}


def is_secret(p) -> bool:
    return any(_SECRET.match(part) for part in Path(p).parts)


def workspace() -> Path:
    return Path(os.environ.get("AGENT_WORKSPACE", "~/Projects")).expanduser().resolve()


_CWD: ContextVar[str | None] = ContextVar("agent_cwd", default=None)


def cwd() -> Path:
    """The calling thread's project dir (set by set_cwd), else AGENT_CWD, else the workspace."""
    return Path(_CWD.get() or os.environ.get("AGENT_CWD") or workspace()).expanduser().resolve()


def set_cwd(p) -> Path:
    """Per-context, so agents for different projects running in different threads of one process don't see each
    other's directory.
    Why: this used to also write os.environ["AGENT_CWD"], a process-global that the last caller won, so two
    conversations could read each other's project. Impact: nothing outside this context changes any more;
    spawned MCP servers and cloud CLIs are handed the directory explicitly (mcp_client, providers) instead of
    inheriting it. AGENT_CWD in the environment is only the startup default for threads that never called this."""
    _CWD.set(str(safe_path(str(p))))
    return cwd()


def safe_path(p: str) -> Path:
    full = (cwd() / Path(p).expanduser()).resolve()  # resolves symlinks, so links can't escape
    if not full.is_relative_to(workspace()):
        raise PermissionError(f"{p}: outside the workspace ({workspace()})")
    rel = full.relative_to(workspace())
    if is_secret(rel) or (".git" in rel.parts and rel.name == "config"):  # .git/config can hold tokens in remote URLs
        raise PermissionError(f"{p}: credential/secret file, access denied")
    return full


def _writable(p: str) -> Path:
    full = safe_path(p)
    if not full.is_relative_to(cwd()): raise PermissionError(f"{p}: outside the project directory ({cwd()}); writes stay inside it")
    if ".git" in full.parts:
        raise PermissionError(f"{p}: writing inside .git is not allowed (use git tools)")
    return full


def atomic_write(p: Path, text: str):
    """Write via a temp file in the same directory + os.replace: a crash never leaves a half-written file. Keeps the mode."""
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.")
    try:
        with os.fdopen(fd, "w", newline="") as f: f.write(text)
        if p.exists(): os.chmod(tmp, p.stat().st_mode & 0o7777)
        else: os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise


@tool(READ)
def list_files(path: str = ".", depth: int = 2) -> str:
    """List files under path (default: project dir) down to depth levels. Directories end with /."""
    root, out = safe_path(path), []

    def walk(d, lvl):
        for e in sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
            if e.name in IGNORE or is_secret(e.name) or len(out) >= 300: continue
            out.append(f"{'  ' * lvl}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir() and not e.is_symlink() and lvl + 1 < depth: walk(e, lvl + 1)
    walk(root, 0)
    return "\n".join(out) or "(empty)"


@tool(READ)
def read_file(path: str, start: int = 1, end: int = 0) -> str:
    """Read a text file with line numbers. Returns up to 200 lines from start; set end for a specific range."""
    p = safe_path(path)
    if p.exists() and not p.is_file(): raise ValueError(f"{path} is not a regular file")  # a FIFO would block forever
    if p.stat().st_size > 2_000_000: raise ValueError("file too large (>2MB); use search_files")
    lines, start = p.read_text(errors="replace").splitlines(), max(1, start)
    end = end or min(len(lines), start + 199)
    body = "\n".join(f"{i}\t{l}" for i, l in enumerate(lines[start - 1:end], start))
    return body + (f"\n... ({len(lines)} lines total; pass start/end to read more)" if end < len(lines) else "")


@tool(READ)
def search_files(pattern: str, path: str = ".", glob: str = "*") -> str:
    """Regex search file contents under path. glob filters file names, e.g. *.py. Returns file:line: text."""
    rx, root, hits = re.compile(pattern), safe_path(path), []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in IGNORE and not is_secret(d)]
        for fn in fns:
            f = Path(dp) / fn
            if is_secret(fn) or not fnmatch(fn, glob): continue
            r = f.resolve()  # a symlink inside the workspace may point at a secret
            if not r.is_relative_to(workspace()) or is_secret(r.relative_to(workspace())): continue
            try:
                if f.stat().st_size > 1_000_000: continue
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for i, l in enumerate(text.splitlines(), 1):
                if rx.search(l): hits.append(f"{os.path.relpath(f, cwd())}:{i}: {l.strip()[:200]}")
                if len(hits) >= 50: break
            if len(hits) >= 50: return "\n".join(hits[:50]) + "\n... (truncated at 50 matches)"
    return "\n".join(hits) or "no matches"


@tool(WRITE)
def create_file(path: str, content: str) -> str:
    """Create a new file with content (fails if it exists; use edit_file to change existing files)."""
    p = _writable(path)
    if p.exists(): raise FileExistsError(f"{path} already exists; use edit_file")
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, content)
    return f"created {path} ({len(content)} chars)"


@tool(WRITE)
def edit_file(path: str, old: str, new: str) -> str:
    """Replace the exact text `old` with `new` in a file. `old` must match exactly once."""
    p = _writable(path)
    with p.open(newline="") as f: text = f.read()  # keep CRLF files CRLF
    n = text.count(old)
    if n != 1: raise ValueError(f"`old` matched {n} times in {path}; it must match exactly once (add context)")
    atomic_write(p, text.replace(old, new))
    return f"edited {path}"


@tool(WRITE)
def multi_edit(edits: list) -> str:
    """Apply several exact-text edits, possibly across files, all or nothing. edits = [{path, old, new}, ...]; edits to
    one file apply in order, each `old` must match exactly once at that point. Nothing is written if any edit fails."""
    texts: dict[Path, str] = {}
    for i, e in enumerate(edits, 1):
        try:
            p = _writable(e["path"])
            if p not in texts:
                with p.open(newline="") as f: texts[p] = f.read()
            if (n := texts[p].count(e["old"])) != 1: raise ValueError(f"`old` matched {n} times; it must match exactly once")
            texts[p] = texts[p].replace(e["old"], e["new"])
        except (KeyError, TypeError) as ex: raise ValueError(f"edit {i}: needs path, old and new ({ex!r})") from ex
        except Exception as ex: raise type(ex)(f"edit {i} ({e.get('path')}): {ex}") from ex
    for p, text in texts.items(): atomic_write(p, text)
    return f"applied {len(edits)} edits to {len(texts)} files, no changes were left half-done"


@tool(READ)
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by glob pattern under path, e.g. **/*.py or src/**/test_*.js. Returns up to 200 paths."""
    root, out = safe_path(path), []
    for f in sorted(root.glob(pattern)):
        rel = f.relative_to(root)
        if f.is_file() and not (set(rel.parts) & IGNORE) and not is_secret(rel): out.append(str(rel))
        if len(out) >= 200: break
    return "\n".join(out) or "no matches"
