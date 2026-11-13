"""@file attachments. `summarise @report.pdf` / `what is in @shot.png`: the @path tokens that name an existing file inside
the workspace are attached to that message.
- cloud agents (claude/codex/agy) get the files themselves: images as --image where the CLI has it, paths + --add-dir otherwise.
- local models get images as base64 (Ollama vision models) and text/PDF as RAG: small files inline, big ones chunked and
  the chunks that best match the message (BM25) injected. ponytail: lexical BM25, no embedding model to install; swap
  rank() for Ollama /api/embed cosine if paraphrased questions start missing."""
import base64
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from tools import READ
from tools.filesystem import cwd, is_secret, safe_path, workspace

from . import config

IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_BYTES = 20_000_000
CHUNK = 1200  # chars per RAG chunk
PASTES = config.STATE / "pastes"  # clipboard images land here; the user's paste is the approval, so no prompt for these


def clipboard_image() -> Path | None:
    """Save the clipboard's image (Wayland wl-paste, else X11 xclip) to PASTES and return its path; None if it holds none."""
    for cmd in (["wl-paste", "--type", "image/png"], ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]):
        try: r = subprocess.run(cmd, capture_output=True, timeout=1)
        except (OSError, subprocess.SubprocessError): continue
        if r.returncode == 0 and r.stdout.startswith(b"\x89PNG"):
            PASTES.mkdir(parents=True, exist_ok=True, mode=0o700)
            p = PASTES / f"paste-{time.strftime('%H%M%S')}.png"
            p.write_bytes(r.stdout)
            return p
    return None


def _candidate(tok: str) -> Path | None:
    """A message word -> an existing file path, or None. Accepts @rel/path and bare /abs or ~/home paths, tolerating the
    quotes / backslash a terminal drag-and-drop leaves in front, and trailing punctuation. Secrets never match."""
    at = tok.startswith("@")
    tok = tok.lstrip("@\"'\\")
    if not (at or tok.startswith(("/", "~/"))): return None
    for cand in (tok, tok.rstrip(".,;:!?)]}\"'")):
        try: p = (cwd() / Path(cand).expanduser()).resolve()
        except (OSError, RuntimeError): continue
        if p.is_file() and not is_secret(p) and p.stat().st_size <= MAX_BYTES: return p
    return None


def is_path(word: str) -> bool:
    """True if a single word is a dropped file path (chat uses it so '/home/me/pic.png' isn't taken for a slash command)."""
    return not word.startswith("@") and _candidate(word) is not None


def parse(task: str, confirm=None) -> list[Path]:
    """Files named in the message. Inside the workspace they attach directly; outside it, only if `confirm` (the same
    approve callback as tool calls) says yes - with no confirm they are skipped."""
    out = []
    for tok in task.split():
        p = _candidate(tok)
        if not p or p in out: continue
        try: safe_path(str(p))
        except PermissionError:
            if p.is_relative_to(PASTES): pass
            elif not (confirm and confirm("attach_file", {"path": str(p)}, READ)): continue
        out.append(p)
    return out


def cloud(task: str, files: list[Path]) -> tuple[str, list[str], list[str]]:
    """-> (task with the file paths appended, image paths, parent dirs). Every cloud CLI reads files itself from a path."""
    if not files: return task, [], []
    listing = "\n".join(f"- {f}" for f in files)
    return (f"{task}\n\nAttached files (read them from these paths):\n{listing}",
            [str(f) for f in files if f.suffix.lower() in IMAGES],
            sorted({str(f.parent) for f in files if not f.is_relative_to(workspace())}))  # in-workspace files are already readable; granting a parent dir would widen the agent's access


def _text(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        r = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, timeout=60)
        if r.returncode: raise ValueError(f"pdftotext failed: {r.stderr.strip()[:100]}")
        return r.stdout
    data = p.read_bytes()
    if b"\0" in data[:4096]: raise ValueError("binary file")
    return data.decode(errors="replace")


def chunks(text: str, size: int = CHUNK) -> list[str]:
    """Greedy line-packed chunks; a single over-long line is cut."""
    out, cur = [], ""
    for line in text.splitlines():
        while len(line) > size: out.append(line[:size]); line = line[size:]
        if len(cur) + len(line) + 1 > size and cur: out.append(cur); cur = ""
        cur += line + "\n"
    return out + [cur] if cur.strip() else out


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def rank(query: str, docs: list[str], k: float = 1.5, b: float = 0.75) -> list[int]:
    """Doc indexes, best BM25 match first (all-zero scores keep document order, so a vague query gets the opening chunks)."""
    toks = [_tok(d) for d in docs]
    avg, df = (sum(map(len, toks)) / len(toks) if toks else 1) or 1, Counter(w for t in toks for w in set(t))
    q = set(_tok(query))

    def score(t):
        tf = Counter(t)
        return sum(math.log(1 + (len(toks) - df[w] + .5) / (df[w] + .5)) * tf[w] * (k + 1) / (tf[w] + k * (1 - b + b * len(t) / avg)) for w in q if w in tf)
    return sorted(range(len(docs)), key=lambda i: -score(toks[i]))


def local(task: str, files: list[Path], num_ctx: int, vision: bool = True) -> tuple[str, list[str]]:
    """-> (task + attached text / retrieved excerpts, base64 images). Attachments get a quarter of the context window."""
    if not files: return task, []
    budget, parts, images = num_ctx // 4 * 3, [], []  # chars (~3 per token)
    texts = {}
    for f in files:
        try:
            if f.suffix.lower() in IMAGES:
                if vision: images.append(base64.b64encode(f.read_bytes()).decode())
                else: parts.append(f"[{f.name}: this backend cannot see images]")
            else: texts[f] = _text(f)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            parts.append(f"[{f.name}: could not be read ({e})]")
    share = budget // max(1, len(texts))
    for f, t in texts.items():
        if len(t) <= share:
            parts.append(f"=== {f.name} ===\n{t}")
            continue
        cs = chunks(t)
        keep, used = [], 0
        for i in rank(task, cs):
            if used + len(cs[i]) > share: continue
            keep.append(i)
            used += len(cs[i])
        body = "\n...\n".join(cs[i] for i in sorted(keep))
        parts.append(f"=== {f.name}: {len(keep)} of {len(cs)} chunks that best match the request (not the whole file; use read_file for more) ===\n{body}")
    return task + "\n\n" + "\n\n".join(parts) if parts else task, images
