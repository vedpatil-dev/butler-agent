"""butler's own record of cloud-agent conversations, so `/resume` works after butler is restarted.

claude, agy and codex each keep their own sessions on disk, but nothing says which of them were
started through butler, what they were about, or which model was used - and their stores are mixed
with everything else you did in those tools. So butler writes a small index of its own: one JSON file
per conversation with the session id to resume plus the text of each turn (for the recap on resume).

Plain files under the state dir, 0600 in a 0700 dir (they contain whatever you typed). Delete the
folder to forget everything. Local-model conversations are persisted separately by agent/memory.py.
"""
import json
import os
import re
import time
import uuid
from pathlib import Path

from . import config

DIR = config.STATE / "butler-history"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
CLAUDE_HIST = Path.home() / ".claude" / "history.jsonl"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
CODEX_INDEX = Path.home() / ".codex" / "session_index.jsonl"
CODEX_HIST = Path.home() / ".codex" / "history.jsonl"
AGY_BRAIN = Path.home() / ".gemini" / "antigravity-cli" / "brain"
MAX_TEXT = 20_000  # per stored message; a pasted log shouldn't make the file unbounded
_ID = re.compile(r"^([0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_USER_REQ = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.S)
_CMD_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.S)
_CLAUDE_NOISE = re.compile(
    r"<(ide_opened_file|ide_selection|system-reminder|local-command-caveat|local-command-stdout)>.*?</\1>|"
    r"</?pasted_content[^>]*>",
    re.S,
)
_SMOKE_PREFIXES = (
    "Reply with exactly: ok",
    "Reply with exactly: pong",
    "Remember: my favorite number is 42",
    "What is my favorite number? Reply with just the number",
    "What model are you? State only the model id",
    "Run the shell command `echo butler-probe`",
    "Use your ask_question tool to ask me whether I prefer tea or coffee",
)


def _is_smoke(turns: list[dict], model: str | None = None) -> bool:
    if model == "no-such-model-xyz" or not turns:
        return True
    u0 = (turns[0].get("user") or "").strip()
    return any(u0.startswith(p) for p in _SMOKE_PREFIXES)


def _file(conv_id: str, base: Path) -> Path:
    if not _ID.match(conv_id): raise ValueError(f"bad conversation id {conv_id!r}")  # ids become file names
    return base / f"{conv_id}.json"


def load(conv_id: str | None, base: Path | None = None) -> dict | None:
    try:
        return json.loads(_file(conv_id or "", base or DIR).read_text())
    except (OSError, ValueError):
        return None


def record_turn(conv_id: str | None, provider: str, model: str | None, session: str | None,
                user: str, reply: str, base: Path | None = None) -> str:
    """Append one exchange to conversation `conv_id` (created on first use) and return its id. The
    stored provider/model/session are the LATEST ones, which is what /resume needs."""
    base = base or DIR
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    conv = load(conv_id, base)
    if conv is None:
        first = next((l for l in user.strip().splitlines() if l.strip()), "(empty)")
        conv = {"id": uuid.uuid4().hex[:12], "title": first.strip()[:80], "created": time.time(), "turns": []}
    conv.update(provider=provider, model=model, session=session or conv.get("session"), updated=time.time())
    conv["turns"].append({"user": user[:MAX_TEXT], "reply": reply[:MAX_TEXT]})
    path = _file(conv["id"], base)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(conv))
    os.chmod(tmp, 0o600)
    tmp.replace(path)  # atomic: a crash mid-write can't leave a half-written conversation
    return conv["id"]


def _jsonl_sessions(path: Path, provider: str, sid_key: str, text_key: str, ts_key: str, ts_scale: float = 1.0, limit: int = 20) -> list[dict]:
    """Group a CLI's history.jsonl (claude or codex) by session id into resumable conversation records."""
    try: lines = path.read_text(errors="replace").splitlines()
    except OSError: return []
    by_sid: dict[str, dict] = {}
    for line in lines:
        try: d = json.loads(line)
        except ValueError: continue
        sid, text = str(d.get(sid_key) or "").strip(), str(d.get(text_key) or "").strip()
        if not sid or not text or text.startswith(("/", "[Pasted text #")): continue
        try: ts = float(d.get(ts_key) or 0) * ts_scale
        except (TypeError, ValueError): ts = 0.0
        rec = by_sid.get(sid)
        if rec is None:
            first = next((l.strip() for l in text.splitlines() if l.strip()), "(empty)")[:80]
            rec = by_sid[sid] = {"id": sid, "provider": provider, "model": None, "session": sid, "title": first, "updated": ts, "turns": []}
        rec["updated"] = max(rec["updated"], ts)
        rec["turns"].append({"user": text[:MAX_TEXT], "reply": ""})
    return [c for c in sorted(by_sid.values(), key=lambda c: -c["updated"]) if not _is_smoke(c["turns"])][:limit]


def _clean_claude_text(c: str) -> str:
    if "<command-name>" in c:
        m = _CMD_ARGS.search(c)
        c = m.group(1).strip() if m else ""
    return _CLAUDE_NOISE.sub("", c).strip()


def _claude_sessions(projects: Path, hist: Path, limit: int = 20) -> list[dict]:
    """Read Claude Code's full session transcripts from ~/.claude/projects/*/*.jsonl (falling back to history.jsonl)."""
    try: files = sorted((f for f in projects.glob("*/*.jsonl") if _UUID.match(f.stem)), key=lambda f: f.stat().st_mtime, reverse=True)
    except OSError: files = []
    out, seen = [], set()
    for f in files[:limit * 3]:
        if len(out) >= limit: break
        try:
            mtime = f.stat().st_mtime
            lines = f.read_text(errors="replace").splitlines()
        except OSError: continue
        turns, model, custom_title = [], None, ""
        for line in lines:
            if not line.strip(): continue
            try: o = json.loads(line)
            except ValueError: continue
            if not isinstance(o, dict): continue
            t = o.get("type")
            if t in ("custom-title", "summary"):
                custom_title = str(o.get("customTitle") or o.get("summary") or custom_title).strip()
            if t not in ("user", "assistant") or o.get("isCompactSummary"): continue
            msg = o.get("message")
            if not isinstance(msg, dict): continue
            if t == "assistant" and msg.get("model"): model = str(msg["model"])
            c = msg.get("content")
            if isinstance(c, list):
                c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if not isinstance(c, str): continue
            c = _clean_claude_text(c)
            if not c or c.startswith(("/", "<local-command", "<bash-", "Caveat:", "[Request interrupted", "This session is being continued")):
                continue
            if t == "user":
                turns.append({"user": c[:MAX_TEXT], "reply": ""})
            elif t == "assistant" and turns:
                turns[-1]["reply"] = c[:MAX_TEXT]
        if not _is_smoke(turns, model):
            first = custom_title or next((l.strip() for l in turns[0]["user"].splitlines() if l.strip()), "(empty)")
            out.append({"id": f.stem, "provider": "claude", "model": model, "session": f.stem, "title": first[:80], "updated": mtime, "turns": turns})
            seen.add(f.stem)
    for c in _jsonl_sessions(hist, "claude", "sessionId", "display", "timestamp", 0.001, limit):
        if c["session"] not in seen and len(out) < limit:
            out.append(c)
    return out


def _codex_sessions(sessions_dir: Path, index_path: Path, hist: Path, limit: int = 20) -> list[dict]:
    """Read Codex CLI's full rollout transcripts from ~/.codex/sessions/**/*.jsonl (+ session_index.jsonl)."""
    titles: dict[str, str] = {}
    try:
        for line in index_path.read_text(errors="replace").splitlines():
            try:
                o = json.loads(line)
                if isinstance(o, dict) and o.get("id") and o.get("thread_name"):
                    titles[str(o["id"])] = str(o["thread_name"]).strip()
            except ValueError: continue
    except OSError: pass
    try: files = sorted(sessions_dir.rglob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    except OSError: files = []
    out, seen = [], set()
    for f in files[:limit * 3]:
        if len(out) >= limit: break
        try:
            mtime = f.stat().st_mtime
            lines = f.read_text(errors="replace").splitlines()
        except OSError: continue
        sid, model, turns = None, None, []
        for line in lines:
            if not line.strip(): continue
            try: o = json.loads(line)
            except ValueError: continue
            if not isinstance(o, dict): continue
            t, pay = o.get("type"), o.get("payload")
            if not isinstance(pay, dict): continue
            if t == "session_meta":
                sid = str(pay.get("id") or pay.get("session_id") or sid or "").strip() or None
            elif t == "turn_context" and pay.get("model"):
                model = str(pay["model"])
            elif t == "response_item" and pay.get("type") == "message" and pay.get("role") in ("user", "assistant"):
                role = pay["role"]
                parts = [b.get("text", "") for b in (pay.get("content") or []) if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text")]
                txt = "\n".join(x for x in parts if x).strip()
                if not txt or txt.startswith(("# AGENTS.md", "<environment_context>", "<permissions", "<app-context>", "<subagent_notification>", "<turn_aborted>")):
                    continue
                if role == "user":
                    turns.append({"user": txt[:MAX_TEXT], "reply": ""})
                elif role == "assistant" and turns:
                    turns[-1]["reply"] = txt[:MAX_TEXT]
        if sid and _ID.match(sid) and not _is_smoke(turns, model):
            first = titles.get(sid) or next((l.strip() for l in turns[0]["user"].splitlines() if l.strip()), "(empty)")
            out.append({"id": sid, "provider": "codex", "model": model, "session": sid, "title": first[:80], "updated": mtime, "turns": turns})
            seen.add(sid)
    for c in _jsonl_sessions(hist, "codex", "session_id", "text", "ts", 1.0, limit):
        if c["session"] not in seen and len(out) < limit:
            out.append(c)
    return out


def _agy_sessions(brain: Path, limit: int = 20) -> list[dict]:
    """Read Antigravity's native conversation transcripts from ~/.gemini/antigravity-cli/brain/<uuid>."""
    try: dirs = sorted((d for d in brain.iterdir() if d.is_dir() and _UUID.match(d.name)), key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError: return []
    out = []
    for d in dirs:
        if len(out) >= limit: break
        t = d / ".system_generated" / "logs" / "transcript.jsonl"
        try:
            mtime = t.stat().st_mtime
            lines = t.read_text(errors="replace").splitlines()
        except OSError: continue
        turns = []
        for line in lines:
            try: e = json.loads(line)
            except ValueError: continue
            if not isinstance(e, dict): continue
            if e.get("type") == "USER_INPUT" and e.get("content"):
                raw = str(e["content"])
                m = _USER_REQ.search(raw)
                u = (m.group(1) if m else raw).strip()
                if u and not u.startswith("/"): turns.append({"user": u[:MAX_TEXT], "reply": ""})
            elif e.get("type") == "PLANNER_RESPONSE" and e.get("content") and turns:
                turns[-1]["reply"] = str(e["content"]).strip()[:MAX_TEXT]
        if not _is_smoke(turns):
            first = next((l.strip() for l in turns[0]["user"].splitlines() if l.strip()), "(empty)")[:80]
            out.append({"id": d.name, "provider": "antigravity", "model": None, "session": d.name, "title": first, "updated": mtime, "turns": turns})
    return out


def native_conversations(limit_per_provider: int = 20) -> list[dict]:
    """Conversations from claude (~/.claude/projects), codex (~/.codex/sessions), and antigravity (~/.gemini/antigravity-cli/brain)."""
    return (_claude_sessions(CLAUDE_PROJECTS, CLAUDE_HIST, limit_per_provider)
            + _codex_sessions(CODEX_SESSIONS, CODEX_INDEX, CODEX_HIST, limit_per_provider)
            + _agy_sessions(AGY_BRAIN, limit_per_provider))


def list_conversations(limit: int = 40, base: Path | None = None, provider: str | None = None) -> list[dict]:
    """Newest first. When `base` is None, merges butler's own cloud records with each cloud CLI's native history."""
    b = base or DIR
    try: files = sorted(b.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    except OSError: files = []
    out, seen_sessions = [], set()
    for f in files:
        try: c = json.loads(f.read_text())
        except (OSError, ValueError): continue
        if isinstance(c, dict) and _ID.match(str(c.get("id", ""))) and c.get("session"):  # nothing to resume without a session
            if provider and provider != "all" and c.get("provider") != provider: continue
            out.append(c)
            seen_sessions.add((c.get("provider"), c.get("session")))
    if base is None:
        for c in native_conversations(limit):
            if provider and provider != "all" and c.get("provider") != provider: continue
            if (c["provider"], c["session"]) not in seen_sessions:
                out.append(c)
    out.sort(key=lambda c: -float(c.get("updated") or 0))
    return out[:limit]


def ago(ts: float, now: float | None = None) -> str:
    s = int((now or time.time()) - ts)
    if s < 90: return "just now"
    for secs, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if s >= secs: return f"{s // secs}{unit} ago"
    return "just now"


def recap(turns: list[dict], n: int = 2, width: int = 150) -> list[tuple[str, str]]:
    """The last `n` exchanges as (speaker, one-line text) pairs, for showing what a resumed conversation was about."""
    out = []
    for t in turns[-n:]:
        out += [("you", clip(t.get("user", ""), width)), ("reply", clip(t.get("reply", ""), width))]
    return out


def clip(text: str, width: int = 150) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")
