"""Conversation memory inside a fixed token budget. Old tool output is shrunk first; if still over budget,
older turns are summarised by the local model. The system prompt, the current task and recent turns stay verbatim."""
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

from tools import clip

from . import config
from .llm import strip_think

SUMMARY_PROMPT = ("Summarise this conversation between a user and a coding agent in under 200 words. Keep: the user's "
                  "goal, decisions made, files read/changed, key command/test results, and what is still open. "
                  "Drop logs and chatter.\n\n")
HISTORY = config.STATE / "conversations"


def _private_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True, mode=0o700)  # transcripts hold whatever you typed and every tool result
    os.chmod(d, 0o700)


def recall(path: Path, query: str, limit: int = 3000) -> str:
    """Search a conversation's Markdown transcript (the full history, which compaction never shortens) and
    return the best-matching entries, oldest first. This is how the model "refetches" what was summarised
    away. ponytail: keyword-count ranking over the whole file, fine for one conversation; use an index if
    transcripts ever reach many MB."""
    try: entries = re.split(r"\n(?=## )", path.read_text())
    except OSError: return "no transcript yet"
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    scored = [(sum(e.lower().count(w) for w in words), i, e) for i, e in enumerate(entries)]
    hits = sorted((t for t in scored if t[0] > 0), reverse=True)[:6] if words else scored[-6:]
    if not hits: return f"nothing in the transcript matches {query!r}"
    return clip("\n\n".join(e.strip() for _, _, e in sorted(hits, key=lambda t: t[1])), limit)


def est_tokens(obj) -> int:
    n = len(json.dumps(obj, ensure_ascii=False)) // 3  # ponytail: chars/3 heuristic, conservative for code
    for m in obj if isinstance(obj, list) else [obj]:  # base64 images are not text tokens: a vision model spends ~1k on one
        if isinstance(m, dict): n -= sum(max(len(i) // 3 - 1000, 0) for i in m.get("images") or [])
    return max(n, 0)


class Memory:
    def __init__(self, system: str, num_ctx: int, tools_tokens: int = 0, cid: str | None = None, persist: bool = True, model: str = "", provider: str = ""):
        self.persist, self.num_ctx, self.tools_tokens, self.scale = persist, num_ctx, tools_tokens, 1.0
        self.system, self.summary, self.messages = system, "", []
        self.id = cid or uuid.uuid4().hex[:12]  # validated by the setter: it becomes a file name
        self.title, self.model, self.provider = "", model, provider  # model/provider: which local backend produced this conversation (shown in /resume)
        self.budget = max(1024, int(num_ctx * 0.85) - 1024 - tools_tokens)  # keep 1K for the reply

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        if not re.fullmatch(r"[\w-]{1,64}", value): raise ValueError(f"invalid conversation id {value!r}")
        self._id = value

    @property
    def md_path(self) -> Path:
        return HISTORY / f"{self.id}.md"

    def used(self) -> int:
        """Estimated prompt tokens, corrected by what the model server actually counted (see calibrate)."""
        return int(est_tokens(self.prompt()) * self.scale)

    def calibrate(self, real_prompt_tokens: int, estimated: int):
        """chars/3 is a guess; the server reports the true count. Blend the observed ratio in (clamped, so a
        cached-prefix report or a stub can't wreck the budget)."""
        if real_prompt_tokens and estimated:
            ratio = (real_prompt_tokens - self.tools_tokens) / estimated
            self.scale = min(2.0, max(0.5, 0.5 * self.scale + 0.5 * ratio))

    def context_pct(self) -> int:
        return min(100, round(100 * (self.used() + self.tools_tokens) / self.num_ctx))

    def add(self, msg: dict):
        if not self.title and msg["role"] == "user": self.title = msg["content"][:60]
        self.messages.append(msg)
        self._log(msg)

    def _log(self, msg: dict):
        """Append to the Markdown transcript on disk immediately (also survives a crash). Unlike self.messages
        this is never compacted, so the model can recall_history() anything the summary dropped."""
        if not self.persist: return
        body = msg.get("content") or json.dumps(msg.get("tool_calls", ""), ensure_ascii=False)
        body = re.sub(r"(?m)^## ", " ## ", clip(body, 6000))  # recall() splits entries on '## ' lines
        who = f"tool {msg.get('tool_name', '')}".strip() if msg["role"] == "tool" else msg["role"]
        _private_dir(HISTORY)
        fd = os.open(self.md_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)  # 0600 from creation, no race
        with os.fdopen(fd, "a") as f: f.write(f"## {who} · {time.strftime('%H:%M:%S')}\n{body}\n\n")

    def prompt(self) -> list:
        head = [{"role": "system", "content": self.system}]
        if self.summary:
            head.append({"role": "system", "content": "Summary of earlier conversation:\n" + self.summary +
                         "\n(Older turns were removed from this window to save memory. For exact details - a path, a "
                         "number, a quote - call recall_history with a few keywords.)"})
        return head + self.messages

    def compact(self, llm) -> bool:
        """Shrink until the prompt fits the budget. Returns True if anything changed."""
        if self.used() <= self.budget: return False
        shrunk = False
        for m in self.messages[:-6]:  # 1) discard bulk of old tool output
            if m["role"] == "tool" and len(m["content"]) > 300: m["content"], shrunk = clip(m["content"], 300), True
        if self.used() <= self.budget: return shrunk
        # 2) summarise old turns; the latest user message (the current task) is pinned verbatim
        cut = self._cut(keep=self.budget // 2)
        task_i = max((j for j, m in enumerate(self.messages) if m["role"] == "user"), default=-1)
        old = [m for j, m in enumerate(self.messages[:cut]) if j != task_i]
        if not old: return shrunk
        pinned = [self.messages[task_i]] if 0 <= task_i < cut else []
        text = (f"Earlier summary: {self.summary}\n" if self.summary else "") + "\n".join(
            f"{m['role']}: {clip(m.get('content') or json.dumps(m.get('tool_calls', '')), 500)}" for m in old)
        summary = strip_think(llm.complete([{"role": "user", "content": SUMMARY_PROMPT + text}], num_predict=400)["content"])
        self.messages, self.summary = pinned + self.messages[cut:], summary  # only after the summary exists: a failure loses nothing
        return True

    def _cut(self, keep: int) -> int:
        """Index from which messages stay verbatim: the newest ones within `keep` tokens, never splitting a tool exchange."""
        used, i = 0, len(self.messages)
        while i > 0 and used + est_tokens(self.messages[i - 1]) <= keep:
            i -= 1
            used += est_tokens(self.messages[i])
        while i < len(self.messages) and self.messages[i]["role"] == "tool": i += 1
        # the last assistant turn and its tool results are what the model must act on next: always keep them
        last_asst = max((j for j, m in enumerate(self.messages) if m["role"] == "assistant"), default=len(self.messages))
        return min(i, last_asst)

    def save(self):
        if not self.persist: return
        _private_dir(HISTORY)
        fd, tmp = tempfile.mkstemp(dir=HISTORY, suffix=".tmp")  # mkstemp is 0600; replace() is atomic
        with os.fdopen(fd, "w") as f:
            json.dump({"id": self.id, "title": self.title, "model": self.model, "provider": self.provider,
                       "updated": time.time(), "summary": self.summary, "messages": self.messages}, f)
        os.replace(tmp, HISTORY / f"{self.id}.json")

    @staticmethod
    def _infer_provider(c: dict) -> str:
        if p := c.get("provider"): return str(p)
        m = str(c.get("model") or "")
        return "npu" if m.startswith(("qwen3.5:2b", "qwen3.5:4b", "llama3.2:1b")) else "ollama"

    @staticmethod
    def list(provider: str | None = None) -> list:
        out = []
        for p in HISTORY.glob("*.json") if HISTORY.exists() else []:
            try:
                c = json.loads(p.read_text())
                prov = Memory._infer_provider(c)
                if provider and provider != "all" and prov != provider: continue
                out.append({k: c[k] for k in ("id", "title", "updated")} | {"model": c.get("model", ""), "provider": prov})
            except (OSError, ValueError, KeyError, TypeError): continue  # one bad file must not break the list
        return sorted(out, key=lambda c: -c["updated"])

    @staticmethod
    def load_raw(cid: str) -> dict:
        if not re.fullmatch(r"[\w-]{1,64}", cid): raise ValueError(f"invalid conversation id {cid!r}")
        return json.loads((HISTORY / f"{cid}.json").read_text())
