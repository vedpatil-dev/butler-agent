"""Conversation bookkeeping for the chat loop: recording turns, rebuilding transcripts, listing earlier
conversations and carrying context across a provider switch. Pure helpers (no provider switching), split out of main.py."""
import sys

from . import history
from .render import BOLD, CYAN, DIM, RESET, _CTRL


def _after_cloud_turn(a, task: str, done: dict) -> None:
    """Keep the session id (so the next message continues the conversation) and write the exchange to
    butler's history (so /resume works after a restart). A turn with no reply - an error - isn't recorded."""
    a.session = done.get("session") or a.session
    if done.get("reply") and not done.get("stopped"):
        a.conv_id = history.record_turn(a.conv_id, a.provider, a.model, a.session, task, done["reply"])
        turns = list(getattr(a, "bridge_turns", None) or [])
        turns.append((task, done["reply"]))
        a.bridge_turns = turns[-6:]
        a.carryover = None


def _turns_from_messages(messages: list[dict]) -> list[dict]:
    """Group a local Agent's raw message list into (user, final assistant reply, tools used) turns."""
    turns: list[dict] = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "user" and content:
            turns.append({"user": content, "reply": "", "tools": []})
        elif role == "assistant" and turns:
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name")
                if fn: turns[-1]["tools"].append(fn)
            if content:
                turns[-1]["reply"] = content
    return turns


def _render_transcript(provider: str, model: str | None, title: str, turns: list[dict]) -> None:
    """Clear the viewport and render the resumed conversation's turns with full Markdown formatting."""
    from .ui import MdStream
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(f"{DIM}resumed: {title}  ({provider}, {model or 'default model'}){RESET}")
    if len(turns) > 10:
        print(f"{DIM}  ... {len(turns) - 10} earlier turns in this conversation ...{RESET}")
    for t in turns[-10:]:
        u = _CTRL.sub("", (t.get("user") or "").strip())
        r = _CTRL.sub("", (t.get("reply") or "").strip())
        tools = t.get("tools") or []
        if u:
            u_lines = u.splitlines()
            body = "\n".join(u_lines[:10]) + (f"\n{DIM}... (+{len(u_lines) - 10} lines){RESET}" if len(u_lines) > 10 else "")
            print(f"\n{BOLD}{CYAN}❯ you:{RESET} {body}")
        if tools:
            print(f"{DIM}  → {', '.join(dict.fromkeys(tools))}{RESET}")
        if r:
            md = MdStream()
            print(md.feed(r + "\n") + md.flush(), end="")
    if turns:
        print(f"{DIM}{'─' * 60}{RESET}")


def _conversation_items(provider: str | None = None) -> list:
    """Earlier conversations as (kind, record, updated), newest first: butler's + native cloud ones
    and the local models'. If `provider` is given (and not 'all'), only that provider's conversations are returned."""
    from .memory import Memory
    from .providers import EXTERNAL_AGENTS
    filt = None if provider in (None, "", "all") else provider
    cloud = [("cloud", c, c["updated"]) for c in history.list_conversations(provider=filt)] if filt is None or filt in EXTERNAL_AGENTS else []
    local = [("local", m, m["updated"]) for m in Memory.list(provider=filt)[:40]] if filt is None or filt not in EXTERNAL_AGENTS else []
    return sorted(cloud + local, key=lambda it: it[2], reverse=True)


def _extract_turns(a, agent) -> list[tuple[str, str]]:
    """Recent (user, assistant) exchange pairs from the active conversation so switching providers mid-chat keeps context."""
    if agent is not None:
        return [(t["user"], t["reply"]) for t in _turns_from_messages(agent.memory.messages)[-6:]]
    if getattr(a, "conv_id", None):
        c = history.load(a.conv_id)
        if c and c.get("turns"):
            return [(t.get("user", ""), t.get("reply", "")) for t in c["turns"][-6:]]
    return list(getattr(a, "bridge_turns", None) or [])[-6:]


def _format_carryover(old_prov: str, turns: list[tuple[str, str]]) -> str:
    lines = [f"[Context carried over from earlier turns in this session ({old_prov}):]"]
    for u, r in turns[-6:]:
        if u: lines.append(f"User: {history.clip(u, 400)}")
        if r: lines.append(f"Assistant: {history.clip(r, 600)}")
    lines.append("[Continue the conversation naturally from the context above.]")
    return "\n".join(lines)
