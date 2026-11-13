"""Append-only record of every tool call and what was decided (STATE/audit.jsonl, 0600). Values are clipped so file
contents and long commands don't end up in the log."""
import json
import os
import time

from . import config

PATH = config.STATE / "audit.jsonl"


def log(tool: str, args: dict, level: str, decision: str, conversation: str = "") -> None:
    short = {k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()}
    line = json.dumps({"ts": round(time.time(), 1), "conversation": conversation, "tool": tool, "args": short,
                       "level": level, "decision": decision}, ensure_ascii=False, default=str)
    try:
        fd = os.open(PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f: f.write(line + "\n")
    except OSError: pass  # an unwritable log must not stop the agent
