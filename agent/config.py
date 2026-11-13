"""Settings from environment variables (optionally loaded from .env in the project root)."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def parse_env(text: str) -> dict:
    """KEY=value lines; `# comment` lines and trailing ` # comments` (outside quotes) are ignored."""
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition("=")
        if not sep or k.lstrip().startswith("#"): continue
        v = v.strip()
        if v[:1] in "\"'" and v.count(v[0]) >= 2: v = v[1:v.index(v[0], 1)]
        else: v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        out[k.strip()] = v
    return out


if (ROOT / ".env").exists():
    for k, v in parse_env((ROOT / ".env").read_text()).items(): os.environ.setdefault(k, v)

LLM_URL = os.environ.get("AGENT_LLM_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("AGENT_MODEL", "qwen3.5:9b")
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "8192"))
THINK = os.environ.get("AGENT_THINK", "false").lower()  # false | true | default (let the model decide)
if THINK not in ("false", "true", "default"): raise ValueError(f"AGENT_THINK must be false|true|default, got {THINK!r}")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "20"))
PORT = int(os.environ.get("AGENT_PORT", "8000"))
NUM_GPU = os.environ.get("AGENT_NUM_GPU")  # unset = Ollama decides; 0 = force CPU
MCP_CONFIG = Path(os.environ.get("AGENT_MCP_CONFIG", ROOT / "config" / "mcp.json"))
STATE = Path(os.environ.get("AGENT_STATE_DIR") or Path.home() / ".local" / "state" / "local-agent")  # conversations, history, logs, pid
STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
