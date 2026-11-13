"""Alternative providers, selected with --provider. Both are duck-typed to the same shape LLM
(agent/llm.py) exposes - .stream()/.complete()/.healthy()/.status()/.num_ctx/.think - so Agent
(agent/orchestrator.py) runs unchanged on any of them. Default stays "ollama" (see config.PROVIDER)."""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx

from tools.filesystem import cwd

from . import attach, audit, config, permissions
from .llm import LLMError


class Provider(Protocol):
    """What Agent needs from a local model backend (LLM and NPUProvider both fit; nothing inherits from this).
    stream() yields ("text", str) | ("tool_calls", list) | ("stats", dict); complete() returns {"content", "tool_calls", "stats"}."""
    model: str
    num_ctx: int
    think: bool | None

    def stream(self, messages: list, tools: list | None = None, **opts): ...
    def complete(self, messages: list, **opts) -> dict: ...
    def healthy(self) -> bool: ...
    def status(self) -> dict: ...

NPU_MODEL = os.environ.get("AGENT_NPU_MODEL", "qwen3.5:2b")  # what `flm serve` has loaded (see flm-npu.service)
NPU_CTX = int(os.environ.get("AGENT_NPU_CTX", "65536"))  # must match `flm serve --ctx-len` in flm-npu.service (RSS: 32k 3.9 GB, 64k 4.1, 128k 4.8 for qwen3.5:2b)
NPU_MAX_TOKENS = int(os.environ.get("AGENT_NPU_MAX_TOKENS", "8192"))  # a 4B model can loop for minutes; 4096 tokens took 400s here
NPU_URL = os.environ.get("AGENT_NPU_URL", "http://127.0.0.1:52625")  # flm serve's default port (FastFlowLM, OpenAI-compatible)
OPENAI_URL = os.environ.get("AGENT_OPENAI_URL", "http://127.0.0.1:8080")  # generic local OpenAI-compat (OpenVINO, Genie/QNN, MLX, llama.cpp, vLLM)
OPENAI_MODEL = os.environ.get("AGENT_OPENAI_MODEL", "")
OPENAI_CTX = int(os.environ.get("AGENT_OPENAI_CTX", str(config.NUM_CTX)))


class OpenAICompatProvider:
    """Generic local OpenAI-compatible provider (/v1/chat/completions + /v1/models).
    Supports OpenVINO (Intel NPU), Qualcomm Genie/QNN (Snapdragon NPU), MLX (Apple Silicon),
    llama.cpp / vLLM (NVIDIA CUDA, AMD ROCm/Vulkan), and FastFlowLM (via NPUProvider)."""

    vision = False
    provider_name = "openai"
    runtime_name = "OpenAI-compatible (local)"

    def __init__(self, model=None, num_ctx=None, think=None, url=None, runtime_name=None):
        self.model, self.num_ctx = model if model is not None else OPENAI_MODEL, num_ctx or OPENAI_CTX
        self.think = None
        if runtime_name: self.runtime_name = runtime_name
        self.http = httpx.Client(base_url=url or OPENAI_URL, timeout=httpx.Timeout(600, connect=5))

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = self.http.get("/v1/models", timeout=3)
            if r.status_code == 200:
                ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
                if ids:
                    self.model = ids[0]
                    return self.model
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        raise LLMError(f"no model configured for {self.runtime_name} (set AGENT_OPENAI_MODEL or use /model)")

    def healthy(self) -> bool:
        try: return self.http.get("/v1/models", timeout=3).status_code == 200
        except httpx.HTTPError: return False

    @staticmethod
    def _openai_messages(messages):
        """butler keeps Ollama-style history; flm's chat template needs OpenAI-style tool turns (call id/type,
        string arguments, tool_call_id on the result) and answers with an {"error"} body otherwise."""
        out, pending = [], []
        for m in messages:
            m = dict(m)
            m.pop("images", None)
            if m.get("tool_calls"):
                calls = []
                for i, c in enumerate(m["tool_calls"]):
                    fn = dict(c["function"])
                    if not isinstance(fn.get("arguments"), str): fn["arguments"] = json.dumps(fn.get("arguments") or {})
                    calls.append({"id": c.get("id") or f"call_{len(out)}_{i}", "type": "function", "function": fn})
                m["tool_calls"], pending = calls, [(c["id"], c["function"]["name"]) for c in calls]
            elif m.get("role") == "tool":
                if not pending: continue  # orphan: compaction dropped its tool_calls turn; flm rejects a null tool_call_id
                tid, name = next(((i, n) for i, n in pending if n == m.get("tool_name")), pending[0] if pending else (None, None))
                pending = [x for x in pending if x[0] != tid]
                m.pop("tool_name", None)
                m.update(tool_call_id=tid, name=name)
            out.append(m)
        return out

    def _body(self, messages, stream, tools=None, **opts):
        body = {"model": self._resolve_model(), "messages": self._openai_messages(messages), "stream": stream}
        if tools: body["tools"] = tools
        body["max_tokens"] = opts.get("num_predict", NPU_MAX_TOKENS)
        return body

    def stream(self, messages, tools=None, **opts):
        try: yield from self._stream(messages, tools, **opts)
        except (httpx.HTTPError, ValueError) as e:  # unreachable / timed out / garbled reply: one error type for callers
            raise LLMError(f"{type(e).__name__}: {e}") from e

    def _stream(self, messages, tools=None, **opts):
        calls: dict[int, dict] = {}
        t0, n_out, usage, finish = time.time(), 0, {}, None
        with self.http.stream("POST", "/v1/chat/completions", json=self._body(messages, True, tools, **opts)) as r:
            if r.status_code != 200:
                r.read()
                raise LLMError(f"{r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if line.startswith('{"error"'): raise LLMError(line[:300])  # flm reports template errors as a bare 200 body
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                chunk = json.loads(line[6:])
                if chunk.get("error"): raise LLMError(str(chunk["error"])[:300])
                usage = chunk.get("usage") or usage
                choice = (chunk.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                if delta.get("content"):
                    n_out += 1
                    yield "text", delta["content"]
                for tc in delta.get("tool_calls") or []:  # OpenAI streams tool calls as fragments by index
                    c = calls.setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                    fn = tc.get("function", {})
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += fn.get("arguments") or ""
        if finish == "length":
            note = "output cut off at the token limit (AGENT_NPU_MAX_TOKENS)"
            if calls: note += "; the unfinished tool call was dropped - ask for a smaller piece (e.g. split the file)"
            calls = {}  # a tool call cut mid-JSON can't run
            yield "text", f"\n[{note}]"
        if calls: yield "tool_calls", list(calls.values())
        dt = time.time() - t0
        gen = usage.get("completion_tokens", n_out)  # flm's final chunk carries real token counts and timings
        pre, dec = usage.get("prefill_duration_ttft", 0.0), usage.get("decoding_duration", dt)
        yield "stats", {"prompt_tokens": usage.get("prompt_tokens", 0), "gen_tokens": gen, "prompt_s": pre, "gen_s": dec,
                        "load_s": 0.0, "total_s": dt, "prompt_tps": round(usage.get("prefill_speed_tps", 0), 1),
                        "gen_tps": round(usage.get("decoding_speed_tps") or (gen / dt if dt else 0), 1)}

    def complete(self, messages, fmt=None, tools=None, **opts) -> dict:
        try:
            r = self.http.post("/v1/chat/completions", json=self._body(messages, False, tools, **opts))
            if r.status_code != 200: raise LLMError(f"{r.status_code}: {r.text[:300]}")
            d = r.json()
            if d.get("error"): raise LLMError(str(d["error"])[:300])
            m = d["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if d["choices"][0].get("finish_reason") == "length" and m.get("tool_calls"):
            m = {**m, "tool_calls": [], "content": (m.get("content") or "") + "\n[output cut off at the token limit; the unfinished tool call was dropped]"}
        return {"content": m.get("content") or "", "tool_calls": [
            {"function": tc["function"]} for tc in m.get("tool_calls") or []], "stats": {}}

    def status(self) -> dict:
        from .llm import meminfo
        up = self.healthy()
        return {"model": self.model, "context": self.num_ctx, "runtime": self.runtime_name, "url": str(self.http.base_url).rstrip("/"),
                "ollama_up": up, "ram": meminfo(), "npu": "OpenAI-compatible endpoint",
                "acceleration": f"{self.runtime_name} ({'up' if up else 'server down'})"}


class NPUProvider(OpenAICompatProvider):
    """FastFlowLM (`flm serve`) on the XDNA2 NPU. Verified 2026-10-15 on this machine with qwen3.5:4b: streaming and
    non-streaming chat work and xrt-smi shows NPU submissions rising. Tool calling is not verified."""

    provider_name = "npu"
    runtime_name = "FastFlowLM (NPU)"

    def __init__(self, model=None, num_ctx=None, think=None, url=None):
        super().__init__(model=model or NPU_MODEL, num_ctx=num_ctx or NPU_CTX, think=think, url=url or NPU_URL)

    def status(self) -> dict:
        from .llm import meminfo, npu_state
        up = self.healthy()
        return {"model": self.model, "context": self.num_ctx, "runtime": self.runtime_name, "url": str(self.http.base_url).rstrip("/"),
                "ollama_up": up, "ram": meminfo(), "npu": npu_state().replace("; not used by Ollama", ""),
                "acceleration": "NPU via FastFlowLM (server up; confirm with xrt-smi submissions)" if up else "server down"}


NPU_UNIT = "flm-npu.service"  # user unit that runs `flm serve`; see README


def _services_enabled() -> bool:
    return not os.environ.get("AGENT_NO_SERVICES")  # tests set this so they never start/stop the real servers


def _systemctl(*args) -> bool:
    try: return subprocess.run(["systemctl", "--user", *args], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError): return False


def _npu_up() -> bool:
    p = NPUProvider()
    try: return p.healthy()
    finally: p.http.close()


def can_start_npu() -> bool:
    """True if the flm-npu user service exists and the NPU device is present."""
    unit = Path.home() / ".config/systemd/user" / NPU_UNIT
    return _services_enabled() and unit.exists() and Path("/dev/accel/accel0").exists()


def ensure_npu_server(wait: float = 45) -> bool:
    """True if the NPU server answers. If it doesn't but the flm-npu user unit exists, start it (the NPU is
    the default provider, so it should come up on its own) and wait for it."""
    if _npu_up(): return True
    if not can_start_npu(): return False
    if not _systemctl("start", NPU_UNIT): return False
    end = time.time() + wait
    while time.time() < end:
        if _npu_up(): return True
        time.sleep(1)
    return False


_NPU_BG = {"state": "idle", "error": ""}


def npu_bg_state() -> dict:
    """Current background NPU startup state: idle | loading | ready | failed."""
    return dict(_NPU_BG)


def start_npu_background(wait: float = 45) -> threading.Thread | None:
    """Start the NPU server on a daemon thread so the interactive UI opens immediately."""
    if _npu_up():
        _NPU_BG.update(state="ready", error="")
        return None
    if not can_start_npu():
        return None
    _NPU_BG.update(state="loading", error="")

    def _run():
        ok = ensure_npu_server(wait=wait)
        _NPU_BG.update(state="ready" if ok else "failed",
                       error="" if ok else f"{NPU_UNIT} did not respond within {wait}s")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def free_local_models(provider: str, model: str | None = None) -> list[str]:
    """Release RAM held by local models the chat is no longer using, so only the selected one stays resident:
    ollama models other than (provider, model) are unloaded; the NPU server is stopped when the chat moves
    to anything but the NPU (it reloads in seconds on return). Returns what was freed, for one status line."""
    if not _services_enabled(): return []
    freed = []
    try:
        for m in httpx.get(f"{config.LLM_URL}/api/ps", timeout=2).json().get("models", []):
            if provider == "ollama" and m["name"] == model: continue
            httpx.post(f"{config.LLM_URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
            freed.append(f"unloaded {m['name']} (ollama)")
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    if provider != "npu" and _npu_up() and _systemctl("stop", NPU_UNIT):
        freed.append("stopped the NPU server")
    return freed


def _run_external_agent(cmd: list[str], parse_line, broker=None):
    """Shared by AgyRunner/ClaudeRunner/CodexRunner: spawn `cmd` in the project dir, parse its NDJSON stdout line by
    line via `parse_line` (-> text/tool/_result event or None), yield the same event shapes Agent.run() does. The
    subprocess is always killed on early exit (Ctrl-C, the caller stopping iteration, a parser error) rather than
    leaving a detached cloud request running. `broker` (permissions.PermissionBroker) adds {"type": "permission"}
    events: the child asks the user through it, so stdout lines and those requests share one queue."""
    # stdin=DEVNULL: the child must never read the user's terminal (it would compete with butler's
    # own input box for keystrokes), and `claude` otherwise stalls ~3s printing "no stdin data
    # received" when stdin isn't a tty (verified).
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             cwd=str(cwd()))  # the calling context's project (-C), not a process-global, applies to cloud agents too
    except OSError as e:
        yield {"type": "error", "error": f"could not start {cmd[0]}: {e}"}
        yield {"type": "done", "content": ""}
        return
    err_parts: list[str] = []  # drained while running: an unread stderr pipe fills up and stalls the child
    drain = threading.Thread(target=lambda: err_parts.append(p.stderr.read()), daemon=True)
    drain.start()
    final, extra, errored = "", {}, False
    q = broker.q if broker else queue.Queue()
    try:
        try:
            threading.Thread(target=lambda: ([q.put(l) for l in p.stdout], q.put(None)), daemon=True).start()
            while (line := q.get()) is not None:
                if isinstance(line, dict):  # a permission request from the broker
                    yield line
                    continue
                ev = parse_line(line)
                if ev is None: continue
                if ev["type"] == "_result":
                    final, extra = ev["content"], {**extra, **{k: v for k, v in ev.items() if k not in ("type", "content")}}
                elif ev["type"] == "_meta":  # e.g. codex announces its thread id first, long before any result
                    extra.update({k: v for k, v in ev.items() if k != "type"})
                else:
                    errored = errored or ev["type"] == "error"
                    if ev["type"] == "tool": audit.log(ev.get("name", "?"), ev.get("args") or {}, "cloud", "delegated")  # the CLI ran it under its own permissions
                    yield ev
        except KeyboardInterrupt:
            p.kill()
            p.wait(timeout=5)
            yield {"type": "done", "content": "(stopped)", "stopped": True}
            return
        except (KeyError, AttributeError, TypeError) as e:  # the CLI's output format changed under the parser
            yield {"type": "error", "error": f"unexpected {cmd[0]} output ({type(e).__name__}: {e})"}
            yield {"type": "done", "content": final}  # the caller always gets a done, so the footer says "failed" instead of nothing
            return
        p.wait()
        drain.join(timeout=5)
        err = "".join(err_parts)
        if p.returncode != 0 and not final and not errored:  # a stream-reported error already said why
            yield {"type": "error", "error": f"{cmd[0]} exited {p.returncode}: {err[:800]}"}
        for d in extra.get("denied") or []: audit.log("permission", {"request": str(d)}, "cloud", "denied-by-cli")
        yield {"type": "done", "content": final, **extra}  # extra: "session" (to resume next turn), "denied"
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


# butler's --approval mode -> each cloud CLI's own permission flags (all verified against `--help`). None = add no flags
# (the user's own config decides). Print mode cannot prompt, so "ask" means "no unattended escalation".
APPROVAL_FLAGS = {
    "claude": {"ask": ["--permission-mode", "manual"], "auto-edit": ["--permission-mode", "acceptEdits"], "readonly": ["--permission-mode", "plan"]},
    "codex": {"ask": ["-s", "workspace-write"], "auto-edit": ["-s", "workspace-write"], "readonly": ["-s", "read-only"]},
    "antigravity": {"ask": [], "auto-edit": [], "readonly": ["--sandbox"]},
}


def approval_flags(provider: str, approval: str | None) -> list[str]:
    return APPROVAL_FLAGS.get(provider, {}).get(approval or "", [])


class ExternalAgent(Protocol):
    """What butler needs from a cloud agent CLI (AgyRunner/ClaudeRunner/CodexRunner fit; nothing inherits from this).
    Why: callers used to special-case provider names (`if provider == "claude"`, EFFORT_AGENTS) for the two
    capabilities that differ; they are now class attributes, so a new runner declares them instead of editing main.py.
    Impact: every runner is constructed the same way; unsupported options are accepted and ignored.
    run() yields the same events as Agent.run(); a runner with can_prompt may also yield {"type": "permission"}."""
    name: str
    supports_effort: bool  # has its own reasoning-effort setting (agy bakes effort into model ids; codex's differs per model)
    can_prompt: bool  # can relay tool-permission requests to the user (rendered by confirm.permission_prompt)

    def __init__(self, model: str | None = None, session: str | None = None, approval: str | None = None,
                 effort: str | None = None, prompt: bool = False): ...
    def healthy(self) -> bool: ...
    def interactive_cmd(self) -> list[str]: ...
    def run(self, task: str, files=()): ...


class AgyRunner:
    """Not a Provider: Antigravity's `agy` CLI is itself a full agent with its own tools (cloud
    models, e.g. Gemini/Claude via Google's backend - NOT local). Selecting it hands the whole
    task to `agy -p`, bypassing this project's Agent/orchestrator and tools entirely."""
    name = "antigravity"

    supports_effort = can_prompt = False

    def __init__(self, model: str | None = None, session: str | None = None, approval: str | None = None,
                 effort: str | None = None, prompt: bool = False):  # effort/prompt: ExternalAgent signature, unused here
        self.model, self.session, self.approval = model, session, approval

    def healthy(self) -> bool:
        return shutil.which("agy") is not None

    def cmd(self, task: str, files=()) -> list[str]:
        task, _, dirs = attach.cloud(task, list(files))  # agy has no image flag: it reads the listed paths itself
        # The model goes through agy's --model flag: its native /model is read-only in print mode
        # ("takes no arguments" - verified). There is deliberately NO effort parameter: agy bakes the
        # effort into its model ids (gemini-4-argon-low/-medium/-high), and --effort alongside such an
        # id is rejected ("--model X conflicts with --effort=Y", verified; xhigh/max don't exist for
        # it either) - so effort for agy IS the model choice.
        task = " " + task if task.startswith("-") else task
        cmd = ["agy", "-p", task, "--output-format", "stream-json"]
        if self.model: cmd += ["--model", self.model]
        cmd += approval_flags("antigravity", self.approval)
        for d in dirs: cmd += ["--add-dir", d]
        # every butler message is a separate `agy -p` process with no memory of the last one, so
        # without this a follow-up answer ("tea") reaches an agent that never asked anything
        # (verified: "unknown" without --conversation, "42" with it)
        if self.session: cmd += ["--conversation", self.session]
        return cmd

    def interactive_cmd(self) -> list[str]:
        """Interactive TUI command (no -p) so native slash commands and pickers work."""
        cmd = ["agy"]
        if self.model: cmd += ["--model", self.model]
        cmd += approval_flags("antigravity", self.approval)
        if self.session: cmd += ["--conversation", self.session]
        return cmd

    def run(self, task: str, files=()):
        t0 = time.time()
        for ev in _run_external_agent(self.cmd(task, files), parse_agy_line):
            if ev["type"] == "done":
                for q in agy_skipped_questions(ev.get("session"), t0): yield {"type": "ask", **q}
            yield ev


class ClaudeRunner:
    """Not a Provider: Claude Code itself as a selectable agent, same idea as AgyRunner - hands
    the whole task to `claude -p`, its own full agent with its own tools, bypassing this
    project's Agent/orchestrator. Cloud, not local. Unlike agy, claude's slash commands are NOT
    available in print mode (verified: `claude -p "/help"` replies "/help isn't available in this
    environment"), so there's no native-command passthrough to wire for this one - this project's
    own /help, /provider, /model apply normally while this provider is selected."""
    name = "claude"

    supports_effort = can_prompt = True

    def __init__(self, model: str | None = None, effort: str | None = None, session: str | None = None,
                 approval: str | None = None, prompt: bool = False):  # positional order kept: callers/tests pass (model, effort)
        # prompt: a human is at the terminal, so permission requests are asked there instead of being denied
        self.model, self.effort, self.session, self.approval, self.prompt = model, effort, session, approval, prompt

    def healthy(self) -> bool:
        return shutil.which("claude") is not None

    def cmd(self, task: str, broker=None, files=()) -> list[str]:
        task, _, dirs = attach.cloud(task, list(files))  # claude's Read tool opens images and PDFs from a path
        task = " " + task if task.startswith("-") else task
        cmd = ["claude", "-p", task, "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
        if self.model: cmd += ["--model", self.model]
        if self.effort: cmd += ["--effort", self.effort]
        cmd += approval_flags("claude", self.approval)
        for d in dirs: cmd += ["--add-dir", d]
        # claude has no ask-the-user tool in print mode (verified: it says so and asks in plain
        # text instead), so a follow-up is just your next message - --resume is what lets it land
        # in the same conversation instead of a brand-new one
        if self.session: cmd += ["--resume", self.session]
        if broker:
            script = str(Path(__file__).resolve().parents[1] / "mcp" / "permission_server.py")
            cmd += ["--mcp-config", broker.mcp_config(sys.executable, script), "--permission-prompt-tool", permissions.TOOL]
        return cmd

    def interactive_cmd(self) -> list[str]:
        """Interactive TUI command (no -p) so native slash commands and pickers work."""
        cmd = ["claude"]
        if self.model: cmd += ["--model", self.model]
        if self.effort: cmd += ["--effort", self.effort]
        cmd += approval_flags("claude", self.approval)
        if self.session: cmd += ["--resume", self.session]
        return cmd

    def run(self, task: str, files=()):
        if not (self.prompt and self.approval in ("ask", "auto-edit")):  # readonly (plan) never needs to ask
            yield from _run_external_agent(self.cmd(task, files=files), parse_claude_line)
            return
        broker = permissions.PermissionBroker()
        try: yield from _run_external_agent(self.cmd(task, broker, files), parse_claude_line, broker)
        finally: broker.close()


class CodexRunner:
    """Not a Provider: OpenAI's `codex` CLI as a selectable agent, same idea as AgyRunner/ClaudeRunner -
    the whole task goes to `codex exec`, which runs its own agent loop with its own tools and
    sandbox (butler passes -s from --approval; see APPROVAL_FLAGS).
    Cloud, not local. Models come from `codex debug models`. No effort setting is wired: codex's
    reasoning levels differ per model (some add "ultra", gpt-5.5 stops at xhigh)."""
    name = "codex"

    supports_effort = can_prompt = False

    def __init__(self, model: str | None = None, session: str | None = None, approval: str | None = None,
                 effort: str | None = None, prompt: bool = False):  # effort/prompt: ExternalAgent signature, unused here
        self.model, self.session, self.approval = model, session, approval

    def healthy(self) -> bool:
        return shutil.which("codex") is not None

    def cmd(self, task: str, files=()) -> list[str]:
        task, images, dirs = attach.cloud(task, list(files))
        task = " " + task if task.startswith("-") else task  # else a message like "- fix it" parses as a flag
        flags = ["--json", "--skip-git-repo-check"] + (["-m", self.model] if self.model else []) + approval_flags("codex", self.approval)
        # forms below are the ones verified against the real CLI; `resume <thread_id>` is what lets a
        # follow-up land in the same conversation (verified: recalls a fact; a fresh `exec` says "unknown")
        for d in dirs: flags += ["--add-dir", d] if not self.session else []  # `exec resume` has no --add-dir
        imgs = [x for i in images for x in ("-i", i)]  # -i is variadic, so the task goes before it
        if self.session: return ["codex", "exec", "resume", self.session, task] + flags + imgs
        if imgs: return ["codex", "exec", task] + flags + imgs
        return ["codex", "exec"] + flags + [task]

    def interactive_cmd(self) -> list[str]:
        """Interactive TUI command (no exec) so native slash commands and pickers work."""
        flags = (["-m", self.model] if self.model else []) + approval_flags("codex", self.approval)
        if self.session: return ["codex", "resume", self.session] + flags
        return ["codex"] + flags

    def run(self, task: str, files=()):
        yield from _run_external_agent(self.cmd(task, files), parse_codex_line)


def parse_agy_line(line: str) -> dict | None:
    """One NDJSON line from `agy --output-format stream-json` -> a text/tool/_result event, or
    None to skip. Pulled out of AgyRunner.run() so the parsing logic is unit-testable without
    spawning the real CLI."""
    try: ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    su = ev.get("step_update") or {}
    if ev.get("event") == "step_update" and su.get("step_type") == "agent_response" and su.get("text_delta"):
        return {"type": "text", "delta": su["text_delta"]}
    if ev.get("event") == "step_update" and su.get("step_type") in ("tool", "tool_call") and su.get("state") != "DONE":
        # verified: step_type "tool", sent ACTIVE then DONE (same step_index); show it once, when it starts
        name, params = su.get("tool_name", "?"), (su.get("tool_info") or {}).get("parameters") or {}
        what = next((str(v) for v in params.values() if isinstance(v, str)), "")  # e.g. run_command -> CommandLine
        return {"type": "tool", "name": name, "args": {}, "text": f"agy: {name} {what[:100]}".strip()}
    if ev.get("event") == "result":
        return {"type": "_result", "content": ev["result"].get("response", ""),
                "session": ev["result"].get("conversation_id") or None}  # "" for slash commands
    return None


def parse_codex_line(line: str) -> dict | None:
    """One JSONL line from `codex exec --json` -> text/tool/error/_meta event, or None to skip.
    Event names verified against the real CLI: thread.started{thread_id}, item.started/completed
    {item:{type: agent_message|command_execution|error(=a non-fatal warning)}}, turn.completed,
    and on failure `error` then `turn.failed` (same message twice, so only turn.failed is used)."""
    try: ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    t, item = ev.get("type"), ev.get("item") or {}
    if t == "thread.started": return {"type": "_meta", "session": ev.get("thread_id")}
    if t == "item.completed" and item.get("type") == "agent_message":
        return {"type": "text", "delta": item.get("text", "") + "\n"}
    if t == "item.started" and item.get("type") not in ("agent_message", "error", "reasoning"):
        cmd = item.get("command")  # "/bin/bash -lc 'echo hi'" -> "echo hi"
        what = re.sub(r"^\S*bash -lc '?(.*?)'?$", r"\1", cmd) if cmd else item.get("type", "?")
        return {"type": "tool", "name": item.get("type", "?"), "args": {}, "text": f"codex: {what[:100]}"}
    if t == "turn.failed":
        msg = (ev.get("error") or {}).get("message", "turn failed")
        try: msg = json.loads(msg)["error"]["message"]  # the message is itself a JSON document
        except (ValueError, KeyError, TypeError): pass
        return {"type": "error", "error": str(msg)}
    return None


def _denial_summary(d: dict) -> str:
    """One entry of claude's result["permission_denials"] ({"tool_name", "tool_use_id", "tool_input"} -
    shape verified against a real denied Write) -> e.g. "Write /tmp/x.txt" / "Bash git push"."""
    inp = d.get("tool_input") or {}
    detail = inp.get("file_path") or inp.get("command") or inp.get("path") or inp.get("url") or ""
    return f"{d.get('tool_name', '?')} {str(detail)[:80]}".strip()


def parse_claude_line(line: str) -> dict | None:
    """One NDJSON line from `claude -p --output-format stream-json --verbose` -> a text/tool/
    _result event, or None to skip (system/hook/init noise - claude's stream-json carries the
    session's own hook/skill config as events, unlike agy's leaner stream). Without
    --include-partial-messages claude only emits whole messages (no word-by-word); with it, text arrives as
    stream_event text_deltas and the final "assistant" text is skipped as a duplicate."""
    try: ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    if ev.get("type") == "stream_event":  # --include-partial-messages: token deltas; the "assistant" event repeats this text
        d = (ev.get("event") or {}).get("delta") or {}
        return {"type": "text", "delta": d["text"]} if d.get("type") == "text_delta" and "text" in d else None
    if ev.get("type") == "assistant":
        blocks = ev["message"].get("content", [])
        tools = [b for b in blocks if b.get("type") == "tool_use"]
        if tools:
            inp = tools[0].get("input") or {}
            what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("path") or inp.get("url") or ""
            return {"type": "tool", "name": tools[0].get("name", "?"), "args": {}, "text": f"claude: {tools[0].get('name', '?')} {str(what).splitlines()[0][:100] if what else ''}".strip()}
        return None
    if ev.get("type") == "result":
        denied = [_denial_summary(d) for d in ev.get("permission_denials") or [] if isinstance(d, dict)]
        return {"type": "_result", "content": ev.get("result", ""), "session": ev.get("session_id"), "denied": denied}
    return None


AGY_BRAIN = Path.home() / ".gemini" / "antigravity-cli" / "brain"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def agy_skipped_questions(conversation_id: str | None, since: float, brain: Path = AGY_BRAIN) -> list[dict]:
    """agy's ask_question tool is auto-skipped in print mode (the agent just replies "You skipped the
    question") and its stream-json never carries the question itself - the step arrives as an opaque
    step_type "unknown". The tool call (question + options) IS in agy's own on-disk transcript, so
    recover it from there: [{"question", "options"}] for calls made at/after `since` (epoch secs).
    Best-effort by design - that's an internal, undocumented file format, so any problem (missing
    file, schema change, odd line) just yields fewer/no questions and the user sees agy's reply as before."""
    if not conversation_id or not _UUID.match(conversation_id): return []  # also keeps the path safe
    out = []
    try: lines = (brain / conversation_id / ".system_generated" / "logs" / "transcript.jsonl").read_text().splitlines()
    except OSError: return []
    for line in lines:
        try:
            e = json.loads(line)
            if datetime.fromisoformat(e["created_at"].replace("Z", "+00:00")).timestamp() < since - 2: continue
            for tc in e.get("tool_calls") or []:
                if tc.get("name") != "ask_question": continue
                qs = tc["args"]["questions"]
                for q in json.loads(qs) if isinstance(qs, str) else qs:
                    out.append({"question": q["question"], "options": list(q.get("options") or [])})
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return out


EXTERNAL_AGENTS = {"antigravity": AgyRunner, "claude": ClaudeRunner, "codex": CodexRunner}


def make_provider(name: str, **kw):
    if name == "ollama":
        from .llm import LLM
        return LLM(**kw)
    if name == "npu":
        return NPUProvider(**kw)
    if name == "openai":
        return OpenAICompatProvider(**kw)
    raise ValueError(f"unknown provider: {name!r} (ollama|npu|openai; "
                     f"{'/'.join(EXTERNAL_AGENTS)} are run separately, see EXTERNAL_AGENTS)")
