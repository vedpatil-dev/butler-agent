"""Thin client for the local Ollama HTTP API (chat, streaming, tool calls, structured output, status)."""
import json
import re
from pathlib import Path

import httpx

from . import config


class LLMError(RuntimeError):
    pass


def meminfo() -> dict:
    m = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())}
    return {"total_gb": round(m["MemTotal"] / 2**20, 1), "available_gb": round(m["MemAvailable"] / 2**20, 1)}


class LLM:
    def __init__(self, url=None, model=None, num_ctx=None, think=None, num_gpu=None):
        self.model = model or config.MODEL
        self.num_ctx = num_ctx or config.NUM_CTX
        self.think = {"false": False, "true": True}.get(think or config.THINK)
        self.options = {"num_ctx": self.num_ctx, "temperature": 0.6, "top_p": 0.9, "top_k": 20}
        ng = num_gpu if num_gpu is not None else config.NUM_GPU
        if ng is not None: self.options["num_gpu"] = int(ng)
        self.http = httpx.Client(base_url=url or config.LLM_URL, timeout=httpx.Timeout(600, connect=5))

    def _body(self, messages, stream, tools=None, fmt=None, **opts):
        body = {"model": self.model, "messages": messages, "stream": stream, "keep_alive": "30m",
                "options": {**self.options, **opts}}
        if self.think is not None: body["think"] = self.think
        if tools: body["tools"] = tools
        if fmt: body["format"] = fmt
        return body

    def healthy(self) -> bool:
        try: return self.http.get("/api/version", timeout=3).status_code == 200
        except httpx.HTTPError: return False

    def stream(self, messages, tools=None, **opts):
        """Yield ("text", str) | ("tool_calls", list) | ("stats", dict). Reasoning tokens are never surfaced."""
        calls = []
        try:
            with self.http.stream("POST", "/api/chat", json=self._body(messages, True, tools, **opts)) as r:
                if r.status_code != 200:
                    r.read()
                    raise LLMError(f"{r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line: continue
                    d = json.loads(line)
                    if "error" in d: raise LLMError(d["error"])
                    m = d.get("message", {})
                    if m.get("content"): yield "text", m["content"]
                    calls += m.get("tool_calls") or []
                    if d.get("done"): yield "stats", _stats(d)
        except (httpx.HTTPError, ValueError) as e:  # unreachable / timed out / garbled reply: one error type for callers
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if calls: yield "tool_calls", calls

    def complete(self, messages, fmt=None, tools=None, **opts) -> dict:
        """Non-streaming call. Returns {"content", "tool_calls", "stats"}. fmt = "json" or a JSON schema."""
        try:
            r = self.http.post("/api/chat", json=self._body(messages, False, tools, fmt, **opts))
            if r.status_code != 200: raise LLMError(f"{r.status_code}: {r.text[:300]}")
            d = r.json()
            m = d["message"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        return {"content": m.get("content", ""), "tool_calls": m.get("tool_calls") or [], "stats": _stats(d)}

    def status(self) -> dict:
        """Model / backend / context / RAM / acceleration, from what Ollama and the kernel actually report."""
        s = {"model": self.model, "context": self.num_ctx, "runtime": "ollama", "ollama_up": self.healthy(),
             "ram": meminfo(), "npu": npu_state()}
        if not s["ollama_up"]: return s
        try:
            info = self.http.post("/api/show", json={"model": self.model}).json().get("details", {})
            s.update(quantization=info.get("quantization_level"), params=info.get("parameter_size"))
        except httpx.HTTPError: pass
        try: running = self.http.get("/api/ps").json().get("models", [])
        except (httpx.HTTPError, ValueError): running = []
        for m in running:
            if m["name"] == self.model:
                pct = round(100 * m.get("size_vram", 0) / m["size"]) if m["size"] else 0
                s.update(loaded=True, model_ram_gb=round(m["size"] / 2**30, 2), gpu_offload_pct=pct,
                         acceleration=f"iGPU (Vulkan) {pct}% of model" if pct else "CPU only")
        s.setdefault("loaded", False)
        return s


def _stats(d: dict) -> dict:
    pe, ge = d.get("prompt_eval_duration", 0) / 1e9, d.get("eval_duration", 0) / 1e9
    return {"prompt_tokens": d.get("prompt_eval_count", 0), "gen_tokens": d.get("eval_count", 0),
            "prompt_s": pe, "gen_s": ge, "load_s": d.get("load_duration", 0) / 1e9,
            "total_s": d.get("total_duration", 0) / 1e9,
            "prompt_tps": round(d.get("prompt_eval_count", 0) / pe, 1) if pe else 0,
            "gen_tps": round(d.get("eval_count", 0) / ge, 1) if ge else 0}


def npu_state() -> str:
    """The XDNA NPU is only reported as detected; Ollama has no NPU backend, so it is never 'used'."""
    dev = Path("/dev/accel/accel0")
    st = Path("/sys/class/accel/accel0/device/power/runtime_status")
    if not dev.exists(): return "not detected"
    return f"detected ({dev}, {st.read_text().strip() if st.exists() else '?'}); not used by Ollama"


def parse_tool_args(a) -> dict:
    """Tool arguments as a dict. Models send "" for no arguments, or cut-off / non-object JSON: all become {} (the
    tool then fails with a clear missing-argument error the model can react to)."""
    if isinstance(a, str):
        try: a = json.loads(a) if a.strip() else {}
        except ValueError: return {}
    return a if isinstance(a, dict) else {}


def strip_think(s: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
