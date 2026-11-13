"""Model lists from every provider (one catalog for /model) and resolving what the user typed."""
import json
import shutil
import subprocess
import threading

from . import config

PROVIDER_NAMES = ("ollama", "npu", "openai", "antigravity", "claude", "codex")


# claude has no model-list command (`claude models` isn't a thing), so this is a curated list: ids
# are the ones Claude Code's own ~/.claude.json carries, labels follow its /model picker. The three
# aliases always resolve to the latest of that family. Order = newest first. It WILL go stale as new
# models ship - the picker also accepts any typed id, and a bad id errors visibly ("unrecognized_model",
# verified) rather than silently falling back.
CLAUDE_MODELS = [
    ("opus", "Opus (alias: latest)"), ("sonnet", "Sonnet (alias: latest)"), ("fable", "Fable (alias: latest)"),
    ("claude-fable-5-1", "Fable 5.1"), ("claude-fable-5", "Fable 5 - most capable, hardest/longest tasks"),
    ("claude-opus-5-5", "Opus 5.5"), ("claude-opus-5", "Opus 5 - everyday complex tasks"),
    ("claude-sonnet-5-5", "Sonnet 5.5"), ("claude-sonnet-5", "Sonnet 5 - efficient for routine tasks"),
    ("claude-haiku-5-5", "Haiku 5.5 - fastest, quick answers"), ("claude-haiku-4-5", "Haiku 4.5"),
    ("claude-opus-4-8", "Opus 4.8"), ("claude-opus-4-7", "Opus 4.7"), ("claude-opus-4-6", "Opus 4.6"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
]


def _list_agy_models() -> list[tuple[str, str]]:
    try:
        r = subprocess.run(["agy", "models"], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
        return [tuple(l.split("\t", 1)) for l in r.stdout.strip().splitlines() if "\t" in l]
    except (OSError, subprocess.TimeoutExpired):
        return []


def _list_ollama_models() -> list[tuple[str, str]]:
    """From the same endpoint butler talks to (config.LLM_URL), NOT the `ollama list` CLI - that one
    queries the default port, which can be a different (running) instance than butler's (down) one,
    so a model could be listed that then can't actually be used."""
    import httpx
    try:
        r = httpx.get(f"{config.LLM_URL}/api/tags", timeout=3)
        return [(m["name"], "") for m in r.json()["models"]] if r.status_code == 200 else []
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def _list_npu_models() -> list[tuple[str, str]]:
    """FastFlowLM models that are downloaded (`flm list` also shows ~40 not-yet-pulled ones, and picking
    one of those would silently start a multi-GB download). Works with the server stopped (it is stopped on other providers)."""
    flm = shutil.which("flm") or "/opt/fastflowlm/bin/flm"
    try:
        out = subprocess.run([flm, "list", "--filter", "installed", "--quiet"], capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [(l.strip()[2:].strip(), "FastFlowLM, NPU") for l in out.splitlines() if l.strip().startswith("- ")]


def _list_openai_models() -> list[tuple[str, str]]:
    """Query the local OpenAI-compatible server (AGENT_OPENAI_URL, e.g. OpenVINO, MLX, Genie, llama.cpp, vLLM)."""
    import httpx
    from .providers import OPENAI_URL
    try:
        r = httpx.get(f"{OPENAI_URL}/v1/models", timeout=3)
        if r.status_code != 200: return []
        return [(m["id"], "OpenAI-compat") for m in r.json().get("data", []) if m.get("id")]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return []


def _list_codex_models() -> list[tuple[str, str]]:
    """`codex debug models` is codex's own model catalog (JSON, instant - it reads the local cache).
    Only the ones codex itself lists in its picker (visibility "list"), best first; hidden/internal
    ones stay reachable by typing `/model codex/<slug>`."""
    try:
        r = subprocess.run(["codex", "debug", "models"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=15)
        models = sorted((m for m in json.loads(r.stdout)["models"] if m.get("visibility") == "list"),
                        key=lambda m: m.get("priority", 99))
        return [(m["slug"], m.get("display_name") or "") for m in models]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return []


_MODEL_CACHE: dict[str, list[tuple[str, str]]] = {}  # per process; only non-empty results are kept, so a
                                                     # provider that was down (or slow) gets re-asked next time


def _fetch_models(provider: str) -> list[tuple[str, str]]:
    from .providers import ClaudeRunner
    if provider not in _MODEL_CACHE:
        rows = {"ollama": _list_ollama_models, "antigravity": _list_agy_models, "npu": _list_npu_models,
                "openai": _list_openai_models, "codex": _list_codex_models,
                "claude": lambda: CLAUDE_MODELS if ClaudeRunner().healthy() else []}[provider]()
        if rows: _MODEL_CACHE[provider] = rows
        return rows
    return _MODEL_CACHE[provider]


def _model_catalog(fetch=_fetch_models) -> list[tuple[str, str, str]]:
    """Every model from every reachable provider as (provider, id, label), local first. Fetched in
    parallel because `agy models` alone takes ~5s."""
    providers = list(PROVIDER_NAMES)
    results: list = [[] for _ in providers]

    def work(i):
        try: results[i] = fetch(providers[i])
        except Exception: pass  # one provider failing must not hide the others

    # daemon threads, not a ThreadPoolExecutor: its workers are joined at interpreter exit, so quitting while
    # `agy models` (~5s) was still running hung and dumped a KeyboardInterrupt traceback
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(providers))]
    for t in threads: t.start()
    for t in threads: t.join()
    return [(p, mid, label) for p, rows in zip(providers, results) for mid, label in rows]


def resolve_model(text: str, current_provider: str, catalog) -> tuple[str, str] | None:
    """`provider/model-id` (any id, even one not in the list yet) or an exact model id from the
    catalog -> (provider, id); None if it's neither. An id offered by several providers resolves to
    the current one's."""
    prov, _, mid = text.partition("/")
    if mid and prov in PROVIDER_NAMES: return prov, mid
    hits = [(p, m) for p, m, _ in catalog if m == text]
    return next((h for h in hits if h[0] == current_provider), hits[0]) if hits else None
