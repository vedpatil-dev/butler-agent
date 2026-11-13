"""`agent` CLI: start | stop | status | chat | run | benchmark."""
import argparse
import atexit
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config, history
from .confirm import cli_confirm, inv
from .conversations import (_after_cloud_turn, _conversation_items, _extract_turns, _format_carryover,  # noqa: F401
                            _render_transcript, _turns_from_messages)
from .llm import LLMError
from .providers import EXTERNAL_AGENTS
from .models import (CLAUDE_MODELS, PROVIDER_NAMES, _MODEL_CACHE, _fetch_models, _list_agy_models, _list_codex_models,  # noqa: F401
                     _list_npu_models, _list_ollama_models, _model_catalog, resolve_model)
from .render import BOLD, CYAN, DIM, GREEN, RED, RESET, SPINNER_FRAMES, _CTRL, _md, _render, _spin, friendly  # noqa: F401

PID = config.STATE / "server.pid"
CLI_NAMES = {"antigravity": "agy", "claude": "claude", "codex": "codex"}  # provider name -> underlying CLI binary
# detect.py's KNOWN_AGENTS name -> provider, for the agents that have a runner. Detection alone ("is the
# binary on PATH") isn't integration: each CLI needs a runner for its own exec command / output format /
# session resume / model list. Anything detected but not in here is reported as "detected only".
AGENT_TO_PROVIDER = {"agy": "antigravity", "claude": "claude", "codex": "codex", "ollama": "ollama"}


def show_status(s: dict):
    r = s.get("ram", {})
    print(f"{BOLD}Model:{RESET}        {s['model']} ({s.get('params', '?')}, {s.get('quantization', '?')})")
    print(f"{BOLD}Backend:{RESET}      {s['runtime']} @ {s.get('url', config.LLM_URL)} ({'up' if s['ollama_up'] else RED + 'DOWN' + RESET})")
    print(f"{BOLD}Context:{RESET}      {s['context']} tokens")
    print(f"{BOLD}RAM:{RESET}          model {s.get('model_ram_gb', '-')} GB | system {r.get('available_gb')} of {r.get('total_gb')} GB free")
    print(f"{BOLD}Acceleration:{RESET} {s.get('acceleration', 'model not loaded yet (unknown until first request)')}")
    print(f"{BOLD}NPU:{RESET}          {s.get('npu', '?')}")


def cmd_status(_):
    from .llm import LLM
    show_status(LLM().status())
    pid = _pid()
    print(f"{BOLD}API:{RESET}          {'http://127.0.0.1:%d (pid %d)' % (config.PORT, pid) if pid else 'stopped'}")


def cmd_providers(_):
    from .detect import detect_agents, detect_gpu, detect_npu
    from .providers import EXTERNAL_AGENTS, NPU_URL, OPENAI_URL
    from .llm import LLM
    print(f"{BOLD}Local LLM providers{RESET}")
    print(f"  ollama       {'up' if LLM().healthy() else RED + 'down' + RESET}   @ {config.LLM_URL}")
    import httpx
    for label, url, note in (("npu", NPU_URL, "FastFlowLM"), ("openai", OPENAI_URL, "OpenVINO/MLX/Genie/llama.cpp")):
        up = False
        try: up = httpx.get(f"{url}/v1/models", timeout=2).status_code == 200
        except httpx.HTTPError: pass
        print(f"  {label:<12} {'up' if up else DIM + 'down' + RESET}   @ {url}  ({note})")
    for name, cls in EXTERNAL_AGENTS.items():
        up = cls().healthy()
        print(f"  {name:<12} {'available' if up else DIM + 'not installed' + RESET}   (cloud, via `{CLI_NAMES[name]}` CLI - not local)")
    print(f"\n{BOLD}NPU hardware{RESET}")
    npu = detect_npu()
    print(f"  platform: {npu['platform']}")
    if not npu["present"]: print("  none detected")
    for n in npu["present"]:
        tag = "" if n["verified_on_this_machine"] else DIM + " (untested detection logic)" + RESET
        print(f"  {n['vendor']} {n['family']}{tag}")
    print(f"\n{BOLD}GPU hardware{RESET}")
    gpu = detect_gpu()
    if not gpu["present"]: print("  none detected")
    for g in gpu["present"]:
        state = "ready" if g.get("driver_ready") else "driver missing"
        print(f"  {g['vendor']} {g['family']} [{g['backend']}, {state}]")
    print(f"\n{BOLD}AI agent CLIs found on PATH{RESET}")
    for ag in detect_agents():
        prov = AGENT_TO_PROVIDER.get(ag["name"])
        status = f"{GREEN}provider: {prov}{RESET}" if prov else f"{DIM}detected only - no runner for it yet{RESET}"
        print(f"  {ag['name']:<16}{ag['version']:<30}{status}")


def cmd_setup(a):
    from .setup import run_setup
    raise SystemExit(run_setup(install=a.install, yes=a.yes, target=a.target, as_json=a.json))


def _pid():
    try:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):  # no file, bad file, process gone, or a pid that isn't ours (PermissionError)
        return None


def cmd_start(_):
    if _pid(): return print("already running")
    log = open(config.STATE / "server.log", "ab")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "agent.server:app", "--host", "127.0.0.1", "--port",
                          str(config.PORT)], stdout=log, stderr=log, start_new_session=True, cwd=config.ROOT)
    PID.write_text(str(p.pid))
    time.sleep(2)
    print(f"API on http://127.0.0.1:{config.PORT}  (web UI at /, log: {config.STATE / 'server.log'})")


def cmd_stop(a):
    pid = _pid()
    if pid:
        try: os.killpg(pid, signal.SIGTERM)
        except OSError: pass
        PID.unlink(missing_ok=True); print("API stopped")
    else: print("API not running")
    if a.unload:
        import httpx
        try: httpx.post(f"{config.LLM_URL}/api/generate", json={"model": config.MODEL, "keep_alive": 0}, timeout=30)
        except httpx.HTTPError as e: return print(f"could not unload the model: {e}")
        print("model unloaded (RAM freed)")


def _pick_provider(wait_npu: bool = True) -> str | None:
    """Auto-selects the first reachable provider, in priority order (npu local > ollama local >
    antigravity > claude), when no --provider was given - no menu, no prompt. Only falls back to
    the interactive menu if NOTHING is reachable, so there's still a way to see what's available
    and what to start, rather than a bare failure. Returns None if the user gives up there."""
    from .detect import detect_npu
    from .providers import EXTERNAL_AGENTS, _npu_up, can_start_npu, ensure_npu_server
    from .llm import LLM
    npu_up = ensure_npu_server() if wait_npu else (_npu_up() or can_start_npu())
    options = [("npu", npu_up, "local, FastFlowLM on the NPU"), ("ollama", LLM().healthy(), f"local, @ {config.LLM_URL}")]
    for name, cls in EXTERNAL_AGENTS.items():
        options.append((name, cls().healthy(), f"cloud via `{CLI_NAMES[name]}` CLI, not local"))
    first_up = next((name for name, up, _ in options if up), None)
    if first_up:
        print(f"{DIM}using {first_up} (first available - /provider <name> to switch){RESET}")
        return first_up
    print(f"{RED}no provider is reachable.{RESET}")
    while True:
        print(f"\n{BOLD}Select a model provider:{RESET}")
        for i, (name, up, desc) in enumerate(options, 1):
            tag = GREEN + "up" + RESET if up else RED + "down" + RESET
            print(f"  {i}) {name:<12} [{tag}]  {desc}")
        try: choice = input(f"{inv(BOLD)}> {inv(RESET)}").strip().lower()
        except (EOFError, KeyboardInterrupt): return None
        if not choice:
            print(f"{DIM}(enter a number or name - Ctrl-C to quit){RESET}")
            continue
        name = options[int(choice) - 1][0] if choice.isdigit() and 1 <= int(choice) <= len(options) else choice
        match = next((o for o in options if o[0] == name), None)
        if not match:
            print(f"{RED}unknown provider{RESET}"); continue
        if not match[1]:
            print(f"{RED}{name} is down right now.{RESET} Start it (see README) and try again, or pick another.")
            continue
        return name


def _init_project(a) -> None:
    """Workspace and project are the directory butler was started from (or -C <dir>): the agent cannot touch
    anything outside it. AGENT_WORKSPACE is not consulted by the CLI."""
    from tools.filesystem import set_cwd
    a.project = str(Path(a.project or os.getcwd()).expanduser().resolve())
    # Why: set once at startup, before any thread exists, as the process default. Impact: per-conversation
    # project changes go through set_cwd (context-local) and never touch the environment again.
    os.environ["AGENT_WORKSPACE"] = os.environ["AGENT_CWD"] = a.project
    set_cwd(a.project)


def _make_agent(a, quiet: bool = False, wait_npu: bool = True):
    from .mcp_client import MCPHub, load_config
    from .orchestrator import Agent
    from .providers import can_start_npu, ensure_npu_server, make_provider, start_npu_background
    llm = make_provider(a.provider, model=a.model, **({"num_gpu": a.num_gpu} if a.provider == "ollama" else {}))
    if not llm.healthy():
        if a.provider == "npu" and not wait_npu and can_start_npu():
            start_npu_background()
            if not quiet: print(f"{DIM}loading NPU model ({llm.model}) in background...{RESET}")
        elif not (a.provider == "npu" and ensure_npu_server()):
            sys.exit(f"{a.provider} provider not reachable (see `agent providers`)")
    hub = MCPHub(load_config(), only=a.mcp) if a.mcp else None
    if not quiet: show_status(llm.status())
    return Agent(llm, cli_confirm({"all": False}), project=a.project, mcp=hub, plan=a.plan, mode=a.approval)


def _run_external(provider: str, task: str, model: str | None, banner: bool = True,
                  effort: str | None = None, session: str | None = None, approval: str | None = None) -> dict:
    """Run one message through a cloud agent. Returns the final event: "session" is the id to pass
    back in next time (so the agent remembers what it asked), "reply" the text it produced. {} if
    the run ended without one."""
    from .providers import EXTERNAL_AGENTS
    # Why: one construction for every runner (ExternalAgent protocol); each ignores the options it doesn't support.
    # Impact: a new runner needs no edit here. prompt=True only matters with a terminal to ask on.
    runner = EXTERNAL_AGENTS[provider](model, session=session, approval=approval, effort=effort, prompt=sys.stdin.isatty())
    cli = CLI_NAMES[provider]
    if not runner.healthy(): sys.exit(f"{cli} CLI not found on PATH")
    if banner: print(f"{DIM}{provider} ({model or 'default model'}{', effort ' + effort if effort and provider in EFFORT_AGENTS else ''}) - cloud, not local{RESET}")
    from . import attach
    return _render(runner.run(task, attach.parse(task, cli_confirm({"all": False}) if sys.stdin.isatty() else None))) or {}


def _drop_unsupported_effort(a):
    if a.effort and a.provider not in EFFORT_AGENTS:
        print(f"{DIM}--effort ignored: {a.provider} has no separate effort setting{RESET}")
        a.effort = None


def cmd_run(a):
    a.provider = a.provider or "ollama"  # non-interactive: keep the old quiet default for scripting
    _drop_unsupported_effort(a)
    from .providers import EXTERNAL_AGENTS
    _init_project(a)
    if a.provider in EXTERNAL_AGENTS:
        return _run_external(a.provider, a.task, a.model, effort=a.effort, approval=a.approval)
    agent = _make_agent(a, quiet=a.json)
    seen = {"error": False, "denied": False}

    def watch(events):  # exit status: 0 ok, 1 error, 2 something was denied (needs --approval or a human)
        for e in events:
            seen["error"] |= e["type"] == "error"
            seen["denied"] |= e["type"] == "result" and e["text"].startswith("DENIED")
            if a.json: print(json.dumps(e, ensure_ascii=False), flush=True)
            else: yield e
    try:
        if a.json: list(watch(agent.run(a.task)))
        else: _render(watch(agent.run(a.task)))
        sys.exit(1 if seen["error"] else 2 if seen["denied"] else 0)
    except KeyboardInterrupt: agent.cancel(); print("\nstopped")
    except LLMError as e: sys.exit(f"{RED}error:{RESET} {friendly(str(e))}")


COMMANDS = {
    "/help": "show this list",
    "/provider [name]": "switch provider (ollama|npu|antigravity|claude), or show current if no name",
    "/model [id]": "pick ANY model from ANY provider in one list - arrows + Enter, type to filter (switches provider too); or /model <id>",
    "/effort [level]": "reasoning effort for claude (low|medium|high|xhigh|max) - arrows + Enter if no level given",
    "/plan [on|off]": "read-only planning mode: investigate, propose a plan, ask before acting (local agent)",
    "/mcp": "MCP servers in this session: tool counts and start errors",
    "/status": "provider, model, approval mode, context use and the current plan",
    "/init": "have the agent write an AGENTS.md for this project (local agent)",
    "/approval [mode]": "ask | auto-edit | readonly - what the agent may do without asking (also passed to claude/codex/agy)",
    "/diff": "show uncommitted changes in the project (git diff)",
    "/compact": "summarise older context now to free the window (local agent only)",
    "/undo": "revert the file edits of the last agent run (local agent only)",
    "/new": "start a fresh conversation (forget earlier context)",
    "/resume [provider|all]": "pick an earlier conversation for this provider (or all/<provider>) and continue it (alias /history)",
    "/native": "open the active cloud CLI (agy/claude/codex) interactively so all native /commands work",
    "/exit": "quit",
}
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")  # same set for `claude --effort` and `agy --effort`
EFFORT_AGENTS = tuple(n for n, c in EXTERNAL_AGENTS.items() if c.supports_effort)  # derived from the runners (agy's effort is part of its model ids)
# Antigravity (agy) is itself a full agent with its own slash commands (/help, /skills, /plugin,
# /agents, /config, /effort, /permissions, /usage, ...) - verified via `agy -p "/help"`, which
# expands them even in one-shot print mode. /model is a partial exception: agy's own native
# version only supports a bare query ("/model takes no arguments" is the literal error if you
# give it one) - actually switching models is done via agy's --model CLI flag instead (see
# AgyRunner), which this project's own /model sets, so "/model <name>" is ours to intercept while
# bare "/model" still passes through to agy's native query. /provider has no agy equivalent at
# all (it doesn't know about other providers), and /exit quits this wrapper, not agy. Everything
# else the user types is passed straight to `agy -p`, native commands included.
# Claude's slash commands are NOT available in print mode (verified: `claude -p "/help"` replies
# "isn't available in this environment"), so claude has no equivalent passthrough below - this
# project's own /help, /provider, /model apply normally while claude is selected.
AGY_RESERVED = ("/provider", "/exit", "/quit", "/new", "/clear", "/resume", "/history")
AGY_NATIVE_CMDS = ["/agents", "/changelog", "/config", "/credits", "/help", "/hooks",
                  "/model", "/permissions", "/plugin", "/skills", "/usage"]


LOCAL_CMDS = [c.split(maxsplit=1)[0] for c in COMMANDS]


def complete_command(text: str, provider: str) -> list[str]:
    """All /command names matching this prefix for the given provider. Pure, so it's testable
    without a real TTY - readline only runs its completer when stdin is an actual terminal."""
    if not text.startswith("/"): return []
    if provider == "antigravity": cmds = AGY_NATIVE_CMDS + list(AGY_RESERVED)
    else: cmds = LOCAL_CMDS if provider in EFFORT_AGENTS else [c for c in LOCAL_CMDS if c != "/effort"]
    return [c for c in cmds if c.startswith(text)]


def help_lines(provider: str) -> list[tuple[str, str]]:
    """The /commands worth advertising for this provider - /effort only where it exists (claude),
    same rule complete_command() applies to tab-completion."""
    return [(c, d) for c, d in COMMANDS.items() if provider in EFFORT_AGENTS or c.split()[0] != "/effort"]


def _print_help(provider: str):
    print(f"\n{BOLD}Commands:{RESET}")
    for cmd, desc in help_lines(provider): print(f"  {cmd:<22}{DIM}{desc}{RESET}")


def box_title(provider: str, model: str | None, effort: str | None = None, ctx: int | None = None) -> str:
    return " · ".join(["you", f"{model or 'default'} [{provider}]"] + ([effort] if effort else []) + ([f"{ctx}% ctx"] if ctx is not None else []))


def _pick_any_model(catalog, current_provider: str, current_model: str | None) -> tuple[str, str] | None:
    """ONE list across every provider, each row tagged [provider] - Up/Down + Enter (type to
    filter, Esc to cancel). Picking a row sets the provider and the model together.
    Returns (provider, id) or None."""
    if not sys.stdin.isatty():
        print(f"{DIM}arrow selection needs a terminal - use /model <id> or /model provider/<id>{RESET}")
        return None
    from .ui import pick_from_list
    cur = next((i for i, (p, m, _) in enumerate(catalog) if (p, m) == (current_provider, current_model)), None)
    i = pick_from_list([(mid, f"[{prov}]", label) for prov, mid, label in catalog], "Select a model", cur)
    return catalog[i][:2] if i is not None else None


def _apply_cloud_conversation(a, agent, conv: dict):
    """Continue a recorded or native cloud conversation: switch provider/model/session and render its transcript."""
    if conv["provider"] != a.provider:
        agent, ok = _switch_provider(a, conv["provider"], agent)
        if not ok: return agent
    a.model, a.session, a.conv_id = conv.get("model"), conv["session"], conv["id"]
    a.carryover = None
    a.bridge_turns = [(t.get("user", ""), t.get("reply", "")) for t in conv.get("turns", [])[-6:]]
    _render_transcript(conv["provider"], a.model, conv["title"], conv.get("turns", []))
    return agent


def _apply_local_conversation(a, agent, cid: str):
    """Load a saved local-model conversation into the Agent's memory and render its transcript."""
    from .memory import Memory
    try: raw = Memory.load_raw(cid)
    except (OSError, ValueError):
        print(f"{RED}couldn't read that conversation{RESET}"); return agent
    prov = raw.get("provider") or a.provider
    if agent is None or (raw.get("provider") and raw["provider"] != a.provider):
        agent, ok = _switch_provider(a, prov, agent)
        if not ok: return agent
    mem = agent.memory
    mem.id, mem.messages, mem.summary, mem.title = cid, raw["messages"], raw.get("summary", ""), raw.get("title", "")
    mem.model, mem.provider = raw.get("model", ""), prov
    if mem.model: agent.llm.model = mem.model
    a.conv_id, a.carryover = None, None
    turns = _turns_from_messages(raw["messages"])
    a.bridge_turns = [(t["user"], t["reply"]) for t in turns[-6:]]
    _render_transcript(prov, mem.model or getattr(agent.llm, "model", ""), mem.title, turns)
    return agent


def _resume_menu(a, agent, filt: str | None = None):
    """/resume [provider|all] (alias /history): arrow list of earlier conversations newest first.
    In interactive chat, defaults to the active provider (`filt=a.provider`), falling back to all
    providers if the current provider has no saved history yet."""
    if not sys.stdin.isatty():
        print(f"{DIM}arrow selection needs a terminal{RESET}"); return agent
    if filt not in (None, "", "all", *PROVIDER_NAMES):
        print(f"{DIM}usage: /resume [all|{'|'.join(PROVIDER_NAMES)}]{RESET}"); return agent
    items = _conversation_items(filt)
    if not items and filt not in (None, "", "all"):
        items = _conversation_items("all")
        if items: print(f"{DIM}no {filt} conversations yet - showing all providers (/resume all){RESET}")
        filt = None
    if not items:
        print(f"{DIM}no earlier conversations yet - they're recorded as you chat{RESET}"); return agent
    from .ui import pick_from_list
    rows = [((c.get("title") or "(untitled)")[:50], f"[{c['provider'] if kind == 'cloud' else (c.get('provider') or 'ollama')}]",
             f"{(c.get('model') or 'default model') + ' · ' if kind == 'cloud' or c.get('model') else ''}{history.ago(when)}")
            for kind, c, when in items]
    title = "Resume a conversation" if filt in (None, "", "all") else f"Resume a {filt} conversation (/resume all for every provider)"
    i = pick_from_list(rows, title)
    if i is None: return agent
    kind, conv, _ = items[i]
    return _apply_cloud_conversation(a, agent, conv) if kind == "cloud" else _apply_local_conversation(a, agent, conv["id"])


def _switch_provider(a, name: str, agent):
    """Switch the chat to provider `name` -> (agent, switched). A local provider must be reachable
    first: _make_agent() sys.exit()s on a dead one, which used to kill butler mid-chat on /provider ollama.
    Carries over active conversation turns so switching providers mid-chat doesn't lose context."""
    _join_ram()
    from .providers import EXTERNAL_AGENTS, ensure_npu_server, make_provider
    if name in EXTERNAL_AGENTS:
        if not EXTERNAL_AGENTS[name]().healthy():
            print(f"{RED}{CLI_NAMES[name]} CLI not found on PATH{RESET}"); return agent, False
    elif not (make_provider(name).healthy() or (name == "npu" and ensure_npu_server())):
        print(f"{RED}{name} is down right now{RESET} (see `butler providers`)"); return agent, False
    old_prov = getattr(a, "provider", None)
    turns = _extract_turns(a, agent)
    old_msgs = ([dict(m) for m in agent.memory.messages if m.get("role") in ("user", "assistant") and m.get("content")]
                if agent is not None else [])
    old_summary = agent.memory.summary if agent is not None else ""
    old_title = agent.memory.title if agent is not None else ""
    old = (a.provider, getattr(a, "model", None), getattr(a, "effort", None), getattr(a, "session", None), getattr(a, "conv_id", None))
    a.provider, a.model, a.effort, a.session, a.conv_id = name, None, None, None, None  # all per-provider
    try: new = None if name in EXTERNAL_AGENTS else _make_agent(a, quiet=True)  # no status block on every switch; /status shows it
    except (SystemExit, Exception) as e:  # keep the chat on the old provider rather than half-switched or dead
        a.provider, a.model, a.effort, a.session, a.conv_id = old
        print(f"{RED}could not switch to {name}:{RESET} {e}"); return agent, False
    _close_hub(agent)
    agent = new
    if agent is not None:
        agent.memory.provider = name
        agent.memory.model = getattr(agent.llm, "model", "")
        if old_msgs:
            agent.memory.messages = old_msgs
            agent.memory.summary = old_summary
            agent.memory.title = old_title
        elif turns:
            for u, r in turns:
                if u: agent.memory.messages.append({"role": "user", "content": u})
                if r: agent.memory.messages.append({"role": "assistant", "content": r})
            if not agent.memory.title and turns[0][0]:
                agent.memory.title = turns[0][0].splitlines()[0][:60]
        a.carryover = None
        a.bridge_turns = turns
    else:
        a.bridge_turns = turns
        a.carryover = _format_carryover(old_prov or "previous provider", turns) if turns else None
    carried = f" · carried {len(turns)} turn{'s' if len(turns) != 1 else ''}" if turns else ""
    print(f"{DIM}switched to {name}{' (cloud) - bypasses local tools' if name in EXTERNAL_AGENTS else ''}{carried}{RESET}")
    _free_ram(name, getattr(agent.llm, "model", None) if agent else None)
    return agent, True


def _close_hub(agent) -> None:
    if agent is not None and agent.mcp: agent.mcp.close()  # MCP servers are subprocesses: don't leave them behind


_RAM_LOCK = threading.Lock()
_RAM_THREAD: threading.Thread | None = None


def _join_ram() -> None:
    if _RAM_THREAD and _RAM_THREAD.is_alive() and threading.current_thread() is not _RAM_THREAD:
        _RAM_THREAD.join(timeout=10)


def _free_ram(provider: str, model: str | None) -> None:
    from .providers import free_local_models
    with _RAM_LOCK:
        if _other_chats(): return  # see _other_chats: don't unload models another chat may be using
        freed = free_local_models(provider, model)
        if freed: print(f"{DIM}freed RAM: {', '.join(freed)}{RESET}")


def _apply_model(a, agent, provider: str, model: str):
    _join_ram()
    if provider != a.provider:
        agent, ok = _switch_provider(a, provider, agent)
        if not ok: return agent
    if agent is None: a.model = model
    else:
        agent.llm.model = model
        agent.memory.model = model
    print(f"{DIM}model: {model} [{provider}]{RESET}")
    _free_ram(provider, model)
    return agent


def _pick_effort(current: str | None) -> str | None:
    """Up/Down + Enter over the effort levels; Esc keeps the current one."""
    if not sys.stdin.isatty():
        print(f"{DIM}arrow selection needs a terminal - use /effort <{'|'.join(EFFORT_LEVELS)}>{RESET}")
        return current
    from .ui import pick_from_list
    i = pick_from_list([(lvl, "", "") for lvl in EFFORT_LEVELS], "Select reasoning effort",
                       EFFORT_LEVELS.index(current) if current in EFFORT_LEVELS else None)
    return EFFORT_LEVELS[i] if i is not None else current


CHATS = config.STATE / "chats"  # one file per live `butler chat` process, named by pid


def _register_chat() -> None:
    CHATS.mkdir(parents=True, exist_ok=True)
    (CHATS / str(os.getpid())).touch()


def _other_chats() -> bool:
    """True if another butler chat is running. Why: freeing RAM unloads the ollama models and stops the NPU server
    machine-wide, so one chat starting, switching or closing used to pull the model out from under every other chat
    (their next message got "Connection refused"). Impact: while another chat lives, nobody frees shared model RAM.
    Stale files left by crashed chats are cleaned up here."""
    live = False
    for f in CHATS.glob("*") if CHATS.exists() else ():
        try: pid = int(f.name)
        except ValueError: continue
        if pid == os.getpid(): continue
        try: os.kill(pid, 0); live = True
        except OSError: f.unlink(missing_ok=True)  # PermissionError can't happen for our own processes; a gone pid is stale
    return live


def _release_on_exit() -> None:
    """Butler is closing: unload every ollama model and stop the NPU server so they stop holding RAM. Also runs on
    SIGTERM/SIGHUP (terminal closed or killed), which would otherwise skip atexit. Not on SIGKILL or a crash of the
    interpreter itself - nothing can run then."""
    from .providers import free_local_models
    from .mcp_client import MCPHub
    for hub in list(MCPHub.live): hub.close()
    (CHATS / str(os.getpid())).unlink(missing_ok=True)
    if _pid() or _other_chats(): return  # the API server / another chat shares the models: unloading would break that session
    _join_ram()
    try: freed = free_local_models("")
    except Exception: return
    if freed: print(f"\n{DIM}freed RAM: {', '.join(freed)}{RESET}")


def _run_turn(agent, task: str):
    """One local-agent turn. In plan mode the turn only investigates; offer to carry the plan out afterwards."""
    from .providers import NPUProvider, ensure_npu_server, npu_bg_state
    if isinstance(agent.llm, NPUProvider) and not agent.llm.healthy():
        bg = npu_bg_state()
        if bg["state"] == "failed":
            print(f"{RED}error:{RESET} NPU service failed to start ({bg['error']}); retrying...")
        else:
            print(f"{DIM}waiting for NPU model ({agent.llm.model}) to finish loading...{RESET}", flush=True)
        if not ensure_npu_server():
            print(f"{RED}error:{RESET} NPU server not reachable (check `systemctl --user status flm-npu.service`)")
            return
    try:
        _render(agent.run(task))
        if agent.plan and sys.stdin.isatty() and input(f"{BOLD}Execute this plan? [y/N] {RESET}").strip().lower() == "y":
            agent.plan = False
            _render(agent.run("Go ahead and carry out the plan."))
    except KeyboardInterrupt: agent.cancel(); print("\nstopped")
    except LLMError as e: print(f"\n{RED}error:{RESET} {friendly(str(e))}")  # the conversation so far is kept; try again or /model


def _status_line(a, agent) -> str:
    if agent is None: return f"{a.provider} (cloud) · model {a.model or 'default'} · approval {a.approval}"
    t = agent.totals
    todo = "".join(f"\n  [{'x' if i['status'] == 'done' else '>' if i['status'] == 'in_progress' else ' '}] {i['step']}" for i in agent.todo)
    return (f"{a.provider} · {agent.llm.model} · approval {agent.mode}{' · PLAN MODE (read-only)' if agent.plan else ''}\n"
            f"  context {agent.memory.context_pct()}% of {agent.llm.num_ctx} · {t['steps']} steps · {t['tool_calls']} tool calls "
            f"({t['tool_errors']} failed) · {t['prompt_tokens'] + t['gen_tokens']} tokens{todo}")


def cmd_chat(a):
    from . import attach
    from .providers import EXTERNAL_AGENTS
    _register_chat()
    atexit.register(_release_on_exit)
    for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *_: sys.exit(0))  # sys.exit -> atexit runs
    if a.provider is None:
        a.provider = _pick_provider(wait_npu=False)
        if a.provider is None: return  # user gave up at the menu
    from .ui import make_box_prompt
    _drop_unsupported_effort(a)
    _init_project(a)
    a.session = None  # cloud agents: id of the conversation to continue, so follow-up answers have context
    a.conv_id = None  # butler's own id for this conversation in history.py (what /resume lists)
    a.carryover = None
    a.bridge_turns = []
    def _live_status() -> str:
        from .llm import meminfo, npu_state
        r = meminfo()
        ram = f"RAM {r.get('available_gb', '?')}/{r.get('total_gb', '?')} GB free"
        if agent is None:
            return f"  {a.provider} (cloud) · {ram} · Enter send · Alt+Enter newline · /native · /help"
        up = agent.llm.healthy()
        bg_err = ""
        if up:
            state = "up"
        elif a.provider == "npu":
            from .providers import npu_bg_state
            bg = npu_bg_state()
            state = "FAILED" if bg["state"] == "failed" else "loading..."
            if bg["state"] == "failed":
                bg_err = f" ({bg['error']})"
        else:
            state = "down"
        rt = getattr(agent.llm, "runtime_name", "Ollama")
        npu_txt = npu_state().replace("; not used by Ollama", "")
        return (f"  Backend: {rt} ({state}{bg_err}) · Context: {agent.llm.num_ctx} tok · {ram}\n"
                f"  NPU: {npu_txt} · Enter send · Alt+Enter newline · /model · /help")

    box_prompt = make_box_prompt(lambda: box_title(a.provider, a.model if agent is None else agent.llm.model, a.effort,
                                                   None if agent is None else agent.memory.context_pct()),
                                 complete_command,
                                 lambda: a.provider, config.STATE / "chat_history", get_status=_live_status)
    agent = None if a.provider in EXTERNAL_AGENTS else _make_agent(a, quiet=True, wait_npu=False)
    # Why: tell the user how to pick the conversation back up. Impact: shown on every exit path (/exit, Ctrl-C/Ctrl-D, SIGHUP)
    # once the chat has a conversation - a recorded cloud turn (conv_id) or a local agent that has taken a user message.
    # Registered after _release_on_exit, so it prints first (atexit is LIFO). `agent` is read late: /provider and /resume rebind it.
    def _resume_hint():
        if getattr(a, "conv_id", None) or (agent and any(m.get("role") == "user" for m in agent.memory.messages)):
            print(f"\n{DIM}resume this chat: butler -C {shlex.quote(a.project)} --continue{RESET}")
    atexit.register(_resume_hint)
    global _RAM_THREAD
    _RAM_THREAD = threading.Thread(target=_free_ram, args=(a.provider, agent.llm.model if agent else None), daemon=True)
    _RAM_THREAD.start()
    threading.Thread(target=_model_catalog, daemon=True).start()  # warm the (slow: agy ~5s) model list so /model is instant
    if agent is None:
        native = " /help shows antigravity's own commands;" if a.provider == "antigravity" else ""
        print(f"{DIM}{a.provider} (cloud, not local).{native} Replies continue one conversation - /new for a fresh "
              f"one. /model to pick any model (any provider), /exit to quit.{RESET}")
    else:
        print(f"{DIM}Local agent{' in ' + a.project}. /model to pick any model (any provider), /help for commands.{RESET}")
    if a.cont:
        if items := (_conversation_items(a.provider) or _conversation_items("all")):
            kind, conv, _ = items[0]
            agent = _apply_cloud_conversation(a, agent, conv) if kind == "cloud" else _apply_local_conversation(a, agent, conv["id"])
        else: print(f"{DIM}nothing to continue - no earlier conversations{RESET}")
    while True:
        task = box_prompt()
        if task is None: break  # Ctrl-C / Ctrl-D on an empty box
        task = task.strip()
        if task in ("/exit", "/quit"): break
        cmd = task.split(maxsplit=1)[0] if task else ""
        dropped = attach.is_path(cmd)  # a dragged-in file path, not a slash command
        if cmd.lstrip("/") in PROVIDER_NAMES and task == cmd:
            agent, _ = _switch_provider(a, cmd.lstrip("/"), agent)
            continue
        # agy's own /model and /effort are read-only (verified live: "/model <name>" and
        # "/effort <level>" both error with "takes no arguments" - switching is done via its
        # --model / --effort CLI flags instead), and this project's versions offer a selection menu
        # when bare - so both are entirely ours to handle, never passed through natively.
        if (agent is None and a.provider == "antigravity" and task.startswith("/") and not dropped
                and not task.startswith(AGY_RESERVED) and not task.startswith(("/model", "/effort"))):
            if task: a.session = _run_external(a.provider, task, a.model, banner=False, effort=a.effort,
                                               session=a.session, approval=a.approval).get("session") or a.session  # native agy command (not history)
            continue
        if task in ("/help", "/"): _print_help(a.provider); continue
        if cmd == "/provider":
            name = (task.split(maxsplit=1) + [""])[1].strip()
            if not name: print(f"current provider: {a.provider}"); continue
            if name not in PROVIDER_NAMES:
                print(f"{DIM}usage: /provider {'|'.join(PROVIDER_NAMES)}{RESET}"); continue
            agent, _ = _switch_provider(a, name, agent)
            continue
        if cmd == "/model":
            arg = (task.split(maxsplit=1) + [""])[1].strip()
            if "antigravity" not in _MODEL_CACHE:  # `agy models` (~5s) is the only slow lister: say why we pause
                print(f"{DIM}loading models from every provider...{RESET}", flush=True)
            catalog = _model_catalog()
            if arg:  # an id not in any list is taken as-is for the current provider (a bad one errors visibly when used)
                target = resolve_model(arg, a.provider, catalog) or (a.provider, arg)
            elif catalog:
                target = _pick_any_model(catalog, a.provider, a.model if agent is None else agent.llm.model)
            else:
                print(f"{RED}no models available{RESET} - nothing is reachable (see `butler providers`)"); target = None
            if target: agent = _apply_model(a, agent, *target)
            continue
        if cmd in ("/resume", "/history"):
            arg = (task.split(maxsplit=1) + [""])[1].strip().lower()
            agent = _resume_menu(a, agent, arg or a.provider)
            continue
        if cmd == "/plan":
            arg = (task.split(maxsplit=1) + [""])[1].strip().lower()
            if agent is None: print(f"{DIM}/plan only works with the local agent{RESET}"); continue
            agent.plan = {"on": True, "off": False}.get(arg, not agent.plan)
            print(f"{DIM}plan mode {'ON: read-only; I will propose a plan and ask before doing it' if agent.plan else 'off'}{RESET}")
            continue
        if task == "/mcp":
            hub = agent.mcp if agent else None
            if not hub: print(f"{DIM}no MCP servers in this session - start with: {sys.argv[0]} chat --mcp <name>  (see config/mcp.json){RESET}"); continue
            for n in hub.cfg:
                cnt = sum(1 for _, (srv, _t, _c) in hub._routes.items() if srv == n)
                print(f"  {n:<14}" + (f"{RED}{hub.errors[n]}{RESET}" if n in hub.errors else f"{GREEN}{cnt} tools{RESET}"))
            continue
        if task == "/status": print(_status_line(a, agent)); continue
        if task == "/init":
            if agent is None: print(f"{DIM}/init only works with the local agent{RESET}"); continue
            _run_turn(agent, "Explore this project (layout, how to build/test/run, conventions) and create AGENTS.md at the project root "
                             "with those facts for future agents. Keep it under 60 lines. If AGENTS.md exists, do not overwrite it.")
            continue
        if cmd == "/approval":
            mode = (task.split(maxsplit=1) + [""])[1].strip()
            if mode and mode not in ("ask", "auto-edit", "readonly"): print(f"{DIM}usage: /approval ask|auto-edit|readonly{RESET}"); continue
            if mode:
                a.approval = mode
                if agent is not None: agent.mode = mode
            print(f"{DIM}approval: {a.approval}{RESET}")
            continue
        if task == "/diff":
            from tools.git import git_diff
            try: out = git_diff()
            except Exception as e: out = f"{e}"
            print("\n".join((GREEN if l.startswith("+") and not l.startswith("+++") else RED if l.startswith("-") and not l.startswith("---") else DIM if l.startswith(("diff", "index", "---", "+++")) else "") + _CTRL.sub("", l) + RESET for l in out.splitlines()))
            continue
        if task == "/compact":
            if agent is None: print(f"{DIM}/compact only works with the local agent{RESET}"); continue
            from .memory import est_tokens
            m = agent.memory; before = est_tokens(m.prompt()); m.budget = min(m.budget, before // 2)
            try: m.compact(agent.llm)
            except LLMError as e: print(f"{RED}error:{RESET} {friendly(str(e))}")
            finally: m.budget = max(1024, int(agent.llm.num_ctx * 0.85) - 1024 - est_tokens(agent.schemas))
            print(f"{DIM}context: ~{before} -> ~{est_tokens(m.prompt())} tokens{RESET}")
            continue
        if task == "/undo":
            done = agent.undo() if agent else None
            print(f"{DIM}{'/undo only works with the local agent' if done is None else ('restored: ' + ', '.join(done)) if done else 'nothing to undo'}{RESET}")
            continue
        if task in ("/new", "/clear"):
            a.session, a.conv_id, a.carryover, a.bridge_turns = None, None, None, []
            if agent is not None:  # local agent: fresh memory too, but keep the model that was selected (a new Agent starts on the default)
                model = agent.llm.model
                _close_hub(agent)
                agent = _make_agent(a, quiet=True)
                agent.llm.model = model
                agent.memory.model = model
            if sys.stdout.isatty(): print("\033[3J\033[2J\033[H", end="", flush=True)  # clear screen + scrollback buffer
            print(f"{DIM}fresh conversation - earlier context forgotten{RESET}")
            continue
        if cmd == "/effort":
            if a.provider not in EFFORT_AGENTS:
                print(f"{DIM}{'antigravity has no separate effort setting - it is part of the model id (...-low / -medium / -high): pick one with /model' if a.provider == 'antigravity' else '/effort is a claude setting'}{RESET}")
                continue
            level = (task.split(maxsplit=1) + [""])[1].strip().lower()
            if level and level not in EFFORT_LEVELS:
                print(f"{RED}unknown effort '{level}'{RESET} - one of: {', '.join(EFFORT_LEVELS)}")
                continue
            a.effort = level or _pick_effort(a.effort)
            print(f"{DIM}{a.provider} effort: {a.effort or 'default'}{RESET}")
            continue
        if cmd == "/native":
            if a.provider not in EXTERNAL_AGENTS:
                print(f"{DIM}/native is for cloud CLIs (antigravity, claude, codex); local providers handle commands directly{RESET}")
                continue
            runner = EXTERNAL_AGENTS[a.provider](model=a.model, effort=a.effort, session=a.session, approval=a.approval)
            if not runner.healthy():
                print(f"{RED}error:{RESET} `{CLI_NAMES[a.provider]}` not found on PATH")
                continue
            icmd = runner.interactive_cmd()
            print(f"{DIM}handing terminal to `{' '.join(icmd)}` (exit it to return to butler)...{RESET}")
            subprocess.run(icmd, cwd=a.project)
            continue
        if task.startswith("/") and not dropped:
            hint = f", or /native to run {a.provider} interactively" if a.provider in EXTERNAL_AGENTS else ""
            print(f"{DIM}unknown command: {cmd} (type /help for commands{hint}){RESET}")
            continue
        if not task: continue
        if agent is None:
            prompt_task = f"{a.carryover}\n\n{task}" if getattr(a, "carryover", None) and not a.session else task
            a.carryover = None
            _after_cloud_turn(a, task, _run_external(a.provider, prompt_task, a.model, banner=False,
                                                     effort=a.effort, session=a.session, approval=a.approval))
        else:
            _run_turn(agent, task)


def cmd_benchmark(a):
    sys.exit(subprocess.call([sys.executable, str(config.ROOT / "scripts" / "benchmark.py"), *a.rest]))


def main():
    p = argparse.ArgumentParser(prog="agent", description="Local agent on Ollama + Qwen3.5-9B")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="start the local API + web UI (127.0.0.1)").set_defaults(f=cmd_start)
    s = sub.add_parser("stop", help="stop the API"); s.add_argument("--unload", action="store_true", help="also unload the model")
    s.set_defaults(f=cmd_stop)
    sub.add_parser("status", help="model / backend / context / RAM / acceleration").set_defaults(f=cmd_status)
    sub.add_parser("providers", help="detected NPU/GPU hardware, AI agent CLIs, and provider availability").set_defaults(f=cmd_providers)
    s = sub.add_parser("setup", help="detect GPUs/NPUs and install required drivers and runtimes")
    s.add_argument("--install", action="store_true", help="execute driver/runtime install commands")
    s.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompts during --install")
    s.add_argument("--target", help="limit setup to one accelerator id (e.g. nvidia-cuda, amd-gpu, intel-npu, apple-silicon)")
    s.add_argument("--json", action="store_true", help="print detected hardware and setup plan as JSON")
    s.set_defaults(f=cmd_setup)
    for name, f in (("chat", cmd_chat), ("run", cmd_run)):
        s = sub.add_parser(name, help="interactive session" if name == "chat" else "run one task and exit")
        if name == "run": s.add_argument("task"); s.add_argument("--json", action="store_true", help="print one JSON event per line (for scripts); exit 0 ok, 1 error, 2 action denied")
        s.add_argument("--project", "-C", help="project dir (inside the workspace)")
        s.add_argument("--mcp", nargs="*", help="MCP servers from config/mcp.json to enable")
        s.add_argument("--plan", action="store_true", help="plan mode: read-only investigation that ends in a plan")
        if name == "chat": s.add_argument("--continue", "-c", dest="cont", action="store_true", help="resume the most recent conversation")
        s.add_argument("--approval", choices=("ask", "auto-edit", "readonly"), default="ask",
                       help="ask: confirm every write/command (default) | auto-edit: file edits in the project run unasked | readonly: refuse all writes and commands")
        s.add_argument("--num-gpu", type=int, help="0 = force CPU")
        s.add_argument("--provider", choices=list(PROVIDER_NAMES), default=None,
                       help="ollama (local) | npu (local, unverified) | antigravity/claude (cloud, "
                            "via agy/claude). run: defaults to ollama. chat: prompts if omitted.")
        s.add_argument("--model", help="model name/id (antigravity/claude only, e.g. claude-sonnet-5-5-high "
                                       "or an alias like 'opus'/'fable' for the claude provider)")
        s.add_argument("--effort", choices=EFFORT_LEVELS, default=None,
                       help="reasoning effort (claude only); in chat, change it with /effort")
        s.set_defaults(f=f)
    s = sub.add_parser("benchmark", help="see scripts/benchmark.py --help"); s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(f=cmd_benchmark)
    argv = sys.argv[1:] or ["chat"]  # bare `butler`/`agent` -> straight into chat, like claude/agy
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
