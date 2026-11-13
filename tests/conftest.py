import subprocess

import pytest

from agent.llm import LLM


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Never touch the real ~/.local/state/local-agent. Agent.run() saves its memory on every run, so
    the suite used to write ~50 junk 'conversations' (titles like "x", "list files") into the user's
    real folder per run - 505 of them by the time anyone looked."""
    monkeypatch.setenv("AGENT_NO_SERVICES", "1")  # never start/stop the user's real flm-npu / unload their ollama models
    import agent.memory
    monkeypatch.setattr(agent.memory, "HISTORY", tmp_path / "_state" / "conversations")
    import tools.filesystem
    tools.filesystem._CWD.set(None)  # a previous test's set_cwd must not outlive it
    import agent.audit
    monkeypatch.setattr(agent.audit, "PATH", tmp_path / "_state" / "audit.jsonl")
    (tmp_path / "_state").mkdir(exist_ok=True)
    try:
        import agent.history
        monkeypatch.setattr(agent.history, "DIR", tmp_path / "_state" / "butler-history")
        monkeypatch.setattr(agent.history, "CLAUDE_PROJECTS", tmp_path / "_state" / "claude_projects")
        monkeypatch.setattr(agent.history, "CLAUDE_HIST", tmp_path / "_state" / "claude_history.jsonl")
        monkeypatch.setattr(agent.history, "CODEX_SESSIONS", tmp_path / "_state" / "codex_sessions")
        monkeypatch.setattr(agent.history, "CODEX_INDEX", tmp_path / "_state" / "codex_index.jsonl")
        monkeypatch.setattr(agent.history, "CODEX_HIST", tmp_path / "_state" / "codex_history.jsonl")
        monkeypatch.setattr(agent.history, "AGY_BRAIN", tmp_path / "_state" / "agy_brain")
    except ImportError:  # history module not written yet in this checkout
        pass


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A sandbox workspace with a git repo in ws/proj; the agent's cwd is proj."""
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_CWD", str(proj))
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(proj), *cmd], check=True)
    (proj / "hello.py").write_text("print('hi')\n")
    subprocess.run(["git", "-C", str(proj), "add", "."], check=True)
    subprocess.run(["git", "-C", str(proj), "commit", "-qm", "init"], check=True)
    return proj


@pytest.fixture(scope="session")
def llm():
    m = LLM()
    if not m.healthy(): pytest.skip("Ollama not reachable")
    return m


class FakeLLM:
    """Scripted stand-in: each script item is (text, [tool_calls]) for one model turn."""
    num_ctx = 8192

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def stream(self, messages, tools=None, **kw):
        self.calls.append(messages)
        text, calls = self.script.pop(0)
        if text: yield "text", text
        yield "stats", {"prompt_tokens": 1, "gen_tokens": 1, "prompt_s": 1, "gen_s": 1, "prompt_tps": 1, "gen_tps": 1}
        if calls: yield "tool_calls", [{"function": {"name": n, "arguments": a}} for n, a in calls]

    def complete(self, messages, **kw):
        return {"content": "SUMMARY", "tool_calls": [], "stats": {}}
