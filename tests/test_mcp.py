import pytest

from agent.mcp_client import MCPHub, load_config
from agent.orchestrator import Agent

from conftest import FakeLLM


@pytest.fixture
def hub(ws):
    h = MCPHub(load_config(), only=["local"])  # `enabled: false` in config only affects the default agent
    yield h
    h.close()


def test_mcp_server_discovery(hub):
    assert not hub.errors
    assert {"local__read_file", "local__git_status", "local__execute_command", "local__list_files"} <= hub.tools
    assert "local__git_push" not in hub.tools  # dangerous tools are not exposed over MCP
    assert all(s["function"]["parameters"]["type"] == "object" for s in hub.schemas)


def test_mcp_tool_execution_and_sandbox(hub, ws):
    assert "print('hi')" in hub.call("local__read_file", {"path": "hello.py"})
    with pytest.raises(RuntimeError, match="outside the workspace"):
        hub.call("local__read_file", {"path": "/etc/passwd"})


def test_mcp_permission_levels(hub):
    assert hub.level("local__read_file", {"path": "x"}) == "read"
    assert hub.level("local__create_file", {}) == "write"
    assert hub.level("local__execute_command", {"command": "git push"}) == "dangerous"
    assert hub.level("local__execute_command", {"command": "ls"}) == "write"  # not in config "read" list: asks unless...


def test_agent_uses_mcp_tool(hub, ws):
    a = Agent(FakeLLM([("", [("local__git_status", {})]), ("clean", [])]), mcp=hub)
    evs = list(a.run("status"))
    r = [e for e in evs if e["type"] == "result"][0]
    assert r["ok"] and evs[-1]["content"] == "clean"


def test_hub_started_in_wrong_project_is_rejected(hub, ws, monkeypatch):
    other = ws.parent / "other"
    other.mkdir()
    monkeypatch.setenv("AGENT_CWD", str(other))  # project changed after the hub started
    with pytest.raises(ValueError, match="MCP hub was started in"):
        Agent(FakeLLM([]), mcp=hub)


def test_mcp_server_runs_in_hub_project_dir(hub, ws):
    assert "hello.py" in hub.call("local__git_status", {}) or "master" in hub.call("local__git_status", {})
    assert hub.cwd == str(ws)
