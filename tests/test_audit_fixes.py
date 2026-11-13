import pytest
from fastapi.testclient import TestClient

from agent.llm import parse_tool_args
from agent.memory import Memory
from tools.shell import command_level


@pytest.mark.parametrize("cmd,level", [
    ("ls & touch pwned", "write"), ("cat $'\\056env'", "write"), ("cat .en?", "write"), ("rg --pre sh x", "write"),
    ("sort -o out in", "write"), ("git branch newb", "write"), ("git -C . push", "dangerous"),
    ("rm --recursive --force d", "dangerous"), ('cat ".e""nv"', "dangerous"), ("cat ../x", "write"), ("ls -la | grep py", "read")])
def test_shell_classifier_bypasses(cmd, level):
    assert command_level(cmd) == level


def test_conversation_id_cannot_escape_history():
    with pytest.raises(ValueError): Memory("s", 8192, cid="../../x")


def test_parse_tool_args_is_forgiving():
    assert parse_tool_args("") == {} and parse_tool_args("null") == {} and parse_tool_args("{bad") == {} and parse_tool_args('{"a": 1}') == {"a": 1}


def test_server_requires_token_and_local_origin():
    from agent import server
    c = TestClient(server.app, base_url="http://127.0.0.1:8000")
    assert c.get("/conversations").status_code == 401
    h = {"X-Agent-Token": server.TOKEN}
    assert c.get("/conversations", headers=h).status_code == 200
    assert c.get("/conversations", headers={**h, "Origin": "http://evil.example"}).status_code == 403
    assert c.post("/agent/run", headers=h, json={"task": "x", "conversation_id": "../x"}).status_code == 422


@pytest.mark.parametrize("cmd,level", [
    ("cat credentials.json", "dangerous"), ("cat secrets.yml", "dangerous"), ("cat .git/config", "dangerous"),
    ("grep -f/etc/passwd x", "write"), ("grep --file=/etc/passwd x", "write"), ("git -C . reset --hard", "dangerous"),
    ("git -C . clean -fd", "dangerous"), ("git -C . branch -D x", "dangerous"), ("cat README.md", "read")])
def test_shell_classifier_secret_and_flag_path_bypasses(cmd, level):
    assert command_level(cmd) == level


def test_null_optional_args_use_defaults(ws):
    import tools
    assert "hello.py" in tools.call("list_files", {"path": None, "depth": None})


def test_empty_api_token_file_is_replaced(tmp_path, monkeypatch):
    from agent import server
    f = tmp_path / "api-token"; f.write_text("")
    monkeypatch.setattr(server, "TOKEN_FILE", f)
    t = server._token()
    assert len(t) >= 16 and f.read_text() == t and oct(f.stat().st_mode & 0o777) == "0o600"


def test_git_add_unstages_secret_beyond_clip_limit(ws):
    import subprocess, tools
    for i in range(300): (ws / f"a_long_file_name_number_{i:04d}.txt").write_text("x")
    (ws / "zz").mkdir(); (ws / "zz" / "credentials.json").write_text("{}")
    tools.call("git_add", {"paths": "."})
    staged = subprocess.run(["git", "-C", str(ws), "diff", "--staged", "--name-only"], capture_output=True, text=True).stdout
    assert "credentials.json" not in staged and "a_long_file_name_number_0299" in staged


def test_at_completion_skips_secrets(ws):
    from agent.ui import at_completions
    (ws / ".env").write_text("K=1")
    assert at_completions("hel", ws) == ["hello.py"] and at_completions(".env", ws) == []


def test_mcp_server_that_hangs_times_out_without_blocking_others(monkeypatch):
    from agent import mcp_client
    monkeypatch.setattr(mcp_client, "START_TIMEOUT", 1)
    hub = mcp_client.MCPHub({"hang": {"command": "sleep", "args": ["30"]}})
    try: assert "hang" in hub.errors and "Timeout" in hub.errors["hang"]
    finally: hub.close()
