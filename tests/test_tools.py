import json

import pytest

import tools
from tools import DANGEROUS, READ, WRITE
from tools.shell import command_level


def test_fs_read_and_search(ws):
    assert "1\tprint('hi')" in tools.call("read_file", {"path": "hello.py"})
    assert "hello.py:1" in tools.call("search_files", {"pattern": "print"})
    assert "hello.py" in tools.call("list_files", {})


def test_fs_write_and_edit(ws):
    tools.call("create_file", {"path": "sub/a.txt", "content": "one two"})
    tools.call("edit_file", {"path": "sub/a.txt", "old": "two", "new": "2"})
    assert (ws / "sub/a.txt").read_text() == "one 2"
    with pytest.raises(FileExistsError): tools.call("create_file", {"path": "sub/a.txt", "content": "x"})
    with pytest.raises(ValueError): tools.call("edit_file", {"path": "sub/a.txt", "old": "zzz", "new": "y"})


def test_sandbox_blocks_outside_and_secrets(ws):
    (ws / ".env").write_text("TOKEN=abc")
    for bad in ("/etc/passwd", "../../../etc/hostname", "~/.ssh/id_rsa", ".env"):
        with pytest.raises(PermissionError): tools.call("read_file", {"path": bad})
    (ws / "link").symlink_to("/etc")
    with pytest.raises(PermissionError): tools.call("list_files", {"path": "link"})
    assert ".env" not in tools.call("list_files", {}) and "abc" not in tools.call("search_files", {"pattern": "TOKEN"})
    with pytest.raises(PermissionError): tools.call("create_file", {"path": ".git/hooks/x", "content": "x"})


def test_shell_captures_everything(ws):
    r = json.loads(tools.call("execute_command", {"command": "echo out; echo err >&2; exit 3"}))
    assert (r["stdout"].strip(), r["stderr"].strip(), r["exit_code"]) == ("out", "err", 3)
    assert "execution_time" in r


def test_shell_timeout_kills_process_group(ws):
    r = json.loads(tools.call("execute_command", {"command": "sleep 30 & sleep 30", "timeout": 1}))
    assert r["exit_code"] == -1 and r["execution_time"] < 5


def test_shell_env_hides_secrets(ws, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "s3cret")
    assert "s3cret" not in tools.call("execute_command", {"command": "echo $MY_API_KEY"})


@pytest.mark.parametrize("cmd,level", [
    ("ls -la | grep py", READ), ("git status", READ), ("git diff HEAD", READ), ("cat hello.py", READ),
    ("pytest -q", WRITE), ("echo hi > f.txt", WRITE), ("find . -delete", WRITE), ("sed -i s/a/b/ f", WRITE),
    ("sudo dnf install x", DANGEROUS), ("rm -rf build", DANGEROUS), ("git push origin main", DANGEROUS),
    ("git push --force", DANGEROUS), ("git reset --hard HEAD~1", DANGEROUS), ("psql -c 'DROP TABLE users'", DANGEROUS),
    ("docker system prune -af", DANGEROUS), ("cat .env", DANGEROUS), ("cat ~/.ssh/id_rsa", DANGEROUS),
    ("ls && rm -r x", DANGEROUS), ("systemctl restart ollama", DANGEROUS), ("echo hi; env", DANGEROUS)])
def test_command_classification(cmd, level):
    assert command_level(cmd) == level


def test_git_status_and_diff(ws):
    assert "master" in tools.call("git_status", {}) or "main" in tools.call("git_status", {})
    (ws / "hello.py").write_text("print('changed')\n")
    (ws / ".env").write_text("TOKEN=abc")
    assert " M hello.py" in tools.call("git_status", {})
    d = tools.call("git_diff", {})
    assert "+print('changed')" in d and "TOKEN" not in d


def test_git_commit_log_and_add_never_stages_secrets(ws):
    (ws / "b.txt").write_text("b")
    (ws / ".env").write_text("TOKEN=abc")
    tools.call("git_add", {"paths": "."})
    tools.call("git_commit", {"message": "add b"})
    assert "add b" in tools.call("git_log", {})
    assert ".env" not in tools.call("execute_command", {"command": "git ls-files"})


def test_git_dangerous_tools_need_confirmation():
    for n in ("git_push", "git_reset_hard", "git_branch_delete"):
        assert tools.level_of(n, {}) == DANGEROUS
    for n in ("git_commit", "create_file", "edit_file"): assert tools.level_of(n, {}) == WRITE
    for n in ("read_file", "list_files", "search_files", "git_status", "git_diff"): assert tools.level_of(n, {}) == READ
