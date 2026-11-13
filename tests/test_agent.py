import json
import sys

import pytest

from agent.memory import Memory, est_tokens
from agent.orchestrator import Agent

from conftest import FakeLLM


def run(agent, task):
    evs = list(agent.run(task))
    return evs, evs[-1]["content"]


def test_one_step_tool_call(ws):
    a = Agent(FakeLLM([("", [("list_files", {})]), ("Found hello.py", [])]))
    evs, final = run(a, "list files")
    assert final == "Found hello.py"
    assert [e["name"] for e in evs if e["type"] == "result"] == ["list_files"]
    assert "hello.py" in [e for e in evs if e["type"] == "result"][0]["text"]


def test_multi_step_tool_use(ws):
    a = Agent(FakeLLM([("", [("list_files", {})]), ("", [("read_file", {"path": "hello.py"})]),
                       ("", [("git_status", {})]), ("print statement", [])]))
    evs, final = run(a, "inspect")
    assert [e["name"] for e in evs if e["type"] == "result"] == ["list_files", "read_file", "git_status"]
    assert final == "print statement" and a.totals["steps"] == 4


def test_failed_tool_recovery(ws):
    fake = FakeLLM([("", [("read_file", {"path": "nope.py"})]), ("", [("read_file", {"path": "hello.py"})]), ("ok", [])])
    evs, final = run(Agent(fake), "read")
    res = [e for e in evs if e["type"] == "result"]
    assert [r["ok"] for r in res] == [False, True] and "FileNotFoundError" in res[0]["text"] and final == "ok"
    assert any("ERROR" in (m.get("content") or "") for m in fake.calls[1])  # model was shown the error


def test_unknown_tool_and_bad_args_are_errors_not_crashes(ws):
    a = Agent(FakeLLM([("", [("nonexistent", {}), ("read_file", {"bogus": 1})]), ("done", [])]))
    evs, final = run(a, "x")
    assert all(not e["ok"] for e in evs if e["type"] == "result") and final == "done"


def test_run_tests_and_fix(ws):
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (ws / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    py = f"{sys.executable} -m pytest -q -p no:cacheprovider 2>&1 | tail -3"
    a = Agent(FakeLLM([("", [("execute_command", {"command": py})]),
                       ("", [("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"})]),
                       ("", [("execute_command", {"command": py})]), ("fixed", [])]),
              confirm=lambda n, args, lv: lv == "write")
    evs, final = run(a, "fix")
    outs = [json.loads(e["text"])["stdout"] if e["name"] == "execute_command" else "" for e in evs if e["type"] == "result"]
    assert "failed" in outs[0] and "passed" in outs[2] and final == "fixed"


def test_permissions_write_denied_by_default_and_dangerous_never_bypassed(ws):
    seen = []
    a = Agent(FakeLLM([("", [("create_file", {"path": "x.txt", "content": "x"}),
                             ("execute_command", {"command": "rm -rf ."}), ("git_push", {})]), ("ok", [])]),
              confirm=lambda n, args, lv: seen.append((n, lv)) or (lv == "write"))  # approves WRITE only
    evs, _ = run(a, "x")
    assert seen == [("create_file", "write"), ("execute_command", "dangerous"), ("git_push", "dangerous")]
    assert (ws / "x.txt").exists()
    res = [e for e in evs if e["type"] == "result"]
    assert res[0]["ok"] and "DENIED" in res[1]["text"] and "DENIED" in res[2]["text"]
    assert (ws / "hello.py").exists()


def test_default_confirm_denies_everything(ws):
    evs, _ = run(Agent(FakeLLM([("", [("create_file", {"path": "y", "content": "y"})]), ("ok", [])])), "x")
    assert not (ws / "y").exists()


def test_max_steps_and_cancel(ws):
    a = Agent(FakeLLM([("", [("list_files", {})])] * 5), max_steps=3)
    assert "Stopped after 3 steps" in run(a, "loop")[1]


def test_context_compaction_keeps_task_and_summarises(ws):
    m = Memory("sys", 4096)  # small window to force compaction
    m.add({"role": "user", "content": "THE TASK"})
    for i in range(12):
        m.add({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}]})
        m.add({"role": "tool", "tool_name": "read_file", "content": "log line\n" * 100})
    assert est_tokens(m.prompt()) > m.budget
    assert m.compact(FakeLLM([]))
    assert est_tokens(m.prompt()) <= m.budget * 1.2
    assert m.summary == "SUMMARY" and m.messages[0]["content"] == "THE TASK"
    assert m.messages[1]["role"] != "tool"  # no orphaned tool result


# ---- live model tests (slow) ----
@pytest.mark.llm
def test_live_one_step(ws, llm):
    a = Agent(llm)
    evs, final = run(a, "List the files in the current project using a tool, then say which .py file exists.")
    assert any(e["type"] == "result" and e["ok"] for e in evs) and "hello" in final.lower()


@pytest.mark.llm
def test_live_multi_step_fix_bug_and_run_tests(ws, llm):
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (ws / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    a = Agent(llm, confirm=lambda n, args, lv: lv == "write", max_steps=12)
    evs, final = run(a, f"The tests in this project fail. Run `{sys.executable} -m pytest -q -p no:cacheprovider`, find the bug, fix it and re-run the tests.")
    assert "a + b" in (ws / "calc.py").read_text().replace("  ", " ") and len([e for e in evs if e["type"] == "tool"]) >= 3
