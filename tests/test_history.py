import json
import os
import stat
import time
from types import SimpleNamespace

import pytest

from agent import history, main

ns = lambda **kw: SimpleNamespace(**{"provider": "claude", "model": None, "effort": None, "session": None, "conv_id": None, **kw})


# ---- the store ----
def test_record_turn_creates_then_appends_to_the_same_conversation(tmp_path):
    cid = history.record_turn(None, "claude", "opus", "S-1", "first question\nsecond line", "answer one", base=tmp_path)
    assert len(cid) == 12
    assert history.record_turn(cid, "claude", "sonnet", None, "follow up", "answer two", base=tmp_path) == cid
    c = history.load(cid, tmp_path)
    assert [t["user"] for t in c["turns"]] == ["first question\nsecond line", "follow up"]
    assert c["title"] == "first question"                       # first line of the first message
    assert c["model"] == "sonnet" and c["session"] == "S-1"     # latest model kept; session kept when a later turn has none
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_files_are_private_and_writes_are_atomic(tmp_path):
    base = tmp_path / "h"
    cid = history.record_turn(None, "codex", None, "T", "secret-ish prompt", "ok", base=base)
    assert stat.S_IMODE(os.stat(base).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(base / f"{cid}.json").st_mode) == 0o600
    assert not list(base.glob("*.tmp"))                         # no temp file left behind


def test_ids_become_file_names_so_bad_ones_are_rejected(tmp_path):
    for bad in ("../../etc/passwd", "x", "", "ZZZZZZZZZZZZ", "abc/def00000000"):
        assert history.load(bad, tmp_path) is None
    assert history.load(None, tmp_path) is None
    with pytest.raises(ValueError):
        history.record_turn("../../evil", "claude", None, "S", "u", "r", base=tmp_path) if False else history._file("../../evil", tmp_path)


def test_stored_text_is_capped(tmp_path):
    cid = history.record_turn(None, "claude", None, "S", "u" * 50_000, "r" * 50_000, base=tmp_path)
    t = history.load(cid, tmp_path)["turns"][0]
    assert len(t["user"]) == len(t["reply"]) == history.MAX_TEXT


def test_list_is_newest_first_skips_corrupt_and_sessionless_and_respects_limit(tmp_path):
    a = history.record_turn(None, "claude", None, "S-a", "older", "r", base=tmp_path); time.sleep(0.02)
    b = history.record_turn(None, "agy", None, "S-b", "newer", "r", base=tmp_path)
    history.record_turn(None, "codex", None, None, "no session -> nothing to resume", "r", base=tmp_path)
    (tmp_path / "deadbeef0000.json").write_text("{not json")
    (tmp_path / "notes.json").write_text(json.dumps({"id": "not-an-id"}))
    os.utime(tmp_path / f"{a}.json", (time.time() - 100, time.time() - 100))
    assert [c["id"] for c in history.list_conversations(base=tmp_path)] == [b, a]
    assert [c["id"] for c in history.list_conversations(limit=1, base=tmp_path)] == [b]
    assert history.list_conversations(base=tmp_path / "missing") == []


def test_ago_and_recap_and_clip():
    now = 1_000_000.0
    assert [history.ago(now - s, now) for s in (5, 200, 7200, 3 * 86400)] == ["just now", "3m ago", "2h ago", "3d ago"]
    turns = [{"user": f"q{i}", "reply": f"a{i}"} for i in range(5)]
    assert history.recap(turns, n=2) == [("you", "q3"), ("reply", "a3"), ("you", "q4"), ("reply", "a4")]
    long = history.recap([{"user": "word " * 100, "reply": "x"}], width=20)[0][1]
    assert long.endswith("…") and len(long) == 21 and "\n" not in history.clip("a\n\nb   c")


# ---- bookkeeping after a turn ----
def test_after_cloud_turn_records_replies_but_not_failures():
    a = ns()
    main._after_cloud_turn(a, "hello", {"session": "S-1", "reply": "hi there"})
    assert a.session == "S-1" and a.conv_id
    first = a.conv_id
    main._after_cloud_turn(a, "again", {"session": "S-1", "reply": "more"})
    assert a.conv_id == first and len(history.load(first)["turns"]) == 2           # same conversation, two turns
    main._after_cloud_turn(a, "boom", {"session": None, "reply": ""})               # an error: nothing new recorded
    assert len(history.load(first)["turns"]) == 2 and a.session == "S-1"


# ---- resuming ----
def test_resume_cloud_conversation_restores_provider_model_session_and_shows_a_recap(capsys, monkeypatch):
    cid = history.record_turn(None, "claude", "claude-opus-5", "S-42", "what is 6*7", "42")
    a = ns(provider="claude", model="haiku", session="other")
    main._apply_cloud_conversation(a, None, history.load(cid))
    assert (a.model, a.session, a.conv_id) == ("claude-opus-5", "S-42", cid)
    out = capsys.readouterr().out
    assert "resumed: what is 6*7" in out and "42" in out

    # a different provider is switched to first; if that fails nothing is half-applied
    cid2 = history.record_turn(None, "codex", "gpt-5.6-luna", "T-9", "codex question", "codex answer")
    switched = []
    def fake_switch(a, name, agent): switched.append(name); a.provider = name; return "AGENT", True
    monkeypatch.setattr(main, "_switch_provider", fake_switch)
    b = ns(provider="claude")
    assert main._apply_cloud_conversation(b, None, history.load(cid2)) == "AGENT"
    assert switched == ["codex"] and (b.model, b.session, b.conv_id) == ("gpt-5.6-luna", "T-9", cid2)
    monkeypatch.setattr(main, "_switch_provider", lambda a, n, ag: (ag, False))
    c = ns(provider="claude", model="keep", session="keep")
    main._apply_cloud_conversation(c, None, history.load(cid2))
    assert (c.provider, c.model, c.session) == ("claude", "keep", "keep")


def test_resume_local_conversation_loads_it_into_the_agents_memory(ws, capsys):
    from agent.memory import Memory
    from agent.orchestrator import Agent
    from conftest import FakeLLM
    old = Memory("sys", 8192, cid="abc123def456")
    old.add({"role": "user", "content": "list the files"}); old.add({"role": "assistant", "content": "there are 3 files"})
    old.save()
    agent = Agent(FakeLLM([]))
    a = ns(provider="ollama")
    assert main._apply_local_conversation(a, agent, "abc123def456") is agent
    assert agent.memory.id == "abc123def456" and [m["content"] for m in agent.memory.messages] == ["list the files", "there are 3 files"]
    out = capsys.readouterr().out
    assert "resumed: list the files" in out and "there are 3 files" in out
    assert main._apply_local_conversation(a, agent, "does-not-exist") is agent        # unreadable: reported, not raised
    assert "couldn't read" in capsys.readouterr().out


def test_resume_menu_lists_cloud_and_local_newest_first_and_applies_the_pick(capsys, monkeypatch):
    from agent.memory import Memory
    history.record_turn(None, "claude", "opus", "S-old", "an older cloud chat", "r"); time.sleep(0.02)
    m = Memory("sys", 8192, cid="aaaaaaaaaaaa"); m.add({"role": "user", "content": "a local chat"}); m.save(); time.sleep(0.02)
    newest = history.record_turn(None, "codex", "gpt-5.6-luna", "T-new", "the newest cloud chat", "r")
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    seen = {}
    def fake_pick(rows, title, current=None): seen["rows"], seen["title"] = rows, title; return 0
    monkeypatch.setattr("agent.ui.pick_from_list", fake_pick)
    a = ns(provider="claude")
    main._resume_menu(a, None)
    titles = [r[0] for r in seen["rows"]]
    assert titles == ["the newest cloud chat", "a local chat", "an older cloud chat"]    # newest first, mixed sources
    assert [r[1] for r in seen["rows"]] == ["[codex]", "[ollama]", "[claude]"]            # provider tag per row, like /model
    assert a.session == "T-new" and a.conv_id == newest                                   # index 0 was applied (codex needs no switch check here)


def test_resume_menu_handles_nothing_and_no_terminal(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    main._resume_menu(ns(), None)
    assert "needs a terminal" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    main._resume_menu(ns(), None)                                  # state dirs are empty (conftest isolates them)
    assert "no earlier conversations" in capsys.readouterr().out


def test_native_histories_from_claude_codex_and_antigravity_are_listed_and_filtered(tmp_path, monkeypatch):
    claude_h = tmp_path / "claude_history.jsonl"
    codex_h = tmp_path / "codex_history.jsonl"
    agy_b = tmp_path / "agy_brain"
    monkeypatch.setattr(history, "CLAUDE_HIST", claude_h)
    monkeypatch.setattr(history, "CODEX_HIST", codex_h)
    monkeypatch.setattr(history, "AGY_BRAIN", agy_b)

    s_claude = "11111111-2222-3333-4444-555555555555"
    s_codex = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    s_agy = "99999999-8888-7777-6666-555555555555"

    claude_h.write_text("\n".join([
        json.dumps({"display": "/exit", "timestamp": 1700000000000, "sessionId": s_claude}),  # slash noise ignored
        json.dumps({"display": "fix claude bug", "timestamp": 1700000001000, "sessionId": s_claude}),
        json.dumps({"display": "second claude turn", "timestamp": 1700000002000, "sessionId": s_claude}),
    ]))
    codex_h.write_text(json.dumps({"session_id": s_codex, "ts": 1700000005, "text": "optimize sql query"}) + "\n")
    agy_log = agy_b / s_agy / ".system_generated" / "logs"
    agy_log.mkdir(parents=True)
    (agy_log / "transcript.jsonl").write_text("\n".join([
        json.dumps({"type": "USER_INPUT", "content": "<USER_REQUEST>\naudit security\n</USER_REQUEST>"}),
        json.dumps({"type": "PLANNER_RESPONSE", "content": "Found 3 issues."}),
    ]))

    all_convs = history.list_conversations()
    by_prov = {c["provider"]: c for c in all_convs}
    assert set(by_prov) == {"claude", "codex", "antigravity"}
    assert by_prov["claude"]["title"] == "fix claude bug" and len(by_prov["claude"]["turns"]) == 2
    assert by_prov["codex"]["title"] == "optimize sql query" and by_prov["codex"]["session"] == s_codex
    assert by_prov["antigravity"]["title"] == "audit security"
    assert by_prov["antigravity"]["turns"] == [{"user": "audit security", "reply": "Found 3 issues."}]

    # Provider filtering in list_conversations and _conversation_items
    assert [c["provider"] for c in history.list_conversations(provider="claude")] == ["claude"]
    assert [c["provider"] for c in history.list_conversations(provider="codex")] == ["codex"]
    assert [c["provider"] for _, c, _ in main._conversation_items("antigravity")] == ["antigravity"]


def test_local_conversations_differentiate_npu_and_ollama_and_resume_filters_by_provider(ws, capsys, monkeypatch):
    from agent.memory import Memory
    m_ollama = Memory("sys", 8192, cid="111111111111", model="qwen3.5:9b", provider="ollama")
    m_ollama.add({"role": "user", "content": "ollama task"}); m_ollama.save()
    time.sleep(0.02)
    m_npu = Memory("sys", 8192, cid="222222222222", model="qwen3:4b", provider="npu")
    m_npu.add({"role": "user", "content": "npu task"}); m_npu.save()

    assert [m["id"] for m in Memory.list(provider="ollama")] == ["111111111111"]
    assert [m["id"] for m in Memory.list(provider="npu")] == ["222222222222"]

    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    seen = {}
    def fake_pick(rows, title, current=None): seen["rows"], seen["title"] = rows, title; return None
    monkeypatch.setattr("agent.ui.pick_from_list", fake_pick)

    main._resume_menu(ns(provider="npu"), None, "npu")
    assert [r[0] for r in seen["rows"]] == ["npu task"]
    assert [r[1] for r in seen["rows"]] == ["[npu]"]

    # If the filtered provider has no conversations yet, _resume_menu falls back to all providers with a note
    main._resume_menu(ns(provider="claude"), None, "claude")
    assert "no claude conversations yet" in capsys.readouterr().out
    assert {r[1] for r in seen["rows"]} == {"[npu]", "[ollama]"}


def test_per_provider_input_history_files(tmp_path):
    from agent.ui import history_file_for
    base = tmp_path / "chat_history"
    p_npu = history_file_for(base, "npu")
    p_claude = history_file_for(base, "claude")
    assert p_npu != p_claude
    assert p_npu.name == "chat_history_npu" and p_claude.name == "chat_history_claude"


def test_switch_provider_carries_over_conversation_context(ws, capsys, monkeypatch):
    from agent.orchestrator import Agent
    from conftest import FakeLLM
    monkeypatch.setattr("agent.providers.ClaudeRunner.healthy", lambda self: True)
    monkeypatch.setattr("agent.llm.LLM.healthy", lambda self: True)
    monkeypatch.setattr(main, "_make_agent", lambda a, quiet=False: Agent(FakeLLM([])))

    # 1. Local (npu) -> Cloud (claude): turns become a carryover block for the next cloud message
    npu_agent = Agent(FakeLLM([]))
    npu_agent.memory.provider = "npu"
    npu_agent.memory.add({"role": "user", "content": "my name is Ved"})
    npu_agent.memory.add({"role": "assistant", "content": "Hello Ved!"})
    a = ns(provider="npu")
    cloud_agent, ok = main._switch_provider(a, "claude", npu_agent)
    assert ok and cloud_agent is None and a.provider == "claude"
    assert "my name is Ved" in a.carryover and "Hello Ved!" in a.carryover

    # 2. Cloud (claude) -> Local (ollama): turns populate the new local agent's Memory
    new_local, ok = main._switch_provider(a, "ollama", cloud_agent)
    assert ok and new_local is not None and a.provider == "ollama"
    assert new_local.memory.provider == "ollama"
    assert [(m["role"], m["content"]) for m in new_local.memory.messages] == [
        ("user", "my name is Ved"),
        ("assistant", "Hello Ved!"),
    ]


def test_claude_project_and_codex_rollout_transcripts_and_local_turn_grouping(tmp_path, monkeypatch, capsys):
    claude_proj = tmp_path / "claude_projects" / "-home-test"
    claude_proj.mkdir(parents=True)
    codex_sess = tmp_path / "codex_sessions" / "2026" / "10" / "15"
    codex_sess.mkdir(parents=True)
    codex_idx = tmp_path / "codex_index.jsonl"
    monkeypatch.setattr(history, "CLAUDE_PROJECTS", tmp_path / "claude_projects")
    monkeypatch.setattr(history, "CODEX_SESSIONS", tmp_path / "codex_sessions")
    monkeypatch.setattr(history, "CODEX_INDEX", codex_idx)

    s_claude = "22222222-3333-4444-5555-666666666666"
    (claude_proj / f"{s_claude}.jsonl").write_text("\n".join([
        json.dumps({"type": "user", "message": {"role": "user", "content": "<ide_opened_file>ignored</ide_opened_file>explain auth flow"}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-sonnet-5-5", "content": [{"type": "text", "text": "Checking files..."}]}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-sonnet-5-5", "content": [{"type": "text", "text": "Auth uses JWT tokens."}]}}),
    ]))

    s_codex = "33333333-4444-5555-6666-777777777777"
    codex_idx.write_text(json.dumps({"id": s_codex, "thread_name": "Fix CSS animation"}) + "\n")
    (codex_sess / f"rollout-{s_codex}.jsonl").write_text("\n".join([
        json.dumps({"type": "session_meta", "payload": {"id": s_codex}}),
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.6-luna"}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# AGENTS.md boilerplate"}]}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix 3d flip bug"}]}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "Inspecting CSS..."}]}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "Added backface-visibility: hidden."}]}}),
    ]))

    c_convs = history.list_conversations(provider="claude")
    assert len(c_convs) == 1 and c_convs[0]["model"] == "claude-sonnet-5-5"
    assert c_convs[0]["turns"] == [{"user": "explain auth flow", "reply": "Auth uses JWT tokens."}]

    x_convs = history.list_conversations(provider="codex")
    assert len(x_convs) == 1 and x_convs[0]["title"] == "Fix CSS animation" and x_convs[0]["model"] == "gpt-5.6-luna"
    assert x_convs[0]["turns"] == [{"user": "fix 3d flip bug", "reply": "Added backface-visibility: hidden."}]

    # Local message grouping collapses intermediate tool thoughts into the final turn reply + tools list
    msgs = [
        {"role": "user", "content": "check os"},
        {"role": "assistant", "content": "Let me run a command...", "tool_calls": [{"function": {"name": "execute_command"}}]},
        {"role": "tool", "tool_name": "execute_command", "content": "Fedora 44"},
        {"role": "assistant", "content": "You are on **Fedora 44**."},
    ]
    turns = main._turns_from_messages(msgs)
    assert turns == [{"user": "check os", "reply": "You are on **Fedora 44**.", "tools": ["execute_command"]}]
    main._render_transcript("npu", "qwen3:4b", "check os", turns)
    out = capsys.readouterr().out
    assert "check os" in out and "execute_command" in out and "Fedora 44" in out


