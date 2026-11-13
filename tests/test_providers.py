import json

import pytest

from agent.detect import KNOWN_AGENTS, detect_agents, detect_npu
from agent.providers import (AgyRunner, ClaudeRunner, EXTERNAL_AGENTS, NPUProvider, make_provider,
                             parse_agy_line, parse_claude_line)


def test_make_provider_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider("bogus")


def test_make_provider_ollama_and_npu_construct_without_network():
    assert make_provider("ollama").model
    assert make_provider("npu").model  # NPUProvider: constructing must not probe the (absent) server


def test_detect_npu_shape():
    d = detect_npu()
    assert "platform" in d and isinstance(d["present"], list)
    for n in d["present"]:
        assert {"vendor", "verified_on_this_machine"} <= n.keys()


def test_detect_agents_returns_known_names_only():
    found = detect_agents()
    assert isinstance(found, list)
    assert all(a["name"] in KNOWN_AGENTS for a in found)
    assert all({"name", "path", "version"} <= a.keys() for a in found)


def test_parse_agy_line_text_delta():
    line = json.dumps({"event": "step_update", "step_update": {
        "step_type": "agent_response", "text_delta": "hi"}})
    assert parse_agy_line(line) == {"type": "text", "delta": "hi"}


def test_parse_agy_line_result():
    line = json.dumps({"event": "result", "result": {"response": "done text"}})
    assert parse_agy_line(line) == {"type": "_result", "content": "done text", "session": None}
    # the conversation id is what lets the next message continue the same conversation
    line = json.dumps({"event": "result", "result": {"response": "x", "conversation_id": "abc-123"}})
    assert parse_agy_line(line)["session"] == "abc-123"


def test_parse_agy_line_tool_call():
    line = json.dumps({"event": "step_update", "step_update": {
        "state": "ACTIVE", "step_type": "tool", "tool_name": "run_command",
        "tool_info": {"parameters": {"CommandLine": "echo hi"}}}})
    ev = parse_agy_line(line)
    assert ev["type"] == "tool" and ev["name"] == "run_command" and ev["text"] == "agy: run_command echo hi"
    assert parse_agy_line(line.replace("ACTIVE", "DONE")) is None  # shown once, at start


def test_parse_agy_line_ignores_junk():
    assert parse_agy_line("not json") is None
    assert parse_agy_line(json.dumps({"event": "init"})) is None


def test_agy_runner_healthy_reflects_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
    assert AgyRunner().healthy()
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert not AgyRunner().healthy()


def test_parse_claude_line_text():
    line = json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}})
    assert parse_claude_line(line) == {"type": "text", "delta": "hi"}
    # the final whole-message text repeats the deltas, so it is skipped
    assert parse_claude_line(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}})) is None


def test_parse_claude_line_result():
    line = json.dumps({"type": "result", "result": "done text"})
    assert parse_claude_line(line) == {"type": "_result", "content": "done text", "session": None, "denied": []}
    line = json.dumps({"type": "result", "result": "x", "session_id": "sess-9", "permission_denials": [
        {"tool_name": "Write", "tool_use_id": "t1", "tool_input": {"file_path": "/tmp/a.txt", "content": "hi"}},
        {"tool_name": "Bash", "tool_input": {"command": "git push"}}]})
    ev = parse_claude_line(line)
    assert ev["session"] == "sess-9" and ev["denied"] == ["Write /tmp/a.txt", "Bash git push"]


def test_parse_claude_line_tool_use():
    line = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {}}]}})
    ev = parse_claude_line(line)
    assert ev["type"] == "tool" and ev["name"] == "Bash"


def test_parse_claude_line_ignores_system_noise():
    # claude's stream-json carries its own hook/session-start events, unlike agy's leaner stream
    assert parse_claude_line("not json") is None
    assert parse_claude_line(json.dumps({"type": "system", "subtype": "hook_started"})) is None


def test_claude_runner_healthy_reflects_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    assert ClaudeRunner().healthy()
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert not ClaudeRunner().healthy()


def test_external_agents_registry():
    from agent.providers import CodexRunner
    assert set(EXTERNAL_AGENTS) == {"antigravity", "claude", "codex"}
    assert EXTERNAL_AGENTS["antigravity"] is AgyRunner and EXTERNAL_AGENTS["claude"] is ClaudeRunner
    assert EXTERNAL_AGENTS["codex"] is CodexRunner


def test_complete_command_local_provider():
    from agent.main import complete_command
    assert complete_command("/pro", "ollama") == ["/provider"]
    from agent.main import COMMANDS
    assert set(complete_command("/", "ollama")) == {c.split()[0] for c in COMMANDS} - {"/effort"}  # /effort is claude-only
    assert complete_command("not-a-command", "ollama") == []


def test_complete_command_antigravity_provider():
    from agent.main import complete_command
    assert complete_command("/mod", "antigravity") == ["/model"]
    assert complete_command("/sk", "antigravity") == ["/skills"]
    matches = complete_command("/", "antigravity")
    assert "/skills" in matches and "/provider" in matches and "/help" in matches


def test_complete_command_claude_provider_gets_local_set():
    # claude has no native print-mode commands (verified separately), so it completes against
    # this project's own commands, same as ollama/npu - not agy's native set.
    from agent.main import complete_command
    assert complete_command("/pro", "claude") == ["/provider"]
    assert complete_command("/sk", "claude") == []  # /skills is agy-only


def test_effort_completion_only_for_claude():
    # effort is a claude setting; agy's is part of its model ids, local providers have none
    from agent.main import complete_command
    assert complete_command("/ef", "claude") == ["/effort"]
    for p in ("antigravity", "ollama", "npu"): assert complete_command("/ef", p) == [], p


def test_runner_cmd_passes_model_and_effort_flags():
    c = ClaudeRunner("claude-opus-5", "high").cmd("hi")
    assert c[:3] == ["claude", "-p", "hi"] and c[-4:] == ["--model", "claude-opus-5", "--effort", "high"]
    # agy: model only, NEVER --effort - its model ids already carry the effort and agy rejects the
    # combination outright ("--model gemini-4-argon-medium conflicts with --effort=high", verified)
    a = AgyRunner("gemini-4-argon-low").cmd("hi")
    assert a[-2:] == ["--model", "gemini-4-argon-low"] and "--effort" not in a
    assert not AgyRunner.supports_effort and "--effort" not in AgyRunner(effort="high").cmd("hi")  # accepted (ExternalAgent signature) but never forwarded
    # nothing pinned -> no flags at all (defaults left to the CLI)
    assert "--model" not in ClaudeRunner().cmd("hi") and "--effort" not in ClaudeRunner().cmd("hi")


def test_claude_model_list_is_sane():
    from agent.main import CLAUDE_MODELS, EFFORT_LEVELS
    ids = [m for m, _ in CLAUDE_MODELS]
    assert len(ids) == len(set(ids))                        # no duplicates
    assert all(m in ("opus", "sonnet", "fable") or m.startswith("claude-") for m in ids)
    assert EFFORT_LEVELS == ("low", "medium", "high", "xhigh", "max")


def test_runner_cmd_resumes_the_same_conversation():
    assert ClaudeRunner(session="s-1").cmd("hi")[-2:] == ["--resume", "s-1"]
    assert AgyRunner(session="c-1").cmd("hi")[-2:] == ["--conversation", "c-1"]
    assert "--resume" not in ClaudeRunner().cmd("hi") and "--conversation" not in AgyRunner().cmd("hi")


def test_external_agent_done_event_carries_session_and_denials():
    # real subprocess, no cloud: a tiny script printing the same NDJSON claude would
    import sys
    from agent.providers import _run_external_agent
    script = ("import json;"
              "print(json.dumps({'type':'stream_event','event':{'delta':{'type':'text_delta','text':'Tea or coffee?'}}}));"
              "print(json.dumps({'type':'result','result':'Tea or coffee?','session_id':'S1',"
              "'permission_denials':[{'tool_name':'Bash','tool_input':{'command':'rm x'}}]}))")
    evs = list(_run_external_agent([sys.executable, "-c", script], parse_claude_line))
    assert [e["type"] for e in evs] == ["text", "done"]
    assert evs[-1]["session"] == "S1" and evs[-1]["denied"] == ["Bash rm x"]


def test_external_agent_child_cannot_read_the_terminal():
    # stdin must be /dev/null: a child that reads stdin sees EOF at once instead of eating keystrokes
    import sys
    from agent.providers import _run_external_agent
    script = "import sys, json; print(json.dumps({'type':'result','result':repr(sys.stdin.read())}))"
    evs = list(_run_external_agent([sys.executable, "-c", script], parse_claude_line))
    assert evs[-1]["content"] == "''"


def _agy_transcript(tmp_path, cid, entries):
    d = tmp_path / cid / ".system_generated" / "logs"
    d.mkdir(parents=True)
    (d / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in entries))


CID = "882e2f12-2a1f-426b-9f18-23cc4827e0e9"


def test_agy_skipped_question_is_recovered_from_its_transcript(tmp_path):
    import time
    from datetime import datetime, timezone
    from agent.providers import agy_skipped_questions
    now = time.time()
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ask = lambda q: {"name": "ask_question", "args": {"questions": json.dumps([{"question": q, "options": ["Tea", "Coffee"]}])}}
    _agy_transcript(tmp_path, CID, [
        {"created_at": ts(now - 600), "tool_calls": [ask("an OLD question from an earlier turn")]},
        {"created_at": ts(now), "tool_calls": [ask("Do you prefer tea or coffee?"), {"name": "run_command", "args": {}}]},
        {"created_at": ts(now), "content": "no tool calls here"},
        "not even a dict",
    ])
    qs = agy_skipped_questions(CID, since=now - 5, brain=tmp_path)
    assert qs == [{"question": "Do you prefer tea or coffee?", "options": ["Tea", "Coffee"]}]


def test_agy_skipped_question_lookup_fails_soft(tmp_path):
    from agent.providers import agy_skipped_questions
    assert agy_skipped_questions(None, 0, tmp_path) == []
    assert agy_skipped_questions("../../etc", 0, tmp_path) == []        # not a uuid -> never touches a path
    assert agy_skipped_questions(CID, 0, tmp_path) == []                 # no transcript on disk
    _agy_transcript(tmp_path, CID, [])
    (tmp_path / CID / ".system_generated" / "logs" / "transcript.jsonl").write_text("{garbage\n[1,2]\n")
    assert agy_skipped_questions(CID, 0, tmp_path) == []                 # unreadable lines are skipped, not fatal


def test_help_hides_effort_for_providers_without_it():
    from agent.main import help_lines
    names = lambda p: {c.split()[0] for c, _ in help_lines(p)}
    for p in ("ollama", "npu", "antigravity"): assert "/effort" not in names(p), p
    assert "/effort" in names("claude")
    assert {"/help", "/provider", "/model", "/new", "/exit"} <= names("ollama")  # everything else stays


def test_render_shows_errors_instead_of_going_silent(capsys):
    # regression: error events were dropped, so a rejected model/effort combo looked like "no response"
    from agent.main import _render
    done = _render(iter([{"type": "error", "error": "invalid model selection: conflicts with --effort=high\n"},
                         {"type": "done", "content": "", "session": None}]))
    out = capsys.readouterr().out
    assert "invalid model selection" in out and "failed" in out and "done" not in out.replace("failed", "")
    assert done["type"] == "done"


def test_render_success_still_says_done(capsys):
    from agent.main import _render
    _render(iter([{"type": "text", "delta": "ok"}, {"type": "done", "content": "ok"}]))
    out = capsys.readouterr().out
    assert "ok" in out and "done" in out and "failed" not in out


def test_unsupported_effort_flag_is_dropped_with_a_note(capsys):
    from types import SimpleNamespace
    from agent.main import _drop_unsupported_effort
    a = SimpleNamespace(provider="antigravity", effort="high")
    _drop_unsupported_effort(a)
    assert a.effort is None and "ignored" in capsys.readouterr().out
    b = SimpleNamespace(provider="claude", effort="high")
    _drop_unsupported_effort(b)
    assert b.effort == "high"


# ---- unified /model catalog ----
CATALOG = [("ollama", "qwen3.5:9b", ""), ("antigravity", "gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
           ("antigravity", "claude-opus-5-5-high", "Claude Opus 5.5 (High)"), ("claude", "claude-opus-5-5", "Opus 5.5"),
           ("claude", "opus", "Opus (alias: latest)")]


def test_resolve_model():
    from agent.main import resolve_model
    assert resolve_model("claude-opus-5-5", "antigravity", CATALOG) == ("claude", "claude-opus-5-5")   # switches provider
    assert resolve_model("qwen3.5:9b", "claude", CATALOG) == ("ollama", "qwen3.5:9b")
    assert resolve_model("claude/claude-opus-9", "ollama", CATALOG) == ("claude", "claude-opus-9")     # id not in any list yet
    assert resolve_model("hf.co/user/model", "ollama", CATALOG) is None   # "/" in an ollama name is not a provider prefix
    assert resolve_model("nope", "claude", CATALOG) is None
    dup = CATALOG + [("claude", "gemini-4-argon-medium", "")]
    assert resolve_model("gemini-4-argon-medium", "claude", dup) == ("claude", "gemini-4-argon-medium")  # same id: current provider wins
    assert resolve_model("gemini-4-argon-medium", "ollama", dup) == ("antigravity", "gemini-4-argon-medium")


def test_model_catalog_merges_providers_in_priority_order_and_skips_unreachable():
    from agent.main import _model_catalog
    rows = {"ollama": [], "npu": [("n1", "NPU one")], "antigravity": [("a1", "A one")],   # ollama is down
            "claude": [("c1", "C one"), ("c2", "")], "codex": [("x1", "X one")]}
    assert _model_catalog(lambda p: rows[p]) == [("npu", "n1", "NPU one"), ("antigravity", "a1", "A one"), ("claude", "c1", "C one"),
                                                 ("claude", "c2", ""), ("codex", "x1", "X one")]
    assert _model_catalog(lambda p: []) == []


def test_box_title_shows_model_and_provider_bracket():
    from agent.main import box_title
    assert box_title("antigravity", "gemini-4-argon-low") == "you · gemini-4-argon-low [antigravity]"
    assert box_title("claude", None, "high") == "you · default [claude] · high"


def test_switching_to_a_dead_local_provider_does_not_kill_butler(capsys, monkeypatch):
    # regression: /provider ollama (or picking an ollama model) with ollama down used to hit
    # _make_agent()'s sys.exit() and quit the whole chat
    from types import SimpleNamespace
    from agent.main import _switch_provider
    monkeypatch.setattr("agent.providers.NPUProvider.healthy", lambda self: False)
    a = SimpleNamespace(provider="claude", model="opus", effort="high", session="s1")
    agent, ok = _switch_provider(a, "npu", None)
    assert ok is False and agent is None
    assert (a.provider, a.model, a.effort, a.session) == ("claude", "opus", "high", "s1")   # nothing changed
    assert "down" in capsys.readouterr().out


def test_ollama_models_come_from_the_endpoint_butler_uses(monkeypatch):
    import httpx
    from agent import main
    seen = []
    def fake_get(url, timeout):
        seen.append(url)
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}, {"name": "llama3:8b"}]})
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(main.config, "LLM_URL", "http://127.0.0.1:11435")
    assert main._list_ollama_models() == [("qwen3.5:9b", ""), ("llama3:8b", "")]
    assert seen == ["http://127.0.0.1:11435/api/tags"]                                           # not the `ollama list` CLI's port
    monkeypatch.setattr(httpx, "get", lambda url, timeout: (_ for _ in ()).throw(httpx.ConnectError("down")))
    assert main._list_ollama_models() == []


# ---- codex ----
def test_parse_codex_line_uses_the_real_event_shapes():
    from agent.providers import parse_codex_line as P
    J = json.dumps
    assert P(J({"type": "thread.started", "thread_id": "T-1"})) == {"type": "_meta", "session": "T-1"}
    assert P(J({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})) == {"type": "text", "delta": "ok\n"}
    tool = P(J({"type": "item.started", "item": {"type": "command_execution", "command": "/bin/bash -lc 'echo butler-probe'"}}))
    assert tool["type"] == "tool" and tool["text"] == "codex: echo butler-probe"
    for noise in ({"type": "turn.started"}, {"type": "turn.completed", "usage": {}},
                  {"type": "item.completed", "item": {"type": "command_execution"}},                  # result of a tool: not re-announced
                  {"type": "item.started", "item": {"type": "reasoning"}},
                  {"type": "item.completed", "item": {"type": "error", "message": "Model metadata ... fallback"}},   # non-fatal warning
                  {"type": "error", "message": "x"}):                                                  # turn.failed carries the same text
        assert P(J(noise)) is None, noise
    assert P("not json") is None


def test_parse_codex_failure_unwraps_the_json_message():
    from agent.providers import parse_codex_line as P
    inner = json.dumps({"type": "error", "status": 400, "error": {"type": "invalid_request_error",
            "message": "The 'x' model is not supported when using Codex with a ChatGPT account."}})
    ev = P(json.dumps({"type": "turn.failed", "error": {"message": inner}}))
    assert ev == {"type": "error", "error": "The 'x' model is not supported when using Codex with a ChatGPT account."}
    assert P(json.dumps({"type": "turn.failed", "error": {"message": "plain text"}}))["error"] == "plain text"


def test_codex_runner_cmd_forms():
    from agent.providers import CodexRunner
    assert CodexRunner().cmd("hi") == ["codex", "exec", "--json", "--skip-git-repo-check", "hi"]
    assert CodexRunner("gpt-5.6-luna").cmd("hi")[2:6] == ["--json", "--skip-git-repo-check", "-m", "gpt-5.6-luna"]
    r = CodexRunner("m", "T-9").cmd("hi")                                      # follow-up -> resume the same thread
    assert r[:5] == ["codex", "exec", "resume", "T-9", "hi"] and r[-2:] == ["-m", "m"]
    assert CodexRunner().cmd("- fix it")[-1] == " - fix it"                      # a leading "-" must not become a flag
    assert not any(a.startswith("--dangerously") or a in ("-s", "--sandbox") for a in r)   # permissions stay the user's own config


def test_shared_runner_merges_meta_and_does_not_report_an_error_twice():
    import sys
    from agent.providers import _run_external_agent, parse_codex_line
    script = ("import json,sys;"
              "print(json.dumps({'type':'thread.started','thread_id':'T-7'}));"
              "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'hello'}}));"
              "print(json.dumps({'type':'turn.failed','error':{'message':'boom'}}));"
              "sys.stderr.write('Reading additional input from stdin...'); sys.exit(1)")
    evs = list(_run_external_agent([sys.executable, "-c", script], parse_codex_line))
    assert [e["type"] for e in evs] == ["text", "error", "done"]                 # exactly ONE error despite the non-zero exit
    assert evs[1]["error"] == "boom" and evs[-1]["session"] == "T-7"             # session from the first event reaches "done"


def test_codex_models_listing_filters_hidden_and_orders_by_priority(monkeypatch):
    from types import SimpleNamespace
    from agent import main
    catalog = {"models": [
        {"slug": "gpt-5.6-luna", "display_name": "GPT-5.6-Luna", "visibility": "list", "priority": 9},
        {"slug": "gpt-reserve", "display_name": "internal", "visibility": "hide", "priority": 4},
        {"slug": "gpt-6-luna", "display_name": "GPT-6-Luna", "visibility": "list", "priority": 4}]}
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(catalog), returncode=0))
    assert main._list_codex_models() == [("gpt-6-luna", "GPT-6-Luna"), ("gpt-5.6-luna", "GPT-5.6-Luna")]
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="garbage", returncode=1))
    assert main._list_codex_models() == []                                       # fails soft, never raises into /model
    def missing(*a, **k): raise FileNotFoundError("codex")
    monkeypatch.setattr(main.subprocess, "run", missing)
    assert main._list_codex_models() == []


def test_providers_output_separates_integrated_from_detected_only(capsys, monkeypatch):
    import httpx
    from agent import main
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    monkeypatch.setattr("agent.detect.detect_agents", lambda: [
        {"name": "codex", "path": "/x/codex", "version": "codex-cli 1"}, {"name": "gemini", "path": "/x/gemini", "version": "9"}])
    main.cmd_providers(None)
    out = capsys.readouterr().out
    assert "provider: codex" in out and "detected only" in out
    assert set(main.AGENT_TO_PROVIDER.values()) <= set(main.PROVIDER_NAMES)


# ---- structural guards: a provider registered in EXTERNAL_AGENTS must work everywhere it's used ----
def test_every_registered_cloud_agent_runs_through_run_external(monkeypatch, capsys):
    # regression: a leftover hard-coded {"antigravity", "claude"} dict made the FIRST message to codex
    # raise KeyError and kill butler. Exercising every registered name catches the next one too.
    from agent import main
    class Fake:
        def __init__(self, model=None, session=None, effort=None, approval=None, prompt=False): self.session = session
        def healthy(self): return True
        def run(self, task, files=()): yield {"type": "text", "delta": "hi"}; yield {"type": "done", "content": "", "session": "S-" + task}
    for name in EXTERNAL_AGENTS:
        monkeypatch.setitem(EXTERNAL_AGENTS, name, Fake)
        done = main._run_external(name, "go", None, banner=True)                       # banner path too
        assert done["session"] == "S-go" and done["reply"] == "hi", name
        assert name in main.CLI_NAMES and name in main.PROVIDER_NAMES, name
    capsys.readouterr()


def test_every_provider_has_a_model_source(monkeypatch):
    from agent import main
    monkeypatch.setattr(main, "_MODEL_CACHE", {})
    monkeypatch.setattr(main, "_list_ollama_models", lambda: [("o1", "")])
    monkeypatch.setattr(main, "_list_agy_models", lambda: [("a1", "")])
    monkeypatch.setattr(main, "_list_codex_models", lambda: [("x1", "")])
    monkeypatch.setattr("agent.providers.ClaudeRunner.healthy", lambda self: True)
    for p in main.PROVIDER_NAMES:
        if p != "npu":                                  # npu has no model list until flm exists
            assert main._fetch_models(p), f"{p} has no model source wired into _fetch_models"


def test_every_cloud_agent_gets_the_shared_commands_and_no_stray_effort():
    from agent.main import complete_command
    for name in EXTERNAL_AGENTS:
        if name != "antigravity":                       # antigravity has its own native command set
            assert "/model" in complete_command("/mo", name), name
            assert ("/effort" in complete_command("/ef", name)) == (name == "claude"), name


def test_npu_status_renders_in_show_status(capsys):
    from agent.main import show_status
    from agent.providers import NPUProvider
    show_status(NPUProvider(url="http://127.0.0.1:1").status())  # unreachable: must still render, not KeyError
    out = capsys.readouterr().out
    assert "FastFlowLM" in out and "127.0.0.1:1" in out


def test_web_search_parses_results(monkeypatch):
    import httpx
    from tools import web
    page = ('<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fa">T&amp;1</a>'
            '<a class="result__snippet" href="x">snip <b>bold</b></a>')
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, text=page, request=httpx.Request("POST", "http://x")))
    assert web.web_search("q") == "T&1\nhttps://ex.com/a\nsnip bold"


def test_npu_messages_are_openai_style_for_tool_turns():
    from agent.providers import NPUProvider
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}]},
            {"role": "tool", "tool_name": "web_search", "content": "r"}]
    out = NPUProvider._openai_messages(msgs)
    call = out[1]["tool_calls"][0]
    assert call["type"] == "function" and call["id"] and call["function"]["arguments"] == '{"query": "x"}'
    assert out[2]["tool_call_id"] == call["id"] and out[2]["name"] == "web_search" and "tool_name" not in out[2]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}  # caller's history untouched


def test_web_search_clips_long_queries(monkeypatch):
    import httpx
    from tools import web
    seen = {}
    def fake(url, data, **k):
        seen["q"] = data["q"]; return httpx.Response(200, text="", request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake)
    web.web_search("profile social media " * 60)
    assert len(seen["q"]) <= 200


def test_fetch_url_follows_redirects_but_rechecks_each_hop(monkeypatch):
    import httpx, pytest
    from tools import web
    ips = {"good.example": "93.184.216.34", "evil.example": "127.0.0.1"}
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", (ips[h], 0))])
    hops = {"http://good.example/a/": httpx.Response(301, headers={"location": "http://good.example/a"}, request=httpx.Request("GET", "http://x")),
            "http://good.example/a": httpx.Response(200, text="<p>hello</p>", request=httpx.Request("GET", "http://x")),
            "http://good.example/r": httpx.Response(302, headers={"location": "http://evil.example/"}, request=httpx.Request("GET", "http://x")),
            "http://good.example/b": httpx.Response(999, request=httpx.Request("GET", "http://x"))}
    import contextlib
    monkeypatch.setattr(httpx, "stream", contextlib.contextmanager(lambda m, url, **k: (yield hops[url])))
    assert web.fetch_url("http://good.example/a/") == "hello"
    with pytest.raises(PermissionError): web.fetch_url("http://good.example/r")  # redirect into loopback
    assert "HTTP 999" in web.fetch_url("http://good.example/b")


def test_web_search_and_fetch_url_need_confirmation():
    import tools
    from tools import web  # noqa: F401
    assert tools.level_of("web_search", {}) == tools.WRITE and tools.level_of("fetch_url", {}) == tools.WRITE


def test_npu_is_preferred_over_ollama_when_both_are_up(monkeypatch):
    from agent import llm, main, providers
    monkeypatch.setattr(llm.LLM, "healthy", lambda self: True)
    monkeypatch.setattr(providers, "ensure_npu_server", lambda *a, **k: True)
    assert main._pick_provider() == "npu"
    monkeypatch.setattr(providers, "ensure_npu_server", lambda *a, **k: False)
    assert main._pick_provider() == "ollama"  # NPU server can't be reached or started -> falls back to ollama



def test_input_box_height_is_exact_not_terminal_filling():
    from agent.ui import MAX_INPUT_LINES, input_lines
    assert input_lines("", 100) == 1 and input_lines("hi", 100) == 1
    assert input_lines("a" * 250, 100) == 3 and input_lines("a\nb\n", 100) == 3   # wrapped + explicit newlines
    assert input_lines("x\n" * 99, 100) == MAX_INPUT_LINES


def test_transcript_is_appended_and_recall_finds_compacted_turns(tmp_path, monkeypatch):
    import agent.memory as am
    monkeypatch.setattr(am, "HISTORY", tmp_path / "h")
    m = am.Memory("sys", 8192)
    m.add({"role": "user", "content": "the deploy key lives in vault path secret/prod/deploy"})
    m.add({"role": "assistant", "content": "noted"})
    m.add({"role": "tool", "tool_name": "web_search", "content": "unrelated pizza results"})
    m.messages.clear(); m.summary = "user talked about deployment"   # simulate compaction dropping everything
    out = am.recall(m.md_path, "where is the deploy key vault")
    assert "secret/prod/deploy" in out and "pizza" not in out
    assert "nothing" in am.recall(m.md_path, "zebra giraffe")
    assert (m.md_path.stat().st_mode & 0o777) == 0o600
    assert "recall_history" in m.prompt()[1]["content"]


def test_free_local_models_unloads_other_ollama_models_and_stops_npu(monkeypatch):
    import httpx
    from agent import providers
    monkeypatch.delenv("AGENT_NO_SERVICES", raising=False)
    unloaded, stopped = [], []
    req = httpx.Request("GET", "http://x")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"models": [{"name": "big:9b"}, {"name": "keep:1b"}]}, request=req))
    monkeypatch.setattr(httpx, "post", lambda url, json, **k: unloaded.append(json["model"]))
    monkeypatch.setattr(providers, "_systemctl", lambda *a: stopped.append(a) or True)
    monkeypatch.setattr(providers.NPUProvider, "healthy", lambda self: True)
    assert providers.free_local_models("ollama", "keep:1b") == ["unloaded big:9b (ollama)", "stopped the NPU server"]
    assert unloaded == ["big:9b"] and stopped == [("stop", providers.NPU_UNIT)]
    unloaded.clear(); stopped.clear()
    assert providers.free_local_models("npu", "q") == ["unloaded big:9b (ollama)", "unloaded keep:1b (ollama)"]  # NPU stays up
    assert stopped == []


def test_approval_mode_becomes_each_clis_own_permission_flags():
    from agent.providers import ClaudeRunner, CodexRunner, AgyRunner
    assert "plan" in ClaudeRunner(approval="readonly").cmd("t") and "acceptEdits" in ClaudeRunner(approval="auto-edit").cmd("t")
    c = CodexRunner(approval="readonly").cmd("t")
    assert c[c.index("-s") + 1] == "read-only" and "--sandbox" in AgyRunner(approval="readonly").cmd("t")
    assert "--permission-mode" not in ClaudeRunner().cmd("t")  # no approval given = the user's own config


def test_parser_failure_still_ends_with_done_and_cloud_tools_are_audited(tmp_path):
    import json, sys
    from agent import audit
    from agent.providers import _run_external_agent
    def bad(line): raise TypeError("format changed")
    evs = list(_run_external_agent([sys.executable, "-c", "print('x')"], bad))
    assert [e["type"] for e in evs] == ["error", "done"]
    tool = lambda line: {"type": "tool", "name": "Bash", "args": {"command": "ls"}, "text": "ls"} if line.strip() == "x" else None
    list(_run_external_agent([sys.executable, "-c", "print('x')"], tool))
    rec = json.loads(audit.PATH.read_text().splitlines()[-1])
    assert rec["tool"] == "Bash" and rec["level"] == "cloud" and rec["decision"] == "delegated"


def test_model_catalog_uses_daemon_threads_and_survives_a_failing_provider():
    import threading
    from agent.models import _model_catalog
    def fetch(p):
        assert threading.current_thread().daemon          # nothing the interpreter must join at exit
        if p == "claude": raise OSError("boom")
        return [(f"{p}-m", "")] if p == "ollama" else []
    assert _model_catalog(fetch) == [("ollama", "ollama-m", "")]


def test_permission_broker_round_trip_and_always():
    import json
    import threading
    from agent.permissions import PermissionBroker, ask
    b = PermissionBroker()
    try:
        def user(*answers):  # plays the render loop: answer each permission event in turn
            for a in answers:
                e = b.q.get(timeout=5)
                assert e["type"] == "permission" and e["tool"] == "Bash"
                e["answer"](*a)
        t = threading.Thread(target=user, args=([(False,), (True, True)]))
        t.start()
        req = {"tool_name": "Bash", "input": {"command": "ls"}}
        assert json.loads(ask(b.path, req)) == {"behavior": "deny", "message": "The user denied this in butler."}
        assert json.loads(ask(b.path, req)) == {"behavior": "allow", "updatedInput": {"command": "ls"}}
        t.join()
        assert json.loads(ask(b.path, req))["behavior"] == "allow" and b.q.empty()  # "always" skips the prompt
    finally:
        b.close()


def test_other_chats_detects_live_peers_and_drops_stale_files(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from agent import main
    monkeypatch.setattr(main, "CHATS", tmp_path / "chats")
    assert not main._other_chats()                                   # nobody else, no dir yet
    main._register_chat()
    assert not main._other_chats()                                   # only ourselves
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (main.CHATS / str(p.pid)).touch()
        assert main._other_chats()                                   # a live peer blocks freeing shared models
    finally:
        p.kill(); p.wait()
    assert not main._other_chats() and not (main.CHATS / str(p.pid)).exists()   # a dead peer is cleaned up
