import base64

from agent import attach
from agent.providers import ClaudeRunner, CodexRunner, NPUProvider


def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_CWD", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def test_parse_finds_only_real_workspace_files(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / ".env").write_text("KEY=1")
    got = attach.parse("read @a.txt, then @missing.txt and @.env and mail me@x.com")
    assert got == [tmp_path / "a.txt"]
    assert attach.parse("see @a.txt.") == [tmp_path / "a.txt"]  # trailing punctuation


def test_big_file_gets_only_the_matching_chunks(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    lines = [f"filler line number {i} about nothing" for i in range(2000)]
    lines[1500] = "the launch code is PURPLE-ELEPHANT"
    f = tmp_path / "big.txt"
    f.write_text("\n".join(lines))
    out, imgs = attach.local("what is the launch code? @big.txt", [f], num_ctx=8192)
    assert "PURPLE-ELEPHANT" in out and imgs == [] and len(out) < 8192 * 3 // 4 + 500
    assert "chunks that best match" in out


def test_small_file_inline_and_image_as_base64(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    (tmp_path / "n.md").write_text("small note")
    (tmp_path / "p.png").write_bytes(b"\x89PNG-fake")
    out, imgs = attach.local("x", [tmp_path / "n.md", tmp_path / "p.png"], 8192)
    assert "small note" in out and imgs == [base64.b64encode(b"\x89PNG-fake").decode()]
    out, imgs = attach.local("x", [tmp_path / "p.png"], 8192, vision=False)
    assert imgs == [] and "cannot see images" in out


def test_cloud_runners_get_paths_and_images(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    img = tmp_path / "p.png"
    c = ClaudeRunner().cmd("look", files=[img])
    assert str(img) in c[2] and "--add-dir" not in c  # in-workspace: no widened access
    x = CodexRunner().cmd("look", files=[img])
    assert x[:3] == ["codex", "exec", x[2]] and x[-2:] == ["-i", str(img)]
    assert "--add-dir" not in ClaudeRunner().cmd("look")


def test_npu_drops_images_from_messages():
    assert "images" not in NPUProvider._openai_messages([{"role": "user", "content": "x", "images": ["AAA"]}])[0]


def test_outside_workspace_file_needs_approval(tmp_path, monkeypatch):
    work, away = tmp_path / "work", tmp_path / "away"
    work.mkdir(); away.mkdir()
    ws(work, monkeypatch)
    f = away / "image.png"
    f.write_bytes(b"x")
    asked = []
    yes = lambda n, a, lv: asked.append((n, a["path"])) or True
    assert attach.parse(f'"\\{f}  read this image') == []                     # no confirm: skipped
    assert attach.parse(f"read {f}", lambda *a: False) == []                   # user said no
    assert attach.parse(f'"\\{f}  read this image', yes) == [f] and asked == [("attach_file", str(f))]
    assert attach.is_path(str(f)) and not attach.is_path("/help")


def test_secrets_outside_never_offered(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    (tmp_path / ".env").write_text("K=1")
    assert attach.parse(f"{tmp_path}/.env", lambda *a: True) == []


def test_clipboard_image_pasted_as_attachment(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    monkeypatch.setattr(attach, "PASTES", tmp_path.parent / "pastes")
    class R: returncode, stdout = 0, b"\x89PNGdata"
    monkeypatch.setattr(attach.subprocess, "run", lambda *a, **k: R)
    p = attach.clipboard_image()
    assert p and p.read_bytes() == b"\x89PNGdata"
    assert attach.parse(f"what is @{p}") == [p]  # outside the workspace but no confirm needed
    R.stdout = b"text"
    assert attach.clipboard_image() is None


def test_outside_file_adds_only_its_dir_and_bash_always_is_per_command(tmp_path, monkeypatch):
    ws(tmp_path, monkeypatch)
    out = tmp_path.parent / "elsewhere" / "a.png"
    c = ClaudeRunner().cmd("look", files=[out])
    assert c[c.index("--add-dir") + 1] == str(out.parent)
    from agent.permissions import PermissionBroker
    b = PermissionBroker()
    try:
        import threading
        from agent.permissions import ask
        threading.Thread(target=lambda: b.q.get(timeout=5)["answer"](True, True)).start()
        ask(b.path, {"tool_name": "Bash", "input": {"command": "ls"}})
        threading.Thread(target=lambda: b.q.get(timeout=5)["answer"](False)).start()
        assert "deny" in ask(b.path, {"tool_name": "Bash", "input": {"command": "rm x"}})  # not auto-allowed
    finally:
        b.close()
