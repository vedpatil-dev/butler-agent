import httpx
import pytest

from agent import detect, models, providers, setup


def test_make_provider_openai_constructs_without_network():
    p = providers.make_provider("openai", model="openvino-qwen", url="http://127.0.0.1:9999")
    assert isinstance(p, providers.OpenAICompatProvider)
    assert p.model == "openvino-qwen"
    assert p.num_ctx == providers.OPENAI_CTX
    assert not p.healthy()


def test_detect_nvidia_drm_without_driver_no_unbound_local(monkeypatch):
    monkeypatch.setattr("agent.detect._find", lambda name: None)
    monkeypatch.setattr("agent.detect._drm_vendors", lambda: {"0x10de"})
    monkeypatch.setattr("agent.detect._lspci_match", lambda pat: "")
    monkeypatch.setattr("agent.detect._amd_gpu", lambda: None)
    monkeypatch.setattr("agent.detect._intel_gpu", lambda: None)
    monkeypatch.setattr("agent.detect._apple_gpu", lambda: None)
    gpus = detect.detect_gpu()["present"]
    assert len(gpus) == 1 and gpus[0]["vendor"] == "NVIDIA" and gpus[0]["driver_ready"] is False


def test_agent_records_openai_provider_name(monkeypatch, tmp_path):
    from agent.orchestrator import Agent
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    p = providers.make_provider("openai", model="m1", url="http://127.0.0.1:9999")
    ag = Agent(p, confirm=lambda *a: True, project=str(tmp_path))
    assert ag.memory.provider == "openai"


def test_list_openai_models_parses_v1_models(monkeypatch):
    def fake_get(url, timeout=3):
        return httpx.Response(200, json={"data": [{"id": "mlx-community/Qwen2.5-7B"}, {"id": "openvino/qwen3"}]})

    monkeypatch.setattr("httpx.get", fake_get)
    assert models._list_openai_models() == [
        ("mlx-community/Qwen2.5-7B", "OpenAI-compat"),
        ("openvino/qwen3", "OpenAI-compat"),
    ]


def test_detect_apple_silicon_npu_and_gpu(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("agent.detect._amd_npu", lambda: None)
    monkeypatch.setattr("agent.detect._intel_npu", lambda: None)
    monkeypatch.setattr("agent.detect._qualcomm_npu", lambda: None)
    monkeypatch.setattr("agent.detect._nvidia_gpu", lambda: None)
    monkeypatch.setattr("agent.detect._amd_gpu", lambda: None)
    monkeypatch.setattr("agent.detect._intel_gpu", lambda: None)

    npus = detect.detect_npu()["present"]
    gpus = detect.detect_gpu()["present"]
    assert len(npus) == 1 and npus[0]["vendor"] == "Apple"
    assert len(gpus) == 1 and gpus[0]["vendor"] == "Apple" and gpus[0]["backend"] == "metal"


def test_setup_plan_covers_all_accelerators(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    hw = {
        "platform": "Linux x86_64",
        "npu": [
            {"vendor": "Intel", "family": "NPU (Meteor Lake+)", "verified_on_this_machine": False},
            {"vendor": "Qualcomm", "family": "Hexagon NPU", "verified_on_this_machine": False},
            {"vendor": "AMD", "family": "XDNA2", "device": "/dev/accel/accel0", "verified_on_this_machine": True},
        ],
        "gpu": [
            {"vendor": "NVIDIA", "family": "RTX 4070", "backend": "cuda", "driver_ready": False, "cuda_toolkit": False},
            {"vendor": "AMD", "family": "Radeon GPU", "backend": "vulkan", "driver_ready": False},
            {"vendor": "Intel", "family": "Arc / iGPU", "backend": "vulkan/oneapi", "driver_ready": False},
        ],
    }
    distro = {"system": "linux", "distro": "fedora", "pkg": "dnf", "pip": "python3"}
    plan = {item["id"]: item for item in setup.setup_plan(hw=hw, distro=distro)}
    assert {"nvidia-cuda", "amd-gpu", "intel-gpu", "intel-npu", "qualcomm-npu", "amd-npu"} <= plan.keys()
    assert any("akmod-nvidia" in c for c in plan["nvidia-cuda"]["driver_cmds"])
    assert any("rocm-hip-devel" in c for c in plan["amd-gpu"]["driver_cmds"])
    assert any("openvino-genai" in c for c in plan["intel-npu"]["runtime_cmds"])
    # Ollama install should appear only once across hybrid GPUs
    ollama_count = sum(1 for item in plan.values() for c in item["runtime_cmds"] if "ollama" in " ".join(c))
    assert ollama_count == 1
    assert "QNN" in plan["qualcomm-npu"]["manual_note"]


def test_setup_plan_apple_silicon(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    hw = {
        "platform": "Darwin arm64",
        "npu": [{"vendor": "Apple", "family": "Apple Silicon (ANE + Metal/MLX)"}],
        "gpu": [{"vendor": "Apple", "family": "Apple Silicon GPU (Metal/MLX)", "backend": "metal", "driver_ready": True}],
    }
    distro = {"system": "darwin", "distro": "darwin", "pkg": "brew", "pip": "python3"}
    plan = setup.setup_plan(hw=hw, distro=distro)
    assert len(plan) == 1 and plan[0]["id"] == "apple-silicon"
    assert plan[0]["ready"] is True
    assert ["brew", "install", "ollama"] in plan[0]["runtime_cmds"]
    assert ["python3", "-m", "pip", "install", "mlx-lm"] in plan[0]["runtime_cmds"]


def test_run_setup_install_executes_commands(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr("agent.setup.detect_hardware", lambda: {
        "platform": "Darwin arm64",
        "npu": [{"vendor": "Apple", "family": "Apple Silicon"}],
        "gpu": [],
    })
    monkeypatch.setattr("agent.setup.distro_info", lambda: {"system": "darwin", "distro": "darwin", "pkg": "brew", "pip": "python3"})
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    class _Res:
        returncode = 0

    def fake_run(cmd, shell=False):
        ran.append(cmd)
        return _Res()

    monkeypatch.setattr("subprocess.run", fake_run)
    rc = setup.run_setup(install=True, yes=True, target="apple-silicon")
    assert rc == 0
    assert ran == [["brew", "install", "ollama"], ["python3", "-m", "pip", "install", "mlx-lm"]]
