"""Hardware-aware driver and local LLM runtime setup for GPUs and NPUs.

Detects NVIDIA CUDA, AMD ROCm/Vulkan, Intel Arc/iGPU, Intel NPU, Qualcomm Snapdragon NPU,
AMD XDNA2 NPU, and Apple Silicon (M-series ANE + Metal/MLX), reports readiness, and can
execute package-manager / pip runtime installation commands when confirmed.
"""
import json
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .detect import detect_hardware


def distro_info() -> dict:
    """Return OS family and available package manager (dnf, apt-get, pacman, brew)."""
    sysname = platform.system().lower()
    pkg = next((m for m in ("dnf", "apt-get", "pacman", "brew") if shutil.which(m)), None)
    distro_id = sysname
    if sysname == "linux" and Path("/etc/os-release").exists():
        for line in Path("/etc/os-release").read_text(errors="ignore").splitlines():
            if line.startswith("ID="):
                distro_id = line.split("=", 1)[1].strip().strip('"').lower()
                break
    return {"system": sysname, "distro": distro_id, "pkg": pkg, "pip": sys.executable}


def _ollama_install_cmds(d: dict) -> list[list[str]]:
    if shutil.which("ollama"):
        return []
    if d["system"] == "darwin" and d["pkg"] == "brew":
        return [["brew", "install", "ollama"]]
    return [["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]]


def _pkg_cmds(d: dict, dnf: list[list[str]], apt: list[list[str]], pacman: list[list[str]]) -> list[list[str]]:
    if d["pkg"] == "dnf": return dnf
    if d["pkg"] == "apt-get": return apt
    if d["pkg"] == "pacman": return pacman
    return []


def setup_plan(hw: dict | None = None, distro: dict | None = None) -> list[dict]:
    """Build per-accelerator driver + runtime install steps from detected hardware."""
    hw = hw or detect_hardware()
    d = distro or distro_info()
    npus = {n["vendor"].lower(): n for n in hw.get("npu", [])}
    gpus = {g["vendor"].lower(): g for g in hw.get("gpu", [])}
    pip_cmd = [d["pip"], "-m", "pip", "install"]
    user = shutil.which("id") and subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
    seen_ollama = False
    plan = []

    def once_ollama() -> list[list[str]]:
        nonlocal seen_ollama
        if seen_ollama: return []
        cmds = _ollama_install_cmds(d)
        if cmds: seen_ollama = True
        return cmds

    if g := gpus.get("nvidia"):
        drv = [] if g.get("driver_ready") else _pkg_cmds(
            d,
            [["sudo", "dnf", "install", "-y", "akmod-nvidia", "xorg-x11-drv-nvidia-cuda"]],
            [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "nvidia-driver-550", "nvidia-cuda-toolkit"]],
            [["sudo", "pacman", "-S", "--noconfirm", "nvidia", "nvidia-utils", "cuda"]],
        )
        if g.get("driver_ready") and not g.get("cuda_toolkit"):
            drv += _pkg_cmds(d, [["sudo", "dnf", "install", "-y", "xorg-x11-drv-nvidia-cuda"]],
                             [["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"]],
                             [["sudo", "pacman", "-S", "--noconfirm", "cuda"]])
        plan.append({
            "id": "nvidia-cuda", "name": f"NVIDIA GPU ({g['family']})", "detected": True,
            "ready": bool(g.get("driver_ready")), "driver_cmds": drv,
            "runtime_cmds": once_ollama(),
            "manual_note": "Reboot after installing kernel NVIDIA drivers." if drv else "",
            "run_hint": "butler --provider ollama  (or llama-server / vLLM with --provider openai)",
        })

    if g := gpus.get("amd"):
        grp = [["sudo", "usermod", "-aG", "render,video", user]] if user else []
        drv = [] if g.get("driver_ready") else _pkg_cmds(
            d,
            [["sudo", "dnf", "install", "-y", "mesa-vulkan-drivers", "rocm-hip-devel", "rocminfo"]] + grp,
            [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "mesa-vulkan-drivers", "rocm-hip-sdk"]] + grp,
            [["sudo", "pacman", "-S", "--noconfirm", "vulkan-radeon", "rocm-hip-sdk"]] + grp,
        )
        plan.append({
            "id": "amd-gpu", "name": f"AMD GPU ({g['family']})", "detected": True,
            "ready": bool(g.get("driver_ready")), "driver_cmds": drv,
            "runtime_cmds": once_ollama(),
            "manual_note": "Log out and back in if added to render/video groups." if drv else "",
            "run_hint": "butler --provider ollama  (Vulkan/ROCm)",
        })

    if g := gpus.get("intel"):
        drv = [] if g.get("driver_ready") else _pkg_cmds(
            d,
            [["sudo", "dnf", "install", "-y", "mesa-vulkan-drivers", "intel-compute-runtime"]],
            [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "mesa-vulkan-drivers", "intel-opencl-icd"]],
            [["sudo", "pacman", "-S", "--noconfirm", "vulkan-intel", "intel-compute-runtime"]],
        )
        plan.append({
            "id": "intel-gpu", "name": f"Intel GPU ({g['family']})", "detected": True,
            "ready": bool(g.get("driver_ready")), "driver_cmds": drv,
            "runtime_cmds": once_ollama(),
            "manual_note": "", "run_hint": "butler --provider ollama",
        })

    if n := npus.get("intel"):
        drv = [] if n.get("device") else _pkg_cmds(
            d,
            [["sudo", "dnf", "install", "-y", "intel-npu-driver"]],
            [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "intel-npu-driver"]],
            [["sudo", "pacman", "-S", "--noconfirm", "intel-npu-driver"]],
        )
        plan.append({
            "id": "intel-npu", "name": f"Intel NPU ({n['family']})", "detected": True,
            "ready": bool(n.get("device")), "driver_cmds": drv,
            "runtime_cmds": [pip_cmd + ["openvino-genai"]],
            "manual_note": "Requires Linux kernel >= 6.8 (intel_vpu) or Intel linux-npu-driver release.",
            "run_hint": "Start OpenVINO OpenAI server on :8080, then: AGENT_OPENAI_URL=http://127.0.0.1:8080 butler --provider openai",
        })

    if n := npus.get("qualcomm"):
        plan.append({
            "id": "qualcomm-npu", "name": f"Snapdragon NPU ({n['family']})", "detected": True,
            "ready": False, "driver_cmds": [],
            "runtime_cmds": [pip_cmd + ["onnxruntime-qnn"]],
            "manual_note": "Qualcomm Hexagon NPU requires the proprietary Qualcomm AI Engine Direct (QNN) / Genie SDK and SoC BSP.",
            "run_hint": "Start Genie/QNN OpenAI-compat server on :8080, then: butler --provider openai",
        })

    if "apple" in npus or "apple" in gpus:
        plan.append({
            "id": "apple-silicon", "name": "Apple Silicon (M-series ANE + Metal/MLX)", "detected": True,
            "ready": True, "driver_cmds": [],
            "runtime_cmds": once_ollama() + [pip_cmd + ["mlx-lm"]],
            "manual_note": "Metal and CoreML/ANE drivers are built into macOS.",
            "run_hint": "butler --provider ollama  (or: mlx_lm.server --port 8080 & butler --provider openai)",
        })

    if n := npus.get("amd"):
        has_flm = bool(shutil.which("flm") or Path("/opt/fastflowlm/bin/flm").exists())
        plan.append({
            "id": "amd-npu", "name": f"AMD NPU ({n['family']})", "detected": True,
            "ready": bool(n.get("device")) and has_flm, "driver_cmds": [],
            "runtime_cmds": [],
            "manual_note": "Requires amdxdna DKMS (with MOK key if Secure Boot is on), XRT, and FastFlowLM (`flm`).",
            "run_hint": "butler --provider npu",
        })

    return plan


def run_setup(install: bool = False, yes: bool = False, target: str | None = None, as_json: bool = False) -> int:
    """Display detected accelerators and driver/runtime status; optionally execute install commands."""
    hw = detect_hardware()
    plan = setup_plan(hw)
    if target:
        plan = [item for item in plan if item["id"] == target]
    if as_json:
        print(json.dumps({"hardware": hw, "plan": plan}, indent=2))
        return 0
    if not plan:
        print("No supported GPU/NPU accelerators detected (or target not found). CPU Ollama works with: butler --provider ollama")
        return 0

    for item in plan:
        state = "ready" if item["ready"] else "needs setup"
        print(f"\n[{item['id']}] {item['name']} — {state}")
        for cmd in item["driver_cmds"]:
            print(f"  driver : {shlex.join(cmd)}")
        for cmd in item["runtime_cmds"]:
            print(f"  runtime: {shlex.join(cmd)}")
        if item["manual_note"]:
            print(f"  note   : {item['manual_note']}")
        print(f"  run    : {item['run_hint']}")

    if not install:
        print("\nRun `butler setup --install` to execute the listed driver/runtime commands.")
        return 0

    rc = 0
    for item in plan:
        if not (item["driver_cmds"] or item["runtime_cmds"]):
            continue
        if not yes:
            ans = input(f"\nInstall drivers/runtime for {item['name']}? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                continue
        driver_ok = True
        for cmd in item["driver_cmds"]:
            print(f"  $ {shlex.join(cmd)}")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  ! driver command failed with exit code {r.returncode}; skipping runtime install for {item['id']}")
                rc = rc or r.returncode
                driver_ok = False
                break
        if not driver_ok:
            continue
        for cmd in item["runtime_cmds"]:
            print(f"  $ {shlex.join(cmd)}")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  ! command failed with exit code {r.returncode}")
                rc = rc or r.returncode
    return rc
