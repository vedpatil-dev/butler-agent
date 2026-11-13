"""Detection helpers: what NPU hardware is present, and what other AI agent CLIs are installed.
Pure read-only probing - no network calls, nothing is started. AMD/XDNA2 is verified against this
machine; Intel/Qualcomm are best-effort from public driver/PCI-ID docs and untested on real hardware."""
import os
import platform
import shutil
import subprocess
from pathlib import Path

# name -> (binary, version flag). Known CLI coding agents.
KNOWN_AGENTS = {
    "claude": ("claude", "--version"), "agy": ("agy", "--version"), "codex": ("codex", "--version"),
    "gemini": ("gemini", "--version"), "cursor-agent": ("cursor-agent", "--version"),
    "aider": ("aider", "--version"), "continue": ("continue", "--version"), "cline": ("cline", "--version"),
    "opencode": ("opencode", "--version"), "goose": ("goose", "--version"), "amazon-q": ("q", "--version"),
    "github-copilot": ("github-copilot-cli", "--version"), "ollama": ("ollama", "--version"),
}

# Checked when a binary isn't on the invoking shell's PATH - e.g. `butler` run from a shell whose
# PATH predates an install, or that never sourced the rc file that added it.
_EXTRA_DIRS = ["~/.local/bin", "~/.npm-global/bin", "~/bin", "/usr/local/bin"]


def _version(binary: str, flag: str) -> str:
    try:
        r = subprocess.run([binary, flag], capture_output=True, text=True, timeout=5)
        return (r.stdout or r.stderr).strip().splitlines()[0][:60] if r.returncode == 0 else "?"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "?"


def _find(binary: str) -> str | None:
    path = shutil.which(binary)
    if path: return path
    for d in _EXTRA_DIRS:
        p = Path(d).expanduser() / binary
        if os.access(p, os.X_OK): return str(p)
    return None


def detect_agents() -> list[dict]:
    """Every known AI agent CLI found - on PATH, or in common install dirs PATH might be missing."""
    found = []
    for name, (binary, flag) in KNOWN_AGENTS.items():
        path = _find(binary)
        if path:
            found.append({"name": name, "path": path, "version": _version(path, flag)})
    return found


def _amd_npu() -> dict | None:
    accel = Path("/dev/accel/accel0")
    if not accel.exists(): return None
    status_f = Path("/sys/class/accel/accel0/device/power/runtime_status")
    return {"vendor": "AMD", "family": "XDNA2 (or earlier)", "device": str(accel),
            "power_state": status_f.read_text().strip() if status_f.exists() else "unknown",
            "verified_on_this_machine": True}


def _intel_npu() -> dict | None:
    """Best-effort, UNTESTED: Intel Meteor Lake+ NPUs use the same /dev/accel/accelN interface
    (intel_vpu driver) as AMD's. Also checks lspci for the known Intel NPU PCI class/vendor."""
    for accel in sorted(Path("/dev/accel").glob("accel*")) if Path("/dev/accel").exists() else []:
        driver = Path(f"/sys/class/accel/{accel.name}/device/driver").resolve().name if \
            Path(f"/sys/class/accel/{accel.name}/device/driver").exists() else ""
        if "vpu" in driver or "intel" in driver.lower():
            return {"vendor": "Intel", "family": "NPU (Meteor Lake+)", "device": str(accel),
                    "driver": driver, "verified_on_this_machine": False}
    try:
        out = subprocess.run(["lspci", "-d", "8086:7d1d"], capture_output=True, text=True, timeout=5).stdout
        if out.strip():
            return {"vendor": "Intel", "family": "NPU (Meteor Lake+)", "pci": out.strip(),
                    "verified_on_this_machine": False}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _qualcomm_npu() -> dict | None:
    """Best-effort, UNTESTED: Qualcomm's Hexagon NPU on Snapdragon X laptops is SoC-integrated
    (ARM devicetree/ACPI, not a discrete PCI device), so detection here is a weak heuristic."""
    if platform.machine() not in ("aarch64", "arm64"): return None
    dt = Path("/sys/firmware/devicetree/base/compatible")
    try:
        if dt.exists() and "qcom" in dt.read_text(errors="ignore").lower():
            return {"vendor": "Qualcomm", "family": "Hexagon NPU (Snapdragon X, heuristic match)",
                    "verified_on_this_machine": False}
    except OSError:
        pass
    return None


def detect_npu() -> dict:
    """{"present": [...], "platform": "Linux x86_64"}. Apple Silicon (ANE) is out of scope: this
    agent targets Linux, and macOS NPU access goes through CoreML/Metal, not a Linux driver path."""
    found = [n for n in (_amd_npu(), _intel_npu(), _qualcomm_npu()) if n]
    return {"platform": f"{platform.system()} {platform.machine()}", "present": found}
