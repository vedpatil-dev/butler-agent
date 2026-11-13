"""Build a standalone single-file binary (Linux ELF, Windows .exe, or macOS .dmg) using PyInstaller."""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main():
    sep = ";" if os.name == "nt" else ":"
    datas = [
        (ROOT / "agent" / "prompts", "agent/prompts"),
        (ROOT / "agent" / "web", "agent/web"),
        (ROOT / "config", "config"),
        (ROOT / "mcp", "mcp"),
    ]
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile",
        "--name", "butler",
        "--collect-all", "uvicorn",
        "--collect-all", "fastapi",
        "--collect-submodules", "mcp.client",
        "--collect-submodules", "mcp.shared",
        "--collect-submodules", "mcp.types",
        "--exclude-module", "mcp.cli",
        "--collect-all", "prompt_toolkit",
        "--paths", str(ROOT),
    ]
    entry = ROOT / "build" / "pyi_entry.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("from agent.main import main\n\nif __name__ == '__main__':\n    main()\n")

    for src, dst in datas:
        if src.exists():
            cmd += ["--add-data", f"{src}{sep}{dst}"]
    cmd.append(str(entry))

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

    if platform.system() == "Darwin":
        dmg_stage = DIST / "dmg-stage"
        dmg_stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DIST / "butler", dmg_stage / "butler")
        dmg_path = DIST / "Butler-macOS.dmg"
        dmg_path.unlink(missing_ok=True)
        subprocess.check_call([
            "hdiutil", "create", "-volname", "Butler",
            "-srcfolder", str(dmg_stage), "-ov", "-format", "UDZO", str(dmg_path),
        ])
        print(f"Created macOS DMG: {dmg_path}")


if __name__ == "__main__":
    main()
