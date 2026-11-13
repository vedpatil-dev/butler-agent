"""Shell tool. Risk is classified from the command text: known read-only -> READ, destructive/system/secret
access -> DANGEROUS, anything else -> WRITE. The classifier is a heuristic confirmation gate; the actual containment
is bubblewrap (sandbox_argv): read-only disk, writable project, no network."""
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import DANGEROUS, READ, WRITE, clip, tool
from .filesystem import is_secret, safe_path

_DANGEROUS = re.compile("|".join([
    r"\bsudo\b", r"\bsu\b", r"\brm\b(?=.*\s(-\w*[rRf]|--recursive|--force))", r"\b(ssh|scp|sftp|rsync|nc|ncat|socat)\b", r"\bgit\s+(push|reset\s+--hard|clean|branch\s+-D)\b",
    r"\bdrop\s+(table|database|schema)\b", r"\btruncate\s+table\b", r"\bdocker\s+(system\s+prune|volume\s+(rm|prune)|rm|rmi)\b",
    r"\b(dnf|yum|rpm|apt(-get)?|pacman|snap|flatpak)\s+(install|remove|erase|upgrade|update|-\w*[eEU])",
    r"\b(systemctl|shutdown|reboot|mkfs\S*|fdisk|parted|dd|chown|chmod\s+-R|mount|umount|sysctl|grubby|efibootmgr)\b",
    r"\bcurl\b.*\|\s*\w*sh\b", r"\bwget\b.*\|\s*\w*sh\b", r"(^|\s)/(etc|boot|root|sys|proc|dev)/",
    r"\.env\b(?!\.example)", r"\.ssh\b", r"\.aws\b", r"\.gnupg\b", r"\bid_(rsa|ed25519)", r"\.pem\b", r"\.netrc\b",
    r"(^|[;&|]\s*)(env|printenv|export)\b",
]), re.I)
_READ_CMDS = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "echo", "which", "tree", "file",
              "stat", "du", "df", "date", "sort", "uniq", "diff", "basename", "dirname", "true"}
_READ_GIT = re.compile(r"^git\s+(status|diff|log|show|branch(\s+-[avr]+)?\s*$)(\s|$)")
# anything the shell would expand or chain differently than the text reads (so a text match proves nothing), plus
# options that make a "read" command write or execute (find -fprint, rg --pre, git diff --output, ...)
_UNSAFE = re.compile(r"[><`\n&$\\*?\[\]{}~!()]|-(exec|execdir|ok|okdir)\b|-delete\b|-fprint|-fls\b|\bsed\s+-i|--pre\b|--output|--ext-diff|--compress-program|\bsort\s+(\S+\s+)*-o")
_GIT_ARG_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def _git_sub(tokens: list[str]) -> str:
    """The git subcommand, skipping global options (`git -C . push`, `git -c k=v push`)."""
    i = 1
    while i < len(tokens):
        if tokens[i] in _GIT_ARG_OPTS: i += 2
        elif tokens[i].startswith("-"): i += 1
        else: return tokens[i]
    return ""


def _operands(arg: str) -> set[str]:
    """What an argument may name as a path: itself, --opt=VALUE's value, or -fVALUE's value (`grep -f/etc/passwd`)."""
    return {arg, arg.split("=", 1)[-1], arg[2:]} if arg.startswith("-") and len(arg) > 2 else {arg}


def command_level(cmd: str) -> str:
    if _DANGEROUS.search(cmd): return DANGEROUS
    segs = []
    for seg in re.split(r"\|\||&&|[;|&\n]", cmd):
        if not seg.strip(): continue
        try: segs.append(shlex.split(seg))
        except ValueError: return WRITE
    for t in segs:  # the same checks on the de-quoted words: `cat ".e""nv"` is `cat .env`
        if t[0] == "git" and (sub := _git_sub(t)) and (sub in ("push", "clean") or (sub == "reset" and "--hard" in t) or (sub == "branch" and "-D" in t)): return DANGEROUS
        if any(is_secret(c) or c.endswith(".git/config") for a in t[1:] for c in _operands(a)): return DANGEROUS  # cat credentials.json, grep -f.env x
        if _DANGEROUS.search(" ".join(t)): return DANGEROUS
    if _UNSAFE.search(cmd): return WRITE
    for t in segs:
        seg = " ".join(t)
        if t[0] not in _READ_CMDS and not _READ_GIT.match(seg): return WRITE
        if t[0] == "uniq" and sum(not a.startswith("-") for a in t[1:]) >= 2: return WRITE  # `uniq in out` overwrites `out`
        if any(c.startswith(("/", "~")) or ".." in c.split("/") for a in t[1:] for c in _operands(a)): return WRITE  # outside the project
    return READ


_ENV_KEEP = re.compile(r"^(PATH|HOME|USER|USERPROFILE|LOGNAME|SHELL|LANG|LANGUAGE|LC_\w+|TERM|TZ|TMPDIR|VIRTUAL_ENV|PWD|NO_COLOR|COLORTERM|CI)$")


def _env():
    """Allow-list: a secret under a name nobody thought to deny-list (MYAPP_CREDS, OPENAI_API_KEY...) never reaches children."""
    env = {k: v for k, v in os.environ.items() if _ENV_KEEP.match(k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # edit + immediate re-run within the same second must not hit a stale .pyc
    return env


_HIDE_DIRS = (".ssh", ".aws", ".gnupg", ".config/gh", ".docker", ".kube")
_HIDE_FILES = (".netrc", ".git-credentials", ".npmrc", ".pypirc", ".pgpass")


def sandbox_mode() -> str:
    """AGENT_SANDBOX: on (default; use bubblewrap if installed) | require (refuse to run without it) | 0 (off)."""
    return os.environ.get("AGENT_SANDBOX", "on").lower()


def sandbox_argv(workdir: Path, readonly: bool = False) -> list[str]:
    """bubblewrap prefix: whole disk read-only, only the project writable (unless readonly=True), credential dirs hidden,
    no network (AGENT_SANDBOX_NET=1 allows it, e.g. for pip/npm). Returns [] when sandboxing is off or bwrap is missing."""
    mode = sandbox_mode()
    if mode in ("0", "off", "false"): return []
    bwrap = shutil.which("bwrap")
    if not bwrap:
        if mode == "require": raise PermissionError("AGENT_SANDBOX=require but bubblewrap (bwrap) is not installed")
        return []
    home = Path.home()
    argv = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp",
            *([] if readonly else ["--bind", str(workdir), str(workdir)]),
            "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session"]
    if os.environ.get("AGENT_SANDBOX_NET") != "1": argv.append("--unshare-net")
    for d in _HIDE_DIRS:
        if (home / d).is_dir() and not workdir.is_relative_to(home / d): argv += ["--tmpfs", str(home / d)]
    for f in _HIDE_FILES:
        if (home / f).is_file(): argv += ["--ro-bind", "/dev/null", str(home / f)]
    return argv + ["--chdir", str(workdir)]


@tool(lambda a: command_level(a.get("command", "")))
def execute_command(command: str, timeout: int = 60, dir: str = ".") -> str:
    """Run a bash command in the project dir (or a subdirectory `dir`). Returns JSON: stdout, stderr,
    exit_code, execution_time. Killed after `timeout` seconds (default 60, max 600)."""
    t0 = time.time()
    wd = safe_path(dir)
    p = subprocess.Popen([*sandbox_argv(wd, readonly=(command_level(command) == READ)), "bash", "-c", command], cwd=wd, env=_env(), text=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=max(1, min(timeout, 600)))
        code = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)  # kill the whole process group, not just bash
        try: out, err = p.communicate(timeout=5)  # a child that left the group can still hold the pipes
        except subprocess.TimeoutExpired: out, err = "", ""
        err += f"\n[timed out after {timeout}s]"
        code = -1
    return json.dumps({"stdout": clip(out), "stderr": clip(err), "exit_code": code,
                       "execution_time": round(time.time() - t0, 2)})
