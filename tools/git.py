"""Git tools operating on the current project dir. Push/reset --hard/branch delete are DANGEROUS (always confirmed)."""
import shlex
import subprocess

from . import DANGEROUS, READ, WRITE, clip, tool
from .filesystem import is_secret, safe_path

_NO_SECRETS = [f":(exclude,glob)**/{g}" for g in (".env", ".env.*", "*.pem", "*.key", "credentials*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", ".npmrc", ".netrc", ".git-credentials", ".envrc", ".pgpass", ".pypirc", "*.p12", "*.pfx", "*.kdbx", "*.tfvars", "secret", "secrets", "secret.*", "secrets.*")]


def _arg(v: str) -> str:
    """A ref/path/remote from the model must not be parseable as an option (`-f`, `--receive-pack=...`)."""
    if v.startswith("-"): raise ValueError(f"{v!r}: arguments may not start with '-'")
    return v


def _git(*args: str, raw: bool = False) -> str:
    r = subprocess.run(["git", "-C", str(safe_path(".")), *args], capture_output=True, text=True,
                       timeout=120, stdin=subprocess.DEVNULL)
    out = (r.stdout + r.stderr).strip()
    if r.returncode: raise RuntimeError(f"git {args[0]} failed: {out}")
    return out if raw else clip(out) or "(no output)"


@tool(READ)
def git_status() -> str:
    """Show branch and changed/untracked files (short format)."""
    return _git("status", "--short", "--branch")


@tool(READ)
def git_diff(path: str = "", staged: bool = False) -> str:
    """Show unstaged changes (or staged if staged=true), optionally for one path. Secret files are excluded."""
    spec = [str(safe_path(path)) if path else ".", *_NO_SECRETS]
    return _git("diff", *(["--staged"] if staged else []), "--", *spec)


@tool(READ)
def git_log(n: int = 10) -> str:
    """Show the last n commits, one line each."""
    return _git("log", f"-{min(n, 50)}", "--oneline", "--decorate")


@tool(READ)
def git_branch() -> str:
    """List local branches; the current one is marked with *."""
    return _git("branch", "-vv")


@tool(WRITE)
def git_checkout(ref: str, create: bool = False) -> str:
    """Switch to a branch or commit; create=true makes a new branch."""
    return _git("checkout", *(["-b", _arg(ref)] if create else [_arg(ref), "--"]))


@tool(WRITE)
def git_add(paths: str) -> str:
    """Stage files. paths is a space-separated list ('.' for everything). Secret files are never staged."""
    out = _git("add", "--", *map(_arg, shlex.split(paths)))
    for f in _git("diff", "--staged", "--name-only", "-z", raw=True).split("\0"):  # -z: names with non-ASCII chars aren't quoted
        if f and is_secret(f): _git("reset", "-q", "--", f)  # e.g. `git add .` picking up an untracked .env
    return out


@tool(WRITE)
def git_commit(message: str) -> str:
    """Commit staged changes with a message."""
    return _git("commit", "-m", message)


@tool(DANGEROUS)
def git_push(remote: str = "origin", branch: str = "", force: bool = False) -> str:
    """Push to a remote. force=true uses --force-with-lease. Always needs user confirmation."""
    return _git("push", *(["--force-with-lease"] if force else []), _arg(remote), *([_arg(branch)] if branch else []))


@tool(DANGEROUS)
def git_reset_hard(ref: str = "HEAD") -> str:
    """Discard ALL uncommitted changes and move HEAD to ref. Always needs user confirmation."""
    return _git("reset", "--hard", _arg(ref))


@tool(DANGEROUS)
def git_branch_delete(name: str) -> str:
    """Force-delete a local branch. Always needs user confirmation."""
    return _git("branch", "-D", _arg(name))
