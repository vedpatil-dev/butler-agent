"""One place that decides whether a tool call runs, needs the user, or is refused. Used by Agent.execute, so the CLI,
the API and sub-agents all share it. The confirm callback only ever sees calls this returns "ask" for."""
import re
import shlex

from tools import DANGEROUS, READ

MODES = ("ask", "auto-edit", "readonly")
FILE_EDIT_TOOLS = ("edit_file", "create_file", "multi_edit")
UNTRUSTED_TOOLS = ("fetch_url", "web_search")  # plus every MCP tool: their output is text someone else wrote
_CHAIN = re.compile(r"[;&|`$()<>\n\\]")


def command_prefix(command: str) -> str | None:
    """'git commit -m x' -> 'git commit'; None when the command chains/expands, so a rule can't be smuggled past."""
    if _CHAIN.search(command): return None
    try: words = shlex.split(command)
    except ValueError: return None
    return " ".join(words[:2]) if words else None


def decide(name: str, args: dict, level: str, mode: str = "ask", rules=(), tainted: bool = False) -> str:
    """-> "allow" | "ask" | "deny". DANGEROUS always asks; so does anything after untrusted content entered the turn."""
    if level == READ: return "allow"
    if mode == "readonly": return "deny"
    if level == DANGEROUS or tainted: return "ask"
    if mode == "auto-edit" and name in FILE_EDIT_TOOLS: return "allow"
    if name == "execute_command" and (p := command_prefix(str(args.get("command", "")))) and p in rules: return "allow"
    return "ask"


def untrusted(text: str, source: str) -> str:
    """Mark text from outside the user/project so the model treats it as data. The closing tag is defanged so the
    content cannot end the block early."""
    return f'<untrusted-data source="{source}">\n{text.replace("</untrusted-data", "</untrusted-data ")}\n</untrusted-data>'
