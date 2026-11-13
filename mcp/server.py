"""MCP server exposing this project's sandboxed filesystem, git and shell tools over stdio.
Run by the agent (see config/mcp.json) or by any MCP client:  python mcp/server.py
NOTE: this directory has no __init__.py on purpose so it never shadows the `mcp` SDK package.
DANGEROUS tools (git push/reset --hard/branch delete) and the network tools are deliberately not exposed, and tools whose
risk depends on their arguments (execute_command) only run when that call classifies as READ: no confirmation step here."""
import functools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make `tools` importable when run as a script

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

import tools  # noqa: E402

server = MCPServer("local-tools")


def _readable_errors(fn, level=None):
    """The v2 SDK hides the text of unexpected exceptions from the model; ToolError text is shown."""
    @functools.wraps(fn)  # keeps the signature, so the JSON schema is unchanged
    def run(*a, **kw):
        try:
            if callable(level) and level(kw) != tools.READ: raise PermissionError("needs approval; this server only runs read-only commands")
            return fn(*a, **kw)
        except Exception as e: raise ToolError(f"{type(e).__name__}: {e}") from e
    return run


for name, e in tools.REGISTRY.items():
    if e["level"] == tools.DANGEROUS or name in ("fetch_url", "web_search"): continue
    server.add_tool(_readable_errors(e["fn"], e["level"]), name=name, description=e["schema"]["function"]["description"])

if __name__ == "__main__":
    server.run("stdio")
