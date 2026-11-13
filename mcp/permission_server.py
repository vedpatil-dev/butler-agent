"""MCP server whose single tool answers claude's --permission-prompt-tool by asking the butler process
(socket path in BUTLER_PERM_SOCK). Launched by claude itself, see agent/permissions.py."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agent.permissions import ask  # noqa: E402

server = MCPServer("butler")


def approve(tool_name: str, input: dict, tool_use_id: str = "") -> str:
    """Ask the user whether claude may run this tool call."""
    return ask(os.environ["BUTLER_PERM_SOCK"], {"tool_name": tool_name, "input": input})


server.add_tool(approve, name="approve", description="Ask the user whether claude may run this tool call.",
                structured_output=False)  # claude requires exactly one text block

if __name__ == "__main__":
    server.run("stdio")
