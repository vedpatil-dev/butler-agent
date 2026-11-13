"""MCP client hub: starts every enabled server from config/mcp.json (stdio) and exposes their tools to the agent
as `<server>__<tool>`. Sync facade over the async SDK (runs its own event loop thread)."""
import asyncio
import json
import os
import re
import sys
import threading
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters

from tools import RANK, READ, WRITE
from tools.filesystem import cwd
from tools.shell import command_level

from . import config

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # a server-supplied description is untrusted text bound for the model
START_TIMEOUT = 15  # seconds per server to start and list its tools
PASS_ENV = ("AGENT_WORKSPACE", "HOME", "PATH")  # AGENT_CWD is passed explicitly from self.cwd below


def load_config(path=None) -> dict:
    p = path or config.MCP_CONFIG
    if not p.exists(): return {}
    txt = p.read_text().replace("${ROOT}", str(config.ROOT)).replace("${PYTHON}", sys.executable)
    return json.loads(txt).get("servers", {})


class MCPHub:
    live: set = set()  # hubs not yet closed, so exit cleanup can stop their server subprocesses

    def __init__(self, servers: dict, only: list[str] | None = None):
        self.cwd = str(cwd())  # servers are spawned with this AGENT_CWD; callers must choose the project first
        self.cfg = {n: c for n, c in servers.items() if (n in only if only is not None else c.get("enabled", True))}
        self.schemas, self.errors, self._routes, self._clients = [], {}, {}, {}
        self._ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._stop = asyncio.run_coroutine_threadsafe(self._main(), self.loop)
        self._ready.wait(60)
        MCPHub.live.add(self)

    async def _main(self):
        self._stopper = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:  # opened and closed in this one task (anyio requirement)
                for name, c in self.cfg.items():
                    try:
                        async with asyncio.timeout(START_TIMEOUT):  # a hung server must not block the others
                            env = {k: os.environ[k] for k in PASS_ENV if k in os.environ} | {"AGENT_CWD": self.cwd} | c.get("env", {})
                            cl = await stack.enter_async_context(Client(
                                StdioServerParameters(command=c["command"], args=c.get("args", []), env=env)))
                            self._clients[name] = cl
                            for t in (await cl.list_tools()).tools:
                                full = f"{name}__{t.name}"
                                self._routes[full] = (name, t.name, c)
                                self.schemas.append({"type": "function", "function": {
                                    "name": full, "description": _CTRL.sub("", t.description or "")[:500], "parameters": t.input_schema}})
                    except Exception as e:  # one broken server must not take the agent down
                        self.errors[name] = f"{type(e).__name__}: {e}"
                self._ready.set()
                await self._stopper.wait()
        finally:
            self._ready.set()

    @property
    def tools(self) -> set:
        return set(self._routes)

    def level(self, full: str, args: dict) -> str:
        """Tools listed under "read" in the config are READ; everything else needs confirmation. Any `command`
        argument is additionally classified like a shell command."""
        _, tool, c = self._routes[full]
        lv = READ if tool in c.get("read", []) else WRITE
        if isinstance(args.get("command"), str):
            lv = max(lv, command_level(args["command"]), key=RANK.get)
        return lv

    def call(self, full: str, args: dict) -> str:
        name, tool, _ = self._routes[full]
        r = asyncio.run_coroutine_threadsafe(self._clients[name].call_tool(tool, args), self.loop).result(300)
        text = "\n".join(getattr(c, "text", str(c)) for c in r.content)
        if r.is_error: raise RuntimeError(text)
        return text

    def close(self):
        MCPHub.live.discard(self)
        self.loop.call_soon_threadsafe(self._stopper.set)
        try: self._stop.result(10)
        except Exception: pass
        self.loop.call_soon_threadsafe(self.loop.stop)
