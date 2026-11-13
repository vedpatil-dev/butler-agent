"""Lets `claude -p` ask the butler user before running a tool. claude calls the MCP tool named by
--permission-prompt-tool (mcp/permission_server.py); that tool forwards the request over a unix socket to this
PermissionBroker, which hands it to the render loop as a {"type": "permission"} event and relays the answer back."""
import json
import os
import queue
import shutil
import socket
import tempfile
import threading

TOOL = "mcp__butler__approve"


def ask(path: str, req: dict) -> str:
    """Client side (runs in the MCP server child): send one request, return claude's JSON verdict string."""
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(path)
        s.sendall(json.dumps(req).encode() + b"\n")
        return s.makefile().readline().strip()


class PermissionBroker:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()  # shared with _run_external_agent: stdout lines and permission events
        self.always: set[str] = set()  # tools the user answered "a" for, this run only
        self._dir = tempfile.mkdtemp(prefix="butler-perm-")
        self.path = os.path.join(self._dir, "sock")
        self._srv = socket.socket(socket.AF_UNIX)
        self._srv.bind(self.path)
        self._srv.listen()
        threading.Thread(target=self._accept, daemon=True).start()

    def mcp_config(self, python: str, script: str) -> str:
        return json.dumps({"mcpServers": {"butler": {"command": python, "args": [script], "env": {"BUTLER_PERM_SOCK": self.path}}}})

    def _accept(self):
        while True:
            try: conn, _ = self._srv.accept()
            except OSError: return  # closed
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn, conn.makefile("rw") as f:
            try: req = json.loads(f.readline())
            except ValueError: return
            tool, inp = req.get("tool_name", "?"), req.get("input") or {}
            key = f"Bash:{inp.get('command', '')}" if tool == "Bash" else tool  # 'always' for one shell command must not approve every later one
            allow = key in self.always
            if not allow:
                done, box = threading.Event(), []
                def answer(ok: bool, always: bool = False):
                    if ok and always: self.always.add(key)
                    box.append(ok); done.set()
                self.q.put({"type": "permission", "tool": tool, "input": inp, "answer": answer})
                done.wait()
                allow = box[0]
            f.write(json.dumps({"behavior": "allow", "updatedInput": inp} if allow
                               else {"behavior": "deny", "message": "The user denied this in butler."}) + "\n")
            f.flush()

    def close(self):
        self._srv.close()
        shutil.rmtree(self._dir, ignore_errors=True)
