"""Local HTTP API. Binds to 127.0.0.1 only. One agent run at a time (one local model; also keeps AGENT_CWD sane).
Streaming = Server-Sent Events, one JSON event per `data:` line."""
import hmac
from contextlib import asynccontextmanager
import json
import os
import queue
import secrets
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import config
from tools.filesystem import cwd, set_cwd, workspace

from .llm import LLM, LLMError
from .mcp_client import MCPHub, load_config
from .memory import Memory
from .orchestrator import SYSTEM, Agent, label

@asynccontextmanager
async def lifespan(_):
    yield
    for h in _hubs.values(): h.close()  # MCP servers are subprocesses: stop them with the API


app = FastAPI(title="local-agent", docs_url=None, redoc_url=None, lifespan=lifespan)
llm = LLM()
_hubs: dict = {}
AGENTS: dict[str, Agent] = {}
CHATS: dict[str, Memory] = {}
CONFIRMS: dict[str, dict] = {}
BUSY = threading.Lock()
ALLOWED_HOSTS = {f"127.0.0.1:{config.PORT}", f"localhost:{config.PORT}"}
ALLOWED_ORIGINS = {f"http://{h}" for h in ALLOWED_HOSTS}
MAX_CACHED = 50  # conversations kept in memory; older ones are reloaded from disk on demand
TOKEN_FILE = config.STATE / "api-token"


def _token() -> str:
    """Random per-install token (0600). The page served at / carries it; other API callers read the file.
    ponytail: any process that can GET / as this user can read it too; per-user sockets would be the next step."""
    try:
        if len(t := TOKEN_FILE.read_text().strip()) >= 16: return t  # an empty file must never become the token ("" == missing header)
    except OSError: pass
    t = secrets.token_urlsafe(32)
    fd, tmp = tempfile.mkstemp(dir=TOKEN_FILE.parent)  # 0600; written fully before it appears under its real name
    with os.fdopen(fd, "w") as f: f.write(t)
    os.replace(tmp, TOKEN_FILE)
    return t


TOKEN = _token()


@app.middleware("http")
async def only_local_hosts(request: Request, call_next):
    if request.headers.get("host") not in ALLOWED_HOSTS:  # blocks DNS-rebinding from web pages
        return JSONResponse({"error": "forbidden host"}, status_code=403)
    if request.headers.get("origin", "http://" + request.headers["host"]) not in ALLOWED_ORIGINS:  # cross-site requests
        return JSONResponse({"error": "forbidden origin"}, status_code=403)
    if request.url.path not in ("/", "/health") and not hmac.compare_digest(request.headers.get("x-agent-token", ""), TOKEN):
        return JSONResponse({"error": "missing or wrong X-Agent-Token (see api-token in the state dir)"}, status_code=401)
    return await call_next(request)


class ChatReq(BaseModel):
    message: str
    conversation_id: str | None = Field(None, pattern=r"^[\w-]{1,64}$")
    stream: bool = True


class RunReq(BaseModel):
    task: str
    conversation_id: str | None = Field(None, pattern=r"^[\w-]{1,64}$")
    cwd: str | None = None
    allow_write: bool = False  # auto-approve WRITE actions; DANGEROUS ones always need /agent/confirm
    approval: str = Field("ask", pattern=r"^(ask|auto-edit|readonly)$")  # allow_write=true is the old spelling of auto-edit
    plan: bool = False
    mcp: list[str] | None = None
    stream: bool = True


class ConfirmReq(BaseModel):
    id: str
    approve: bool


def _sse(events):
    for e in events:
        yield f"data: {json.dumps(e)}\n\n"


def _drain(q: queue.Queue):
    while (e := q.get()) is not None:
        yield e


@app.get("/health")
def health():
    return {"status": "ok" if llm.healthy() else "llm_down", "llm_url": config.LLM_URL, "busy": BUSY.locked()}


@app.get("/model")
def model():
    return llm.status()


@app.post("/chat")
def chat(req: ChatReq):
    if not BUSY.acquire(blocking=False): raise HTTPException(409, "model busy with another request")
    held = [True]

    def release():  # idempotent: also runs as a response background task, since a never-started generator skips its finally
        if held[0]: held[0] = False; BUSY.release()

    cid = req.conversation_id or uuid.uuid4().hex[:12]
    try:
        mem = CHATS.get(cid)
        if mem is None:
            mem = Memory("You are a helpful assistant running locally. Be concise.", llm.num_ctx, cid=cid)
            try:
                raw = Memory.load_raw(cid)
                mem.messages, mem.summary, mem.title = raw["messages"], raw["summary"], raw["title"]
            except (OSError, ValueError, KeyError): pass  # new conversation, or an unreadable file
            _remember(CHATS, cid, mem)
    except Exception:
        release()
        raise

    def events():
        try:
            mem.add({"role": "user", "content": req.message})
            mem.compact(llm)
            text = ""
            for kind, val in llm.stream(mem.prompt()):
                if kind == "text":
                    text += val
                    yield {"type": "text", "delta": val}
                elif kind == "stats": yield {"type": "stats", **val}
            mem.add({"role": "assistant", "content": text})
            mem.save()
            yield {"type": "done", "content": text, "conversation_id": cid}
        except Exception as e:  # LLMError, httpx errors, bad JSON from the model server
            yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            release()
    if req.stream: return StreamingResponse(_sse(events()), media_type="text/event-stream", background=BackgroundTask(release))
    done = [e for e in events() if e["type"] in ("done", "error")][-1]
    return done


def _remember(cache: dict, key: str, value):
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > MAX_CACHED: cache.pop(next(iter(cache)))  # oldest first; the saved file is the source of truth


def _agent_for(req: RunReq) -> Agent:
    cid = req.conversation_id
    if cid in AGENTS and (not req.cwd or Path(req.cwd).expanduser().resolve() == AGENTS[cid].project):
        AGENTS[cid].plan = req.plan
        return AGENTS[cid]
    if req.cwd:
        if not Path(req.cwd).expanduser().is_dir(): raise ValueError(f"cwd is not a directory: {req.cwd}")
        set_cwd(req.cwd)
    else:
        set_cwd(workspace())
    servers = load_config()
    mcp = sorted(n for n in req.mcp or () if servers.get(n, {}).get("enabled", True))  # "enabled": false stays off over the API
    key = (str(cwd()), tuple(mcp))
    if mcp and key not in _hubs: _hubs[key] = MCPHub(servers, only=mcp)
    a = Agent(llm, mcp=_hubs.get(key), plan=req.plan)
    if cid:
        a.memory.id = cid
        try:
            raw = Memory.load_raw(cid)
            a.memory.messages, a.memory.summary, a.memory.title = raw["messages"], raw["summary"], raw["title"]
        except (OSError, ValueError, KeyError):
            pass
    _remember(AGENTS, a.memory.id, a)
    return a


@app.post("/agent/run")
def agent_run(req: RunReq):
    if not BUSY.acquire(blocking=False): raise HTTPException(409, "agent busy with another run")
    try:
        agent = _agent_for(req)
    except Exception as e:
        BUSY.release()
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    q: queue.Queue = queue.Queue()
    mine: set = set()

    def confirm(name, args, level):
        if not req.stream: return False  # nobody to ask
        cid = uuid.uuid4().hex[:8]
        CONFIRMS[cid] = {"event": threading.Event(), "approve": False}
        mine.add(cid)
        q.put({"type": "confirm", "id": cid, "level": level, "name": name, "args": args,
               "text": f"{name} {json.dumps(args, ensure_ascii=False)}"})  # the approver sees everything, not a 120-char label
        CONFIRMS[cid]["event"].wait(300)  # no answer in 5 min = denied
        return CONFIRMS.pop(cid)["approve"]

    agent.confirm = confirm
    agent.mode = "auto-edit" if req.allow_write and req.approval == "ask" else req.approval

    def work():
        try:
            for e in agent.run(req.task): q.put({**e, "conversation_id": agent.memory.id} if e["type"] == "done" else e)
        except Exception as e:
            q.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            BUSY.release()
            q.put(None)
    threading.Thread(target=work, daemon=True).start()
    def drain():
        try: yield from _drain(q)
        finally:  # client gone (or finished): stop the run and deny what it was waiting on, instead of holding BUSY for 5 min
            agent.cancel()
            for c in list(mine):
                if c in CONFIRMS: CONFIRMS[c]["event"].set()
    if req.stream: return StreamingResponse(_sse(drain()), media_type="text/event-stream")
    evs = list(_drain(q))
    return {"conversation_id": agent.memory.id,
            "final": next((e["content"] for e in evs if e["type"] == "done"), None),
            "errors": [e["error"] for e in evs if e["type"] == "error"],
            "activity": [e["text"] for e in evs if e["type"] == "tool"], "totals": agent.totals}


@app.post("/agent/confirm")
def agent_confirm(req: ConfirmReq):
    c = CONFIRMS.get(req.id)
    if not c: raise HTTPException(404, "no such pending confirmation")
    c["approve"] = req.approve
    c["event"].set()
    return {"ok": True}


@app.post("/agent/stop")
def agent_stop(conversation_id: str | None = None):
    for cid, a in list(AGENTS.items()):
        if conversation_id in (None, cid): a.cancel()
    for c in list(CONFIRMS.values()): c["event"].set()  # unblock pending confirmations as denied
    return {"ok": True}


@app.get("/conversations")
def conversations():
    return Memory.list()


@app.get("/conversations/{cid}")
def conversation(cid: str):
    try: return Memory.load_raw(cid)
    except (OSError, ValueError, KeyError): raise HTTPException(404, "not found")


@app.get("/")
def index():
    page = (config.ROOT / "agent" / "web" / "index.html").read_text().replace("__TOKEN__", TOKEN)
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})
