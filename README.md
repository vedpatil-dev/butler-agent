# local-agent

A local coding agent: **Qwen3.5-9B (Q4_K_M) on Ollama**, tool calling, MCP, sandboxed filesystem/shell/git tools,
permission levels, context management, local HTTP API, CLI and a small web UI. No cloud LLM is used anywhere.

## Quick start

```bash
cd ~/Projects/local-agent
uv sync                      # Python 3.12 venv (.venv)
.venv/bin/agent status       # model / backend / context / RAM / acceleration
.venv/bin/agent run "List the files in the current project" -C ~/Projects/local-agent
.venv/bin/agent chat -C ~/Projects/my-app
.venv/bin/agent start        # API + web UI on http://127.0.0.1:8000
.venv/bin/agent stop [--unload]
.venv/bin/agent benchmark --raw cpu=http://127.0.0.1:11434 igpu=http://127.0.0.1:11435
.venv/bin/python -m pytest -m "not llm"   # fast tests (no model)
.venv/bin/python -m pytest                # + live-model tests (needs Ollama)
```

## Runtime & acceleration (measured on Ryzen AI 7 350 / Radeon 860M / 24 GB, Fedora 44)

| Backend | Prompt tok/s | Gen tok/s | TTFT | Verified how |
|---|---|---|---|---|
| CPU (system Ollama service) | 57 | 5.8 | 10.6 s | `size_vram = 0` in `/api/ps` |
| **iGPU, Vulkan (RADV)** | **126** | 5.2 | **5.1 s** | `size_vram` 100 % **and** amdgpu `gpu_busy_percent` peak 100 % |
| NPU (XDNA 2) | not usable | | | see below |

Generation is memory-bandwidth-bound (~5-6 tok/s either way); the iGPU wins on prompt processing, which dominates
agent loops (tool output is re-read every turn). Full data: `benchmarks/`.

### Using the iGPU
The system Ollama service (`/etc/systemd/system/ollama.service`) runs CPU-only: it has `OLLAMA_VULKAN=true`, but Ollama
drops integrated GPUs unless `OLLAMA_IGPU_ENABLE=1`, and its ROCm backend rejects this chip (`gfx1152`, no rocblas).
Instead of editing the system service, a **user-level instance** is used (no root):

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=$HOME/.ollama/models OLLAMA_IGPU_ENABLE=1 OLLAMA_VULKAN=1 ollama serve &
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3.5:9b        # separate 6.6 GB model store
echo AGENT_LLM_URL=http://127.0.0.1:11435 >> .env
```
Alternative (system-wide, needs sudo): `sudo systemctl edit ollama` and add `Environment="OLLAMA_IGPU_ENABLE=1"`.
Run only one of the two Ollama instances with the model loaded at a time (RAM).

### NPU (AMD XDNA 2) - in progress, blocked on Secure Boot

Ollama/llama.cpp have no XDNA backend, so the NPU path is a separate runtime: **FastFlowLM**, now an official
AMD ROCm project, talking to the NPU through XRT + the `amdxdna` DKMS driver. GGUF models (what Ollama uses) do not
run on it; FastFlowLM fetches its own optimized kernels per model (`flm pull qwen3.5:9b` - the same model/size, repackaged).

**Done:** `npu/src/xdna-driver` built from source and installed (XRT base/base-devel/npu + the amdxdna plugin RPM).
Two real Fedora packaging bugs were hit and fixed (patches kept local, not upstreamed):
- `ocl_icd.h` (legacy `ocl-icd` project) doesn't exist on Fedora, which ships Khronos' `OpenCL-ICD-Loader` instead.
  The only code needing it, `xocl` (legacy Alveo PCIe-card OpenCL bindings), is irrelevant to NPU use, so it's now
  skipped for `-npu` builds (`runtime_src/CMakeLists.txt`), and the stale `ocl-icd` RPM `Requires` was dropped to
  match (`CMake/cpackLin.cmake`) rather than fighting a dead package or faking an ABI-incompatible shim.
- Fedora 44's CMake 4 has no `cmake3` and rejects XRT's old `cmake_minimum_required()`; worked around with a
  `cmake3 -> cmake` symlink (`~/.local/xrt-build/bin`, on PATH) and `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
- `-lstdc++` static link needed `libstdc++-static`; RPM packaging needed `rpm-build`. Both just missing packages.

**Blocked:** this machine has **Secure Boot enabled**. The DKMS driver builds and installs, but the kernel refuses
to load a self-signed module whose key isn't enrolled, and installing it briefly took `/dev/accel/accel0` offline
(caught and reverted the same session - the in-tree signed driver was reloaded, DKMS registration undone, a reboot
right now boots clean). To actually use the NPU:
```bash
sudo mokutil --import /var/lib/dkms/mok.pub   # sets a one-time enrollment password
sudo reboot
# at the blue MokManager screen (before Fedora loads): Enroll MOK -> enter the password -> confirm -> continue
```
Then re-run `sudo dkms install xrt-amdxdna/2.27.0 -k $(uname -r)` (it's still `built`, just not installed) and
`sudo depmod -a`, build FastFlowLM itself (source already cloned to `npu/src/FastFlowLM`, not yet built),
`flm pull qwen3.5:9b`, and verify with `flm validate` / `xrt-smi examine` / `xrt-smi examine -r telemetry` counter
deltas while running inference (Linux has no NPU busy-percent counter the way amdgpu has one for the iGPU).

**Rollback** (if the DKMS driver misbehaves after enrollment): `sudo dkms uninstall xrt-amdxdna/2.27.0 -k $(uname -r)`
restores the original signed in-tree module and reruns depmod - no reboot required, verified working mid-session.

`agent/providers.py`'s `NPUProvider` is written against FastFlowLM's documented OpenAI-compatible API
(`flm serve`, port 52625) but **has never been run against a live server** - treat it as a sketch, not a
confirmed integration, until `flm` is actually built and serving.

**Full feature and behaviour reference: [docs/FEATURES.md](docs/FEATURES.md).**

## Providers

`butler` (same as `agent chat`) auto-selects the first reachable provider - npu > ollama > antigravity > claude > codex -
and says which; there is no startup menu (one appears only if *nothing* is reachable). Force one with
`--provider ollama|npu|antigravity|claude`, or switch mid-chat with `/provider <name>`. `butler providers` shows what's
detected: NPU hardware (AMD verified on this machine; Intel/Qualcomm detection is best-effort and **untested**; Apple
Silicon out of scope, this targets Linux), every known AI-agent CLI on PATH (`agent/detect.py`'s `KNOWN_AGENTS`), and
which providers are reachable right now.

- **ollama** - local, this project's own Agent/tools loop (the only fully local one that's been run end to end).
- **npu** - local, same loop, FastFlowLM backend.
- **antigravity** / **claude** / **codex** - **not local**: the whole task is handed to the `agy` / `claude` / `codex exec`
  CLI in print mode, each running its *own* agent loop with its *own* tools. Butler maps `--approval` (`ask` / `auto-edit` / `readonly`)
  onto each CLI's own permission flags (`APPROVAL_FLAGS` in `agent/providers.py`: claude `--permission-mode`, codex `-s`, agy `--sandbox`)
  and logs their tool calls and permission denials to `audit.jsonl`. Codex models come from `codex debug models`; no effort setting is
  wired for it, since its reasoning levels differ per model.

**Detected is not the same as integrated.** `butler providers` lists every known agent CLI found on PATH, and marks each as
`provider: <name>` (has a runner) or `detected only - no runner for it yet`. A runner is needed per CLI because each has its
own non-interactive command, output format, session-resume mechanism, model list and error shape - there is no common
protocol to auto-wire. To add one: write a `Runner` (`cmd()`, `run()`, `healthy()`), a `parse_<name>_line()`, register it in
`EXTERNAL_AGENTS` + `PROVIDER_NAMES` + `CLI_NAMES` + `AGENT_TO_PROVIDER`, and give `_fetch_models` a model source - the
structural tests in `tests/test_providers.py` fail if any of those is missed.

### In-chat commands
`/model` - **one list of every model from every reachable provider, each row tagged `[provider]`**
(`gemini-4-argon-low [antigravity]`, `claude-opus-5-5 [claude]`, `qwen3.5:9b [ollama]`). Picking a row sets the provider
*and* the model together - there is no separate provider step. Navigate with **Up/Down** (wraps; PgUp/PgDn/Home/End
too), **Enter** to select, **Esc** to cancel; just start typing to filter the list live (matches id, label and
provider). The current model is marked and the cursor starts on it. `/model <id>` jumps straight to an exact id, and
`/model provider/<id>` takes an id that isn't in the list yet. `/effort` (claude) uses the same widget. It needs a real
terminal; with piped stdin it tells you to use `/model <id>` instead. Sources:
`agy models` (live), butler's own ollama endpoint (`/api/tags`, so a listed model is one you can actually use), `flm list --filter installed` (NPU), `codex debug models`, and a
curated claude list (claude has no list API; see `CLAUDE_MODELS`). Providers that are down or missing are simply absent.
The slow one (`agy models`, ~5s) is fetched in parallel and warmed in the background at startup. The input box title
always shows what's active: `you · <model> [provider] · <ctx>% ctx`. Also: `/effort` (low|medium|high|xhigh|max - **claude only**),
`/provider <name>`, `/plan [on|off]`, `/approval [ask|auto-edit|readonly]`, `/status`, `/diff`, `/compact`, `/undo`, `/mcp`, `/init`,
`/new`, `/resume` (alias `/history`), `/help`, `/exit`. Tab completes `/commands` and `@file` paths with a dropdown. While on antigravity, agy's own slash commands
(`/skills`, `/plugin`, `/agents`, `/config`, ...) pass straight through; `/model` is ours because agy's native one is
read-only in print mode ("takes no arguments" - the switch is its `--model` flag). Claude's slash commands don't exist in
print mode at all.

**Effort is claude-only.** agy bakes effort into its model ids (`gemini-4-argon-low` / `-medium` / `-high`), so for agy the
effort *is* the model choice - and passing `--effort` alongside such an id is rejected outright ("conflicts with
--effort=high"; `xhigh`/`max` don't exist for it). butler therefore never sends `--effort` to agy; `/effort` there just
explains this. Errors from either CLI are printed (`error: ...`, `✘ failed`) rather than swallowed.

### Conversation history (`/resume` and `--continue`)
Butler records every cloud-agent exchange (claude / antigravity / codex) in its own small files - one JSON per conversation
in `~/.local/state/local-agent/butler-history/` (0600 in a 0700 dir; they hold whatever you typed, so delete the folder to
forget everything). `/resume` (or `/history`) opens the same arrow-key list used by `/model`: butler's cloud conversations
and the local model's saved ones (`.../conversations/`), newest first, each tagged `[provider]` with its model and age.
Picking one switches to its provider and model, prints a short recap of the last exchanges, and **continues it** - the
session id is stored, so after a restart the follow-up reaches the same claude/agy/codex conversation (verified across two
separate butler processes). Pass `--continue` (`-c`) to `butler chat` to automatically resume the newest conversation at startup.
Boundaries: only conversations started *through butler* are listed; an exchange that errored (no reply) is not recorded.
The input box's up-arrow recall is a separate file (`chat_history`). Set `AGENT_STATE_DIR` to keep all of butler's own files somewhere else (handy for tests).

### Follow-up questions and conversation continuity
Every message is a separate `agy -p` / `claude -p` process, so by default an agent has no memory of what it just asked.
butler therefore remembers the session id from each reply and passes `--resume` (claude) / `--conversation` (agy) on the
next message - your answer lands in the same conversation. `/new` starts a fresh one. Details, all verified live:
- claude has no ask-the-user tool in print mode; it asks in plain text, so continuity is all that's needed.
- agy's `ask_question` tool is **auto-skipped** in print mode and the stream never carries the question. butler recovers
  the question + options from agy's own transcript (`~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/`) and
  shows it. That file format is internal/undocumented, so this is best-effort: if it changes you just stop seeing the
  hint, nothing breaks.
- Print mode can't prompt you for *permissions* either. If claude is denied a tool, butler lists what was denied;
  allow it via claude's own `permissions.allow` or do it yourself. butler never passes a skip-permissions flag.
- codex continues via `codex exec resume <thread_id>` (verified: a follow-up recalls earlier context; a fresh `exec` doesn't).
- Children run with stdin = /dev/null so they can never read your terminal or compete with the input box.

## Architecture

```
user -> Agent (agent/orchestrator.py) -> LLM (agent/llm.py, Ollama /api/chat, streaming)
     -> tool calls -> policy/permission check (agent/policy.py) -> execute (tools/ or MCP) -> result -> LLM ... -> final answer
```
* `agent/memory.py` - token budget (`AGENT_NUM_CTX`, 8192 unless `.env` sets it; `.env.example` uses 65536), calibrated from server `prompt_tokens`. Old tool output is shrunk first; if still over, older turns are summarised by the
  local model. The system prompt (plus `AGENTS.md` / `CLAUDE.md` from the project root), the current task and the last tool exchange always stay verbatim.
* `agent/policy.py` & `agent/audit.py` - centralized approval policy (`ask` | `auto-edit` | `readonly`), `<untrusted-data>` taint tracking, session command-prefix allow-rules, and `audit.jsonl` logging.
* `tools/` - `list_files read_file search_files glob_files create_file edit_file multi_edit`, `execute_command`, `git_*`, `fetch_url`, `web_search` (plus built-in `recall_history`, `update_plan`, and `explore` read-only sub-agent).
* `mcp/server.py` - MCP server (official SDK) exposing the filesystem/git/shell tools; `agent/mcp_client.py` connects to
  any servers in `config/mcp.json` (postgres/github/playwright/docker are pre-filled, disabled). Enable with
  `agent run --mcp local ...`. MCP tools appear as `<server>__<tool>` (and enabled MCP search/fetch tools are automatically preferred over the built-in DuckDuckGo scrape).

## Security model
| Level | Examples | Behaviour |
|---|---|---|
| READ | list/read/search/glob, git status/diff/log, read-only shell (`ls`, `grep`, `git diff`...) | automatic (runs in read-only `bwrap` sandbox) |
| WRITE | create/edit/multi_edit file, git add/commit/checkout, other shell commands, `fetch_url`, `web_search` | asks in `ask` mode (`y/N/a/p`), auto-allows file edits in `auto-edit` mode (unless turn is tainted by untrusted data) |
| DANGEROUS | sudo, `rm -rf`, git push/reset --hard/branch -D, `DROP`, `docker system prune`, package/system changes | **always** asks (`y/N`), no "always allow" |

* Filesystem is confined to the directory `butler`/`agent` was started in (or `-C <dir>`); the API server uses `AGENT_WORKSPACE` (default `~/Projects`), symlinks resolved, with per-thread `cwd` via `contextvars`. `.env`, `.ssh`, keys, `.aws`,
  credentials etc. are unreadable, hidden from listings/search/glob, excluded from `git diff`, never staged by `git add`.
* Shell commands run inside **bubblewrap (`bwrap`)** when installed: `/` mounted read-only, only the project dir writable (and read-only for `READ` commands), credential directories masked with `tmpfs`/`/dev/null`, `--unshare-net` (override with `AGENT_SANDBOX_NET=1`), and an allow-listed environment (`_env()`). Set `AGENT_SANDBOX=require` to fail closed if `bwrap` is missing, or `AGENT_SANDBOX=0` to disable.
* Untrusted output from `fetch_url`, `web_search`, and MCP tools is wrapped in `<untrusted-data>` and taints the turn so any subsequent write or shell command requires explicit confirmation even in `auto-edit` mode.
* API binds to `127.0.0.1` only, checks `Host`, `Origin`, and `X-Agent-Token` (`api-token` in state dir, `0600`), and runs one agent at a time.

## API (`http://127.0.0.1:8000`)
`GET /health`, `GET /model`, `POST /chat {message, conversation_id?, stream}`,
`POST /agent/run {task, cwd?, conversation_id?, approval?, allow_write?, plan?, mcp?, stream}` (SSE events: `text`, `tool`, `result`,
`confirm`, `stats`, `done`), `POST /agent/confirm {id, approve}`, `POST /agent/stop`, `GET /conversations[/id]`.
`GET /` serves the web UI (chat, streaming, current task, tool activity/results, Stop, history, inline approve/deny).
Only concise actions are shown; model reasoning is disabled (`AGENT_THINK=false`) and never surfaced.

## Known limitations
* ~5 tok/s generation: multi-step tasks take minutes. `AGENT_THINK=true` makes it much slower.
* `.env.example` sets a 64K context (`AGENT_NUM_CTX`, 8K without it; the NPU's is `AGENT_NPU_CTX`, default 65536, and must match `flm serve --ctx-len` in `~/.config/systemd/user/flm-npu.service`). Measured, qwen3.5:9b on ollama: 8K 5.3 GB, 32K 6.0, 64K 7.1, 128K 9.9 (allocated at load, used or not); qwen3.5:2b on the NPU: 32K 3.9 GB, 64K 4.1, 128K 4.8. Tool schemas take ~1.6K.
* RAM follows the selected model: starting butler, `/provider` and `/model` unload ollama models you are not using, and stop the NPU server (`flm-npu` user unit) when you leave the NPU; picking the NPU starts it again (~10 s). Set `AGENT_NO_SERVICES=1` to disable this.
* Every message is also appended to a Markdown transcript (`~/.local/state/local-agent/conversations/<id>.md`, 0600) that compaction never shortens; the model's `recall_history` tool (read-only) searches it, so older turns can drop out of the context window without being lost.
  (KV cache grows; this machine has little RAM headroom).
* 9B-class models sometimes over-use tools or misjudge tasks; they are not a substitute for review.
* Prompt tok/s in per-turn stats undercounts when Ollama reuses its prompt cache (only new tokens are counted).
