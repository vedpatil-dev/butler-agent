# Repository Guidelines

## Project Structure & Modules

This Python 3.12 project packages the `agent/` runtime and `tools/` tool implementations. The CLI and `butler` entry points are defined in `agent/main.py`; orchestration, providers, memory, and configuration live in focused modules under `agent/`. The FastAPI service and browser UI are in `agent/server.py` and `agent/web/`. `mcp/server.py` exposes tools over MCP, `config/mcp.json` contains server examples, `tests/` holds pytest coverage, and `benchmarks/` stores benchmark tasks and results. NPU experiments and local runtime notes are under `npu/`.

## Build, Test, and Development

- `uv sync` creates the project environment and installs dependencies.
- `.venv/bin/agent status` reports model/backend availability and runtime details.
- `.venv/bin/agent run "<task>" -C <project-dir>` runs a task against a chosen workspace.
- `.venv/bin/agent chat -C <project-dir>` starts an interactive session; `.venv/bin/agent start` launches the local API and web UI.
- `.venv/bin/python -m pytest -m "not llm"` runs fast tests without a model; `.venv/bin/python -m pytest` also runs tests marked `llm` and requires a live Ollama model.
- `.venv/bin/agent benchmark --raw <name>=<url>` records backend performance.

## Style & Naming

Follow the existing Python style: four spaces, type hints for public interfaces, descriptive `snake_case` functions and modules, and `PascalCase` classes. Keep provider and tool behavior in their respective modules rather than growing the CLI entry point. Name tests `test_*.py` and test functions `test_<behavior>`. No separate formatter or linter is configured; keep changes consistent with nearby code.

## Testing Guidelines

Use pytest. Add or update focused tests in `tests/` for changed behavior, and prefer mocked providers for ordinary unit coverage. Mark tests requiring a live local model with `@pytest.mark.llm`; run those only when Ollama and the configured model are available. The repository does not specify a coverage threshold.

## Commits & Pull Requests

Recent history uses short, imperative summaries (for example, `Fix: ...` and `/model: ...`). Keep commit subjects concise and focused. Pull requests should explain the user-visible or operational change, list validation commands and results, link related issues when applicable, and include screenshots for web UI changes. Document configuration or security impact, especially changes to workspace boundaries, permissions, or provider behavior.

## Configuration & Security

Copy `.env.example` to `.env` for local settings; never commit credentials or machine-specific secrets. The agent is designed to confine file access to `AGENT_WORKSPACE` and to require approval for writes and dangerous commands. Preserve these checks when adding tools or API operations, and treat configured MCP servers as trusted code.
