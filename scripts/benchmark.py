#!/usr/bin/env python
"""Benchmark. Part A (--raw): speed per backend with the accelerator verified from sysfs while generating.
Part B (--tasks): the six agent tasks end to end.
  python scripts/benchmark.py --raw cpu=http://127.0.0.1:11434 igpu=http://127.0.0.1:11435
  python scripts/benchmark.py --tasks http://127.0.0.1:11435 [--num-gpu 0]"""
import argparse
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agent import config  # noqa: E402
from agent.llm import LLM, meminfo  # noqa: E402

OUT = ROOT / "benchmarks"


def _read(p, cast=float):
    try: return cast(Path(p).read_text().strip())
    except (OSError, ValueError): return None


class Sampler(threading.Thread):
    """Samples GPU busy %, shared-memory use, temp, power and the NPU power state every 0.5s."""
    def __init__(self):
        super().__init__(daemon=True)
        d = next(iter(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")), None)
        self.dev = Path(d).parent if d else None
        hw = glob.glob(f"{self.dev}/hwmon/hwmon*") if self.dev else []
        self.hw = hw[0] if hw else None
        self.s = {"gpu_busy": [], "gtt_gb": [], "temp_c": [], "power_w": [], "npu": set()}
        self.stop_ev = threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            if self.dev:
                self.s["gpu_busy"].append(_read(self.dev / "gpu_busy_percent"))
                self.s["gtt_gb"].append((_read(self.dev / "mem_info_gtt_used") or 0) / 2**30)
            if self.hw:
                self.s["temp_c"].append((_read(f"{self.hw}/temp1_input") or 0) / 1000)
                self.s["power_w"].append((_read(f"{self.hw}/power1_average") or _read(f"{self.hw}/power1_input") or 0) / 1e6)
            self.s["npu"].add(_read("/sys/class/accel/accel0/device/power/runtime_status", str))
            self.stop_ev.wait(0.5)

    def summary(self):
        f = lambda k, fn: round(fn([x for x in self.s[k] if x is not None] or [0]), 1)
        return {"gpu_busy_peak_pct": f("gpu_busy", max), "gpu_busy_avg_pct": f("gpu_busy", st.mean),
                "gtt_used_peak_gb": f("gtt_gb", max), "temp_peak_c": f("temp_c", max),
                "gpu_power_avg_w": f("power_w", st.mean), "npu_states_seen": sorted(x for x in self.s["npu"] if x)}


def unload_all(urls):
    for u in urls:
        try: httpx.post(f"{u}/api/generate", json={"model": config.MODEL, "keep_alive": 0}, timeout=60)
        except httpx.HTTPError: pass
    time.sleep(3)


def prompt_text():
    code = (ROOT / "tools" / "filesystem.py").read_text()[:1800]
    return f"[run {uuid.uuid4().hex}] Explain what this Python code does in about 100 words.\n\n{code}"  # nonce defeats prompt cache


def raw_bench(name, url, num_gpu=None, reps=3):
    llm = LLM(url=url, num_gpu=num_gpu)
    if not llm.healthy(): return {"backend": name, "error": f"{url} not reachable"}
    t0 = time.time()
    list(llm.stream([{"role": "user", "content": "hi"}], num_predict=2))  # load + warm up
    load_s = round(time.time() - t0, 1)
    mem0 = meminfo()["available_gb"]
    sm, runs = Sampler(), []
    sm.start()
    for _ in range(reps):
        t, first, stats = time.time(), None, None
        for kind, val in llm.stream([{"role": "user", "content": prompt_text()}], num_predict=128, temperature=0, seed=1):
            if kind == "text" and first is None: first = time.time() - t
            if kind == "stats": stats = val
        runs.append({"ttft_s": round(first or 0, 2), "latency_s": round(time.time() - t, 1), **stats})
    sm.stop_ev.set()
    sm.join()
    status = llm.status()
    med = lambda k: round(st.median(r[k] for r in runs), 2)
    return {"backend": name, "url": url, "num_gpu_opt": num_gpu, "model": llm.model, "quant": status.get("quantization"),
            "context": llm.num_ctx, "runtime": "ollama", "load_s": load_s,
            "prompt_tokens": runs[0]["prompt_tokens"], "prompt_tps": med("prompt_tps"), "gen_tps": med("gen_tps"),
            "ttft_s": med("ttft_s"), "latency_s": med("latency_s"), "model_ram_gb": status.get("model_ram_gb"),
            "gpu_offload_pct_reported_by_ollama": status.get("gpu_offload_pct"),
            "sys_ram_available_before_gb": mem0, "sys_ram_available_after_gb": meminfo()["available_gb"],
            **sm.summary(), "runs": runs}


def verdict(r):
    """Only call an accelerator active if two independent signals agree."""
    gpu = (r.get("gpu_offload_pct_reported_by_ollama") or 0) > 0 and r["gpu_busy_peak_pct"] >= 20
    npu = "active" in r["npu_states_seen"]
    return {"CPU": "yes" if not gpu else "partial (non-offloaded layers/sampling)", "iGPU": "VERIFIED ACTIVE" if gpu else "not used",
            "NPU": "ACTIVE (unexpected)" if npu else "not used (stayed suspended)"}


def raw_report(results):
    L = ["## Part A - raw speed (same prompt, ctx 8192, 128 tokens, temperature 0, cold prompt cache)\n",
         "| Backend | Model / quant | Prompt tok/s | Gen tok/s | TTFT s | Latency s | Model RAM GB | GPU busy peak % | GTT peak GB | Temp peak C | GPU avg W | Verified: CPU / iGPU / NPU |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r: L.append(f"| {r['backend']} | {r['error']} |"); continue
        v = verdict(r)
        L.append(f"| {r['backend']} | {r['model']} {r['quant']} | {r['prompt_tps']} | {r['gen_tps']} | {r['ttft_s']} | {r['latency_s']} | "
                 f"{r['model_ram_gb']} | {r['gpu_busy_peak_pct']} | {r['gtt_used_peak_gb']} | {r['temp_peak_c']} | {r['gpu_power_avg_w']} | "
                 f"{v['CPU']} / {v['iGPU']} / {v['NPU']} |")
    return "\n".join(L)


# ---------------- Part B: agent tasks ----------------
def make_project():
    base = Path.home() / "Projects" / "_bench" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    proj = base / "buggy_project"
    shutil.copytree(ROOT / "scripts" / "fixtures" / "buggy_project", proj)
    for c in (["init", "-q"], ["config", "user.email", "bench@local"], ["config", "user.name", "bench"], ["add", "."], ["commit", "-qm", "initial"]):
        subprocess.run(["git", "-C", str(proj), *c], check=True, capture_output=True)
    return proj


def tasks_bench(url, num_gpu, only=None):
    from agent.mcp_client import MCPHub, load_config
    from agent.orchestrator import Agent
    llm = LLM(url=url, num_gpu=num_gpu)
    proj = make_project()
    py = f"{sys.executable} -m pytest -q -p no:cacheprovider"
    denied = []

    def confirm(name, args, level):  # benchmark policy: file edits and the pytest command, inside the throwaway copy only
        ok = level == "write" and (name in ("edit_file", "create_file") or (name == "execute_command" and args.get("command", "").startswith(py)))
        if not ok: denied.append((name, level))
        return ok
    from tools.filesystem import set_cwd
    os.environ["AGENT_WORKSPACE"] = str(proj)  # else the whole of ~/Projects is writable, real repos included
    set_cwd(proj)
    hub = MCPHub(load_config(), only=["local"])  # after set_cwd: the MCP server inherits the project dir
    sm = Sampler(); sm.start()
    used = lambda evs, pre: [e["name"] for e in evs if e["type"] == "tool" and e["name"].startswith(pre)]
    T = [
        ("1 Java palindrome", "Write a Java function that checks whether a string is a palindrome.", None,
         lambda evs, fin: "boolean" in fin and "String" in fin),
        ("2 List files", "List the files in the current project.", None,
         lambda evs, fin: "list_files" in used(evs, "") and "stats.py" in fin),
        ("3 TODO summary", "Find all TODO comments in this project and summarize them.", None,
         lambda evs, fin: "search_files" in used(evs, "") and "empty" in fin.lower() and "key" in fin.lower()),
        ("4 Fix bug + run tests", f"Inspect the test project, identify a simple bug, fix it, and run the tests. (Run tests with: {py})", None,
         lambda evs, fin: subprocess.run(py.split(), cwd=proj, capture_output=True).returncode == 0),
        ("5 Git status", "Show the current Git status and summarize the changes.", None,
         lambda evs, fin: "git_status" in used(evs, "") and "stats.py" in fin),
        ("6 MCP tool", "Use an MCP tool (the tools prefixed with local__) to show the git status, and report the result.", hub,
         lambda evs, fin: any(e["type"] == "result" and e["name"].startswith("local__") and e["ok"] for e in evs)),
    ]
    rows = []
    for name, task, mcp, check in T:
        if only and name[0] not in only: continue
        agent = Agent(llm, confirm, project=proj, mcp=mcp, max_steps=15)
        agent.memory.persist = False
        t0 = time.time()
        evs, err = [], None
        try: evs = list(agent.run(task))
        except Exception as e: err = f"{type(e).__name__}: {e}"
        fin = next((e["content"] for e in evs if e["type"] == "done"), err or "")
        tt = agent.totals
        rows.append({"task": name, "ok": bool(evs) and bool(check(evs, fin)), "seconds": round(time.time() - t0, 1), "steps": tt["steps"],
                     "tool_calls": tt["tool_calls"], "tool_errors": tt["tool_errors"], "prompt_tokens": tt["prompt_tokens"],
                     "gen_tokens": tt["gen_tokens"], "gen_tps": round(tt["gen_tokens"] / tt["gen_s"], 1) if tt["gen_s"] else 0,
                     "tools": [e["name"] for e in evs if e["type"] == "tool"], "final": fin[:400], "error": err})
        print(f"[{'PASS' if rows[-1]['ok'] else 'FAIL'}] {name} {rows[-1]['seconds']}s tools={rows[-1]['tools']}", flush=True)
    hub.close(); sm.stop_ev.set(); sm.join()
    status = llm.status()
    calls = sum(r["tool_calls"] for r in rows); errs = sum(r["tool_errors"] for r in rows)
    tool_tasks = [r for r in rows if r["tool_calls"]]
    return {"url": url, "model": llm.model, "quant": status.get("quantization"), "runtime": "ollama", "context": llm.num_ctx,
            "backend": status.get("acceleration"), "model_ram_gb": status.get("model_ram_gb"),
            "tokens_per_sec_avg": round(st.mean([r["gen_tps"] for r in rows if r["gen_tps"]] or [0]), 1),
            "tool_call_success": f"{calls - errs}/{calls}", "tasks_passed": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
            "multi_step_success": f"{sum(r['ok'] and r['tool_calls'] >= 3 for r in rows)}/{sum(r['task'][0] in '34' for r in rows)} (tasks 3-4 need >=3 tool calls)",
            "dangerous_actions_blocked": denied, "hardware": sm.summary(), "rows": rows, "project": str(proj)}


def tasks_report(r):
    L = [f"## Part B - agent tasks\n", f"Model: {r['model']} {r['quant']} | Runtime: {r['runtime']} | Backend: {r['backend']} | Context: {r['context']} | "
         f"Model RAM: {r['model_ram_gb']} GB | Avg gen: {r['tokens_per_sec_avg']} tok/s\n",
         f"Tool-call success: {r['tool_call_success']} | Tasks passed: {r['tasks_passed']} | Multi-step: {r['multi_step_success']}\n",
         f"Hardware while running: {json.dumps(r['hardware'])}\n",
         "| Task | Pass | Time s | Steps | Tool calls (errors) | Prompt tok | Gen tok | Tools used |", "|---|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['task']} | {'PASS' if x['ok'] else 'FAIL'} | {x['seconds']} | {x['steps']} | {x['tool_calls']} ({x['tool_errors']}) | "
                 f"{x['prompt_tokens']} | {x['gen_tokens']} | {', '.join(x['tools'])} |")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", metavar="NAME=URL[,num_gpu=N]")
    ap.add_argument("--tasks", metavar="URL")
    ap.add_argument("--num-gpu", type=int)
    ap.add_argument("--only", help="task numbers, e.g. 16")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if a.raw:
        specs = []
        for s in a.raw:
            n, _, rest = s.partition("=")
            u, *o = rest.split(",")
            specs.append((n, u, int(o[0].split("=")[1]) if o else None))
        res = []
        for n, u, ng in specs:
            unload_all([x[1] for x in specs])
            print(f"== {n} ==", flush=True)
            res.append(raw_bench(n, u, ng))
            print(json.dumps({k: v for k, v in res[-1].items() if k != "runs"}), flush=True)
        unload_all([x[1] for x in specs])
        (OUT / f"raw-{stamp}.json").write_text(json.dumps(res, indent=1))
        (OUT / f"raw-{stamp}.md").write_text(raw_report(res))
        print(raw_report(res))
    if a.tasks:
        r = tasks_bench(a.tasks, a.num_gpu, a.only)
        (OUT / f"tasks-{stamp}.json").write_text(json.dumps(r, indent=1))
        (OUT / f"tasks-{stamp}.md").write_text(tasks_report(r))
        print(tasks_report(r))
