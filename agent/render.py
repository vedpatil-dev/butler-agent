"""Terminal output for agent events: spinner, streamed markdown, tool lines, the end-of-turn footer, error hints."""
import re
import threading
import time

DIM, BOLD, RED, GREEN, CYAN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[36m", "\033[0m"


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MD_BOLD, _MD_CODE = re.compile(r"\*\*(.+?)\*\*"), re.compile(r"`([^`\n]+?)`")


def _md(text: str) -> str:
    """Inline markdown -> ANSI, applied per streamed chunk (not the whole response). ponytail:
    a **bold**/`code` span split across two chunks won't render (chunks are usually word/phrase
    sized in practice, so this is rare) - upgrade path: buffer until a closing marker if it bites."""
    text = _MD_BOLD.sub(rf"{BOLD}\1{RESET}", text)
    return _MD_CODE.sub(rf"\033[36m\1{RESET}", text)


def _spin(stop: threading.Event, t0: float):
    i = 0
    while not stop.wait(0.08):
        i += 1
        print(f"\r{DIM}{SPINNER_FRAMES[i % len(SPINNER_FRAMES)]} thinking... {time.time() - t0:.0f}s{RESET}",
              end="", flush=True)


_CTRL = re.compile(r"\x1b|[\x00-\x08\x0b-\x1a\x1c-\x1f\x7f]")  # keeps \n and \t; drops ESC so text can't drive the terminal


def _render(events):
    sp = {}

    def spin_on():  # the model is working: show it. Never while a tool waits for an approval (input() would clash)
        sp["stop"] = threading.Event()
        sp["thread"] = threading.Thread(target=_spin, args=(sp["stop"], time.time()), daemon=True)
        sp["thread"].start()

    def spin_off():
        if "stop" in sp and not sp["stop"].is_set():
            sp["stop"].set(); sp["thread"].join(timeout=0.5)
            print("\r" + " " * 30 + "\r", end="", flush=True)  # erase the spinner line

    spin_on()
    from .ui import MdStream
    md, t0, tools_used, tokens = MdStream(), time.time(), 0, 0
    streamed, done, failed, parts = False, None, False, []  # some providers (antigravity) take 8-16s before the first token
    try:
        for e in events:
            spin_off()
            t = e["type"]
            if t == "text": streamed = True; e["delta"] = _CTRL.sub("", e["delta"]); parts.append(e["delta"]); print(md.feed(e["delta"]), end="", flush=True)
            elif t == "tool": tools_used += 1; print(md.flush() + f"\n{DIM}→ {e['text']}{RESET}", flush=True)
            elif t == "result":
                if e.get("name") == "update_plan" and e["ok"]:
                    body = e["text"]
                else:
                    lines = e["text"].splitlines() or [""]
                    preview = "\n    ".join(l[:160] for l in lines[:3])
                    extra = f"\n    (+{len(lines) - 3} more lines)" if len(lines) > 3 else ""
                    body = preview + extra
                mark = f"{GREEN}✓{RESET}" if e["ok"] else f"{RED}✗{RESET}"
                print(f"  {mark} {DIM}{body} ({e['seconds']}s){RESET}"); spin_on()  # next: the model reads the result
            elif t == "permission":  # claude asks via permissions.PermissionBroker; the child waits for the answer
                from .confirm import permission_prompt
                permission_prompt(e); spin_on()
            elif t == "activity": print(f"{DIM}· {e['text']}{RESET}")
            elif t == "error":  # was silently dropped: a rejected model/effort combo looked like "no response"
                failed = True
                print(f"\n{RED}error:{RESET} {friendly(e['error'].strip())}")
            elif t == "ask":  # agy's ask_question tool, auto-skipped in print mode (see agy_skipped_questions)
                opts = f"  {DIM}[{' / '.join(e['options'])}]{RESET}" if e["options"] else ""
                print(f"\n{BOLD}? agy asked:{RESET} {e['question']}{opts}\n{DIM}  (print mode can't wait for an answer - "
                      f"just type it as your next message, it continues the same conversation){RESET}")
            elif t == "stats": tokens += e.get("prompt_tokens", 0) + e.get("gen_tokens", 0); print(f"\n{DIM}[{e['prompt_tokens']} prompt tok, {e['gen_tokens']} gen tok @ {e['gen_tps']} tok/s]{RESET}")
            elif t == "done":
                # nothing was streamed as "text" (e.g. agy's native slash commands come back as a
                # single command_result the parser doesn't unpack) - the content only lives here.
                print(md.flush(), end="")
                if not streamed and e.get("content"): print(_md(_CTRL.sub("", e["content"])))
                if e.get("denied"):  # claude's permission requests - print mode has nobody to ask
                    print(f"\n{RED}claude was denied permission for:{RESET} " + "; ".join(e["denied"])
                          + f"\n{DIM}  print mode can't prompt you. Allow it in claude's own permissions "
                            f"(permissions.allow in ~/.claude/settings.json), or do it yourself.{RESET}")
                bits = [f"{tools_used} tool call{'s' * (tools_used != 1)}"] * bool(tools_used) + [f"{tokens} tok"] * bool(tokens) + [f"{time.time() - t0:.0f}s"]
                print(f"\n{RED}✘ failed{RESET}" if failed else f"\n{GREEN}✔ done{RESET} {DIM}· {' · '.join(bits)}{RESET}")
                done = {**e, "reply": "".join(parts) or e.get("content", "")}
    finally:
        spin_off()  # never leave it printing into the next prompt
    return done  # callers read done["session"] (cloud agents: continue the conversation) and done["reply"] (history)


def friendly(err: str) -> str:
    """Append what to do about the common failures (the raw 'ConnectError: ...' alone doesn't say)."""
    hints = (("Connect", "the model server isn't reachable - start it (`ollama serve`) or check `agent providers`"),
             ("Timeout", "the model took too long - try a smaller model or a shorter task"),
             ("not found", "model missing on the server - `ollama pull <model>` or pick another with /model"))
    return err + next((f"\n  {DIM}hint: {h}{RESET}" for k, h in hints if k.lower() in err.lower()), "")
