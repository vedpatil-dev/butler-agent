"""Tool registry. @tool(level) registers a function; its JSON schema is derived from the signature."""
import inspect

READ, WRITE, DANGEROUS = "read", "write", "dangerous"
RANK = {READ: 0, WRITE: 1, DANGEROUS: 2}
REGISTRY: dict[str, dict] = {}
_JSON = {str: "string", int: "integer", bool: "boolean", list: "array"}


def _prop(t) -> dict:
    """list params are lists of edit objects ({path, old, new}); nothing else takes a list yet."""
    if t is list: return {"type": "array", "items": {"type": "object", "properties": {k: {"type": "string"} for k in ("path", "old", "new")}, "required": ["path", "old", "new"]}}
    return {"type": _JSON[t]}


def tool(level):
    """level: READ/WRITE/DANGEROUS, or callable(args) -> level when risk depends on the arguments."""
    def deco(fn):
        params = inspect.signature(fn).parameters
        REGISTRY[fn.__name__] = {
            "fn": fn, "level": level,
            "schema": {"type": "function", "function": {
                "name": fn.__name__, "description": inspect.cleandoc(fn.__doc__ or ""),
                "parameters": {"type": "object",
                               "properties": {n: _prop(p.annotation) for n, p in params.items()},
                               "required": [n for n, p in params.items() if p.default is p.empty]}}},
        }
        return fn
    return deco


def level_of(name: str, args: dict) -> str:
    lv = REGISTRY[name]["level"]
    return lv(args) if callable(lv) else lv


def _coerce(v, t):
    # small models often send "5" or "true" as strings
    if t == "integer": return int(v)
    if t == "boolean": return v if isinstance(v, bool) else str(v).lower() == "true"
    return v


def call(name: str, args: dict) -> str:
    e = REGISTRY[name]
    props = e["schema"]["function"]["parameters"]["properties"]
    # a model that sends null for an optional parameter means "use the default"
    return e["fn"](**{k: _coerce(v, props.get(k, {}).get("type")) for k, v in args.items() if v is not None})


def clip(s: str, n: int = 3000) -> str:
    """Keep head and tail of long output so the model context is not flooded."""
    if len(s) <= n: return s
    h = n * 2 // 3
    return f"{s[:h]}\n... [{len(s) - n} chars omitted] ...\n{s[-(n - h):]}"


from . import filesystem, git, shell, web  # noqa: E402,F401  (register tools)
