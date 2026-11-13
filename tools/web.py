"""Web tools. Both need confirmation: fetch_url sends a model-chosen URL, web_search a model-chosen query (which can carry
secrets read from the workspace). fetch_url refuses loopback/private addresses."""
import html
import ipaddress
import re
import socket
import time
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from . import WRITE, clip, tool

_TAGS = re.compile(r"<[^>]+>")


@tool(WRITE)
def fetch_url(url: str) -> str:
    """Fetch a public http(s) page and return its text (HTML tags stripped, truncated)."""
    for _ in range(6):  # follow redirects by hand so every hop is re-checked against the private-address rule
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname: raise ValueError("only http(s) URLs")
        for *_, sa in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(sa[0])
            if not ip.is_global: raise PermissionError(f"{u.hostname} is not a public address")  # private, loopback, link-local, CGNAT, ...
        # trust_env=False: no proxy from the environment (it would resolve the name itself). Body capped in size and time.
        # ponytail: the name is resolved again by httpx, so DNS rebinding between check and connect is not closed.
        with httpx.stream("GET", url, timeout=20, follow_redirects=False, trust_env=False,
                          headers={"User-Agent": "Mozilla/5.0 (compatible; local-agent)", "Accept-Encoding": "identity"}) as r:
            if r.status_code not in (301, 302, 303, 307, 308) or not r.headers.get("location"):
                body, deadline = b"", time.monotonic() + 30
                for chunk in r.iter_bytes():
                    body += chunk
                    if len(body) > 2_000_000 or time.monotonic() > deadline: break
                break
            url = urljoin(url, r.headers["location"])
    else:
        raise RuntimeError("too many redirects")
    if r.status_code >= 400: return f"HTTP {r.status_code} from {u.hostname}: the site refused or blocked this request (login wall or bot protection), so there is no page content."
    text = re.sub(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", " ", body.decode(r.encoding or "utf-8", "replace"))
    return clip(re.sub(r"\s+", " ", text).strip(), 4000)


@tool(WRITE)  # the query is model-chosen text leaving the machine, so it is approved like fetch_url
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) and return the top results as title, URL and snippet. The query must be a few plain
    keywords taken from the user's own words (e.g. "ved patil"); do not add extra terms. Use fetch_url to read a result."""
    query = " ".join(query.split())[:200].rsplit(" ", 1)[0] if len(query) > 200 else query  # DuckDuckGo 302s long queries
    r = httpx.post("https://html.duckduckgo.com/html/", data={"q": query}, timeout=20,
                   headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200: raise RuntimeError(f"DuckDuckGo refused the search (HTTP {r.status_code}); try a shorter query")
    hits = []
    for chunk in r.text.split('class="result__a"')[1:]:  # per result, so one without a snippet can't eat the next
        m = re.match(r' href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.S)
        s = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', chunk, re.S)
        if m: hits.append((m[1], m[2], s[1] if s else ""))
    out = []
    for href, title, snip in hits[:8]:
        if "uddg=" in href: href = unquote(parse_qs(urlparse(html.unescape(href)).query).get("uddg", [href])[0])
        out.append(f"{html.unescape(_TAGS.sub('', title)).strip()}\n{href}\n{html.unescape(_TAGS.sub('', snip)).strip()}")
    return clip("\n\n".join(out), 4000) or "no results"
