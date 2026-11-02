"""SSRF-safe URL fetching for website ingestion.

A "fetch this URL" feature is a request-forgery primitive unless every hop is checked.
The rules, in order:

1. Scheme ``http``/``https`` only; no credentials in the URL; ports 80/443 only.
2. The host is resolved *before* connecting and every resolved address must be public:
   loopback, private (RFC 1918), link-local (incl. cloud metadata 169.254.169.254),
   CGNAT, multicast, reserved, unspecified and IPv6 equivalents are all refused,
   including IPv4-mapped IPv6 forms.
3. Redirects are never followed automatically. Each ``Location`` is re-validated from
   step 1, up to ``SETU_URL_MAX_REDIRECTS`` hops.
4. After connecting, the *actual* peer address is checked again before any body byte is
   read. That closes the DNS-rebinding window between resolution and connection.
5. ``robots.txt`` is honoured for the SETU user agent.
6. Only ``text/html``, ``application/xhtml+xml``, ``text/plain`` and ``application/pdf``
   bodies are accepted, streamed with a hard byte cap and timeout. Nothing else is
   downloaded, so the feature can't be used to pull arbitrary binaries.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from ..config import settings

USER_AGENT = "SETU-KnowledgeBot/2.0 (+https://github.com/; research ingestion; respects robots.txt)"
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/pdf")
ALLOWED_PORTS = {80, 443}

_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_robots_cache: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
_last_hit: dict[str, float] = {}


class FetchError(ValueError):
    """The URL was refused or could not be fetched. The message is safe to show users."""


@dataclass
class FetchResult:
    url: str
    final_url: str
    content: bytes
    content_type: str
    status: int
    redirects: list[str]


def is_public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified or not ip.is_global):
        return False
    return not (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT)


def validate_url(url: str) -> tuple[str, str, int]:
    """Return ``(scheme, host, port)`` or raise :class:`FetchError`."""
    if len(url) > 2048:
        raise FetchError("URL is too long")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http and https URLs can be imported")
    if parts.username or parts.password:
        raise FetchError("URLs with embedded credentials are not allowed")
    host = (parts.hostname or "").strip().rstrip(".").lower()
    if not host:
        raise FetchError("URL has no host")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        raise FetchError("local and internal hostnames are not allowed")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise FetchError("invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise FetchError("only ports 80 and 443 are allowed")
    return parts.scheme, host, port


def resolve_public(host: str, port: int) -> list[str]:
    try:
        ipaddress.ip_address(host)
        addrs = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise FetchError(f"could not resolve host '{host}'") from exc
        addrs = sorted({i[4][0] for i in infos})
    if not addrs:
        raise FetchError(f"could not resolve host '{host}'")
    bad = [a for a in addrs if not is_public_ip(a)]
    if bad:
        raise FetchError("the host resolves to a private, loopback or reserved address")
    return addrs


def _robots_allowed(client: httpx.Client, scheme: str, host: str, url: str) -> bool:
    key = f"{scheme}://{host}"
    cached = _robots_cache.get(key)
    if cached and time.time() - cached[0] < 3600:
        rp = cached[1]
    else:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = client.get(f"{key}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=8.0)
            if r.status_code in (401, 403):
                rp = None  # robots itself forbidden: treat as disallow-all (RFC 9309 §2.3.1.3)
            elif r.status_code >= 400:
                rp.parse([])  # no robots.txt: everything allowed
            else:
                rp.parse(r.text.splitlines())
        except httpx.HTTPError:
            rp.parse([])
        _robots_cache[key] = (time.time(), rp)
    if rp is None:
        return False
    return rp.can_fetch(USER_AGENT, url)


def _check_peer(resp: httpx.Response) -> None:
    stream = resp.extensions.get("network_stream")
    if stream is None:
        return
    try:
        peer = stream.get_extra_info("server_addr")
    except Exception:  # noqa: BLE001
        peer = None
    if peer and not is_public_ip(str(peer[0])):
        raise FetchError("connection landed on a non-public address (possible DNS rebinding)")


def safe_fetch(url: str, *, max_bytes: int | None = None, check_robots: bool = True,
               transport: httpx.BaseTransport | None = None) -> FetchResult:
    max_bytes = max_bytes or int(settings.max_url_mb * 1024 * 1024)
    redirects: list[str] = []
    current = url.strip()
    with httpx.Client(follow_redirects=False, timeout=settings.url_timeout_s, transport=transport,
                      headers={"User-Agent": USER_AGENT, "Accept": ", ".join(ALLOWED_TYPES)}) as client:
        for _hop in range(settings.url_max_redirects + 1):
            scheme, host, port = validate_url(current)
            resolve_public(host, port)
            if check_robots and not _robots_allowed(client, scheme, host, current):
                raise FetchError(f"robots.txt on {host} disallows fetching this page")
            # Politeness: at most one request per second per host.
            wait = 1.0 - (time.time() - _last_hit.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            _last_hit[host] = time.time()
            try:
                with client.stream("GET", current) as resp:
                    _check_peer(resp)
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("redirect without a Location header")
                        nxt = urljoin(current, loc)
                        redirects.append(nxt)
                        current = nxt
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"the site returned HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype not in ALLOWED_TYPES:
                        raise FetchError(f"content type '{ctype or 'unknown'}' is not importable")
                    declared = int(resp.headers.get("content-length") or 0)
                    if declared and declared > max_bytes:
                        raise FetchError(f"page is larger than the {max_bytes // 1048576} MB limit")
                    buf = bytearray()
                    for chunk in resp.iter_bytes():
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise FetchError(f"page is larger than the {max_bytes // 1048576} MB limit")
                    return FetchResult(url=url, final_url=current, content=bytes(buf), content_type=ctype,
                                       status=resp.status_code, redirects=redirects)
            except httpx.TimeoutException as exc:
                raise FetchError("the site did not respond in time") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"network error: {type(exc).__name__}") from exc
    raise FetchError(f"too many redirects (more than {settings.url_max_redirects})")


__all__ = ["FetchError", "FetchResult", "is_public_ip", "resolve_public", "safe_fetch", "validate_url"]
