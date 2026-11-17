from __future__ import annotations

import ipaddress
import math
from collections import Counter
from urllib.parse import urlsplit

import tldextract

from app.schemas.common import Evidence

# Use the list of web suffixes bundled with the library; never fetch from the network.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SUSPICIOUS_TOKENS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "password", "banking", "wallet", "support", "webscr", "billing",
)
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly"}


class InvalidUrlError(ValueError):
    """A URL we refuse to analyze. The message is safe to show to users."""


def normalize_url(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise InvalidUrlError("URL is empty")
    if len(raw) > 2048:
        raise InvalidUrlError("URL is too long")
    if any(c.isspace() or ord(c) < 32 for c in raw):
        raise InvalidUrlError("URL contains spaces or control characters")
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parts = urlsplit(raw)
        _ = parts.port  # raises ValueError when the port is not a number
    except ValueError as exc:
        raise InvalidUrlError("URL is malformed") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidUrlError("Only http and https URLs are supported")
    if not parts.hostname:
        raise InvalidUrlError("URL has no hostname")
    return raw


def registered_domain(host: str) -> str:
    ext = _EXTRACT(host)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain or host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _entropy(s: str) -> float:
    """How random-looking a string is. Machine-generated names score higher."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def extract_features(raw_url: str, host_only: bool = False) -> dict[str, float]:
    """Lexical features. The scheme (http vs https) is deliberately ignored."""
    url = normalize_url(raw_url)
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    rest = url.split("://", 1)[1]  # drop "http://" or "https://"

    if host_only:  # score the hostname alone
        rest, path, query, port = host, "", "", None
    else:
        path, query, port = p.path or "", p.query or "", p.port

    lower = rest.lower()
    is_ip = _is_ip(host)
    ext = _EXTRACT(host)
    sub_labels = [s for s in ext.subdomain.split(".") if s] if not is_ip else []
    digits_host = sum(ch.isdigit() for ch in host)

    return {
        "url_length": len(rest),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots_host": host.count("."),
        "num_hyphens_host": host.count("-"),
        "num_digits_host": digits_host,
        "digit_ratio_host": digits_host / len(host) if host else 0.0,
        "num_subdomains": len(sub_labels),
        "host_is_ip": int(is_ip),
        "has_at_symbol": int("@" in rest),
        "has_port": int(port is not None),
        "num_percent_encoded": lower.count("%"),
        "has_punycode": int("xn--" in host),
        "host_entropy": _entropy(host),
        "path_depth": len([s for s in path.split("/") if s]),
        "num_query_params": len([s for s in query.split("&") if s]),
        "suspicious_token_count": sum(lower.count(t) for t in SUSPICIOUS_TOKENS),
        "is_shortener": int(registered_domain(host) in SHORTENERS),
        "tld_length": len(ext.suffix.split(".")[-1]) if ext.suffix else 0,
        "longest_host_label": max((len(x) for x in host.split(".")), default=0),
    }


def _short(v: str, n: int = 80) -> str:
    return v if len(v) <= n else v[: n - 3] + "..."


def build_evidence(raw_url: str) -> list[Evidence]:
    """Plain-language OBSERVATIONS about the URL. These are facts, not model output."""
    url = normalize_url(raw_url)
    f = extract_features(url)
    host = (urlsplit(url).hostname or "").lower()
    out: list[Evidence] = []

    def add(id_: str, desc: str, sev: str, val: str | None = None) -> None:
        out.append(Evidence(id=id_, description=desc, severity=sev,
                            observed_value=_short(val) if val else None,
                            kind="observation", source="url_features"))

    if f["host_is_ip"]:
        add("url.host_is_ip", "Hostname is a raw IP address instead of a domain name", "high", host)
    if f["has_at_symbol"]:
        add("url.has_at_symbol", "URL contains '@', which can hide the real destination", "medium", url)
    if f["num_subdomains"] >= 4:
        add("url.many_subdomains", f"Hostname has {int(f['num_subdomains'])} subdomain levels",
            "medium", host)
    if f["has_punycode"]:
        add("url.punycode", "Hostname uses punycode (xn--), which can disguise lookalike characters",
            "low", host)
    if f["suspicious_token_count"] >= 2:
        add("url.suspicious_tokens",
            f"URL contains {int(f['suspicious_token_count'])} account-related words (e.g. login, verify)",
            "medium", url)
    if f["num_percent_encoded"] >= 3:
        add("url.heavy_encoding", "URL contains many percent-encoded characters", "low", url)
    if f["is_shortener"]:
        add("url.shortener", "URL uses a link shortener, so the final destination is hidden", "low", host)
    if f["url_length"] > 100:
        add("url.long_url", f"URL is unusually long ({int(f['url_length'])} characters)", "low")
    return out