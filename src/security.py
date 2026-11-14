"""Security helpers: secret masking, URL policy, identifier validation, export escaping."""

from __future__ import annotations

import functools
import ipaddress
import os
import re
import socket
import time
from urllib.parse import urlparse

# --- Secret masking ---------------------------------------------------------
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-(?:proj-|or-v1-)?[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)[\"'\s:=]+)[^\s\"',}]{6,}"),
    re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"),
]

MASK = "***REDACTED***"


def mask_secrets(text: object, extra_secrets: list[str] | None = None) -> str:
    """Return ``text`` as string with anything resembling a credential replaced.

    ``extra_secrets`` are literal values (e.g. keys typed into the UI) that must be
    removed even if they do not match a known pattern.
    """
    out = str(text)
    for secret in extra_secrets or []:
        if secret and len(secret) >= 4:
            out = out.replace(secret, MASK)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            out = pattern.sub(lambda m: m.group(1) + MASK, out)
        else:
            out = pattern.sub(MASK, out)
    return out


_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def scrubbed_environ() -> dict[str, str]:
    """Copy of ``os.environ`` without variables that look like credentials."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(hint in k.upper() for hint in _SECRET_ENV_HINTS)
    }


_SANDBOX_ENV_NAMES = frozenset({
    "PATH", "HOME", "USERPROFILE", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
})


def sandbox_environ() -> dict[str, str]:
    """Minimal environment for the analysis sandbox: an allowlist, not a blocklist.

    A blocklist of secret-looking names misses variables such as ``GH_PAT`` or
    ``DATABASE_URL``; the sandbox child only needs what Python and the numeric libraries
    read.
    """
    return {k: v for k, v in os.environ.items()
            if k.upper() in _SANDBOX_ENV_NAMES or k.upper().startswith("LC_")}


# --- URL policy for custom endpoints ---------------------------------------
_ALLOWED_HOSTNAMES = {"localhost", "host.docker.internal"}


class UrlPolicyError(ValueError):
    """Raised when an endpoint URL violates the network policy."""


_IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
# Hosts such as "0xA9FEA9FE", "2852039166" or "169.254.43518": not valid dotted IPv4 for
# ``ipaddress``, but the operating system's resolver still reads them as IPv4 addresses.
_NUMERIC_HOST_RE = re.compile(r"(?:0[xX][0-9a-fA-F]*|[0-9]+)(?:\.(?:0[xX][0-9a-fA-F]*|[0-9]+))*\.?")


def _parse_ip(host: str) -> _IpAddress | None:
    """IP address of a literal host, with IPv4-mapped IPv6 (``::ffff:a.b.c.d``) unwrapped.

    Older Python patch releases report ``::ffff:169.254.169.254`` as private but not as
    link-local, which would let the cloud metadata address through.
    """
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def _is_forbidden_ip(ip: _IpAddress) -> bool:
    return ip.is_link_local or ip.is_multicast or ip.is_unspecified


def is_private_host(host: str) -> bool:
    """True for loopback, RFC1918/ULA private ranges and known local hostnames."""
    host = host.strip("[]").lower()
    if host in _ALLOWED_HOSTNAMES:
        return True
    ip = _parse_ip(host)
    if ip is None:
        return False
    return ip.is_loopback or (ip.is_private and not _is_forbidden_ip(ip))


_DNS_CACHE_SECONDS = 60  # the UI re-validates saved endpoints on every rerun


@functools.lru_cache(maxsize=256)
def _resolve(host: str, _time_bucket: int) -> tuple[str, ...]:
    """Addresses a host name resolves to (empty if it does not resolve at all)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return ()  # unreachable anyway; the request itself will report the error
    return tuple(sorted({str(info[4][0]).split("%", 1)[0] for info in infos}))


def _check_resolved_addresses(host: str) -> None:
    """Reject a public host name that resolves to a link-local (metadata) address.

    Best effort: the name is resolved again when the request is sent, so a hostile DNS
    server can still answer differently then (DNS rebinding).
    """
    for address in _resolve(host.lower(), int(time.monotonic() // _DNS_CACHE_SECONDS)):
        ip = _parse_ip(address)
        if ip is not None and _is_forbidden_ip(ip):
            raise UrlPolicyError(f"Host '{host}' resolves to {ip}, which is not allowed "
                                 "(link-local/metadata range)")


def check_endpoint_url(url: str, allow_public: bool = False) -> str:
    """Validate an OpenAI-compatible base URL and return it normalised.

    Public hosts are rejected unless ``allow_public`` is True, because prompts would
    leave the machine and an attacker-controlled URL could be used for SSRF probing.
    Link-local addresses (e.g. cloud metadata 169.254.169.254) are always rejected.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UrlPolicyError("URL must start with http:// or https://")
    if not parsed.hostname:
        raise UrlPolicyError("URL has no host name")
    if parsed.username or parsed.password:
        raise UrlPolicyError("Do not put credentials in the URL; use api_key_env instead")
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url.strip()):
        raise UrlPolicyError("URL must not contain spaces or control characters")
    host = parsed.hostname
    ip = _parse_ip(host)
    if ip is not None and _is_forbidden_ip(ip):
        raise UrlPolicyError(f"Address {host} is not allowed (link-local/metadata range)")
    if ip is None and _NUMERIC_HOST_RE.fullmatch(host):
        raise UrlPolicyError(f"Host '{host}' is not a plain IP address; write it as four "
                             "decimal numbers, e.g. 192.168.1.20")
    if not allow_public and not is_private_host(host):
        raise UrlPolicyError(
            f"Host '{host}' is not local/private. Enable 'Allow public endpoint' if you "
            "really want to send prompts to this server."
        )
    if ip is None and host.lower() not in _ALLOWED_HOSTNAMES:
        _check_resolved_addresses(host)  # only reachable with allow_public
    return url.strip().rstrip("/")


# --- Identifier validation -------------------------------------------------
OLLAMA_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]{0,63}(/[a-z0-9][a-z0-9._\-]{0,63})?"
                           r"(:[A-Za-z0-9][A-Za-z0-9._\-]{0,63})?$")
HF_GGUF_RE = re.compile(
    r"^(hf\.co|huggingface\.co)/[A-Za-z0-9][A-Za-z0-9._\-]{0,95}/[A-Za-z0-9][A-Za-z0-9._\-]{0,95}"
    r"(:[A-Za-z0-9][A-Za-z0-9._\-]{0,127})?$"
)
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$")
CUSTOM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,47}$")


def is_valid_ollama_tag(tag: str) -> bool:
    """Ollama library tag (``name[:tag]``, optional namespace) or Hugging Face GGUF path."""
    return bool(OLLAMA_TAG_RE.fullmatch(tag) or HF_GGUF_RE.fullmatch(tag))


def is_valid_model_name(name: str) -> bool:
    return bool(MODEL_NAME_RE.fullmatch(name)) and ".." not in name


# --- Spreadsheet formula injection -----------------------------------------
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def escape_spreadsheet_cell(value: object) -> object:
    """Neutralise strings that a spreadsheet would interpret as a formula.

    Leading spaces and line breaks are skipped for the check, because some spreadsheet
    programs trim them on import and would then still see the formula.
    """
    if isinstance(value, str) and value.lstrip(" \n").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value
