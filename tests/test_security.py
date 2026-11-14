"""Tests for secret masking, URL policy, identifier validation and export escaping."""

import pytest

from src.security import (
    MASK,
    UrlPolicyError,
    check_endpoint_url,
    escape_spreadsheet_cell,
    is_valid_model_name,
    is_valid_ollama_tag,
    mask_secrets,
    scrubbed_environ,
)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "sk-ant-api03-abcdefghijklmnop",
        "AIzaSyA1234567890abcdefghijklmnop",
        "Bearer abcdefghijklmnop.qrstuv",
    ],
)
def test_mask_known_patterns(secret):
    out = mask_secrets(f"AuthenticationError: invalid key {secret} provided")
    assert secret not in out and MASK in out


def test_mask_literal_session_secret():
    out = mask_secrets("error with key myweirdkey!!", extra_secrets=["myweirdkey!!"])
    assert "myweirdkey" not in out


def test_mask_keeps_normal_text():
    assert mask_secrets("Rate limit exceeded, retry in 3s") == "Rate limit exceeded, retry in 3s"


def test_scrubbed_environ(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("HARMLESS_VAR", "ok")
    env = scrubbed_environ()
    assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HARMLESS_VAR"] == "ok"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:1234/v1/",
        "http://host.docker.internal:1234/v1",
        "http://192.168.1.20:8000/v1",
        "http://[::1]:11434",
    ],
)
def test_private_urls_allowed(url):
    assert check_endpoint_url(url).startswith("http")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
        "http://user:pw@127.0.0.1:8080",
        "ftp://127.0.0.1",
    ],
)
def test_bad_urls_rejected(url):
    with pytest.raises(UrlPolicyError):
        check_endpoint_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:169.254.169.254]/v1",  # IPv4-mapped IPv6 metadata address
        "http://0xA9FEA9FE/v1",  # 169.254.169.254 written as one hex number
        "http://2852039166/v1",  # ... as one decimal number
        "http://169.254.43518/v1",  # ... in the short three-part form
        "http://127.0.0.1:8080/v1\r\nx",  # control characters
        "http://127.0.0.1:8080/a b",
    ],
)
@pytest.mark.parametrize("allow_public", [False, True])
def test_disguised_addresses_rejected(url, allow_public):
    with pytest.raises(UrlPolicyError):
        check_endpoint_url(url, allow_public=allow_public)


def test_public_host_resolving_to_metadata_rejected(monkeypatch):
    import socket

    from src import security

    security._resolve.cache_clear()
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    with pytest.raises(UrlPolicyError, match="resolves to"):
        check_endpoint_url("https://metadata.example.test/v1", allow_public=True)
    security._resolve.cache_clear()
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert check_endpoint_url("https://api.example.test/v1", allow_public=True)
    security._resolve.cache_clear()


def test_mapped_loopback_counts_as_private():
    from src.security import is_private_host

    assert is_private_host("::ffff:127.0.0.1")
    assert not is_private_host("::ffff:169.254.169.254")


def test_public_url_with_opt_in():
    assert check_endpoint_url("https://api.example.com/v1", allow_public=True)
    with pytest.raises(UrlPolicyError):
        check_endpoint_url("http://169.254.169.254", allow_public=True)


@pytest.mark.parametrize(
    "tag,ok",
    [
        ("gemma3:270m", True),
        ("llama3.2:3b-instruct-q4_K_M", True),
        ("myuser/mymodel:latest", True),
        ("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", True),
        ("huggingface.co/user/repo", True),
        ("model; rm -rf /", False),
        ("$(whoami)", False),
        ("../../etc/passwd", False),
        ("", False),
    ],
)
def test_ollama_tag_validation(tag, ok):
    assert is_valid_ollama_tag(tag) is ok


def test_model_name_validation():
    assert is_valid_model_name("my-finetune-q4")
    assert not is_valid_model_name("a/../b")
    assert not is_valid_model_name("bad name")


@pytest.mark.parametrize("cell", ["=HYPERLINK(\"x\")", "+1+1", "-2", "@SUM(A1)", "\tx"])
def test_formula_escape(cell):
    assert escape_spreadsheet_cell(cell) == "'" + cell


@pytest.mark.parametrize("cell", [" =1+1", "\n=cmd|' /C calc'!A0", "  @SUM(A1)"])
def test_formula_escape_ignores_leading_blanks(cell):
    assert escape_spreadsheet_cell(cell) == "'" + cell


def test_formula_escape_leaves_numbers_and_text():
    assert escape_spreadsheet_cell(-2.5) == -2.5
    assert escape_spreadsheet_cell("hello") == "hello"
