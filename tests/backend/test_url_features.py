import pytest
from app.services.url_features import (
    InvalidUrlError,
    build_evidence,
    extract_features,
    normalize_url,
)


def test_ip_host_detected():
    assert extract_features("http://192.168.1.1/login")["host_is_ip"] == 1


def test_subdomain_count():
    assert extract_features("http://a.b.c.example.com/")["num_subdomains"] == 3


def test_at_symbol():
    assert extract_features("http://good.com@evil.example.com/")["has_at_symbol"] == 1


def test_https_does_not_change_features():
    assert extract_features("http://example.com/x") == extract_features("https://example.com/x")


def test_missing_scheme_gets_default():
    assert normalize_url("example.com").startswith("http://")


def test_host_only_ignores_path():
    f = extract_features("http://example.com/very/long/path?a=1", host_only=True)
    assert f["path_length"] == 0 and f["query_length"] == 0


@pytest.mark.parametrize("bad", ["", "   ", "ftp://example.com", "http://",
                                 "http://exa mple.com", "http://example.com:abc/"])
def test_invalid_urls_rejected(bad):
    with pytest.raises(InvalidUrlError):
        normalize_url(bad)


def test_evidence_for_ip_host():
    ids = [e.id for e in build_evidence("http://10.0.0.1/a")]
    assert "url.host_is_ip" in ids