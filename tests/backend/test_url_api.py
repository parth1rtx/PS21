from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_missing_model_means_unavailable_not_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("PHISHGUARD_URL_MODEL_DIR", str(tmp_path))
    r = client.post("/api/v1/analyze/url", json={"url": "http://192.168.0.1/login"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["label"] == "unknown"
    assert body["probability"] is None
    assert any(e["id"] == "url.host_is_ip" for e in body["evidence"])


def test_invalid_scheme_returns_422():
    assert client.post("/api/v1/analyze/url", json={"url": "ftp://example.com"}).status_code == 422


def test_blank_url_returns_422():
    assert client.post("/api/v1/analyze/url", json={"url": "   "}).status_code == 422