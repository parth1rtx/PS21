from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_missing_model_means_unavailable_not_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("PHISHGUARD_EMAIL_MODEL_DIR", str(tmp_path))
    r = client.post("/api/v1/analyze/email",
                    json={"subject": "Urgent", "body": "Please verify your account within 24 hours"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["label"] == "unknown"
    assert body["probability"] is None
    assert any(e["id"] == "email.urgency" for e in body["evidence"])


def test_empty_email_rejected():
    assert client.post("/api/v1/analyze/email", json={"subject": "", "body": ""}).status_code == 422


def test_oversized_body_rejected():
    r = client.post("/api/v1/analyze/email", json={"subject": "x", "body": "a" * 200_001})
    assert r.status_code == 422