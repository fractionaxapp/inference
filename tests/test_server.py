import pytest
from fastapi.testclient import TestClient

from fractionax_inference.config import get_settings
from fractionax_inference.server import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_complete_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        resp = client.post("/v1/complete", json={"prompt": "hi"})
        assert resp.status_code == 503
    finally:
        get_settings.cache_clear()
