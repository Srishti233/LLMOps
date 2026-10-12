"""API tests with a fake backend (no model, no torch). Skipped if fastapi/httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pydantic = pytest.importorskip("pydantic")

from fastapi.testclient import TestClient  # noqa: E402

from src.model_io import GenResult  # noqa: E402
from src.serve.app import create_app  # noqa: E402
from src.serve.schemas import FeedbackRequest, GenerateRequest  # noqa: E402
from src.utils.config import load_config  # noqa: E402


class FakeBackend:
    def __init__(self, ready=True, fail=False):
        self.ready, self.fail, self.error = ready, fail, None
        self.model_version, self.prompt_version = "fake:v1", "v1"

    def generate(self, prompt, max_new_tokens, temperature):
        if self.fail:
            raise RuntimeError("boom")
        return GenResult(text=f"echo: {prompt}", tokens=7, gen_seconds=0.5, input_tokens=3)


def make_client(tmp_path, **kw):
    cfg = load_config()
    cfg["data"]["monitoring_db"] = str(tmp_path / "m.db")
    cfg["monitoring"]["drift"]["report_json"] = str(tmp_path / "drift.json")
    return TestClient(create_app(FakeBackend(**kw), cfg))


def test_schema_validation():
    assert GenerateRequest(prompt="hi").max_new_tokens == 128
    for bad in ({"prompt": ""}, {"prompt": "x", "max_new_tokens": 0}, {"prompt": "x", "temperature": 5}):
        with pytest.raises(pydantic.ValidationError):
            GenerateRequest(**bad)
    with pytest.raises(pydantic.ValidationError):
        FeedbackRequest(request_id="a", rating="meh")


def test_health_ready_and_index(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["model_version"] == "fake:v1"
    assert "X-Instance" in ready.headers
    assert "AuroraSync" in client.get("/").text


def test_not_ready_returns_503(tmp_path):
    client = make_client(tmp_path, ready=False)
    assert client.get("/ready").status_code == 503
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 503


def test_generate_contract_logging_and_feedback(tmp_path):
    client = make_client(tmp_path)
    r = client.post("/generate", json={"prompt": "mail bob@example.com", "max_new_tokens": 16, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"response", "model_version", "prompt_version", "latency_ms", "tokens_generated", "request_id"}
    assert body["tokens_generated"] == 7 and body["prompt_version"] == "v1"
    fb = client.post("/feedback", json={"request_id": body["request_id"], "rating": "up", "comment": "great"})
    assert fb.status_code == 200
    assert client.post("/feedback", json={"request_id": "unknown", "rating": "up"}).status_code == 404
    assert client.post("/feedback", json={"request_id": body["request_id"], "rating": "bad"}).status_code == 422
    metrics = client.get("/metrics").text
    assert 'llm_requests_total{endpoint="/generate",status="ok"} 1.0' in metrics
    assert 'llm_feedback_total{rating="up"} 1.0' in metrics
    assert "llm_tokens_generated_total 7.0" in metrics and "llm_drift_score" in metrics
    from src.monitor import store
    prompts = store.recent_prompts(tmp_path / "m.db")
    assert prompts == ["mail [EMAIL]"]


def test_generation_error_is_counted(tmp_path):
    client = make_client(tmp_path, fail=True)
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 500
    metrics = client.get("/metrics").text
    assert 'llm_errors_total{endpoint="/generate"} 1.0' in metrics


def test_drift_gauge_reads_report(tmp_path):
    (tmp_path / "drift.json").write_text('{"drift_score": 0.42}')
    client = make_client(tmp_path)
    assert "llm_drift_score 0.42" in client.get("/metrics").text
