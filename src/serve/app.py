"""FastAPI service.

    uvicorn src.serve.app:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from src.monitor import store
from src.serve.backend import LLMBackend
from src.serve.metrics import Metrics
from src.serve.schemas import FeedbackRequest, FeedbackResponse, GenerateRequest, GenerateResponse
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger, log_fields, setup_logging

log = get_logger("api")
STATIC_DIR = Path(__file__).parent / "static"
INSTANCE = socket.gethostname()


def create_app(backend: Any | None = None, cfg: dict[str, Any] | None = None) -> FastAPI:
    """App factory. Pass a fake `backend` in tests; in production the real LLMBackend loads in the background."""
    setup_logging()
    cfg = cfg or load_config()
    backend = backend or LLMBackend(cfg)
    metrics = Metrics()
    db_path = resolve_path(cfg["data"]["monitoring_db"])
    drift_report = resolve_path(cfg["monitoring"]["drift"]["report_json"])
    store.init_db(db_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if hasattr(backend, "start"):
            backend.start()
        yield

    app = FastAPI(title="llmops-lab support assistant", version="1.0.0", lifespan=lifespan)
    app.state.metrics, app.state.backend = metrics, backend

    @app.middleware("http")
    async def add_instance_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Instance"] = INSTANCE
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "instance": INSTANCE, "model_ready": bool(backend.ready)}

    @app.get("/ready")
    def ready():
        body = {"ready": bool(backend.ready), "instance": INSTANCE, "model_version": backend.model_version,
                "prompt_version": backend.prompt_version, "error": getattr(backend, "error", None)}
        return JSONResponse(body, status_code=200 if backend.ready else 503)

    @app.get("/metrics")
    def prometheus_metrics():
        metrics.refresh_drift(drift_report)
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        if not backend.ready:
            metrics.requests.labels(endpoint="/generate", status="unavailable").inc()
            raise HTTPException(status_code=503, detail=getattr(backend, "error", None) or "model is still loading")
        request_id = uuid.uuid4().hex
        metrics.inflight.inc()
        metrics.input_length.observe(len(req.prompt))
        started = time.perf_counter()
        try:
            result = backend.generate(req.prompt, req.max_new_tokens, req.temperature)
        except Exception:  # noqa: BLE001
            log.exception("generation failed", extra={"fields": {"request_id": request_id}})
            metrics.errors.labels(endpoint="/generate").inc()
            metrics.requests.labels(endpoint="/generate", status="error").inc()
            raise HTTPException(status_code=500, detail="generation failed") from None
        finally:
            metrics.inflight.dec()
        latency_s = time.perf_counter() - started

        metrics.latency.observe(latency_s)
        metrics.tokens_generated.inc(result.tokens)
        if result.gen_seconds > 0 and result.tokens > 0:
            metrics.tokens_per_second.observe(result.tokens / result.gen_seconds)
        metrics.requests.labels(endpoint="/generate", status="ok").inc()
        metrics.model_info.labels(model_version=backend.model_version, prompt_version=backend.prompt_version).set(1)

        try:  # monitoring must never break serving
            store.log_request(db_path, request_id=request_id, prompt=req.prompt, response=result.text,
                              latency_ms=latency_s * 1000, tokens_generated=result.tokens,
                              model_version=backend.model_version, prompt_version=backend.prompt_version)
        except Exception:  # noqa: BLE001
            log.exception("could not store request log", extra={"fields": {"request_id": request_id}})
        log_fields(log, 20, "generate", request_id=request_id, latency_ms=round(latency_s * 1000, 1),
                   tokens=result.tokens, input_chars=len(req.prompt), model_version=backend.model_version)
        return GenerateResponse(response=result.text, model_version=backend.model_version,
                                prompt_version=backend.prompt_version, latency_ms=round(latency_s * 1000, 2),
                                tokens_generated=result.tokens, request_id=request_id)

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        if not store.request_exists(db_path, req.request_id):
            metrics.requests.labels(endpoint="/feedback", status="not_found").inc()
            raise HTTPException(status_code=404, detail="unknown request_id")
        store.log_feedback(db_path, request_id=req.request_id, rating=req.rating, comment=req.comment)
        metrics.feedback.labels(rating=req.rating).inc()
        metrics.requests.labels(endpoint="/feedback", status="ok").inc()
        log_fields(log, 20, "feedback", request_id=req.request_id, rating=req.rating)
        return FeedbackResponse(request_id=req.request_id)

    return app
