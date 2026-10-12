"""Prometheus metrics for the API (one private registry per app instance, so tests can create many apps)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 120)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("llm_requests_total", "Requests by endpoint and outcome",
                                ["endpoint", "status"], registry=r)
        self.errors = Counter("llm_errors_total", "Failed requests by endpoint", ["endpoint"], registry=r)
        self.latency = Histogram("llm_request_latency_seconds", "End-to-end /generate latency",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.tokens_per_second = Histogram("llm_tokens_per_second", "Generation speed per request",
                                           buckets=(0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128), registry=r)
        self.tokens_generated = Counter("llm_tokens_generated_total", "Total generated tokens", registry=r)
        self.input_length = Histogram("llm_input_length_chars", "Prompt length in characters",
                                      buckets=(20, 50, 100, 200, 400, 800, 1600, 3200), registry=r)
        self.inflight = Gauge("llm_inflight_requests", "Requests currently being processed", registry=r)
        self.feedback = Counter("llm_feedback_total", "User feedback by rating", ["rating"], registry=r)
        self.drift_score = Gauge("llm_drift_score", "Latest data drift score (0-1) from reports/drift.json",
                                 registry=r)
        self.model_info = Gauge("llm_model_info", "Loaded model and prompt version",
                                ["model_version", "prompt_version"], registry=r)
        for rating in ("up", "down"):               # make the series exist at 0
            self.feedback.labels(rating=rating)
        self.errors.labels(endpoint="/generate")
        self._drift_mtime = 0.0

    def refresh_drift(self, report_path: str | os.PathLike) -> None:
        """Update llm_drift_score from the JSON written by `make drift` (re-read only when it changes)."""
        path = Path(report_path)
        try:
            mtime = path.stat().st_mtime
            if mtime == self._drift_mtime:
                return
            self.drift_score.set(float(json.loads(path.read_text(encoding="utf-8")).get("drift_score", 0.0)))
            self._drift_mtime = mtime
        except (OSError, ValueError):
            return

    def render(self) -> bytes:
        return generate_latest(self.registry)
