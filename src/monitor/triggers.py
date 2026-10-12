"""Decide whether to retrain (drift score or negative-feedback ratio above thresholds)."""
from __future__ import annotations

from typing import Any


def should_retrain(drift_summary: dict[str, Any] | None, feedback: dict[str, int],
                   thresholds: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    drift_score = (drift_summary or {}).get("drift_score", 0.0)
    drift_ok = (drift_summary or {}).get("status") == "ok"
    if drift_ok and drift_score >= thresholds["drift_score_threshold"]:
        reasons.append(f"drift score {drift_score:.3f} >= {thresholds['drift_score_threshold']}")

    total = feedback.get("up", 0) + feedback.get("down", 0)
    ratio = feedback.get("down", 0) / total if total else 0.0
    if total >= thresholds["min_feedback"] and ratio >= thresholds["negative_feedback_ratio"]:
        reasons.append(f"negative feedback ratio {ratio:.2f} >= {thresholds['negative_feedback_ratio']} "
                       f"({feedback.get('down', 0)}/{total})")
    return {"trigger": bool(reasons), "reasons": reasons, "drift_score": drift_score,
            "feedback_total": total, "negative_ratio": ratio}
