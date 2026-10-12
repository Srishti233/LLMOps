"""Quality gate: pure decision logic (no I/O) so it is easy to unit test."""
from __future__ import annotations

from typing import Any

HIGHER_IS_BETTER = ("rouge_l", "keyword_match_rate")


def evaluate_gate(metrics: dict[str, float], thresholds: dict[str, float],
                  production_metrics: dict[str, float] | None, max_regression: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, value: float, op: str, limit: float) -> None:
        ok = value >= limit if op == ">=" else value <= limit
        checks.append({"check": name, "value": round(value, 4), "op": op, "limit": limit, "passed": bool(ok)})

    check("rouge_l", metrics["rouge_l"], ">=", thresholds["min_rouge_l"])
    check("keyword_match_rate", metrics["keyword_match_rate"], ">=", thresholds["min_keyword_match_rate"])
    check("avg_latency_s", metrics["avg_latency_s"], "<=", thresholds["max_avg_latency_s"])
    check("tokens_per_sec", metrics["tokens_per_sec"], ">=", thresholds["min_tokens_per_sec"])

    if production_metrics:
        for name in HIGHER_IS_BETTER:
            if name in production_metrics:
                check(f"no_regression_vs_production:{name}", metrics[name], ">=",
                      production_metrics[name] - max_regression)

    failed = [c for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "checks": checks,
        "reasons": [f"{c['check']}: {c['value']} not {c['op']} {c['limit']}" for c in failed],
        "compared_with_production": bool(production_metrics),
    }
