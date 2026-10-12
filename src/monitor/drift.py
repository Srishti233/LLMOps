"""Data-drift detection: reference window (training prompts) vs recent window (logged production prompts).

Signals
  * prompt length (characters): two-sample Kolmogorov-Smirnov test
  * token distribution: Population Stability Index (PSI, small-sample bias corrected) and Jensen-Shannon
    distance over a vocabulary built from the reference (everything else falls into an OOV bucket)
  * optional: Evidently's DataDriftPreset if it is installed (extra HTML report, never required)

Outputs reports/drift.json + reports/drift.html and feeds the `llm_drift_score` Prometheus gauge.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from src.utils.logging_utils import get_logger

log = get_logger("drift")
TOKEN_RE = re.compile(r"[a-z0-9']+")
EPS = 1e-4


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def prompt_lengths(prompts: list[str]) -> np.ndarray:
    return np.array([len(p) for p in prompts], dtype=float)


def ks_length_test(ref_prompts: list[str], cur_prompts: list[str]) -> tuple[float, float]:
    result = ks_2samp(prompt_lengths(ref_prompts), prompt_lengths(cur_prompts))
    return float(result.statistic), float(result.pvalue)


def build_vocab(ref_prompts: list[str], size: int = 300) -> list[str]:
    counts = Counter(tok for p in ref_prompts for tok in tokenize(p))
    return [tok for tok, _ in counts.most_common(size)]


def token_histogram(prompts: list[str], vocab: list[str]) -> np.ndarray:
    """Counts per vocab token; the final bucket is out-of-vocabulary."""
    index = {tok: i for i, tok in enumerate(vocab)}
    hist = np.zeros(len(vocab) + 1, dtype=float)
    for prompt in prompts:
        for tok in tokenize(prompt):
            hist[index.get(tok, len(vocab))] += 1
    return hist


def _to_probs(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    probs = counts / total if total > 0 else np.full_like(counts, 1.0 / len(counts))
    probs = np.clip(probs, EPS, None)
    return probs / probs.sum()


def psi(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    p, q = _to_probs(ref_counts), _to_probs(cur_counts)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_bias_correction(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Expected PSI between two samples of the *same* distribution: (bins-1) * (1/N_ref + 1/N_cur).

    Raw PSI is positively biased on small samples (a 20-prompt window would otherwise always look drifted).
    """
    n_ref, n_cur = max(ref_counts.sum(), 1.0), max(cur_counts.sum(), 1.0)
    return float((len(ref_counts) - 1) * (1.0 / n_ref + 1.0 / n_cur))


def js_distance(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Jensen-Shannon distance with base 2, bounded in [0, 1]."""
    return float(jensenshannon(_to_probs(ref_counts), _to_probs(cur_counts), base=2))


def _length_stats(prompts: list[str]) -> dict[str, float]:
    arr = prompt_lengths(prompts)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}


def compute_drift(ref_prompts: list[str], cur_prompts: list[str], settings: dict[str, Any]) -> dict[str, Any]:
    """Compare two prompt windows and return a JSON-serialisable summary including `drift_score` in [0, 1]."""
    min_samples = int(settings.get("min_samples", 20))
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference": len(ref_prompts),
        "n_current": len(cur_prompts),
        "thresholds": {k: settings[k] for k in
                       ("psi_threshold", "ks_pvalue_threshold", "js_threshold", "score_threshold")},
    }
    if len(ref_prompts) < min_samples or len(cur_prompts) < min_samples:
        summary.update(status="insufficient_data", drift_score=0.0, drift_detected=False,
                       message=f"need at least {min_samples} prompts in each window")
        return summary

    vocab = build_vocab(ref_prompts, int(settings.get("vocab_size", 300)))
    ref_hist, cur_hist = token_histogram(ref_prompts, vocab), token_histogram(cur_prompts, vocab)
    ks_stat, ks_p = ks_length_test(ref_prompts, cur_prompts)
    psi_raw, js_value = psi(ref_hist, cur_hist), js_distance(ref_hist, cur_hist)
    psi_value = max(0.0, psi_raw - psi_bias_correction(ref_hist, cur_hist))
    oov_rate = float(cur_hist[-1] / max(cur_hist.sum(), 1.0))

    score = 0.30 * ks_stat + 0.35 * min(psi_value / 1.0, 1.0) + 0.35 * js_value
    th = summary["thresholds"]
    flags = {
        "length_drift": ks_p < th["ks_pvalue_threshold"] and ks_stat > 0.2,
        "psi_drift": psi_value > th["psi_threshold"],
        "js_drift": js_value > th["js_threshold"],
    }
    top_shifted = _top_shifted_tokens(ref_hist, cur_hist, vocab)
    summary.update(
        status="ok",
        ks_statistic=ks_stat, ks_pvalue=ks_p, psi=psi_value, psi_raw=psi_raw, js_distance=js_value, oov_rate=oov_rate,
        drift_score=float(score), drift_detected=bool(score > th["score_threshold"] or any(flags.values())),
        flags=flags,
        length_stats={"reference": _length_stats(ref_prompts), "current": _length_stats(cur_prompts)},
        top_shifted_tokens=top_shifted,
    )
    return summary


def _top_shifted_tokens(ref_hist: np.ndarray, cur_hist: np.ndarray, vocab: list[str], k: int = 15):
    p, q = _to_probs(ref_hist), _to_probs(cur_hist)
    labels = vocab + ["<out-of-vocabulary>"]
    order = np.argsort(-np.abs(q - p))[:k]
    return [{"token": labels[i], "reference": float(p[i]), "current": float(q[i])} for i in order]


# ---------------------------------------------------------------------------------------------- reports
def render_html(summary: dict[str, Any]) -> str:
    esc = html.escape
    status_color = "#c0392b" if summary.get("drift_detected") else "#27ae60"
    status_text = "DRIFT DETECTED" if summary.get("drift_detected") else "No significant drift"
    if summary.get("status") != "ok":
        status_text, status_color = "INSUFFICIENT DATA", "#7f8c8d"

    def fmt(key: str, digits: int = 4) -> str:
        value = summary.get(key)
        return "n/a" if value is None else f"{value:.{digits}f}"

    rows = "".join(
        f"<tr><td>{esc(str(name))}</td><td>{esc(str(value))}</td></tr>"
        for name, value in [
            ("Reference prompts", summary["n_reference"]), ("Current prompts", summary["n_current"]),
            ("Drift score (0-1)", fmt("drift_score")), ("KS statistic (length)", fmt("ks_statistic")),
            ("KS p-value", fmt("ks_pvalue", 6)), ("PSI (tokens, bias-corrected)", fmt("psi")), ("PSI (raw)", fmt("psi_raw")),
            ("Jensen-Shannon distance", fmt("js_distance")), ("Out-of-vocabulary rate", fmt("oov_rate")),
        ]
    )
    stats = summary.get("length_stats")
    length_html = ""
    if stats:
        length_html = "<h2>Prompt length (characters)</h2><table><tr><th></th><th>mean</th><th>median</th><th>p95</th></tr>" + "".join(
            f"<tr><td>{name}</td><td>{s['mean']:.1f}</td><td>{s['median']:.1f}</td><td>{s['p95']:.1f}</td></tr>"
            for name, s in stats.items()
        ) + "</table>"
    tokens_html = ""
    if summary.get("top_shifted_tokens"):
        body = "".join(
            f"<tr><td>{esc(t['token'])}</td><td>{t['reference']:.4f}</td><td>{t['current']:.4f}</td>"
            f"<td><div class='bar' style='width:{min(abs(t['current'] - t['reference']) * 400, 100):.0f}%'></div></td></tr>"
            for t in summary["top_shifted_tokens"]
        )
        tokens_html = ("<h2>Most shifted tokens</h2><table><tr><th>token</th><th>reference share</th>"
                       f"<th>current share</th><th>shift</th></tr>{body}</table>")
    thresholds = ", ".join(f"{esc(str(k))}={esc(str(v))}" for k, v in summary["thresholds"].items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>llmops-lab drift report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%;margin-bottom:1.5rem}}td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f4f4f4}}.badge{{display:inline-block;padding:6px 14px;border-radius:6px;color:#fff;font-weight:600;background:{status_color}}}
.bar{{height:10px;background:#3498db}}small{{color:#666}}</style></head><body>
<h1>Data drift report</h1><p><span class="badge">{status_text}</span></p>
<small>Generated {esc(summary['generated_at'])}. Thresholds: {thresholds}</small>
<h2>Summary</h2><table>{rows}</table>{length_html}{tokens_html}</body></html>"""


def try_evidently(ref_prompts: list[str], cur_prompts: list[str], out_html: Path) -> str | None:
    """Optional Evidently report on simple text features. Returns the path or None if unavailable."""
    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception:  # noqa: BLE001 - Evidently is optional and its API changes between versions
        return None

    def features(prompts: list[str]) -> "pd.DataFrame":
        rows = []
        for p in prompts:
            words = p.split()
            rows.append({
                "n_chars": len(p), "n_words": len(words),
                "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
                "digit_ratio": sum(c.isdigit() for c in p) / max(len(p), 1),
                "question_marks": p.count("?"),
            })
        return pd.DataFrame(rows)

    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=features(ref_prompts), current_data=features(cur_prompts))
        out_html.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_html))
        return str(out_html)
    except Exception as exc:  # noqa: BLE001
        log.warning("evidently report failed: %s", exc)
        return None


# ------------------------------------------------------------------------------------------------- runner
def run_drift(cfg: dict[str, Any], write: bool = True) -> dict[str, Any]:
    from src.data.dataset import read_jsonl
    from src.monitor.store import init_db, recent_prompts
    from src.utils.config import resolve_path

    settings = cfg["monitoring"]["drift"]
    reference = [r["instruction"] for r in read_jsonl(resolve_path(cfg["data"]["train_file"]))]
    db_path = resolve_path(cfg["data"]["monitoring_db"])
    init_db(db_path)
    current = recent_prompts(db_path, int(settings["window"]))
    summary = compute_drift(reference, current, settings)

    if write:
        json_path, html_path = resolve_path(settings["report_json"]), resolve_path(settings["report_html"])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        html_path.write_text(render_html(summary), encoding="utf-8")
        if summary["status"] == "ok":
            evidently_path = try_evidently(reference, current, resolve_path(settings["evidently_html"]))
            summary["evidently_report"] = evidently_path
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logging

    setup_logging()
    summary = run_drift(load_config())
    print(json.dumps({k: v for k, v in summary.items() if k != "top_shifted_tokens"}, indent=2))


if __name__ == "__main__":
    main()
