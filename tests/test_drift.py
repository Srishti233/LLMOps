import numpy as np

from src.data.dataset import generate_examples
from src.monitor.drift import (compute_drift, js_distance, ks_length_test, psi, render_html, token_histogram,
                               build_vocab)
from src.utils.config import load_config

SETTINGS = load_config()["monitoring"]["drift"]
TRAIN = [e["instruction"] for e in generate_examples(300, 42)][:255]
OFF_TOPIC = ["Explain how to photograph the Orion nebula with a small telescope and a camera tripod. "] * 5


def test_psi_and_js_are_zero_for_identical_and_positive_for_different():
    a = np.array([10.0, 20.0, 30.0])
    assert psi(a, a) < 1e-9 and js_distance(a, a) < 1e-9
    b = np.array([30.0, 20.0, 10.0])
    assert psi(a, b) > 0.1 and 0 < js_distance(a, b) <= 1


def test_ks_detects_length_shift():
    stat, p = ks_length_test(TRAIN, ["x" * 500] * 30)
    assert stat > 0.9 and p < 0.001


def test_vocab_histogram_has_oov_bucket():
    vocab = build_vocab(TRAIN, 20)
    hist = token_histogram(["zzzqqq unknownword"], vocab)
    assert len(hist) == 21 and hist[-1] == 2


def test_no_false_alarm_on_same_distribution_for_many_seeds_and_window_sizes():
    for n in (20, 40, 100):
        for seed in range(15):
            cur = [e["instruction"] for e in generate_examples(n, 7000 + seed)]
            summary = compute_drift(TRAIN, cur, SETTINGS)
            assert summary["status"] == "ok"
            assert not summary["drift_detected"], (n, seed, summary["drift_score"])
            assert summary["drift_score"] < SETTINGS["score_threshold"]


def test_detects_shifted_traffic():
    summary = compute_drift(TRAIN, OFF_TOPIC * 8, SETTINGS)
    assert summary["drift_detected"] and summary["drift_score"] > 0.6
    assert summary["oov_rate"] > 0.3 and summary["top_shifted_tokens"]


def test_detects_partial_contamination():
    cur = [e["instruction"] for e in generate_examples(40, 9000)]
    cur[:12] = (OFF_TOPIC * 3)[:12]
    assert compute_drift(TRAIN, cur, SETTINGS)["drift_detected"]


def test_insufficient_data_is_reported_not_crashed():
    summary = compute_drift(TRAIN, TRAIN[:3], SETTINGS)
    assert summary["status"] == "insufficient_data" and summary["drift_score"] == 0.0


def test_html_report_renders_for_both_states():
    ok = compute_drift(TRAIN, OFF_TOPIC * 8, SETTINGS)
    page = render_html(ok)
    assert "DRIFT DETECTED" in page and "<table>" in page
    assert "INSUFFICIENT DATA" in render_html(compute_drift(TRAIN, TRAIN[:2], SETTINGS))
