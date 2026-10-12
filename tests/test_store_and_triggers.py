import sqlite3

from src.monitor import store
from src.monitor.triggers import should_retrain


def _log(db, rid, prompt="hi", response="ok"):
    store.log_request(db, request_id=rid, prompt=prompt, response=response, latency_ms=12.5, tokens_generated=3,
                      model_version="m:v1", prompt_version="v1")


def test_request_logging_masks_pii(tmp_path):
    db = tmp_path / "m.db"
    store.init_db(db)
    _log(db, "r1", prompt="mail me at bob@example.com or +44 20 7946 0958", response="call 415 555 2671 now")
    row = sqlite3.connect(db).execute("select prompt, response from requests").fetchone()
    assert "bob@" not in row[0] and "[EMAIL]" in row[0] and "[PHONE]" in row[0]
    assert "[PHONE]" in row[1]


def test_feedback_roundtrip_and_upsert(tmp_path):
    db = tmp_path / "m.db"
    store.init_db(db)
    _log(db, "r1")
    _log(db, "r2")
    assert store.request_exists(db, "r1") and not store.request_exists(db, "nope")
    store.log_feedback(db, request_id="r1", rating="up")
    store.log_feedback(db, request_id="r2", rating="down", comment="wrong, mail a@b.com")
    store.log_feedback(db, request_id="r2", rating="up")           # changes the earlier rating
    assert store.feedback_counts(db) == {"up": 2, "down": 0}
    rows = store.feedback_rows(db)
    assert len(rows) == 2 and rows[1]["comment"] is None or "a@b.com" not in (rows[1]["comment"] or "")


def test_recent_prompts_orders_newest_first(tmp_path):
    db = tmp_path / "m.db"
    store.init_db(db)
    for i in range(5):
        store.log_request(db, request_id=f"r{i}", prompt=f"p{i}", response="x", latency_ms=1, tokens_generated=1,
                          model_version="m", prompt_version="v1", ts=f"2026-01-01T00:00:0{i}Z")
    assert store.recent_prompts(db, 3) == ["p4", "p3", "p2"]
    assert store.count_requests(db) == 5


TH = {"drift_score_threshold": 0.4, "negative_feedback_ratio": 0.4, "min_feedback": 10}


def test_retrain_trigger_on_drift():
    d = should_retrain({"status": "ok", "drift_score": 0.7}, {"up": 0, "down": 0}, TH)
    assert d["trigger"] and "drift" in d["reasons"][0]


def test_retrain_trigger_on_negative_feedback_needs_min_samples():
    assert not should_retrain({"status": "ok", "drift_score": 0.1}, {"up": 1, "down": 5}, TH)["trigger"]
    assert should_retrain({"status": "ok", "drift_score": 0.1}, {"up": 5, "down": 8}, TH)["trigger"]


def test_no_trigger_when_healthy_or_drift_data_insufficient():
    assert not should_retrain({"status": "ok", "drift_score": 0.1}, {"up": 20, "down": 2}, TH)["trigger"]
    assert not should_retrain({"status": "insufficient_data", "drift_score": 0.9}, {"up": 0, "down": 0}, TH)["trigger"]
