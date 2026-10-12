import json
from pathlib import Path

from src.gate import evaluate_gate
from src.model_io import ModelNotFound, resolve_source
from src.registry import remap_artifact_path
from src.utils.config import load_config

TH = {"min_rouge_l": 0.35, "min_keyword_match_rate": 0.3, "max_avg_latency_s": 60, "min_tokens_per_sec": 1.0}
GOOD = {"rouge_l": 0.6, "keyword_match_rate": 0.7, "avg_latency_s": 5.0, "tokens_per_sec": 8.0}


def test_gate_passes_good_metrics():
    d = evaluate_gate(GOOD, TH, None, 0.02)
    assert d["passed"] and not d["reasons"] and not d["compared_with_production"]


def test_gate_fails_each_threshold():
    for key, bad in [("rouge_l", 0.1), ("keyword_match_rate", 0.0), ("avg_latency_s", 99.0), ("tokens_per_sec", 0.1)]:
        d = evaluate_gate({**GOOD, key: bad}, TH, None, 0.02)
        assert not d["passed"] and key in d["reasons"][0]


def test_gate_blocks_regression_vs_production_but_allows_small_tolerance():
    prod = {"rouge_l": 0.8, "keyword_match_rate": 0.9}
    assert not evaluate_gate(GOOD, TH, prod, 0.02)["passed"]
    assert evaluate_gate({**GOOD, "rouge_l": 0.79, "keyword_match_rate": 0.89}, TH, prod, 0.02)["passed"]


def test_remap_artifact_path(tmp_path):
    root = tmp_path
    stored = "/home/someone/llmops-lab/mlruns/1/abc/artifacts/adapter"
    assert remap_artifact_path(stored, root) == root / "mlruns/1/abc/artifacts/adapter"
    assert remap_artifact_path("file://" + stored, root) == root / "mlruns/1/abc/artifacts/adapter"
    existing = tmp_path / "real"
    existing.mkdir()
    assert remap_artifact_path(str(existing), root) == existing


def _cfg_with_tmp_production(tmp_path):
    cfg = load_config()
    cfg["model"]["local_production_dir"] = str(tmp_path / "production")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/does-not-exist.db"   # registry absent
    return cfg


def test_resolution_falls_back_to_base_when_nothing_exists(tmp_path):
    r = resolve_source(_cfg_with_tmp_production(tmp_path), "auto")
    assert r.kind == "base" and r.label.startswith("base:")


def test_resolution_prefers_local_production_adapter(tmp_path):
    cfg = _cfg_with_tmp_production(tmp_path)
    prod = Path(cfg["model"]["local_production_dir"])
    prod.mkdir()
    (prod / "adapter_config.json").write_text("{}")
    (prod / "meta.json").write_text(json.dumps({"version": 3, "run_id": "r", "prompt_version": "v2"}))
    r = resolve_source(cfg, "auto")
    assert r.kind == "local" and r.version == "3" and r.prompt_version == "v2" and "v3" in r.label


def test_strict_aliases_raise_when_missing(tmp_path):
    for source in ("candidate", "production"):
        try:
            resolve_source(_cfg_with_tmp_production(tmp_path), source)
        except ModelNotFound:
            continue
        raise AssertionError("expected ModelNotFound")


def test_train_label_masking_and_feedback_merge():
    from scripts.export_feedback import split_feedback
    from scripts.merge_feedback import merge
    from src.train import mask_prompt_labels

    assert mask_prompt_labels(2, [5, 6, 7, 8]) == [-100, -100, 7, 8]
    rows = [{"request_id": "a", "rating": "up", "prompt": "p1", "response": "r1", "comment": None, "model_version": "m"},
            {"request_id": "b", "rating": "down", "prompt": "p2", "response": "bad", "comment": "no", "model_version": "m"},
            {"request_id": "c", "rating": "up", "prompt": "p3", "response": "r3", "comment": None, "model_version": "m"}]
    train, review = split_feedback(rows, existing_ids={"c"})
    assert [t["request_id"] for t in train] == ["a"] and [r["request_id"] for r in review] == ["b"]
    merged, added = merge([{"instruction": "p0"}], train + [{"instruction": "p0"}], [{"instruction": "pv"}])
    assert added == 1 and len(merged) == 2
