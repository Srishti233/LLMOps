"""Drift/feedback-triggered retraining.

Runs the drift check, counts feedback, and (if thresholds in configs/config.yaml are exceeded) runs `make retrain`.

    python scripts/check_and_retrain.py --dry-run     # decide only
    python scripts/check_and_retrain.py --force       # retrain regardless
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.drift import run_drift  # noqa: E402
from src.monitor.store import feedback_counts, init_db  # noqa: E402
from src.monitor.triggers import should_retrain  # noqa: E402
from src.utils.config import ROOT, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("check_and_retrain")


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="pass --smoke to train and eval (fast pipeline test)")
    args = parser.parse_args()

    cfg = load_config()
    db_path = resolve_path(cfg["data"]["monitoring_db"])
    init_db(db_path)
    drift = run_drift(cfg)
    decision = should_retrain(drift, feedback_counts(db_path), cfg["retrain"])
    if args.force and not decision["trigger"]:
        decision.update(trigger=True, reasons=["--force"])
    print(json.dumps(decision, indent=2))
    if not decision["trigger"]:
        log.info("no retraining needed")
        return 0
    if args.dry_run:
        log.info("retraining would be triggered (dry run)")
        return 0
    if not shutil.which("make"):
        print("`make` not found; run the retrain steps manually (see Makefile).", file=sys.stderr)
        return 2
    cmd = ["make", "retrain"] + (["TRAIN_ARGS=--smoke", "EVAL_ARGS=--smoke"] if args.smoke else [])
    log_fields(log, 20, "triggering retraining", reasons=decision["reasons"], cmd=" ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
