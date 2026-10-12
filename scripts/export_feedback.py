"""Turn logged feedback into data: thumbs-up -> new training examples, thumbs-down -> a review file.

Thumbs-up rows are APPENDED to data/feedback_train.jsonl (deduplicated by request_id).
Thumbs-down rows are written to reports/feedback_review.jsonl for a human to correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import read_jsonl, write_jsonl  # noqa: E402
from src.monitor.store import feedback_rows, init_db  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("export_feedback")


def split_feedback(rows: list[dict], existing_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (new training examples, review rows)."""
    train, review = [], []
    for row in rows:
        if row["rating"] == "up" and row["request_id"] not in existing_ids and row["response"].strip():
            train.append({"id": f"fb-{row['request_id'][:12]}", "request_id": row["request_id"], "topic": "feedback",
                          "instruction": row["prompt"], "response": row["response"], "keywords": [],
                          "source": "feedback_thumbs_up"})
        elif row["rating"] == "down":
            review.append({"request_id": row["request_id"], "prompt": row["prompt"], "bad_response": row["response"],
                           "comment": row["comment"], "model_version": row["model_version"],
                           "corrected_response": ""})
    return train, review


def main() -> None:
    setup_logging()
    cfg = load_config()
    db_path = resolve_path(cfg["data"]["monitoring_db"])
    init_db(db_path)
    rows = feedback_rows(db_path)

    fb_path = resolve_path(cfg["data"]["feedback_train_file"])
    existing = read_jsonl(fb_path)
    new_train, review = split_feedback(rows, {r.get("request_id", "") for r in existing})
    if new_train:
        write_jsonl(fb_path, existing + new_train)
    review_path = resolve_path(cfg["data"]["feedback_review_file"])
    write_jsonl(review_path, review)
    log_fields(log, 20, "feedback exported", feedback_rows=len(rows), new_training_examples=len(new_train),
               total_feedback_training_examples=len(existing) + len(new_train), review_rows=len(review),
               review_file=str(review_path))
    print(json.dumps({"new_training_examples": len(new_train), "review_rows": len(review)}))


if __name__ == "__main__":
    main()
