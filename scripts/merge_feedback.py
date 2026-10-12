"""Merge feedback-derived examples into the training set (the validation set is never touched)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import read_jsonl, write_jsonl  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("merge_feedback")


def merge(train: list[dict], feedback: list[dict], val: list[dict]) -> tuple[list[dict], int]:
    """Append feedback examples whose instruction is not already in train or val. Returns (rows, n_added)."""
    known = {r["instruction"] for r in train} | {r["instruction"] for r in val}
    added = []
    for row in feedback:
        if row["instruction"] not in known:
            known.add(row["instruction"])
            added.append(row)
    return train + added, len(added)


def main() -> None:
    setup_logging()
    cfg = load_config()
    train_path = resolve_path(cfg["data"]["train_file"])
    train = read_jsonl(train_path)
    merged, n_added = merge(train, read_jsonl(resolve_path(cfg["data"]["feedback_train_file"])),
                            read_jsonl(resolve_path(cfg["data"]["val_file"])))
    if n_added:
        write_jsonl(train_path, merged)
    log_fields(log, 20, "feedback merged into training set", added=n_added, train_size=len(merged))
    print(json.dumps({"added": n_added, "train_size": len(merged)}))


if __name__ == "__main__":
    main()
