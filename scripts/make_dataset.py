"""Generate the synthetic customer-support dataset (train/val JSONL)."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import generate_examples, split_examples, write_jsonl  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("make_dataset")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="number of examples (default from config)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = args.n or cfg["data"]["n_examples"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    train, val = split_examples(generate_examples(n, seed), cfg["data"]["val_fraction"], seed)
    train_path, val_path = resolve_path(cfg["data"]["train_file"]), resolve_path(cfg["data"]["val_file"])
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    log_fields(log, 20, "dataset written", train=len(train), val=len(val), train_path=str(train_path),
               val_path=str(val_path), train_sha256=sha256_file(train_path)[:16])


if __name__ == "__main__":
    main()
