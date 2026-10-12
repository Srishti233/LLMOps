"""Download and cache the base model locally (free, from the Hugging Face Hub)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("download_model")


def main() -> None:
    setup_logging()
    from huggingface_hub import snapshot_download

    cfg = load_config()
    name, cache = cfg["model"]["base_name"], resolve_path(cfg["model"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    log_fields(log, 20, "downloading model (no API key needed)", model=name, cache_dir=str(cache))
    path = snapshot_download(repo_id=name, cache_dir=str(cache),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
    log_fields(log, 20, "model cached", path=path)
    print(path)


if __name__ == "__main__":
    main()
