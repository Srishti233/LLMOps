"""Version the dataset and the production model with DVC (local remote ./.dvc-storage).

Best effort: if DVC is not installed it prints a warning and exits 0, so the pipeline still runs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import ROOT, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("dvc_snapshot")


def run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        log_fields(log, 30, "command failed", cmd=" ".join(cmd), stderr=proc.stderr.strip()[-400:])
    return proc.returncode


def in_git_repo() -> bool:
    return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True).returncode == 0


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--message", default="data: new dataset version")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--init-only", action="store_true", help="only run `dvc init` and configure the local remote")
    args = parser.parse_args()

    if not shutil.which("dvc"):
        log.warning("dvc not installed; skipping data versioning (pip install dvc==3.55.2)")
        return 0
    cfg = load_config()
    git = bool(shutil.which("git")) and in_git_repo()
    if not (ROOT / ".dvc").exists():
        run(["dvc", "init"] + ([] if git else ["--no-scm"]))
    run(["dvc", "remote", "add", "-d", "-f", "localstore", ".dvc-storage"])
    if args.init_only:
        return 0

    targets = [t for t in (cfg["data"]["train_file"], cfg["data"]["val_file"],
                           cfg["model"]["local_production_dir"]) if resolve_path(t).exists()]
    if not targets:
        log.warning("nothing to version yet")
        return 0
    if run(["dvc", "add"] + targets) != 0:
        return 0
    if not args.no_push:
        run(["dvc", "push"])
    if git:
        candidates = [".dvc/config", ".gitignore", "data/.gitignore", "models/.gitignore",
                      *[f"{t}.dvc" for t in targets]]
        run(["git", "add", *[c for c in candidates if (ROOT / c).exists()]])
        identity = [] if subprocess.run(["git", "config", "user.email"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip() else ["-c", "user.name=llmops-lab",
                                                                         "-c", "user.email=llmops@example.com"]
        run(["git", *identity, "commit", "-m", args.message])
    log_fields(log, 20, "dvc snapshot done", tracked=targets, pushed=not args.no_push, git_commit=git)
    return 0


if __name__ == "__main__":
    sys.exit(main())
