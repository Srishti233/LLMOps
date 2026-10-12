"""Configuration loading. configs/config.yaml is the single source of truth."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("LLMOPS_HOME") or Path(__file__).resolve().parents[2]).resolve()


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("LLMOPS_CONFIG") or ROOT / "configs" / "config.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base_override = os.environ.get("LLMOPS_BASE_MODEL")
    if base_override:
        cfg["model"]["base_name"] = base_override
    prompt_override = os.environ.get("PROMPT_VERSION")
    if prompt_override:
        cfg["prompt"]["version"] = prompt_override
    return cfg


def resolve_path(value: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def tracking_uri(cfg: dict[str, Any]) -> str:
    """MLflow tracking URI; relative sqlite paths are anchored at the project root."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        return prefix + (ROOT / uri[len(prefix):]).as_posix()
    return uri
