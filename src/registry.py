"""MLflow Model Registry helpers. The registered artifact is the LoRA adapter directory (a few MB).

MLflow is imported lazily so unit tests and light-weight scripts do not need it installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.utils.config import ROOT, resolve_path, tracking_uri
from src.utils.logging_utils import get_logger

log = get_logger("registry")


def sqlite_file(uri: str) -> Path | None:
    prefix = "sqlite:///"
    return Path(uri[len(prefix):]) if uri.startswith(prefix) else None


def registry_available(cfg: dict[str, Any]) -> bool:
    """Cheap pre-check so we never create an empty mlflow.db just by trying to read the registry."""
    uri = tracking_uri(cfg)
    db = sqlite_file(uri)
    return db.is_file() if db else True


def get_client(cfg: dict[str, Any]):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def setup_experiment(cfg: dict[str, Any]) -> str:
    """Create the experiment with an artifact root anchored in the project (./mlruns) and return its id."""
    import mlflow

    client = get_client(cfg)
    name = cfg["mlflow"]["experiment"]
    exp = client.get_experiment_by_name(name)
    if exp is None:
        artifact_root = resolve_path(cfg["mlflow"]["artifact_root"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        exp_id = client.create_experiment(name, artifact_location=str(artifact_root))
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(name)
    return exp_id


def ensure_registered_model(client, name: str) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description="LoRA adapters for the AuroraSync support assistant")


def register_adapter(client, name: str, run_id: str, source: str, tags: dict[str, str]):
    ensure_registered_model(client, name)
    mv = client.create_model_version(name=name, source=source, run_id=run_id, tags=tags)
    return mv


def get_version_by_alias(client, name: str, alias: str):
    """Return the ModelVersion for an alias or None."""
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def remap_artifact_path(source: str, root: Path | None = None) -> Path:
    """Map an artifact path stored in the registry to this machine.

    Registry rows store absolute paths of the machine that trained the model (e.g. /home/me/llmops-lab/mlruns/1/ab/artifacts/adapter).
    Inside a container the project is mounted elsewhere, so if the stored path does not exist we re-anchor
    everything after the last `mlruns/` segment at <root>/mlruns.
    """
    root = root or ROOT
    raw = source[len("file://"):] if source.startswith("file://") else source
    path = Path(raw)
    if path.exists():
        return path
    marker = "mlruns/"
    norm = raw.replace("\\", "/")
    if marker in norm:
        tail = norm.rsplit(marker, 1)[1]
        return root / "mlruns" / tail
    return path


def resolve_adapter_dir(mv, root: Path | None = None) -> Path:
    """Return a local directory holding the adapter of a registered model version."""
    local = remap_artifact_path(mv.source, root)
    if local.is_dir():
        return local
    import mlflow

    dst = (root or ROOT) / "models" / "registry_downloads" / f"{mv.name}-v{mv.version}"
    dst.mkdir(parents=True, exist_ok=True)
    downloaded = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(dst))
    return Path(downloaded)


def metric_tags(version) -> dict[str, float]:
    """Evaluation metrics previously stored as model-version tags (eval_<metric>)."""
    out: dict[str, float] = {}
    for key, value in (version.tags or {}).items():
        if key.startswith("eval_") and key not in ("eval_mode", "eval_samples"):
            try:
                out[key[len("eval_"):]] = float(value)
            except ValueError:
                continue
    return out


def configure_fast_failure() -> None:
    """Don't hang for minutes retrying an unreachable tracking server."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
