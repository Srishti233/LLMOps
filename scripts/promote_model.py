"""Quality gate: promote the `candidate` model version to `production` only if it passes.

Reads reports/eval.json (written by `python -m src.evaluate --model candidate`).
Exit codes: 0 promoted | 1 gate failed | 2 stale/missing evaluation or no candidate.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import registry  # noqa: E402
from src.gate import evaluate_gate  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("promote")


def export_local_copy(cfg, mv, adapter_dir: Path, report: dict) -> Path:
    """Copy the production adapter to models/production so the API can start without the registry."""
    dest = resolve_path(cfg["model"]["local_production_dir"])
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(adapter_dir, dest)
    meta = {"version": int(mv.version), "run_id": mv.run_id, "prompt_version": report.get("prompt_version"),
            "base_model": cfg["model"]["base_name"], "metrics": report["metrics"],
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def main() -> int:
    setup_logging()
    cfg = load_config()
    report_path = resolve_path("reports/eval.json")
    if not report_path.is_file():
        print("reports/eval.json not found. Run `make eval` first.", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))

    client = registry.get_client(cfg)
    name = cfg["model"]["registry_name"]
    aliases = cfg["mlflow"]["aliases"]
    candidate = registry.get_version_by_alias(client, name, aliases["candidate"])
    if candidate is None:
        print("No model version with alias 'candidate'. Run `make train` first.", file=sys.stderr)
        return 2
    if report.get("model_version") != int(candidate.version):
        print(f"eval.json is for version {report.get('model_version')} but the candidate is v{candidate.version}. "
              "Re-run `make eval`.", file=sys.stderr)
        return 2

    mode = report.get("mode", "full")
    thresholds = cfg["gate"]["smoke_thresholds" if mode == "smoke" else "thresholds"]
    production = registry.get_version_by_alias(client, name, aliases["production"])
    production_metrics = registry.metric_tags(production) if production else None
    decision = evaluate_gate(report["metrics"], thresholds, production_metrics, cfg["gate"]["max_regression"])
    decision.update(candidate_version=int(candidate.version), mode=mode,
                    production_version=int(production.version) if production else None)

    if decision["passed"]:
        client.set_registered_model_alias(name, aliases["production"], candidate.version)
        client.set_model_version_tag(name, candidate.version, "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        local = export_local_copy(cfg, candidate, registry.resolve_adapter_dir(candidate), report)
        decision["local_copy"] = str(local)
        log_fields(log, 20, "model promoted to production", version=candidate.version, mode=mode)
    else:
        log_fields(log, 40, "quality gate FAILED, candidate not promoted", reasons=decision["reasons"])

    out = resolve_path("reports/promotion.json")
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
