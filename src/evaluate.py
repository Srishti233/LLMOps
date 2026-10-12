"""Evaluate a model on the validation set: ROUGE-L, keyword-match rate, latency and tokens/sec.

    python -m src.evaluate --model candidate [--smoke]     # evaluate the registry `candidate` and tag the result
    python -m src.evaluate --model base --out reports/eval_base.json
Writes reports/eval.json (consumed by the quality gate, scripts/promote_model.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from src.data.dataset import read_jsonl
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger, log_fields, setup_logging
from src.utils.text_metrics import keyword_score, rouge_l

log = get_logger("evaluate")

# generate_fn(question, max_new_tokens) -> object with .text .tokens .gen_seconds
GenerateFn = Callable[[str, int], Any]


def evaluate_samples(samples: list[dict[str, Any]], generate_fn: GenerateFn, max_new_tokens: int) -> dict[str, Any]:
    """Pure evaluation loop (model-agnostic, unit-testable with a fake generate_fn)."""
    per_sample, total_tokens, total_seconds = [], 0, 0.0
    for sample in samples:
        out = generate_fn(sample["instruction"], max_new_tokens)
        coverage, exact = keyword_score(out.text, sample.get("keywords", []))
        total_tokens += out.tokens
        total_seconds += out.gen_seconds
        per_sample.append({"id": sample.get("id"), "topic": sample.get("topic"), "instruction": sample["instruction"],
                           "prediction": out.text, "reference": sample["response"],
                           "rouge_l": rouge_l(out.text, sample["response"]), "keyword_coverage": coverage,
                           "keyword_exact": exact, "latency_s": out.gen_seconds, "tokens": out.tokens})
    n = max(len(per_sample), 1)
    metrics = {
        "rouge_l": sum(s["rouge_l"] for s in per_sample) / n,
        "keyword_match_rate": sum(1 for s in per_sample if s["keyword_exact"]) / n,   # all keywords present
        "keyword_coverage": sum(s["keyword_coverage"] for s in per_sample) / n,       # partial credit
        "avg_latency_s": sum(s["latency_s"] for s in per_sample) / n,
        "tokens_per_sec": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "avg_tokens": total_tokens / n,
    }
    return {"metrics": metrics, "per_sample": per_sample, "n_samples": len(per_sample)}


def log_to_registry(cfg: dict[str, Any], resolved, result: dict[str, Any], mode: str, out_path: Path) -> None:
    """Attach the metrics to the training run and to the model version (used by the quality gate)."""
    import mlflow

    from src import registry

    client = registry.get_client(cfg)
    name = cfg["model"]["registry_name"]
    with mlflow.start_run(run_id=resolved.run_id):
        mlflow.log_metrics({f"eval_{k}": float(v) for k, v in result["metrics"].items()})
        mlflow.log_artifact(str(out_path), artifact_path="eval")
    for key, value in result["metrics"].items():
        client.set_model_version_tag(name, resolved.version, f"eval_{key}", f"{value:.6f}")
    client.set_model_version_tag(name, resolved.version, "eval_mode", mode)
    client.set_model_version_tag(name, resolved.version, "eval_samples", str(result["n_samples"]))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="candidate",
                        help="candidate | production | local | base | path:<adapter dir>")
    parser.add_argument("--smoke", action="store_true", help="evaluate only a handful of samples")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="reports/eval.json")
    parser.add_argument("--no-log", action="store_true", help="do not write metrics back to MLflow")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    from src.model_io import load_model, resolve_source
    from src.utils.prompts import load_prompt

    cfg = load_config(args.config)
    mode = "smoke" if args.smoke else "full"
    limit = args.limit or (cfg["eval"]["smoke_samples"] if args.smoke else cfg["eval"]["num_samples"])
    samples = read_jsonl(resolve_path(cfg["data"]["val_file"]))[:limit]
    if not samples:
        raise SystemExit("Validation set missing. Run `make data` first.")

    resolved = resolve_source(cfg, args.model)
    prompt_version, system_prompt = load_prompt(cfg, resolved.prompt_version or None)
    model = load_model(cfg, resolved)
    # warm-up so one-off initialisation cost does not distort latency
    model.generate(samples[0]["instruction"], system_prompt, max_new_tokens=4)

    def generate_fn(question: str, max_new_tokens: int):
        return model.generate(question, system_prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    result = evaluate_samples(samples, generate_fn, cfg["eval"]["max_new_tokens"])
    report = {
        "model_label": resolved.label, "model_name": cfg["model"]["registry_name"],
        "model_version": int(resolved.version) if resolved.kind == "registry" and resolved.version else None,
        "run_id": resolved.run_id, "source_kind": resolved.kind, "mode": mode, "prompt_version": prompt_version,
        "base_model": cfg["model"]["base_name"], "device": model.device, "n_samples": result["n_samples"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "metrics": result["metrics"],
        "per_sample": result["per_sample"],
    }
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_fields(log, 20, "evaluation finished", **{k: round(v, 4) for k, v in result["metrics"].items()},
               model=resolved.label, mode=mode)

    if resolved.kind == "registry" and not args.no_log:
        log_to_registry(cfg, resolved, result, mode, out_path)
    print(json.dumps({"model": resolved.label, "mode": mode, "n_samples": result["n_samples"],
                      "metrics": {k: round(v, 4) for k, v in result["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
