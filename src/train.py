"""LoRA fine-tuning with Hugging Face transformers + peft, tracked in MLflow.

    python -m src.train --smoke      # ~20 optimisation steps on tiny sequences (CPU, a few minutes)
    python -m src.train              # full run (3 epochs by default; CPU: tens of minutes)
    python -m src.train --merge      # additionally save a merged full model to models/merged/<run>

The adapter is logged to MLflow, registered in the Model Registry and given the alias `candidate`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src import registry
from src.data.dataset import read_jsonl
from src.utils.config import ROOT, load_config, resolve_path
from src.utils.logging_utils import get_logger, log_fields, setup_logging
from src.utils.prompts import build_messages, load_prompt

log = get_logger("train")


# ----------------------------------------------------------------------------------------- pure helpers
def mask_prompt_labels(prompt_len: int, full_ids: list[int]) -> list[int]:
    """Labels = token ids, with the prompt part replaced by -100 so loss is only computed on the answer."""
    return [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]


def sha256_of(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "no-git"
    except Exception:  # noqa: BLE001
        return "no-git"


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, item, out)
    else:
        out[prefix] = ",".join(map(str, value)) if isinstance(value, list) else value
    return out


# ------------------------------------------------------------------------------------------ main logic
def tokenize_rows(tokenizer, rows: list[dict[str, Any]], system_prompt: str, max_len: int) -> tuple[list[dict], int]:
    examples, skipped = [], 0
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(build_messages(system_prompt, row["instruction"]),
                                                    tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + row["response"] + tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"][:max_len]
        if len(prompt_ids) >= len(full_ids):   # nothing left to learn from after truncation
            skipped += 1
            continue
        examples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
                         "labels": mask_prompt_labels(len(prompt_ids), full_ids)})
    return examples, skipped


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlflow
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                              set_seed)

    cfg = load_config(args.config)
    mode = "smoke" if args.smoke else "full"
    tcfg = cfg["train"][mode]
    seed = cfg["seed"]
    set_seed(seed)

    prompt_version, system_prompt = load_prompt(cfg)
    train_path, val_path = resolve_path(cfg["data"]["train_file"]), resolve_path(cfg["data"]["val_file"])
    train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    if not train_rows or not val_rows:
        raise SystemExit("Dataset missing. Run `make data` first.")
    if args.smoke:
        train_rows, val_rows = train_rows[: tcfg["train_samples"]], val_rows[: tcfg["eval_samples"]]

    base, cache = cfg["model"]["base_name"], str(resolve_path(cfg["model"]["cache_dir"]))
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    log_fields(log, 20, "starting training", mode=mode, base_model=base, device=device_name,
               n_train=len(train_rows), n_val=len(val_rows), prompt_version=prompt_version)

    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, cache_dir=cache, torch_dtype=torch.float32)
    model.config.use_cache = False
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"], target_modules=lcfg["target_modules"],
        bias="none", task_type="CAUSAL_LM"))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    train_ds, skipped_train = tokenize_rows(tokenizer, train_rows, system_prompt, tcfg["max_seq_len"])
    val_ds, _ = tokenize_rows(tokenizer, val_rows, system_prompt, tcfg["max_seq_len"])
    if not train_ds:
        raise SystemExit("No trainable examples after tokenisation; increase max_seq_len.")

    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict[str, Any]:
        longest = max(len(item["input_ids"]) for item in batch)
        pad = lambda seq, value: seq + [value] * (longest - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }

    run_name = args.run_name or f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    work_dir = resolve_path(cfg["model"]["adapters_dir"]) / "_work"
    training_args = TrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=tcfg["batch_size"], per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"], learning_rate=tcfg["lr"],
        num_train_epochs=tcfg.get("epochs", 1), max_steps=tcfg.get("max_steps", -1),
        warmup_ratio=cfg["train"]["warmup_ratio"], lr_scheduler_type="cosine",
        logging_steps=tcfg["logging_steps"], eval_strategy="steps", eval_steps=tcfg["eval_steps"],
        save_strategy="no", report_to="none", seed=seed, use_cpu=not use_cuda, bf16=use_bf16,
        fp16=use_cuda and not use_bf16, dataloader_pin_memory=False, remove_unused_columns=False,
        label_names=["labels"], disable_tqdm=True,
    )

    class MlflowLossCallback(TrainerCallback):
        """Logs the loss curve (train + eval) to MLflow at every logging step."""

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            numeric = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float)) and k != "epoch"}
            if numeric:
                mlflow.log_metrics(numeric, step=state.global_step)
                log_fields(log, 20, "train log", step=state.global_step, **numeric)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate, callbacks=[MlflowLossCallback()])

    registry.setup_experiment(cfg)
    client = registry.get_client(cfg)
    result: dict[str, Any] = {}
    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        params = flatten("", {"mode": mode, "base_model": base, "prompt_version": prompt_version, "seed": seed,
                              "lora": lcfg, "train": tcfg, "device": device_name, "n_train": len(train_ds),
                              "n_val": len(val_ds), "skipped_too_long": skipped_train,
                              "trainable_params": n_trainable, "total_params": n_total,
                              "train_data_sha256": sha256_of([train_path])[:16],
                              "val_data_sha256": sha256_of([val_path])[:16], "git_sha": git_sha()}, {})
        mlflow.log_params(params)
        mlflow.set_tags({"prompt_version": prompt_version, "mode": mode})

        started = time.perf_counter()
        train_output = trainer.train()
        train_seconds = time.perf_counter() - started
        final_eval = trainer.evaluate()
        mlflow.log_metrics({"final_train_loss": float(train_output.training_loss),
                            "final_eval_loss": float(final_eval["eval_loss"]),
                            "train_seconds": train_seconds})

        adapter_dir = resolve_path(cfg["model"]["adapters_dir"]) / run_id[:8]
        model.config.use_cache = True
        model.save_pretrained(str(adapter_dir))
        (adapter_dir / "train_meta.json").write_text(json.dumps(
            {"base_model": base, "prompt_version": prompt_version, "mode": mode, "run_id": run_id}, indent=2))
        shutil.rmtree(work_dir, ignore_errors=True)

        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        mlflow.log_artifact(str(resolve_path(args.config) if args.config else ROOT / "configs" / "config.yaml"), "config")
        mlflow.log_artifact(str(resolve_path(cfg["prompt"]["dir"]) / f"{prompt_version}.txt"), "prompt")

        result = {"run_id": run_id, "adapter_dir": str(adapter_dir), "mode": mode, "prompt_version": prompt_version,
                  "final_eval_loss": float(final_eval["eval_loss"]), "train_seconds": round(train_seconds, 1)}

        if not args.no_register:
            name = cfg["model"]["registry_name"]
            mv = registry.register_adapter(
                client, name, run_id, f"{active_run.info.artifact_uri}/adapter",
                tags={"prompt_version": prompt_version, "base_model": base, "mode": mode,
                      "train_data_sha256": params["train_data_sha256"], "git_sha": params["git_sha"]})
            client.set_registered_model_alias(name, cfg["mlflow"]["aliases"]["candidate"], mv.version)
            result.update(model_name=name, model_version=int(mv.version), alias="candidate")
            log_fields(log, 20, "registered model", name=name, version=mv.version, alias="candidate")

        if args.merge:
            merged_dir = resolve_path(cfg["model"]["merged_dir"]) / run_id[:8]
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            result["merged_dir"] = str(merged_dir)
            log_fields(log, 20, "merged model saved", path=str(merged_dir))
    return result


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run (20 steps)")
    parser.add_argument("--merge", action="store_true", help="also save a merged full model")
    parser.add_argument("--no-register", action="store_true", help="skip Model Registry registration")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
