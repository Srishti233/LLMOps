"""Model resolution and loading shared by evaluation and the API.

Resolution order for source="auto" (used by the API):
    1. MLflow registry, alias `production`
    2. local fallback directory models/production (written by the promote step)
    3. the base model from Hugging Face  -> the app always starts
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import registry
from src.utils.config import resolve_path
from src.utils.logging_utils import get_logger
from src.utils.prompts import build_messages

log = get_logger("model_io")


class ModelNotFound(RuntimeError):
    pass


@dataclass
class ResolvedSource:
    kind: str                       # registry | local | path | base
    label: str                      # human readable, returned by the API as model_version
    adapter_dir: Path | None = None
    version: str | None = None
    run_id: str | None = None
    prompt_version: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


def local_production_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["model"]["local_production_dir"])


def _resolve_local(cfg: dict[str, Any]) -> ResolvedSource | None:
    folder = local_production_dir(cfg)
    meta_file = folder / "meta.json"
    if not (meta_file.is_file() and (folder / "adapter_config.json").is_file()):
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return ResolvedSource(
        kind="local", adapter_dir=folder, version=str(meta.get("version")), run_id=meta.get("run_id"),
        prompt_version=meta.get("prompt_version"),
        label=f"{cfg['model']['registry_name']}:v{meta.get('version')}@local-production",
    )


def _resolve_registry(cfg: dict[str, Any], alias: str) -> ResolvedSource | None:
    if not registry.registry_available(cfg):
        return None
    registry.configure_fast_failure()
    client = registry.get_client(cfg)
    name = cfg["model"]["registry_name"]
    mv = registry.get_version_by_alias(client, name, alias)
    if mv is None:
        return None
    return ResolvedSource(
        kind="registry", adapter_dir=registry.resolve_adapter_dir(mv), version=str(mv.version), run_id=mv.run_id,
        prompt_version=(mv.tags or {}).get("prompt_version"), tags=dict(mv.tags or {}),
        label=f"{name}:v{mv.version}@{alias}",
    )


def _base(cfg: dict[str, Any]) -> ResolvedSource:
    return ResolvedSource(kind="base", label=f"base:{cfg['model']['base_name']}")


def resolve_source(cfg: dict[str, Any], source: str = "auto") -> ResolvedSource:
    """source: auto | production | candidate | local | base | path:<adapter dir>"""
    if source == "base":
        return _base(cfg)
    if source.startswith("path:"):
        folder = Path(source[5:])
        if not (folder / "adapter_config.json").is_file():
            raise ModelNotFound(f"no adapter_config.json in {folder}")
        return ResolvedSource(kind="path", adapter_dir=folder, label=f"adapter:{folder}")
    if source in ("production", "candidate"):
        resolved = _resolve_registry(cfg, cfg["mlflow"]["aliases"][source])
        if resolved is None:
            raise ModelNotFound(f"no registered model version with alias '{source}'")
        return resolved
    if source == "local":
        return _resolve_local(cfg) or _base(cfg)
    if source != "auto":
        raise ValueError(f"unknown model source: {source}")

    try:
        resolved = _resolve_registry(cfg, cfg["mlflow"]["aliases"]["production"])
        if resolved is not None:
            return resolved
    except Exception as exc:  # noqa: BLE001 - registry problems must never stop the API from starting
        log.warning("registry lookup failed, using local fallback: %s", exc)
    return _resolve_local(cfg) or _base(cfg)


@dataclass
class GenResult:
    text: str
    tokens: int
    gen_seconds: float
    input_tokens: int


class LoadedModel:
    def __init__(self, tokenizer, model, device: str, resolved: ResolvedSource, cfg: dict[str, Any]):
        self.tokenizer, self.model, self.device = tokenizer, model, device
        self.resolved, self.cfg = resolved, cfg

    def generate(self, question: str, system_prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.0) -> GenResult:
        import torch

        tok = self.tokenizer
        text = tok.apply_chat_template(build_messages(system_prompt, question), tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens),
                                  "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed = time.perf_counter() - start
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return GenResult(text=tok.decode(new_tokens, skip_special_tokens=True).strip(), tokens=int(new_tokens.shape[0]),
                         gen_seconds=elapsed, input_tokens=int(inputs["input_ids"].shape[1]))


def load_model(cfg: dict[str, Any], resolved: ResolvedSource) -> LoadedModel:
    """Load base weights (+ adapter merged in) onto the best available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    threads = os.environ.get("TORCH_NUM_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base, cache = cfg["model"]["base_name"], str(resolve_path(cfg["model"]["cache_dir"]))
    log.info("loading model", extra={"fields": {"base": base, "device": device, "source": resolved.label}})
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(base, cache_dir=cache, torch_dtype=dtype)
    if resolved.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resolved.adapter_dir))
        model = model.merge_and_unload()
    model.to(device).eval()
    return LoadedModel(tokenizer, model, device, resolved, cfg)
