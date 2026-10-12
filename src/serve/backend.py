"""LLM backend: loads the model in a background thread (so /health answers immediately) and serialises generation."""
from __future__ import annotations

import os
import threading
from typing import Any

from src.model_io import GenResult, load_model, resolve_source
from src.utils.logging_utils import get_logger, log_fields
from src.utils.prompts import load_prompt

log = get_logger("backend")


class LLMBackend:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.ready = False
        self.error: str | None = None
        self.model_version = "loading"
        self.prompt_version = cfg["prompt"]["version"]
        self.system_prompt = ""
        self._model = None
        self._lock = threading.Lock()    # a single torch model must not run concurrent generate() calls

    def start(self) -> None:
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            resolved = resolve_source(self.cfg, os.environ.get("MODEL_SOURCE", "auto"))
            explicit_prompt = bool(os.environ.get("PROMPT_VERSION"))
            wanted = self.cfg["prompt"]["version"] if explicit_prompt else (resolved.prompt_version or self.cfg["prompt"]["version"])
            self.prompt_version, self.system_prompt = load_prompt(self.cfg, wanted)
            self._model = load_model(self.cfg, resolved)
            self.model_version = resolved.label
            self.ready = True
            log_fields(log, 20, "model ready", model_version=self.model_version, prompt_version=self.prompt_version,
                       device=self._model.device, source=resolved.kind)
        except Exception as exc:  # noqa: BLE001 - keep the API up and report the reason through /ready
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenResult:
        if not self.ready or self._model is None:
            raise RuntimeError("model not ready")
        with self._lock:
            return self._model.generate(prompt, self.system_prompt, max_new_tokens=max_new_tokens,
                                        temperature=temperature)
