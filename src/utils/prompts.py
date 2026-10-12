"""Versioned prompt templates (prompts/v1.txt, prompts/v2.txt)."""
from __future__ import annotations

from typing import Any

from src.utils.config import resolve_path


def prompt_dir(cfg: dict[str, Any]):
    return resolve_path(cfg["prompt"]["dir"])


def list_versions(cfg: dict[str, Any]) -> list[str]:
    return sorted(p.stem for p in prompt_dir(cfg).glob("*.txt"))


def load_prompt(cfg: dict[str, Any], version: str | None = None) -> tuple[str, str]:
    """Return (version, system_prompt_text)."""
    version = version or cfg["prompt"]["version"]
    path = prompt_dir(cfg) / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt version '{version}' not found. Available: {list_versions(cfg)}")
    return version, path.read_text(encoding="utf-8").strip()


def build_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]
