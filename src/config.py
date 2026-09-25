"""Config loading. OPENAI_API_KEY is read from the environment (optionally via .env), never from config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class ConfigError(RuntimeError):
    pass


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("llm", "classifier_llm", "embeddings", "paths"):
        if section not in cfg:
            raise ConfigError(f"config.yaml is missing the '{section}' section")
    return cfg


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def resolve_path(cfg: dict[str, Any], name: str) -> Path:
    return ROOT / cfg["paths"][name]
