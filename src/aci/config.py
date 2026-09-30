"""Configuration loading. YAML file + optional dict overrides (dotted keys)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key, val in (overrides or {}).items():  # {"retrieval.dense_weight": 0.4}
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return copy.deepcopy(cfg)
