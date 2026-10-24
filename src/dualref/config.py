"""Small YAML configuration layer; importing it never loads ML dependencies."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def config_path(name: str) -> Path:
    installed = Path(__file__).parent / "configs" / name
    source = Path(__file__).resolve().parents[2] / "configs" / name
    return installed if installed.is_file() else source


def read_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def write_config(path: str | Path, value: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)
    return path


def merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = read_config(config_path("default.yaml"))
    if path:
        cfg = merge(cfg, read_config(path))
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Expected key=value override, got {expression!r}")
        key, raw = expression.split("=", 1)
        try:
            value = yaml.safe_load(raw) if raw else ""
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML override: {expression!r}") from error
        target = cfg
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"Invalid override key: {key!r}")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot descend into {key!r}")
        target[parts[-1]] = value
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    objective = cfg.get("objective", {})
    if not 0 < objective.get("alpha", 0.5) < 1:
        raise ValueError("objective.alpha must be strictly between 0 and 1")
    if objective.get("beta", 1) <= 0:
        raise ValueError("objective.beta must be positive")
    for key in ("num_fast", "num_slow"):
        if cfg.get("bank", {}).get(key, 1) < 1:
            raise ValueError(f"bank.{key} must be positive")
    for key in ("steps", "batch_size", "gradient_accumulation"):
        if cfg.get("training", {}).get(key, 1) < 1:
            raise ValueError(f"training.{key} must be positive")
    if cfg.get("model", {}).get("precision", "bf16") not in ("bf16", "fp32"):
        raise ValueError("Supported precision values: bf16, fp32")
