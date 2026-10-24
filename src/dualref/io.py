"""Atomic artifacts, fingerprints, and execution telemetry."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Iterable, Iterator

from .config import write_config


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def write_json(path: str | Path, value: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_rows(rows: list[dict], limit: int | None, seed: int, key: str = "id") -> list[dict]:
    """Stable selection independent of the input's storage order."""
    ordered = sorted(rows, key=lambda row: fingerprint([seed, str(row[key])]))
    return ordered if limit is None else ordered[:limit]


@contextlib.contextmanager
def telemetry(output_dir: str | Path, stage: str, cfg: dict):
    """Measure stage duration and GPU memory across loading and computation."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_config(output / "config.yaml", cfg)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status = "failed"
    timing = {"excluded_seconds": 0.0}
    try:
        yield timing
        status = "complete"
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory = [
            {"device": i, "name": torch.cuda.get_device_name(i),
             "allocated_gb": torch.cuda.max_memory_allocated(i) / 1e9,
             "reserved_gb": torch.cuda.max_memory_reserved(i) / 1e9}
            for i in range(torch.cuda.device_count())
        ]
        elapsed = time.perf_counter() - started
        timing.update(seconds=max(0.0, elapsed - timing["excluded_seconds"]),
                      wall_seconds_including_evaluation=elapsed)
        write_json(output / f"telemetry-{stage}.json", {
            "stage": stage, "status": status, **timing,
            "devices": memory, "platform": platform.platform(), "torch": torch.__version__,
            "rank": int(os.environ.get("RANK", "0")),
        })
