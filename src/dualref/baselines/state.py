"""External run identity and checkpoint-aligned evaluation history."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

from ..io import write_jsonl


MODEL_REVISIONS = {
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Qwen/Qwen3-VL-8B-Thinking": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}


def run_identity(config: dict, rows: list[dict], source: dict) -> str:
    settings = copy.deepcopy({key: config.get(key, {}) for key in
                              ("model", "training", "external", "generation", "evaluation")})
    for key in ("output_dir", "train_file", "resume", "resume_from_checkpoint"):
        settings["training"].pop(key, None)
    for key in ("root", "source_dir", "python", "install", "fetch", "prepare_only"):
        settings["external"].pop(key, None)
    model = settings["external"].setdefault("model", source["model"])
    settings["external"]["model_revision"] = settings["external"].get("model_revision") or MODEL_REVISIONS.get(model)
    settings["external"]["revision"] = source["revision"]
    settings["evaluation"].pop("output", None)
    bank = config.get("training", {}).get("bank_file") or config.get("bank", {}).get("path")
    settings["training"].pop("bank_file", None)
    for key, path in (("evaluation_input", settings["evaluation"].pop("input", None)), ("bank", bank)):
        settings[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    value = {"examples": rows, "seed": config.get("seed", 42), "source": source, "settings": settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(type(row.get("step")) is not int or row["step"] < 0 for row in rows):
        raise ValueError("Evaluation history requires nonnegative integer steps")
    return rows


def prepare_history(config: dict, output: Path, method: str, identity: str) -> None:
    training = config.get("training", {})
    resume = training.get("resume_from_checkpoint") or training.get("resume")
    history = output / "history.jsonl"
    if not resume:
        if history.exists() or any(output.rglob("checkpoint-*")) or (output / "result.json").exists():
            raise ValueError("External training outputs already exist; resume a supported checkpoint or choose a new output directory")
        return
    if method == "fast":
        raise ValueError("FAST checkpoint resume is not supported by this runner")
    checkpoint = Path(resume).resolve()
    manifest = next((parent / "external_manifest.json" for parent in (checkpoint, *checkpoint.parents)
                     if (parent / "external_manifest.json").is_file()), None)
    if manifest is None or json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint") != identity:
        raise ValueError("Resume checkpoint does not have the same verified external run identity")
    state = checkpoint / ("progress.json" if method == "dualmind" else "trainer_state.json")
    progress = json.loads(state.read_text(encoding="utf-8"))
    step = progress.get("step" if method == "dualmind" else "global_step")
    if type(step) is not int or step < 0:
        raise ValueError("Resume checkpoint has no valid training step")
    rows = _history(history)
    if history.exists():
        index = 0
        backup = output / "history.before-resume.jsonl"
        while backup.exists():
            index += 1
            backup = output / f"history.before-resume-{index}.jsonl"
        shutil.copy2(history, backup)
        by_step = {row["step"]: row for row in rows if row["step"] <= step}
        write_jsonl(history, [by_step[key] for key in sorted(by_step)])


def record_evaluation(path: Path, record: dict) -> None:
    """Replace an existing checkpoint evaluation instead of appending duplicate steps."""
    if type(record.get("step")) is not int or record["step"] < 0:
        raise ValueError("Evaluation step must be a nonnegative integer")
    by_step = {row["step"]: row for row in _history(path)}
    by_step[record["step"]] = record
    write_jsonl(path, [by_step[key] for key in sorted(by_step)])
