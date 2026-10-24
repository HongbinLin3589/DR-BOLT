"""Validate stage outputs before publishing or reusing their results."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .analysis import REPORT_VERSION, benchmark_metrics, calibration_metrics, cross_likelihood_metrics


FILE_KEYS = {"bank", "history", "records", "predictions", "metrics", "summary", "report",
             "router", "train", "validation", "test", "replay_bank", "matrix_bank", "manifest",
             "coverage", "token_budget", "_result"}
SUCCESS = {"complete", "completed", "measured"}


def _file(value: object, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Output {key} requires a file path")
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Output {key} is not a nonempty file: {path}")
    return path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _records(path: Path):
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object records: {path}")
                count += 1
                yield row
    if not count:
        raise ValueError(f"No records in {path}")


def _checkpoint(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("Checkpoint requires a local path")
    path = Path(value)
    if path.is_file() and path.suffix == ".pt":
        _file(value, "checkpoint")
        return
    config = path / ("adapter_config.json" if (path / "adapter_config.json").is_file() else "config.json")
    metadata = _json(_file(str(config), "checkpoint config"))
    identity_key = "base_model_name_or_path" if config.name == "adapter_config.json" else "model_type"
    if not isinstance(metadata.get(identity_key), str) or not metadata[identity_key]:
        raise ValueError(f"Checkpoint config requires {identity_key}: {config}")
    weights = [p for pattern in ("*.safetensors", "pytorch_model*.bin", "adapter_model.bin")
               for p in path.glob(pattern) if p.is_file() and p.stat().st_size]
    if not weights:
        raise ValueError(f"Checkpoint contains no model weights: {path}")
    for index in path.glob("*.index.json"):
        mapping = _json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid checkpoint shard index: {index}")
        for shard in set(mapping.values()):
            _file(str(path / shard), "checkpoint shard")


def validate_result(stage: str, cfg: dict, result: dict, referenced: set[str] | None = None) -> str:
    """Return completed/partial; reject failures, unusable artifacts and unknown states."""
    if not isinstance(result, dict):
        raise ValueError("A stage must return a dictionary of artifacts")
    status = result.get("status")
    if status is not None and status not in SUCCESS | {"partial"}:
        raise ValueError(f"Stage returned a non-success status: {status}")
    if status == "partial" and stage != "analyze":
        raise ValueError("Only analysis reports can publish partial results")
    required = set(referenced or ()) - {"_result", "_fingerprint"}
    required.update({
        "bank": {"bank"}, "score": {"bank"}, "weights": {"bank"},
        "train": {"checkpoint"}, "evaluate": {"predictions", "metrics"},
        "router": {"router", "checkpoint"}, "diagnostics": {"records"},
        "analyze": {"summary", "report", "status"}, "prepare": set(),
    }[stage])
    if stage == "diagnostics" and cfg.get("diagnostics", {}).get("kind", "cross_likelihood") == "cross_likelihood":
        required.update({"bank", "matrix_bank", "replay_bank"})
    if stage == "prepare":
        splits = {key for key, value in result.items() if not key.startswith("_") and isinstance(value, str) and value.endswith(".jsonl")}
        if not splits:
            raise ValueError("Dataset preparation produced no splits")
        required.update(splits)
    absent = required - result.keys()
    if absent:
        raise ValueError(f"Stage {stage} did not produce required outputs: {sorted(absent)}")
    for key in required | (FILE_KEYS & result.keys()):
        value = result[key]
        if key in {"checkpoint", "resume"}:
            if key == "checkpoint":
                _checkpoint(value)
            elif not isinstance(value, str) or not Path(value).is_dir():
                raise ValueError("Resume output requires a checkpoint directory")
        elif key in FILE_KEYS or (stage == "prepare" and key != "status"):
            path = _file(value, key)
            if path.suffix == ".json":
                _json(path)
            elif path.suffix == ".jsonl":
                for row in _records(path):
                    if key in {"bank", "replay_bank", "matrix_bank"}:
                        candidates = row.get("candidates", [])
                        if not row.get("id") or not candidates or {c.get("source") for c in candidates} != {"fast", "slow"}:
                            raise ValueError(f"Bank group requires an ID and both candidate sources: {path}")
                        if any(not isinstance(c.get("response"), str) for c in candidates):
                            raise ValueError(f"Bank candidates require response text: {path}")
                    elif key == "history" and (type(row.get("step")) is not int or row["step"] < 0):
                        raise ValueError(f"History requires nonnegative integer steps: {path}")
                    elif stage == "prepare":
                        from .data import _validate_example

                        _validate_example(row)
                        for image in row["images"]:
                            resolved = Path(image) if Path(image).is_absolute() else path.parent / image
                            if not resolved.is_file():
                                raise ValueError(f"Prepared image file not found: {resolved}")
        elif key != "status":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Referenced output {key} requires a positive finite number")
    for key in ("plots", "images"):
        for value in result.get(key, []):
            _file(value, key)
    if stage == "evaluate":
        metrics = _json(Path(result["metrics"]))
        if benchmark_metrics([metrics])["status"] != "measured":
            raise ValueError("Evaluation metrics are incomplete")
        totals = {key: 0.0 for key in ("score", "length", "slow_ratio")}
        n = 0
        for row in _records(Path(result["predictions"])):
            measured = benchmark_metrics([row])
            if measured["status"] != "measured" or measured["n"] != 1:
                raise ValueError("Evaluation prediction lacks valid response metrics")
            n += 1
            for key in totals:
                totals[key] += measured[key]
        if metrics["n"] != n:
            raise ValueError("Evaluation sample count does not match predictions")
        if any(not math.isclose(metrics[key], totals[key] / n, rel_tol=1e-9, abs_tol=1e-12) for key in totals):
            raise ValueError("Evaluation metrics disagree with predictions")
    if stage in {"bank", "score", "weights"}:
        manifest = _json(_file(str(Path(result["bank"]).parent / "bank-manifest.json"), "bank manifest"))
        if not manifest.get("fingerprint") or not isinstance(manifest.get("identity"), dict):
            raise ValueError("Bank manifest requires an identity and fingerprint")
    if stage == "diagnostics":
        kind = cfg.get("diagnostics", {}).get("kind", "cross_likelihood")
        path = Path(result["records"])
        if kind in {"coverage", "oracle_coverage"}:
            observed = _json(path) if path.suffix == ".json" else next(_records(path))
            if (type(observed.get("prompts")) is not int or observed["prompts"] <= 0
                    or type(observed.get("candidates")) is not int or observed["candidates"] <= 0
                    or not isinstance(observed.get("oracle_coverage"), (int, float))
                    or not 0 <= observed["oracle_coverage"] <= 1):
                raise ValueError("Coverage diagnostic contains no valid measured counts")
        else:
            rows = list(_records(path))
            if kind in {"calibration", "routing_calibration"}:
                observed = calibration_metrics(rows)
                if not observed["n_prompts"] or observed["rejected_rows"]:
                    raise ValueError("Calibration diagnostic contains invalid observations")
            elif kind == "cross_likelihood":
                observed = cross_likelihood_metrics(rows)
                if not observed["n"] or observed["excluded_rows"] or any(not row["n"] for row in observed["mechanisms"]):
                    raise ValueError("Likelihood diagnostic requires valid observations from both sources")
    if stage == "analyze":
        summary = _json(Path(result["summary"]))
        if summary.get("report_version") != REPORT_VERSION or result.get("report_version") != REPORT_VERSION:
            raise ValueError("Report format changed; regenerate this analysis")
        if summary.get("kind") != cfg.get("analysis", {}).get("kind", "benchmark_table") or summary.get("status") != status:
            raise ValueError("Report kind or status disagrees with its stage result")
        count = summary.get("valid_observations")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count <= 0:
            raise ValueError("Report contains no valid observations")
        if status == "complete" and summary.get("missing"):
            raise ValueError("Complete report contains unresolved source errors")
        def incomplete(value):
            if isinstance(value, dict):
                return (value.get("status") is not None and value["status"] not in SUCCESS
                        or any(incomplete(item) for key, item in value.items() if key not in {"source_artifacts", "resolved_config"}))
            if isinstance(value, list):
                return any(incomplete(item) for item in value)
            return False
        if status == "complete" and incomplete(summary):
            raise ValueError("Complete report contains incomplete measurements")
        for key in ("plots",):
            for value in summary.get(key, []):
                _file(value, key)
        for case in summary.get("cases", []):
            for value in case.get("image_artifacts", []):
                _file(value, "case image")
        if status == "partial":
            return "partial"
    return "completed"
