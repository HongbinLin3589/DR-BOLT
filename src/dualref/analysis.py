"""Compute metrics and export figures and reports from experiment artifacts."""
from __future__ import annotations

import csv
import html
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPORT_VERSION = 2
REPORT_KINDS = {"benchmark_table", "sensitivity", "calibration", "cross_likelihood", "correction",
                "trajectory", "coverage", "difficulty", "qualitative", "resources", "parser_cases", "protocol", "report"}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _field(record: dict, *keys: str) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _load(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = next((value[key] for key in ("records", "predictions", "history", "rows") if isinstance(value.get(key), list)), [value])
        else:
            raise ValueError("Expected a JSON object/list or JSONL records")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("All artifact records must be JSON objects")
    expanded = []
    for row in rows:
        if isinstance(row.get("candidates"), list):
            context = {key: value for key, value in row.items() if key != "candidates"}
            for index, candidate in enumerate(row["candidates"]):
                expanded.append({**context, **candidate, "prompt_id": str(row["id"]), "sample_id": candidate.get("sample_id", f"{row['id']}:{index}")})
        else:
            expanded.append(row)
    return expanded


def _read_sources(settings: dict) -> tuple[dict[str, list[dict]], dict[str, dict], list[dict]]:
    data, metadata, missing = {}, {}, []
    for name, descriptor in settings.get("sources", {}).items():
        descriptor = {"path": descriptor} if isinstance(descriptor, str) else descriptor
        metadata[name] = descriptor
        path = descriptor.get("path")
        if not path or not Path(path).is_file():
            missing.append({"source": name, "path": path, "reason": "Source path is required" if not path else "Artifact does not exist"})
            continue
        try:
            if settings.get("kind") == "report":
                summary = json.loads(Path(path).read_text(encoding="utf-8"))
                if (not isinstance(summary, dict) or summary.get("kind") not in REPORT_KINDS
                        or summary.get("status") not in {"complete", "partial", "missing", "unavailable", "failed"}
                        or summary.get("report_version") != REPORT_VERSION):
                    raise ValueError("Expected a current report summary object with a recognized kind and status")
                data[name] = [summary]
            else:
                data[name] = _load(path)
            if not data[name]:
                missing.append({"source": name, "path": path, "reason": "Artifact contains no records"})
        except (ValueError, OSError) as error:
            missing.append({"source": name, "path": path, "reason": str(error)})
    return data, metadata, missing


def _id(row: dict) -> str:
    return str(_field(row, "prompt_id", "example_id", "input_id", "id") or row.get("example", {}).get("id", ""))


def _source(row: dict) -> str:
    value = str(_field(row, "source", "mode", "generation_source") or "").lower()
    return {"f": "fast", "s": "slow", "<fast>": "fast", "<slow>": "slow"}.get(value, value)


def _slow(row: dict) -> float | None:
    explicit = _field(row, "slow", "is_slow")
    if explicit is not None:
        if isinstance(explicit, str) and explicit.lower() in {"true", "false"}:
            return float(explicit.lower() == "true")
        return float(explicit) if isinstance(explicit, (int, float, bool)) and explicit in (0, 1) else None
    mode = str(row.get("mode", "")).lower()
    if mode in ("fast", "slow"):
        return float(mode == "slow")
    response = _field(row, "response", "text", "output")
    return float("<think>" in response.lower() or response.lstrip().startswith("<SLOW>")) if isinstance(response, str) else None


def _words(row: dict) -> float | None:
    count = _number(_field(row, "words", "word_count", "response_words", "length"))
    if count is not None:
        return count if count >= 0 else None
    response = _field(row, "response", "text", "output")
    return float(len(response.split())) if isinstance(response, str) else None


def benchmark_metrics(rows: list[dict]) -> dict:
    """Aggregate responses, weighting every response equally across benchmarks."""
    summaries = ["score" in row and "correct" not in row for row in rows]
    if any(summaries):
        if not all(summaries):
            raise ValueError("Cannot combine metric summaries with response records")
        valid = [row for row in rows if not isinstance(row.get("n"), bool) and _number(row.get("n")) is not None
                 and row["n"] > 0 and float(row["n"]).is_integer()
                 and all(_number(row.get(key)) is not None for key in ("score", "length", "slow_ratio"))
                 and 0 <= row["score"] <= 1 and row["length"] >= 0 and 0 <= row["slow_ratio"] <= 1
                 and row.get("status", "measured") in {"measured", "complete"}]
        n = sum(row["n"] for row in valid)
        return {"n": n, **{key: sum(row[key] * row["n"] for row in valid) / n if n else None
                           for key in ("score", "length", "slow_ratio")},
                "status": "measured" if len(valid) == len(rows) and n else "incomplete",
                "rejected_summaries": len(rows) - len(valid)}
    correct = [_number(int(value) if isinstance(value, bool) else value)
               for row in rows for value in [_field(row, "correct", "accuracy", "r_acc")]]
    words = [_words(row) for row in rows]
    slow = [_slow(row) for row in rows]
    correct = [value if value is not None and 0 <= value <= 1 else None for value in correct]
    words = [value if value is not None and value >= 0 else None for value in words]
    values = {"score": correct, "length": words, "slow_ratio": slow}
    result = {name: _mean([v for v in items if v is not None]) for name, items in values.items()}
    result.update({"n": len(rows), "status": "measured" if rows and all(all(v is not None for v in items) for items in values.values()) else "incomplete", "observed_counts": {name: sum(v is not None for v in items) for name, items in values.items()}})
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy) if xx > 0 and yy > 0 else None


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + end - 1) / 2 + 1
        start = end
    return result


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))


def calibration_metrics(rows: list[dict], bins: int = 5) -> dict:
    grouped: dict[str, dict] = {}
    rejected = 0
    for index, row in enumerate(rows):
        probability = _number(_field(row, "predicted_fast_probability", "predicted_fast", "p_fast"))
        if probability is None and "log_z_fast" in row and "log_z_slow" in row:
            alpha = row.get("alpha", .5)
            a, b = math.log(alpha) + row["log_z_fast"], math.log1p(-alpha) + row["log_z_slow"]
            probability = math.exp(a - _logadd(a, b))
        if probability is None or not 0 <= probability <= 1:
            rejected += 1
            continue
        identifier = _id(row) or str(index)
        point = grouped.setdefault(identifier, {"prompt_id": identifier, "predicted": probability, "fast": 0., "total": 0.})
        if not math.isclose(point["predicted"], probability, abs_tol=1e-10):
            raise ValueError(f"Conflicting predicted probabilities for prompt {identifier}")
        if "fast_count" in row and _field(row, "total", "repeats", "n") is not None:
            fast, total = float(row["fast_count"]), float(_field(row, "total", "repeats", "n"))
        elif _slow(row) is not None:
            fast, total = 1. - _slow(row), 1.
        elif "empirical_fast_rate" in row:
            count = _number(_field(row, "total", "repeats", "n"))
            if count is None:
                raise ValueError("Empirical fast rates require an observed total, repeats, or n")
            total = count
            fast = float(row["empirical_fast_rate"]) * total
        else:
            rejected += 1
            continue
        if not math.isfinite(total) or not math.isfinite(fast) or total <= 0 or not total.is_integer() or not 0 <= fast <= total:
            raise ValueError("Invalid calibration fast count or trial count")
        point["fast"] += fast
        point["total"] += total
    points = [{**point, "observed": point["fast"] / point["total"]} for point in grouped.values() if point["total"] > 0]
    points.sort(key=lambda point: (point["predicted"], point["prompt_id"]))
    grouped_bins = []
    bin_count = min(max(1, bins), len(points))
    if bin_count:
        size, remainder = divmod(len(points), bin_count)
        offset = 0
        for index in range(bin_count):
            subset = points[offset:offset + size + (index < remainder)]
            offset += len(subset)
            grouped_bins.append({"bin": index + 1, "n": len(subset), "min_predicted": subset[0]["predicted"], "max_predicted": subset[-1]["predicted"], "predicted": statistics.fmean(point["predicted"] for point in subset), "observed": statistics.fmean(point["observed"] for point in subset)})
    xs, ys = [point["predicted"] for point in points], [point["observed"] for point in points]
    return {"n_prompts": len(points), "n_decodes": sum(point["total"] for point in points), "pearson": pearson(xs, ys), "spearman": pearson(_ranks(xs), _ranks(ys)), "mae": _mean([abs(x - y) for x, y in zip(xs, ys)]), "grouped_ece": sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in grouped_bins) / len(points) if points else None, "slow_ratio": 1. - sum(point["fast"] for point in points) / sum(point["total"] for point in points) if points else None, "rejected_rows": rejected, "bins": grouped_bins, "points": points}


def cross_likelihood_metrics(rows: list[dict]) -> dict:
    normalized = []
    for row in rows:
        fast = _number(_field(row, "logp_fast", "log_prob_fast", "fast_logprob"))
        slow = _number(_field(row, "logp_slow", "log_prob_slow", "slow_logprob"))
        tokens = _number(_field(row, "response_tokens", "token_count", "tokens"))
        source = _source(row)
        if fast is not None and slow is not None and tokens is not None and tokens > 0 and source in ("fast", "slow"):
            normalized.append({"source": source, "fast": fast, "slow": slow, "tokens": tokens, "margin": (fast - slow) if source == "fast" else (slow - fast), "sample_id": str(_field(row, "sample_id", "id") or "")})
    matrix, mechanisms = [], []
    for source in ("fast", "slow"):
        subset = [row for row in normalized if row["source"] == source]
        for policy in ("fast", "slow"):
            matrix.append({"source": source, "policy": policy, "n": len(subset), "sequence_log_likelihood": _mean([row[policy] for row in subset]), "per_token_log_likelihood": _mean([row[policy] / row["tokens"] for row in subset])})
        margins = [row["margin"] for row in subset]
        mechanisms.append({"source": source, "n": len(subset), "mean_tokens": _mean([row["tokens"] for row in subset]), "margin_mean": _mean(margins), "margin_std": statistics.pstdev(margins) if margins else None, "margin_min": min(margins) if margins else None, "margin_max": max(margins) if margins else None, "length_cross_logratio_pearson": pearson([row["tokens"] for row in subset], [-row["margin"] for row in subset]), "nonseparated_fraction": _mean([float(value <= 0) for value in margins])})
    return {"n": len(normalized), "excluded_rows": len(rows) - len(normalized), "matrix": matrix, "mechanisms": mechanisms, "records": normalized}


def correction_metrics(rows: list[dict], alpha: float = .5, q: float = .7) -> dict:
    if not 0 < alpha < 1 or not 0 < q < 1:
        raise ValueError("alpha and q must lie strictly between zero and one")
    normalized = cross_likelihood_metrics(rows)["records"]
    result = []
    for row in normalized:
        fast, slow = row["fast"], row["slow"]
        exact = _logadd(math.log(alpha) + fast, math.log1p(-alpha) + slow) - _logadd(math.log(q) + fast, math.log1p(-q) + slow)
        theory = math.log(alpha / q) if row["source"] == "fast" else math.log((1 - alpha) / (1 - q))
        log_mu = _logadd(math.log(q) + fast, math.log1p(-q) + slow)
        result.append({"sample_id": row["sample_id"], "source": row["source"], "log_exact": exact, "log_modewise": theory, "log_single_ratio": fast - slow, "log_single_correction": fast - log_mu, "logp_fast": fast, "logp_slow": slow, "error": theory - exact, "margin": row["margin"]})
    errors = [abs(row["error"]) for row in result]
    singles = []
    for source in ("fast", "slow"):
        values = [row["log_single_ratio"] for row in result if row["source"] == source]
        singles.append({"source": source, "n": len(values), "mean": _mean(values), "std": statistics.pstdev(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None, "fraction_below_exp_minus_100": _mean([float(v < -100) for v in values]), "fraction_below_exp_minus_200": _mean([float(v < -200) for v in values]), "estimated_bf16_underflow_count": sum(v < -134 * math.log(2) for v in values), "estimated_float64_underflow_count": sum(v < -1075 * math.log(2) for v in values)})
    return {"alpha": alpha, "q": q, "n": len(result), "log_weight_pearson": pearson([r["log_modewise"] for r in result], [r["log_exact"] for r in result]), "correlation_note": "Undefined when both weights are constant (alpha equals q)" if alpha == q else None, "fraction_absolute_log_error_above_001": _mean([float(e > .01) for e in errors]), "maximum_absolute_log_error": max(errors) if errors else None, "single_reference": singles, "records": result, "underflow_note": "single_reference.estimated_* fields use analytical round-to-nearest thresholds including subnormals. Actual dtype/device exponentiation is recorded separately by the analysis stage in underflow_measurement."}


def _measure_underflow(records: list[dict], settings: dict) -> dict:
    """Measure exponentiation in actual dtypes without importing torch in pure metrics."""
    requested_device = str(settings.get("underflow_device") or settings.get("device") or "cpu")
    dtype_names = settings.get("underflow_dtypes", ["bfloat16", "float32", "float64"])
    allowed = {"bfloat16", "float16", "float32", "float64"}
    if not isinstance(dtype_names, list) or not dtype_names or any(name not in allowed for name in dtype_names):
        raise ValueError("analysis.underflow_dtypes must list supported floating-point dtype names")
    measurement = {"requested_device": requested_device, "operation": "torch.exp(torch.tensor(log_values, dtype=dtype, device=device))", "definitions": {"log_single_ratio": "log pi_fast - log pi_slow", "log_single_correction": "log pi_fast - log(q*pi_fast + (1-q)*pi_slow)", "logp_fast": "Raw fast-reference sequence log probability", "logp_slow": "Raw slow-reference sequence log probability"}, "rows": []}
    if not records:
        return {**measurement, "status": "missing", "reason": "No valid likelihood observations"}
    try:
        import torch
    except (ImportError, OSError) as error:
        return {**measurement, "status": "unavailable", "error": str(error)}
    measurement["torch_version"] = str(torch.__version__)
    for dtype_name in dtype_names:
        for source in ("fast", "slow"):
            selected = [row for row in records if row["source"] == source]
            if not selected:
                continue
            for field in measurement["definitions"]:
                row = {"dtype": dtype_name, "source": source, "quantity": field, "n": len(selected), "device": requested_device}
                try:
                    with torch.inference_mode():
                        values = torch.tensor([record[field] for record in selected], dtype=getattr(torch, dtype_name), device=requested_device)
                        exponentials = torch.exp(values)
                        finite_input = torch.isfinite(values)
                        zeros = exponentials == 0
                        positive_finite = exponentials[(exponentials > 0) & torch.isfinite(exponentials)]
                        row.update({"status": "measured", "device": str(exponentials.device), "output_dtype": str(exponentials.dtype), "zero_count": int(zeros.sum().item()), "zero_fraction": float(zeros.float().mean().item()), "finite_input_underflow_count": int((zeros & finite_input).sum().item()), "nonfinite_input_count": int((~finite_input).sum().item()), "infinite_output_count": int(torch.isinf(exponentials).sum().item()), "nan_output_count": int(torch.isnan(exponentials).sum().item()), "minimum_positive_output": float(positive_finite.min().item()) if positive_finite.numel() else None})
                except (RuntimeError, TypeError, ValueError) as error:
                    row.update({"status": "unavailable", "error": str(error)})
                measurement["rows"].append(row)
    measurement["status"] = "measured" if measurement["rows"] and all(row["status"] == "measured" for row in measurement["rows"]) else "partial"
    return measurement


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def _trajectory(data: dict[str, list[dict]]) -> dict:
    rows, points = [], {}
    for name, history in data.items():
        values = sorted([row for row in history if _number(row.get("step")) is not None], key=lambda row: row["step"])
        points[name] = values
        gradients = [(row["step"], _number(_field(row, "grad_norm", "gradient_norm"))) for row in values]
        gradients = [(step, value) for step, value in gradients if value is not None]
        auc = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(gradients, gradients[1:]))
        row = {"method": name, "n_points": len(values)}
        losses = [_number(_field(value, "loss", "train_loss")) for value in values]
        losses = [value for value in losses if value is not None]
        if losses:
            row["final_loss"] = losses[-1]
        if gradients:
            row["gradient_peak"] = max(value for _, value in gradients)
        if len(gradients) >= 2:
            row["gradient_auc"] = auc
        for step in (25, 50, 100, 500):
            value = next((_number(_field(point, "implicit_kl", "kl", "sequence_kl")) for point in values if point["step"] == step), None)
            if value is not None:
                row[f"implicit_kl_step_{step}"] = value
        rows.append(row)
    comparison = {}
    if "mode_wise" in points and "exact" in points:
        a = {row["step"]: _field(row, "implicit_kl", "kl", "sequence_kl") for row in points["mode_wise"]}
        b = {row["step"]: _field(row, "implicit_kl", "kl", "sequence_kl") for row in points["exact"]}
        differences = [abs(a[step] - b[step]) for step in a.keys() & b.keys() if a[step] is not None and b[step] is not None]
        if differences:
            comparison["maximum_stepwise_implicit_kl_difference"] = max(differences)
        areas = {row["method"]: row["gradient_auc"] for row in rows if "gradient_auc" in row}
        if areas.get("exact") and areas.get("mode_wise") is not None:
            comparison["relative_gradient_auc_difference"] = abs(areas["mode_wise"] - areas["exact"]) / abs(areas["exact"])
    return {"rows": rows, "points": points, "comparison": comparison}


def _coverage(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    missing_ids = 0
    for row in rows:
        identifier = _id(row)
        if identifier:
            grouped[identifier].append(row)
        else:
            missing_ids += 1
    observed = []
    for identifier, candidates in grouped.items():
        correctness = [_number(_field(row, "correct", "r_acc")) for row in candidates]
        known = [value for value in correctness if value is not None]
        observed.append({"prompt_id": identifier, "candidates": len(candidates), "correct_candidates": sum(value > 0 for value in known), "complete": len(known) == len(candidates), "covered": any(value > 0 for value in known) if known else None})
    complete = [row for row in observed if row["complete"]]
    return {"n_inputs": len(grouped), "n_complete_inputs": len(complete), "oracle_coverage": _mean([float(row["covered"]) for row in complete]), "missing_id_rows": missing_ids, "rows": observed}


def _benchmark_table(data: dict, metadata: dict, settings: dict) -> dict:
    rows, transfer = [], defaultdict(list)
    for name, values in data.items():
        meta = metadata[name]
        if meta.get("kind") == "calibration":
            continue
        dataset = meta.get("dataset", name.split("/")[0])
        method = meta.get("method", name.split("/")[-1])
        rows.append({"dataset": dataset, "method": method, **benchmark_metrics(values)})
        if dataset in ("mathvision", "dynamath", "mmmu_pro", "mmmu-pro"):
            transfer[method].extend(values)
    aggregate = [{"method": method, **benchmark_metrics(values)} for method, values in transfer.items()]
    for row in aggregate:
        row["slow_ratio_percent"] = 100 * row["slow_ratio"] if row["slow_ratio"] is not None else None
    return {"rows": rows, "transfer_aggregate": aggregate, "aggregation": "Each generated response has equal weight; ScienceQA is excluded from transfer slow ratio"}


def _difficulty(data: dict, metadata: dict) -> dict:
    rates = defaultdict(list)
    for name, values in data.items():
        if metadata[name].get("kind") == "bank":
            for row in values:
                correct = _number(_field(row, "correct", "r_acc"))
                if _source(row) == "fast" and correct is not None:
                    rates[_id(row)].append(correct)
    order = sorted(rates, key=lambda identifier: (-statistics.fmean(rates[identifier]), identifier))
    difficulty_by_id = {}
    size, remainder = divmod(len(order), 3)
    offset = 0
    for index, label in enumerate(("easy", "medium", "hard")):
        subset = order[offset:offset + size + (index < remainder)]
        difficulty_by_id.update({identifier: label for identifier in subset})
        offset += len(subset)
    grouped = defaultdict(list)
    missing = 0
    for name, values in data.items():
        method = metadata[name].get("method", name)
        if metadata[name].get("kind") == "bank":
            continue
        for row in values:
            difficulty = row.get("difficulty")
            identifier = _id(row)
            if difficulty is None:
                difficulty = difficulty_by_id.get(identifier)
            words = _words(row)
            if difficulty not in ("easy", "medium", "hard") or words is None:
                missing += 1
                continue
            grouped[(method, difficulty)].append(words)
    return {"rows": [{"method": method, "difficulty": difficulty, "n": len(values), "mean": _mean(values), "q25": _quantile(values, .25), "median": _quantile(values, .5), "q75": _quantile(values, .75), "min": min(values), "max": max(values)} for (method, difficulty), values in grouped.items()], "distributions": {f"{method}/{difficulty}": values for (method, difficulty), values in grouped.items()}, "assignments": [{"prompt_id": identifier, "fast_correctness_rate": statistics.fmean(rates[identifier]), "difficulty": difficulty_by_id[identifier]} for identifier in order], "missing_difficulty_rows": missing, "definition": "Equal-frequency thirds by fast-candidate correctness rate, descending; ties break by stable prompt ID. Supplied difficulty labels take precedence."}


def _qualitative(data: dict, metadata: dict, settings: dict) -> dict:
    def normalize(value: Any) -> str:
        return " ".join(str(value).casefold().split()).strip(". ")

    def matches(dataset: str, row: dict) -> bool:
        question = normalize(row.get("question", ""))
        answer = normalize(_field(row, "answer", "ground_truth", "target") or "")
        choices = normalize(" ".join(str(choice) for choice in row.get("choices", [])))
        if dataset == "virl39k":
            return all(term in question for term in ("clocks", "lily", "tuesday")) and answer == "b"
        if dataset == "mathvision":
            return "which number should be written in place of the question mark" in question and answer in ("60", "60.0")
        if dataset == "dynamath":
            return "cape town" in question and "hottest" in question and answer in ("sep", "september")
        return "symbol" in question and "time waits for no man" in choices and answer == "h"

    cases = []
    ids = settings.get("case_ids", {})
    datasets = sorted({meta.get("dataset", name.split("/")[0]) for name, meta in metadata.items()} | set(ids))
    for dataset in datasets:
        values = [{**row.get("example", {}), **row, "artifact_path": metadata[name].get("path")} for name, rows in data.items() if metadata[name].get("dataset", name.split("/")[0]) == dataset for row in rows]
        requested = ids.get(dataset)
        selected = [row for row in values if _id(row) == str(requested)] if requested is not None else [row for row in values if matches(dataset, row)]
        matched_ids = sorted({_id(row) for row in selected})
        if matched_ids:
            selected = sorted([row for row in selected if _id(row) == matched_ids[0]], key=lambda row: (str(_field(row, "candidate_id", "sample_id") or ""), row.get("repeat", 0)))
        if dataset == "virl39k":
            selected = [next((row for row in selected if _source(row) == source), {}) for source in ("fast", "slow")]
            selected = [row for row in selected if row]
        else:
            selected = selected[:1]
        missing_fields = sorted({field for row in selected for field in ("question", "images", "choices", "answer", "response") if field not in row})
        asset_issues = []
        for row in selected:
            for image in row.get("images", []):
                if str(image).startswith(("https://", "http://")):
                    asset_issues.append(f"Remote image has not been verified: {image}")
                elif not str(image).startswith("data:image/") and not _image_path(row, image).is_file():
                    asset_issues.append(f"Image file not found: {image}")
        complete = bool(selected) and not missing_fields and not asset_issues and (dataset != "virl39k" or len(selected) == 2)
        cases.append({"dataset": dataset, "status": "measured" if complete else ("partial" if selected else "missing"), "requested_id": requested, "matched_ids": matched_ids, "selection": "explicit ID" if requested is not None else "Match question, answer and options; use the first stable ID when several match", "missing_fields": missing_fields, "asset_issues": asset_issues, "reason": None if complete else ("; ".join(asset_issues) if asset_issues else "No matching example in the supplied artifacts" if not selected else "Matched example lacks required fields or one teacher response"), "records": selected})
    return {"cases": cases}


def _image_path(context: dict, image: str) -> Path:
    source = Path(image)
    if not source.is_absolute() and not source.is_file():
        source = Path(context.get("artifact_path") or ".").parent / source
    return source


def _hardware_inventory(telemetry: list[dict]) -> list[dict]:
    """Avoid counting every visible GPU once for each distributed worker."""
    by_rank = {}
    for item in telemetry:
        devices = item.get("devices", [])
        rank = item.get("rank", 0)
        if len(devices) > len(by_rank.get(rank, [])):
            by_rank[rank] = devices
    inventories = list(by_rank.values())
    if not inventories:
        return []
    if all(device.get("uuid") for devices in inventories for device in devices):
        return list({device["uuid"]: device for devices in inventories for device in devices}.values())
    if all(len(devices) == 1 for devices in inventories):
        # With one visible device per rank, device 0 denotes different GPUs.
        return [devices[0] for devices in inventories]
    # With a shared visible-device namespace, every rank reports the same inventory.
    return max(inventories, key=len)


def _resources(data: dict, metadata: dict) -> dict:
    rows = []
    for name, values in data.items():
        for value in values:
            outputs = value.get("outputs", {})
            reported_seconds = _number(outputs.get("seconds"))
            seconds = reported_seconds if reported_seconds is not None else _number(value.get("elapsed_seconds"))
            directory = Path(metadata[name]["path"]).parent
            telemetry = []
            for path in sorted(directory.rglob("telemetry-*.json")):
                try:
                    telemetry.extend(_load(path))
                except (ValueError, OSError):
                    continue
            devices = [device for item in telemetry for device in item.get("devices", [])]
            allocated = [_number(device.get("allocated_gb")) for device in devices]
            reserved = [_number(device.get("reserved_gb")) for device in devices]
            reported_memory = str(outputs.get("memory_measurement", outputs.get("memory_metric", "")))
            generic_bytes = _number(outputs.get("peak_gpu_bytes"))
            allocated_bytes = _number(outputs.get("peak_gpu_allocated_bytes"))
            reserved_bytes = _number(outputs.get("peak_gpu_reserved_bytes"))
            nvml_bytes = _number(outputs.get("peak_gpu_nvml_bytes"))
            if generic_bytes is not None:
                if "nvml" in reported_memory.casefold():
                    nvml_bytes = generic_bytes if nvml_bytes is None else max(nvml_bytes, generic_bytes)
                elif "reserved" in reported_memory.casefold():
                    reserved_bytes = generic_bytes if reserved_bytes is None else max(reserved_bytes, generic_bytes)
                elif "allocated" in reported_memory.casefold():
                    allocated_bytes = generic_bytes if allocated_bytes is None else max(allocated_bytes, generic_bytes)
            allocated = [value for value in allocated if value is not None] + ([allocated_bytes / 1e9] if allocated_bytes is not None else [])
            reserved = [value for value in reserved if value is not None] + ([reserved_bytes / 1e9] if reserved_bytes is not None else [])
            peak_allocated = max(allocated) if allocated else None
            peak_reserved = max(reserved) if reserved else None
            peak_nvml = nvml_bytes / 1e9 if nvml_bytes is not None else None
            if peak_allocated is not None:
                measured_memory, memory_metric = peak_allocated, "torch.cuda.max_memory_allocated"
            elif peak_reserved is not None:
                measured_memory, memory_metric = peak_reserved, "torch.cuda.max_memory_reserved"
            elif peak_nvml is not None:
                measured_memory = peak_nvml
                memory_metric = "NVML process-tree resident memory" if "process-tree" in reported_memory.casefold() else "NVML reported GPU memory"
            else:
                measured_memory = _number(_field(outputs, "peak_gpu_memory_gb", "peak_memory_gb"))
                if measured_memory is None and generic_bytes is not None:
                    measured_memory = generic_bytes / 1e9
                memory_metric = outputs.get("memory_metric", "reported_unspecified") if measured_memory is not None else None
            inventory = _hardware_inventory(telemetry)
            hardware = {"gpu_count": len(inventory), "device_names": sorted(device.get("name", "unknown") for device in inventory)} if inventory else outputs.get("hardware")
            rows.append({"job": name, "stage": value.get("stage", metadata[name].get("stage")), "status": value.get("status"), "seconds": seconds, "seconds_source": "stage report excluding diagnostic evaluation" if reported_seconds is not None else "suite wall-clock measurement", "wall_seconds_including_evaluation": _number(outputs.get("wall_seconds_including_evaluation")), "suite_elapsed_seconds": _number(value.get("elapsed_seconds")), "hours": seconds / 3600 if seconds is not None else None, "peak_gpu_memory_gb": measured_memory, "memory_metric": memory_metric, "memory_measurement_source": reported_memory or None, "peak_gpu_allocated_gb": peak_allocated, "peak_gpu_reserved_gb": peak_reserved, "peak_gpu_nvml_gb": peak_nvml, "hardware": hardware, "telemetry": telemetry})
    by_id = {row["job"]: row for row in rows}
    comparisons = []
    internal = ("mix_sft", "reward_sft", "single_ref_fast", "single_ref_slow", "dr_bolt", "mr_dpo", "cond_rw")
    methods = (*internal, "grpo", "fast_grpo", "ton", "dualmind")
    datasets = {name[6:-(len(method) + 1)] for name in metadata for method in methods
                if name.startswith("train_") and name.endswith("_" + method)}
    for dataset in sorted(datasets):
        for method in methods:
            identifiers = [f"train_{dataset}_{method}"]
            if identifiers[0] not in metadata:
                continue
            if method in internal:
                identifiers = [f"bank_{dataset}", f"score_{dataset}", f"warmup_{dataset}", f"weights_{dataset}_{method}", *identifiers]
                if method in ("single_ref_fast", "single_ref_slow", "mr_dpo"):
                    identifiers.append(f"references_score_{dataset}")
            values = [by_id[identifier] for identifier in identifiers if identifier in by_id]
            complete = len(values) == len(identifiers) and all(value["status"] == "completed" and value["hours"] is not None for value in values)
            memory = [value["peak_gpu_memory_gb"] for value in values if value["peak_gpu_memory_gb"] is not None]
            memory_metrics = {value["memory_metric"] for value in values if value["peak_gpu_memory_gb"] is not None}
            signatures = {json.dumps(value["hardware"], sort_keys=True) for value in values if value["hardware"]}
            comparisons.append({"dataset": dataset, "method": method, "status": "measured" if complete else "missing", "end_to_end_hours": sum(value["hours"] for value in values) if complete else None, "peak_gpu_memory_gb": max(memory) if memory and len(memory_metrics) == 1 else None, "memory_metric": next(iter(memory_metrics)) if len(memory_metrics) == 1 else ("mixed; see stage measurements" if memory_metrics else None), "hardware_signature": next(iter(signatures)) if len(signatures) == 1 else None, "included_jobs": identifiers})
    for row in comparisons:
        baseline = next((value for value in comparisons if value["dataset"] == row["dataset"] and value["method"] == "grpo"), None)
        if baseline is not None:
            row["speedup"] = baseline["end_to_end_hours"] / row["end_to_end_hours"] if baseline["end_to_end_hours"] is not None and row["end_to_end_hours"] and baseline["hardware_signature"] and row["hardware_signature"] == baseline["hardware_signature"] else None
    return {"rows": rows, "comparisons": comparisons, "total_measured_hours": sum(row["hours"] for row in rows if row["hours"] is not None), "note": "Method totals include shared bank generation, reward computation, warm-up, weights and optimization. Reported training seconds exclude diagnostic evaluation when supplied; raw wall times remain available. Memory uses decimal GB (bytes/1e9), preferring peak per-device PyTorch allocated memory. Reserved and NVML memory remain separate; unlike metrics are never merged into a method peak. Speedup requires matching recorded hardware."}


def _parser_cases(data: dict, metadata: dict) -> dict:
    selected = []
    for name, values in data.items():
        dataset = metadata[name].get("dataset", name.split("/")[0])
        for category in ("correct", "incorrect", "parse_failure"):
            candidates = [row for row in values if ("correct" if row.get("correct") else ("incorrect" if row.get("format") else "parse_failure")) == category]
            if candidates:
                row = min(candidates, key=_id)
                response = str(row.get("response", ""))
                selected.append({"dataset": dataset, "prompt_id": _id(row), "category": category, "answer_span": response.rsplit("</think>", 1)[-1].strip(), "prediction": row.get("prediction"), "correct": row.get("correct"), "format": row.get("format"), "response": response})
    return {"rows": selected, "selection": "One observed response per benchmark and parsing outcome, selected by stable ID; absent outcomes are not synthesized"}


PALETTE = ("#145da0", "#e07a28", "#36845b", "#a5448f", "#7b6c42", "#666666")


def _plots(directory: Path, kind: str, result: dict) -> list[str]:
    plot_kinds = {"calibration", "trajectory", "cross_likelihood", "correction", "difficulty", "benchmark_table", "sensitivity"}
    if kind not in plot_kinds:
        return []
    measurements = result.get("rows") or result.get("records") or result.get("points") or result.get("calibration")
    if kind == "cross_likelihood":
        measurements = result.get("n", 0)
    if not measurements:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    paths = []

    def save(figure, name: str) -> None:
        safe_name = name.replace("/", "_")
        figure.tight_layout()
        for extension in ("svg", "pdf"):
            path = directory / f"{safe_name}.{extension}"
            figure.savefig(path, bbox_inches="tight")
            paths.append(str(path))
        plt.close(figure)

    def legend(axis) -> None:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.))

    def chart(name: str, series: dict, title: str, xlabel: str, ylabel: str, scatter: bool = False) -> None:
        series = {name: [(float(x), float(y)) for x, y in values if _number(x) is not None and _number(y) is not None] for name, values in series.items()}
        series = {name: values for name, values in series.items() if values}
        if not series:
            return
        figure, axis = plt.subplots(figsize=(8, max(4, len(series) * .18)))
        for index, (label, values) in enumerate(series.items()):
            xs, ys = zip(*values)
            color = PALETTE[index % len(PALETTE)]
            if scatter:
                axis.scatter(xs, ys, s=18, alpha=.65, color=color, label=label)
            else:
                axis.plot(xs, ys, linewidth=1.6, marker="o" if len(values) < 30 else None, markersize=3, color=color, label=label)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
        legend(axis)
        save(figure, name)

    def density(name: str, series: dict[str, list[float]], title: str, xlabel: str) -> None:
        from scipy.stats import gaussian_kde

        series = {key: values for key, values in series.items() if values}
        if not series:
            return
        figure, axis = plt.subplots(figsize=(8, 4))
        for index, (label, values) in enumerate(series.items()):
            color = PALETTE[index % len(PALETTE)]
            if len(values) < 2 or max(values) - min(values) <= 1e-12 * max(1., max(abs(value) for value in values)):
                axis.axvline(values[0], color=color, label=f"{label} (constant)")
            else:
                bandwidth = statistics.pstdev(values) * .3
                xs = np.linspace(min(values) - bandwidth, max(values) + bandwidth, 256)
                axis.plot(xs, gaussian_kde(values)(xs), color=color, label=label)
        axis.set(title=title, xlabel=xlabel, ylabel="Density")
        axis.grid(alpha=.2)
        legend(axis)
        save(figure, name)

    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        if kind == "calibration":
            for name, value in result.get("calibration", {}).items():
                bins = value["bins"]
                if not bins:
                    continue
                figure, axis = plt.subplots(figsize=(8, 4.5))
                xs = np.arange(len(bins))
                axis.bar(xs, [item["observed"] for item in bins], color=PALETTE[0], alpha=.7, label="Empirical fast rate")
                axis.plot(xs, [item["predicted"] for item in bins], color=PALETTE[1], marker="o", linestyle="none", markersize=7, label="Mean predicted fast probability")
                axis.set_xticks(xs, [f"{item['min_predicted']:.2f}–{item['max_predicted']:.2f}" for item in bins])
                axis.set(title="Partition-based routing calibration", xlabel="Predicted fast-probability groups", ylabel="Fast probability / rate", ylim=(0, 1))
                statistics_text = "; ".join(f"{key}: {value[key]:.3f}" for key in ("pearson", "spearman", "mae", "grouped_ece") if value.get(key) is not None)
                figure.text(.12, -.01, statistics_text, fontsize=9)
                legend(axis)
                save(figure, f"calibration_{name}")
        elif kind == "trajectory":
            fields = (("implicit_kl", ("implicit_kl", "kl", "sequence_kl")), ("loss", ("loss", "train_loss")), ("grad_norm", ("grad_norm", "gradient_norm")), ("score", ("score", "accuracy")), ("length", ("length", "mean_words")))
            for field, aliases in fields:
                series = {name: [(row["step"], _field(row, *aliases)) for row in values if _field(row, *aliases) is not None] for name, values in result["points"].items()}
                chart(field, series, field.replace("_", " ").title(), "Optimizer steps", field.replace("_", " "))
            if any(any(row.get("score") is not None and row.get("length") is not None for row in values) for values in result["points"].values()):
                figure, accuracy_axis = plt.subplots(figsize=(8, 4.5))
                length_axis = accuracy_axis.twinx()
                for index, (name, values) in enumerate(result["points"].items()):
                    color = PALETTE[index % len(PALETTE)]
                    for axis, key, style in ((accuracy_axis, "score", "-"), (length_axis, "length", "--")):
                        points = [(row["step"], row[key]) for row in values if _number(row.get(key)) is not None]
                        if points:
                            xs, ys = zip(*points)
                            axis.plot(xs, ys, color=color, linestyle=style, label=f"{name}: {key}")
                accuracy_axis.set(title="ScienceQA score–length dynamics", xlabel="Optimizer steps", ylabel="Score")
                length_axis.set_ylabel("Mean response words")
                handles_a, labels_a = accuracy_axis.get_legend_handles_labels()
                handles_b, labels_b = length_axis.get_legend_handles_labels()
                accuracy_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left", bbox_to_anchor=(1.12, 1.))
                accuracy_axis.grid(alpha=.2)
                save(figure, "score_length_dynamics")
        elif kind == "cross_likelihood":
            figure, axes = plt.subplots(1, 2, figsize=(10, 4))
            for axis, field in zip(axes, ("per_token_log_likelihood", "sequence_log_likelihood")):
                matrix = [[next((row[field] for row in result["matrix"] if row["source"] == source and row["policy"] == policy), None) for source in ("fast", "slow")] for policy in ("fast", "slow")]
                array = np.array([[np.nan if value is None else value for value in row] for row in matrix])
                image = axis.imshow(array, cmap="Blues")
                axis.set_xticks([0, 1], ["Fast response", "Slow response"])
                axis.set_yticks([0, 1], ["Fast policy", "Slow policy"])
                axis.set_title(field.replace("_", " "))
                for row_index, row in enumerate(matrix):
                    for column, value in enumerate(row):
                        axis.text(column, row_index, "N/A" if value is None else f"{value:.3f}", ha="center", va="center")
                figure.colorbar(image, ax=axis, label="Nats", shrink=.8)
            save(figure, "cross_likelihood_heatmaps")
        elif kind == "correction":
            rows = result["records"]
            chart("weight_scatter", {source: [(row["log_modewise"], row["log_exact"]) for row in rows if row["source"] == source] for source in ("fast", "slow")}, "Exact versus mode-wise correction", "Mode-wise log correction", "Exact log correction", scatter=True)
            density("weight_density", {key: [row[key] for row in rows] for key in ("log_exact", "log_modewise")}, "Correction log-weight density", "Log correction")
            density("single_reference_density", {source: [row["log_single_ratio"] for row in rows if row["source"] == source] for source in ("fast", "slow")}, "Single-reference log-ratio diagnostic", "Log likelihood ratio")
        elif kind == "difficulty":
            methods = sorted({row["method"] for row in result["rows"]})
            if methods:
                figure, axis = plt.subplots(figsize=(9, 4.5))
                distributions, positions, colors = [], [], []
                for group_index, difficulty in enumerate(("easy", "medium", "hard")):
                    for method_index, method in enumerate(methods):
                        values = result["distributions"].get(f"{method}/{difficulty}", [])
                        if values:
                            distributions.append(values)
                            positions.append(group_index * (len(methods) + 1) + method_index)
                            colors.append(PALETTE[method_index % len(PALETTE)])
                boxes = axis.boxplot(distributions, positions=positions, widths=.7, patch_artist=True, showfliers=False)
                for patch, color in zip(boxes["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(.6)
                axis.set_xticks([index * (len(methods) + 1) + (len(methods) - 1) / 2 for index in range(3)], ["Easy", "Medium", "Hard"])
                axis.set(title="Difficulty-wise response-length distributions", xlabel="Difficulty group", ylabel="Response words")
                axis.legend(handles=[Patch(facecolor=PALETTE[index % len(PALETTE)], alpha=.6, label=method) for index, method in enumerate(methods)], loc="upper left", bbox_to_anchor=(1.01, 1.))
                axis.grid(axis="y", alpha=.2)
                save(figure, "difficulty_length_distributions")
                for difficulty in ("easy", "medium", "hard"):
                    density(f"difficulty_{difficulty}_density", {key.split("/")[0]: values for key, values in result["distributions"].items() if key.endswith("/" + difficulty)}, f"Response length distributions: {difficulty}", "Response words")
        elif kind in ("benchmark_table", "sensitivity"):
            chart("score_length", {row["method"] + "/" + row["dataset"]: [(row["length"], row["score"])] for row in result["rows"] if row.get("length") is not None and row.get("score") is not None}, "Measured score-length trade-offs", "Mean response words", "Score", scatter=True)
    return paths


def _export_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key], (dict, list))))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _qualitative_html(result: dict, directory: Path) -> str:
    sections = []
    for number, case in enumerate(result["cases"], 1):
        parts = [f'<article><h2>Case {number}: {html.escape(case["dataset"])}</h2><p>Status: {case["status"]}</p>']
        if case.get("reason"):
            parts.append(f'<p>{html.escape(case["reason"])}</p>')
        if not case["records"]:
            sections.append("".join(parts) + "</article>")
            continue
        context = case["records"][0]
        parts.append(f'<p><strong>Example:</strong> {html.escape(_id(context))}</p>')
        parts.append(f'<p class="question">{html.escape(str(context.get("question", "")))}</p>')
        images = context.get("images", [])
        assets = []
        for index, image in enumerate(images):
            image = str(image)
            if image.startswith(("https://", "http://", "data:image/")):
                location = image
            else:
                source = _image_path(context, image)
                if not source.is_file():
                    parts.append(f'<p>Image file not found: {html.escape(image)}</p>')
                    continue
                destination = directory / "assets" / f"case-{number}-image-{index}{source.suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                location = destination.relative_to(directory).as_posix()
                assets.append(str(destination))
            parts.append(f'<figure><img class="case-image" src="{html.escape(location, quote=True)}" alt="Case {number}, source image {index + 1}"></figure>')
        case["image_artifacts"] = assets
        if not images:
            parts.append("<p>No image was present in the supplied artifact.</p>")
        choices = context.get("choices", [])
        if choices:
            parts.append('<ol type="A">' + "".join(f'<li>{html.escape(str(choice))}</li>' for choice in choices) + "</ol>")
        parts.append(f'<p><strong>Ground truth:</strong> {html.escape(str(context.get("answer", "")))}</p>')
        for row in case["records"]:
            label = _source(row) + " teacher" if case["dataset"] == "virl39k" else "DR-BOLT"
            parts.append(f'<h3>{html.escape(label)}</h3><pre>{html.escape(str(row.get("response", "")))}</pre>')
        parts.append("</article>")
        sections.append("".join(parts))
    return "".join(sections)


def _measurement_status(kind: str, result: dict, data: dict) -> tuple[str, int]:
    """Classify usable observations separately from readable input files."""
    partial = False
    if kind in {"benchmark_table", "sensitivity"}:
        rows = result["rows"]
        count = sum(row["n"] for row in rows if row.get("score") is not None)
        partial = any(row["status"] != "measured" for row in rows)
        if kind == "sensitivity":
            partial |= any(not value["n_prompts"] or value["rejected_rows"] for value in result.get("calibration", {}).values())
    elif kind == "calibration":
        values = list(result["calibration"].values())
        count = sum(value["n_prompts"] for value in values)
        partial = any(not value["n_prompts"] or value["rejected_rows"] for value in values)
    elif kind in {"cross_likelihood", "correction"}:
        count = result["n"]
        sources = {row["source"] for row in (result["records"] if kind == "correction" else result["mechanisms"]) if row.get("n", 1)}
        partial = sources != {"fast", "slow"} or count < sum(map(len, data.values()))
        if kind == "correction":
            partial |= result["underflow_measurement"]["status"] != "measured"
    elif kind == "coverage":
        count = result["n_complete_inputs"]
        partial = bool(result["missing_id_rows"] or count < result["n_inputs"])
    elif kind == "difficulty":
        count = sum(row["n"] for row in result["rows"])
        partial = bool(result["missing_difficulty_rows"])
    elif kind == "qualitative":
        count = sum(bool(case["records"]) for case in result["cases"])
        partial = any(case["status"] != "measured" for case in result["cases"])
    elif kind == "resources":
        count = sum(row["status"] == "completed" and row["seconds"] is not None for row in result["rows"])
        partial = count < len(result["rows"]) or any(row["status"] != "measured" for row in result["comparisons"])
        partial |= any(record.get("outputs", {}).get("memory_status") in {"unavailable", "partial"} for rows in data.values() for record in rows)
    elif kind == "trajectory":
        fields = ("loss", "train_loss", "score", "accuracy", "length", "grad_norm", "implicit_kl", "kl")
        valid = [sum(any(_number(row.get(key)) is not None for key in fields) for row in rows) for rows in result["points"].values()]
        count = sum(valid)
        partial = any(not value for value in valid)
    elif kind == "report":
        reports = list(result["reports"].values())
        count = sum(report.get("valid_observations", 0) for report in reports)
        partial = any(report["status"] != "complete" for report in reports)
    elif kind == "protocol":
        return "complete", 1
    else:
        count = len(result.get("rows", []))
    return ("missing" if not count else "partial" if partial else "complete"), count


def run_analysis(config: dict) -> dict:
    settings = config.get("analysis", {})
    kind = settings.get("kind", "benchmark_table")
    directory = Path(config.get("output_dir", "outputs/analysis"))
    directory.mkdir(parents=True, exist_ok=True)
    data, metadata, missing = _read_sources(settings)
    flattened = [row for rows in data.values() for row in rows]
    if kind in ("benchmark_table", "sensitivity"):
        result = _benchmark_table(data, metadata, settings)
        if kind == "sensitivity":
            calibrations = {metadata[name].get("method", name): calibration_metrics(rows, settings.get("bins", 5)) for name, rows in data.items() if metadata[name].get("kind") == "calibration"}
            result["calibration"] = calibrations
            for row in result["rows"]:
                calibration = calibrations.get(row["method"], {})
                if calibration.get("slow_ratio") is not None:
                    row["calibration_slow_ratio"] = calibration["slow_ratio"]
    elif kind == "calibration":
        result = {"calibration": {name: calibration_metrics(rows, settings.get("bins", 5)) for name, rows in data.items()}}
    elif kind == "cross_likelihood":
        result = cross_likelihood_metrics(flattened)
    elif kind == "correction":
        result = correction_metrics(flattened, settings.get("alpha", .5), settings.get("q", .7))
        result["underflow_measurement"] = _measure_underflow(result["records"], settings)
    elif kind == "trajectory":
        result = _trajectory(data)
    elif kind == "coverage":
        result = _coverage(flattened)
    elif kind == "difficulty":
        result = _difficulty(data, metadata)
    elif kind == "qualitative":
        result = _qualitative(data, metadata, settings)
    elif kind == "resources":
        result = _resources(data, metadata)
    elif kind == "parser_cases":
        result = _parser_cases(data, metadata)
    elif kind == "protocol":
        result = {"resolved_config": settings.get("resolved_config", config), "note": "Experiment configuration snapshot"}
    elif kind == "report":
        result = {"reports": {name: rows[0] for name, rows in data.items() if rows}}
    else:
        raise ValueError(f"Unknown analysis kind: {kind}")
    status, observations = _measurement_status(kind, result, data)
    if missing and status == "complete":
        status = "partial"
    result.update({"kind": kind, "status": status, "valid_observations": observations,
                   "report_version": REPORT_VERSION, "missing": missing, "source_artifacts": metadata})
    plots = _plots(directory, kind, result)
    result["plots"] = plots
    qualitative_html = _qualitative_html(result, directory) if kind == "qualitative" else ""
    summary = directory / "summary.json"
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("rows", "matrix", "mechanisms", "single_reference", "transfer_aggregate", "records", "assignments", "comparisons"):
        if isinstance(result.get(key), list):
            _export_csv(directory / f"{key}.csv", result[key])
    if kind == "calibration":
        for name, value in result["calibration"].items():
            safe_name = name.replace("/", "_")
            _export_csv(directory / f"{safe_name}_bins.csv", value["bins"])
            _export_csv(directory / f"{safe_name}_prompts.csv", value["points"])
    if kind == "correction":
        _export_csv(directory / "measured_underflow.csv", result["underflow_measurement"]["rows"])
    report = directory / "index.html"
    plot_html = "".join(f'<figure><img src="{html.escape(Path(path).name)}" alt="{html.escape(Path(path).stem)}"><figcaption><a href="{html.escape(Path(path).with_suffix(".pdf").name)}">Download PDF</a></figcaption></figure>' for path in plots if Path(path).suffix == ".svg")
    compact = {key: value for key, value in result.items() if key not in {"records", "points", "distributions", "cases"}}
    report.write_text('<!doctype html><meta charset="utf-8"><title>DualRef measured analysis</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}pre{white-space:pre-wrap;background:#f4f5f7;padding:20px}img{max-width:100%}.case-image{max-height:520px;object-fit:contain}.question{white-space:pre-wrap}article{margin:40px 0;border-top:1px solid #ccc;padding-top:20px}</style>' + f'<h1>{html.escape(kind.replace("_", " "))}</h1><p>Status: {result["status"]}</p>' + plot_html + qualitative_html + '<pre>' + html.escape(json.dumps(compact, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    images = [image for case in result.get("cases", []) for image in case.get("image_artifacts", [])]
    return {"summary": str(summary), "report": str(report), "plots": plots, "images": images,
            "status": result["status"], "report_version": REPORT_VERSION}


analyze = run_analysis
