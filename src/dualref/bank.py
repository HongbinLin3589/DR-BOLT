"""Immutable, resumable candidate collection and offline reference scoring."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .io import append_jsonl, file_hash, fingerprint, read_jsonl, seed_everything, select_rows, telemetry, write_json, write_jsonl
from .objectives import reward_group, weight_group


def _output(cfg: dict) -> Path:
    output = Path(cfg.get("output_dir", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    return output


def _identity(cfg: dict, source: str, kind: str) -> dict:
    return {"kind": kind, "input_sha256": file_hash(source), "seed": cfg.get("seed", 42),
            "model": cfg.get("model", {}), "generation": cfg.get("generation", {}),
            "bank": {k: v for k, v in cfg.get("bank", {}).items() if k not in ("path", "output")},
            "objective": cfg.get("objective", {})}


def _claim(output: Path, identity: dict) -> None:
    manifest = output / "bank-manifest.json"
    value = {"fingerprint": fingerprint(identity), "identity": identity}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["fingerprint"] != value["fingerprint"]:
            raise ValueError(f"Refusing incompatible bank reuse in {output}; choose a new output directory")
    else:
        if (output / "bank.jsonl").exists() or any(output.glob("*.partial.jsonl")):
            raise ValueError(f"Existing candidate artifacts have no manifest in {output}; use a new output directory")
        write_json(manifest, value)


def _write_bank(path: Path, groups) -> None:
    def checked():
        count = 0
        seen = set()
        for group in groups:
            count += 1
            identifier = group.get("id")
            if not identifier or identifier in seen:
                raise ValueError(f"Bank group ID must be nonempty and unique: {identifier!r}")
            seen.add(identifier)
            if {candidate.get("source") for candidate in group.get("candidates", [])} != {"fast", "slow"}:
                raise ValueError(f"Candidate group {identifier} requires both fast and slow sources")
            yield group
        if not count:
            raise ValueError("Candidate processing retained 0 groups; no final bank was committed")
    write_jsonl(path, checked())


def _validate_bank(cfg: dict, destination: Path) -> None:
    from .artifacts import validate_result

    validate_result("bank", cfg, {"bank": str(destination)})


def _policy_example(example: dict, source: str, variant: str) -> dict:
    example = {key: value for key, value in example.items() if key != "candidates"}
    if variant == "ablation" and source == "fast":
        example = copy.deepcopy(example)
        example.setdefault("metadata", {})["direct_prompt"] = True
    return example


def _bank_result(cfg: dict, output: Path) -> dict:
    destination = output / "bank.jsonl"
    _validate_bank(cfg, destination)
    if cfg.get("bank", {}).get("score_references"):
        scoring_cfg = copy.deepcopy(cfg)
        scoring_cfg["bank"]["path"] = str(destination)
        scoring_cfg["output_dir"] = str(output / "reference-scores")
        return score_bank(scoring_cfg)
    return {"bank": str(destination), "manifest": str(output / "bank-manifest.json")}


def build_bank(cfg: dict) -> dict:
    from .data import load_examples
    from .models import ModelBackend

    settings = cfg.get("bank", {})
    if settings.get("action") == "subset":
        return subset_bank(cfg)
    train_file = cfg.get("data", {}).get("train_file") or settings.get("input")
    if not train_file:
        raise ValueError("Candidate collection requires data.train_file")
    output = _output(cfg)
    _claim(output, _identity(cfg, train_file, "generation"))
    destination = output / "bank.jsonl"
    if destination.exists():
        return _bank_result(cfg, output)
    seed = cfg.get("seed", 42)
    examples = select_rows(load_examples(train_file), settings.get("limit"), seed)
    if not examples:
        raise ValueError("Cannot build a candidate bank from an empty input file")
    variant = settings.get("path_variant", "main")
    with telemetry(output, "bank", cfg):
        paths = {}
        for source, key in (("fast", "num_fast"), ("slow", "num_slow")):
            paths[source] = output / f"{source}.partial.jsonl"
            done = {row["id"] for row in read_jsonl(paths[source])} if paths[source].exists() else set()
            remaining = [example for example in examples if example["id"] not in done]
            if not remaining:
                continue
            reference = "slow" if variant == "ablation" and source == "fast" else source
            backend = ModelBackend(cfg, role=reference)
            try:
                for example in remaining:
                    local_seed = (seed + int(fingerprint([example["id"], source])[:8], 16)) % (2**32)
                    seed_everything(local_seed)
                    generated = backend.generate([_policy_example(example, source, variant)],
                                                 num_return_sequences=settings.get(key, 7 if source == "fast" else 3))[0]
                    # Source records the generating policy, independently of response format.
                    candidates = [dict(result, candidate_id=f"{example['id']}:{source}:{index}", source=source)
                                  for index, result in enumerate(generated)]
                    append_jsonl(paths[source], {"id": example["id"], "candidates": candidates})
            finally:
                backend.close()
        pools = {source: {row["id"]: row["candidates"] for row in read_jsonl(path)}
                 for source, path in paths.items()}
        def groups():
            for example in examples:
                group = dict(example, candidates=pools["fast"][example["id"]] + pools["slow"][example["id"]])
                group["bank_variant"] = variant
                yield reward_group(group, cfg.get("objective", {}))
        _write_bank(destination, groups())
        write_json(output / "coverage.json", coverage(destination))
    return _bank_result(cfg, output)


def subset_bank(cfg: dict) -> dict:
    from .data import load_examples

    settings = cfg["bank"]
    source = settings["path"]
    output = _output(cfg)
    _claim(output, _identity(cfg, source, "subset"))
    destination = output / "bank.jsonl"
    def groups():
        for group in load_examples(source):
            group = copy.deepcopy(group)
            chosen = []
            for mode, key in (("fast", "num_fast"), ("slow", "num_slow")):
                pool = [c for c in group["candidates"] if c["source"] == mode]
                count = settings.get(key, 7 if mode == "fast" else 3)
                if len(pool) < count:
                    raise ValueError(f"Insufficient {mode} candidates for {group['id']}")
                chosen.extend(select_rows(pool, count, cfg.get("seed", 42), key="candidate_id"))
            group["candidates"] = chosen
            # The length reward's denominator belongs to the selected group.
            yield reward_group(group, cfg.get("objective", {}))
    _write_bank(destination, groups())
    return {"bank": str(destination)}


def score_bank(cfg: dict) -> dict:
    from .data import load_examples
    from .models import ModelBackend

    source_file = cfg["bank"]["path"]
    output = _output(cfg)
    _claim(output, _identity(cfg, source_file, "reference-scoring"))
    destination = output / "bank.jsonl"
    if destination.exists():
        _validate_bank(cfg, destination)
        return {"bank": str(destination)}
    if not cfg.get("bank", {}).get("score_references", False):
        _write_bank(destination, (reward_group(group, cfg.get("objective", {})) for group in load_examples(source_file)))
        return {"bank": str(destination)}
    current = source_file
    with telemetry(output, "reference-scoring", cfg):
        for mode in ("fast", "slow"):
            partial = output / f"{mode}.partial.jsonl"
            done = {row["id"] for row in read_jsonl(partial)} if partial.exists() else set()
            backend = None
            try:
                for group in load_examples(current):
                    if group["id"] in done:
                        continue
                    variant = group.get("bank_variant", cfg.get("bank", {}).get("path_variant", "main"))
                    if any(f"logp_{mode}" not in candidate for candidate in group["candidates"]):
                        if backend is None:
                            role = "slow" if mode == "fast" and variant == "ablation" else mode
                            backend = ModelBackend(cfg, role=role)
                        scores = backend.sequence_log_probs(
                            _policy_example(group, mode, variant), [c["response"] for c in group["candidates"]],
                            response_token_ids=[c.get("token_ids") for c in group["candidates"]])
                        for candidate, score in zip(group["candidates"], scores):
                            candidate[f"logp_{mode}"] = score["logp"]
                            candidate[f"logp_tokens_{mode}"] = score["token_count"]
                    append_jsonl(partial, reward_group(group, cfg.get("objective", {})))
            finally:
                if backend is not None:
                    backend.close()
            current = str(partial)
        _write_bank(destination, read_jsonl(current))
    return {"bank": str(destination)}


def construct_weights(cfg: dict) -> dict:
    from .data import load_examples

    source = cfg["bank"]["path"]
    output = _output(cfg)
    _claim(output, _identity(cfg, source, "weights"))
    destination = output / "bank.jsonl"
    method = cfg.get("training", {}).get("method", "dr_bolt")
    objective = cfg.get("objective", {})
    def groups():
        total = retained = 0
        for group in load_examples(source):
            total += 1
            weighted = weight_group(group, objective, method)
            if objective.get("drop_all_wrong", False) and weighted["all_wrong"]:
                continue
            retained += 1
            yield weighted
        if not retained:
            raise ValueError(f"Weight construction retained 0 of {total} input groups; check candidate rewards and filtering")
    _write_bank(destination, groups())
    write_json(output / "coverage.json", coverage(destination))
    return {"bank": str(destination), "coverage": str(output / "coverage.json")}


def coverage(path: str | Path) -> dict:
    total, covered, candidates = 0, 0, 0
    for group in read_jsonl(path):
        total += 1
        candidates += len(group["candidates"])
        covered += int(any(c["correct"] for c in group["candidates"]))
    return {"prompts": total, "candidates": candidates,
            "oracle_coverage": covered / total if total else None}
