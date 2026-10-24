"""White-box measurements and repeated router calibration."""

from __future__ import annotations

import copy
from pathlib import Path

from .io import read_jsonl, seed_everything, select_rows, write_json, write_jsonl
from .objectives import partition, reward_group


def run_diagnostics(cfg: dict) -> dict:
    kind = cfg.get("diagnostics", {}).get("kind", "cross_likelihood")
    if kind == "cross_likelihood":
        return cross_likelihood(cfg)
    if kind in ("calibration", "routing_calibration"):
        return calibration(cfg)
    if kind in ("coverage", "oracle_coverage"):
        from .bank import coverage

        output = Path(cfg.get("output_dir", "outputs/diagnostics")) / "coverage.json"
        write_json(output, coverage(cfg["bank"]["path"]))
        return {"records": str(output), "coverage": str(output)}
    raise ValueError(f"Unknown diagnostic kind {kind!r}")


def validate_groups(groups: list[dict]) -> None:
    seen = set()
    for group in groups:
        identifier = group.get("id")
        if not identifier or identifier in seen:
            raise ValueError(f"Diagnostic bank requires unique, nonempty IDs: {identifier!r}")
        seen.add(identifier)
        sources = {candidate.get("source") for candidate in group.get("candidates", [])}
        if sources != {"fast", "slow"}:
            raise ValueError(f"Diagnostic group {identifier} requires both fast and slow candidates")
    if not groups:
        raise ValueError("Diagnostic bank contains no candidate groups")


def cross_likelihood(cfg: dict) -> dict:
    from .bank import build_bank, score_bank

    output = Path(cfg.get("output_dir", "outputs/diagnostics"))
    settings = cfg.get("diagnostics", {})
    source = cfg.get("bank", {}).get("path")
    working = copy.deepcopy(cfg)
    if not source or settings.get("path") == "ablation":
        working["output_dir"] = str(output / "generation")
        working.setdefault("bank", {}).update(limit=settings.get("samples_per_mode", 1000),
                                               path_variant=settings.get("path", "main"), path=None,
                                               score_references=False, action="generate")
        source = build_bank(working)["bank"]
    working["output_dir"] = str(output / "scores")
    working.setdefault("bank", {}).update(path=source, score_references=True)
    scored_file = score_bank(working)["bank"]
    groups = list(read_jsonl(scored_file))
    validate_groups(groups)
    groups = select_rows(groups, settings.get("samples_per_mode", 1000), cfg.get("seed", 42))
    validate_groups(groups)
    records = []
    selected_groups = []
    for group in groups:
        selected = []
        for mode in ("fast", "slow"):
            candidates = [c for c in group["candidates"] if c["source"] == mode]
            row = select_rows(candidates, 1, cfg.get("seed", 42), "candidate_id")[0]
            selected.append(row)
            records.append({**row, "sample_id": row["candidate_id"], "prompt_id": group["id"],
                            "response_tokens": row.get("logp_tokens_fast", row["token_count"]),
                            "question": group["question"], "dataset": group["dataset"]})
        selected_group = copy.deepcopy(group)
        selected_group["candidates"] = selected
        selected_groups.append(reward_group(selected_group, cfg.get("objective", {})))
    path = write_jsonl(output / "cross-likelihood.jsonl", records)
    # Optimization uses the original 7:3 groups, not the balanced matrix sample.
    training_bank = write_jsonl(output / "bank.jsonl", groups)
    result = {"records": str(path), "bank": str(training_bank), "matrix_bank": str(output / "matrix-bank.jsonl")}
    write_jsonl(result["matrix_bank"], selected_groups)
    replay_input = settings.get("replay_input")
    if replay_input:
        replay = copy.deepcopy(cfg)
        replay["data"]["train_file"] = replay_input
        replay["output_dir"] = str(output / "replay-generation")
        replay["bank"].update(path=None, action="generate", num_fast=1, num_slow=1,
                              limit=settings.get("replay_samples", 200) // 2,
                              path_variant=settings.get("path", "main"), score_references=False)
        result["replay_bank"] = build_bank(replay)["bank"]
    else:
        raise ValueError("cross_likelihood requires diagnostics.replay_input from a held-out split")
    write_json(output / "counts.json", {mode: sum(row["source"] == mode for row in records) for mode in ("fast", "slow")})
    return result


def calibration(cfg: dict) -> dict:
    from .data import load_examples
    from .models import ModelBackend
    from .scoring import is_slow

    settings = cfg.get("diagnostics", {})
    evaluation = cfg.get("evaluation", {})
    repeats = settings.get("repeats", 20)
    seed = cfg.get("seed", 42)
    examples = select_rows(load_examples(evaluation["input"]), settings.get("prompts", 100), seed)
    groups = {g["id"]: g for g in read_jsonl(cfg["bank"]["path"])}
    missing = [row["id"] for row in examples if row["id"] not in groups]
    if missing:
        raise ValueError(f"Calibration bank does not cover {len(missing)} evaluation prompts")
    backend = ModelBackend(cfg, checkpoint=evaluation.get("checkpoint"))
    records = []
    generations = []
    try:
        seed_everything(seed)
        for example in examples:
            group = reward_group(groups[example["id"]], cfg.get("objective", {}))
            stats = partition(group["candidates"], cfg.get("objective", {}).get("alpha", .5),
                              cfg.get("objective", {}).get("beta", 1))
            samples = backend.generate([example], num_return_sequences=repeats)[0]
            fast_count = sum(not is_slow(sample["response"]) for sample in samples)
            records.append({"prompt_id": example["id"], "predicted_fast_probability": stats["predicted_fast_probability"],
                            "fast_count": fast_count, "total": repeats})
            for index, sample in enumerate(samples):
                generations.append({"example_id": example["id"], "repeat": index,
                                    "mode": "slow" if is_slow(sample["response"]) else "fast", **sample})
    finally:
        backend.close()
    output = Path(cfg.get("output_dir", "outputs/calibration"))
    path = write_jsonl(output / "calibration.jsonl", records)
    predictions = write_jsonl(output / "predictions.jsonl", generations)
    return {"records": str(path), "predictions": str(predictions)}
