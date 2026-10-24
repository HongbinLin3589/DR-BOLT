"""Evaluate a checkpoint with deterministic answer scoring."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .config import write_config
from .data import load_examples
from .scoring import score_response


def evaluate(config: dict) -> dict:
    from .models import ModelBackend
    from transformers import set_seed

    settings = config.get("evaluation", {})
    if settings.get("router") or settings.get("router_kind"):
        from .routing import evaluate_router

        return evaluate_router(config)
    required = {"input", "output", "checkpoint"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"evaluation requires {sorted(missing)}")
    examples = load_examples(settings["input"])
    seed = int(settings.get("seed", config.get("seed", 42)))
    limit = settings.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("evaluation.limit must be positive")
        if len(examples) > limit:
            chosen = set(random.Random(seed).sample(range(len(examples)), limit))
            examples = [example for index, example in enumerate(examples) if index in chosen]
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset")
    repeats = int(settings.get("repeats", 1))
    batch_size = int(settings.get("batch_size", 1))
    if repeats < 1 or batch_size < 1:
        raise ValueError("evaluation.repeats and batch_size must be positive")
    output = Path(settings["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_config(output / "config.yaml", config)
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for repeat in range(repeats):
            run_config = copy.deepcopy(config)
            run_config["seed"] = seed + repeat
            run_config.setdefault("generation", {})["seed"] = seed + repeat
            set_seed(seed + repeat)
            backend = ModelBackend(run_config, checkpoint=settings["checkpoint"], role="policy")
            try:
                for start in range(0, len(examples), batch_size):
                    batch = examples[start : start + batch_size]
                    maximum = settings.get("max_new_tokens", run_config.get("generation", {}).get("max_new_tokens"))
                    predictions = backend.generate(batch, num_return_sequences=1, max_new_tokens=maximum)
                    if len(predictions) != len(batch):
                        raise ValueError("Generation returned a different number of examples")
                    for example, candidates in zip(batch, predictions):
                        if len(candidates) != 1:
                            raise ValueError("Evaluation requires exactly one response per input")
                        candidate = candidates[0]
                        response = candidate["response"]
                        row = {"example_id": example["id"], "dataset": example["dataset"],
                               "question": example["question"], "choices": example.get("choices", []),
                               "images": example.get("images", []), "answer": example["answer"],
                               "response": response, **score_response(example, response),
                               "token_count": int(candidate["token_count"]), "repeat": repeat,
                               "finish_reason": candidate.get("finish_reason")}
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
            finally:
                backend.close()
    count = len(rows)
    metrics = {
        "score": sum(row["correct"] for row in rows) / count,
        "length": sum(row["words"] for row in rows) / count,
        "slow_ratio": sum(row["slow"] for row in rows) / count,
        "n": count,
        "mean_tokens": sum(row["token_count"] for row in rows) / count,
        "format_rate": sum(row["format"] for row in rows) / count,
        "unique_examples": len(examples),
        "repeats": repeats,
        "seed": seed,
        "dataset": settings.get("dataset", examples[0]["dataset"]),
        "checkpoint": settings["checkpoint"],
        "per_repeat": [
            {"repeat": repeat, "score": sum(row["correct"] for row in rows if row["repeat"] == repeat) / len(examples), "length": sum(row["words"] for row in rows if row["repeat"] == repeat) / len(examples)}
            for repeat in range(repeats)
        ],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**{key: value for key, value in metrics.items() if key != "checkpoint"},
            "predictions": str(output / "predictions.jsonl"), "metrics": str(output / "metrics.json")}
