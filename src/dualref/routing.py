"""Frozen-feature two-expert and one-checkpoint prompt-router controls."""

from __future__ import annotations

import copy
from pathlib import Path

from .io import seed_everything, telemetry, write_json, write_jsonl
from .objectives import partition, reward_group


def train_router(cfg: dict) -> dict:
    import torch
    from torch.nn import functional as F

    from .data import load_examples
    from .models import ModelBackend

    output = Path(cfg.get("output_dir", "outputs/router"))
    output.mkdir(parents=True, exist_ok=True)
    settings = cfg.get("router", {})
    seed_everything(cfg.get("seed", 42))
    groups = load_examples(cfg["bank"]["path"])
    features, labels = [], []
    with telemetry(output, "router", cfg):
        backend = ModelBackend(cfg, role="fast")
        try:
            for group in groups:
                weighted = reward_group(group, cfg.get("objective", {}))
                stats = partition(weighted["candidates"], cfg.get("objective", {}).get("alpha", .5), cfg.get("objective", {}).get("beta", 1))
                features.append(backend.prompt_features(group))
                labels.append(int(stats["predicted_fast_probability"] < .5))
        finally:
            backend.close()
        if not features:
            raise ValueError("No router training inputs")
        x, y = torch.stack(features), torch.tensor(labels)
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / scale
        classifier = torch.nn.Linear(x.shape[1], 2)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=settings.get("learning_rate", .001))
        history = []
        for epoch in range(settings.get("epochs", 20)):
            order = torch.randperm(len(x))
            total_loss = 0.0
            for indices in order.split(settings.get("batch_size", 256)):
                loss = F.cross_entropy(classifier(x[indices]), y[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            history.append({"epoch": epoch + 1, "loss": total_loss / len(x)})
        path = output / "router.pt"
        torch.save({"weight": classifier.weight.detach(), "bias": classifier.bias.detach(),
                    "mean": mean, "scale": scale}, path)
        write_json(output / "router.json", {"features": "fast_reference_last_prompt_hidden_state",
                    "model": cfg.get("model", {}), "labels": {"0": "fast", "1": "slow"},
                    "training_examples": len(groups), "seed": cfg.get("seed", 42)})
        write_jsonl(output / "history.jsonl", history)
    return {"router": str(path), "checkpoint": str(path), "history": str(output / "history.jsonl")}


def evaluate_router(cfg: dict) -> dict:
    import torch
    from torch.nn import functional as F

    from .models import ModelBackend
    from .data import load_examples
    from .scoring import score_response

    settings = cfg.get("evaluation", {})
    router_path = settings.get("router") or cfg.get("router", {}).get("checkpoint")
    if isinstance(router_path, dict):
        router_path = router_path.get("checkpoint") or router_path.get("path")
    state = torch.load(router_path, map_location="cpu", weights_only=True)
    kind = settings.get("router_kind", "two_policy")
    prompt_router = kind in ("prompt", "prompt_router", "thinking_prompt", "thinking_prompt_router")
    examples = load_examples(settings["input"])
    if settings.get("limit"):
        examples = examples[:settings["limit"]]
    seed_everything(cfg.get("seed", 42))
    fast = ModelBackend(cfg, role="fast")
    assignments, records = {}, {}
    try:
        for example in examples:
            feature = (fast.prompt_features(example) - state["mean"]) / state["scale"]
            with torch.no_grad():
                assignments[example["id"]] = int(F.linear(feature, state["weight"], state["bias"]).argmax())
        if not prompt_router:
            for example in examples:
                if assignments[example["id"]] == 0:
                    records[example["id"]] = fast.generate([example])[0][0]
    finally:
        fast.close()
    slow = ModelBackend(cfg, role="slow")
    try:
        for example in examples:
            if not prompt_router and assignments[example["id"]] == 0:
                continue
            modified = copy.deepcopy(example)
            if prompt_router:
                modified.setdefault("metadata", {})["direct_prompt" if assignments[example["id"]] == 0 else "reasoning_prompt"] = True
            records[example["id"]] = slow.generate([modified])[0][0]
    finally:
        slow.close()
    predictions = []
    for example in examples:
        result = records[example["id"]]
        score = score_response(example, result["response"])
        predictions.append({"example_id": example["id"], "dataset": example["dataset"],
                            "route": "fast" if assignments[example["id"]] == 0 else "slow", **result, **score})
    n = len(predictions)
    if not n:
        raise ValueError("Empty router evaluation set")
    fast_routes = [p for p in predictions if p["route"] == "fast"]
    metrics = {"n": n, "score": sum(p["correct"] for p in predictions) / n,
               "length": sum(p["words"] for p in predictions) / n,
               "slow_ratio": sum(p["slow"] for p in predictions) / n,
               "fast_prompt_compliance": (sum(not p["slow"] and p["token_count"] <= 30 for p in fast_routes) / len(fast_routes)
                                          if fast_routes and prompt_router else None)}
    output = Path(settings.get("output") or cfg.get("output_dir", "outputs/router-evaluation"))
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "metrics.json", metrics)
    return {**metrics, "predictions": str(output / "predictions.jsonl"), "metrics": str(output / "metrics.json")}
