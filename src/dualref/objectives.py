"""Prompt-normalized DR-BOLT weights and related offline objectives."""

from __future__ import annotations

import copy
import math


def logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot normalize an empty set")
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def log_mixture(log_fast: float, log_slow: float, prior: float) -> float:
    if not 0 < prior < 1:
        raise ValueError("Mixture prior must be in (0, 1)")
    return logsumexp([math.log(prior) + log_fast, math.log1p(-prior) + log_slow])


def exact_log_correction(log_fast: float, log_slow: float, alpha: float, q: float) -> float:
    """Correct the empirical fast sampling ratio q to the target fast prior alpha."""
    if alpha == q:
        return 0.0
    return log_mixture(log_fast, log_slow, alpha) - log_mixture(log_fast, log_slow, q)


def partition(candidates: list[dict], alpha: float = 0.5, beta: float = 1.0) -> dict:
    """Estimate source partitions and q from the observed candidate sources."""
    if beta <= 0:
        raise ValueError("BOLT temperature must be positive")
    fast = [row["reward"] / beta for row in candidates if row["source"] == "fast"]
    slow = [row["reward"] / beta for row in candidates if row["source"] == "slow"]
    if not fast or not slow:
        raise ValueError("Each candidate group needs both known sources")
    log_zf = logsumexp(fast) - math.log(len(fast))
    log_zs = logsumexp(slow) - math.log(len(slow))
    log_z = log_mixture(log_zf, log_zs, alpha)
    return {"log_z_fast": log_zf, "log_z_slow": log_zs, "log_z": log_z,
            "predicted_fast_probability": math.exp(math.log(alpha) + log_zf - log_z),
            "q": len(fast) / (len(fast) + len(slow))}


def reward_group(group: dict, objective: dict) -> dict:
    from .scoring import score_response

    group = copy.deepcopy(group)
    candidates = group["candidates"]
    if not candidates:
        raise ValueError(f"Empty group: {group['id']}")
    for candidate in candidates:
        score = score_response(group, candidate["response"])
        candidate.update(score)
        candidate["word_count"] = score["words"]
    length_scale = max(1, max(c["word_count"] for c in candidates))
    for candidate in candidates:
        candidate["normalized_length"] = candidate["word_count"] / length_scale
        candidate["reward"] = (candidate["correct"]
                               + objective.get("gamma", 0.1) * candidate["format"]
                               - objective.get("length_penalty", 0.05) * candidate["normalized_length"])
    return group


def weight_group(group: dict, objective: dict, method: str = "dr_bolt") -> dict:
    """Use the stratified partition estimate, with optional diagnostic group normalization."""
    group = reward_group(group, objective)
    candidates = group["candidates"]
    alpha, beta = objective.get("alpha", 0.5), objective.get("beta", 1.0)
    stats = partition(candidates, alpha, beta)
    q = stats["q"]
    correction = objective.get("correction", "modewise").replace("-", "").replace("_", "")
    log_weights = []
    for row in candidates:
        source = row["source"]
        if source not in ("fast", "slow"):
            raise ValueError(f"Unknown candidate source {source!r}")
        if method in ("mix_sft", "warmup"):
            log_weight = 0.0
        elif method == "reward_sft":
            log_weight = row["reward"] / beta
        elif method == "cond_rw":
            log_weight = row["reward"] / beta - logsumexp([c["reward"] / beta for c in candidates])
        elif method in ("single_ref_fast", "single_ref_slow"):
            target = "fast" if method == "single_ref_fast" else "slow"
            if "logp_fast" not in row or "logp_slow" not in row:
                raise ValueError(f"{method} requires both frozen reference likelihoods")
            log_mu = log_mixture(row["logp_fast"], row["logp_slow"], q)
            log_kappa = row[f"logp_{target}"] - log_mu
            log_weight = log_kappa + row["reward"] / beta - stats[f"log_z_{target}"]
            row["log_correction"] = log_kappa
        else:
            if correction in ("exact", "single"):
                if "logp_fast" not in row or "logp_slow" not in row:
                    raise ValueError("Reference scoring is required for exact/single correction")
                if correction == "exact":
                    log_kappa = exact_log_correction(row["logp_fast"], row["logp_slow"], alpha, q)
                else:
                    log_kappa = row["logp_fast"] - log_mixture(row["logp_fast"], row["logp_slow"], q)
            elif correction in ("none", "no", "nocorrection"):
                log_kappa = 0.0
            elif correction == "modewise":
                log_kappa = math.log(alpha / q) if source == "fast" else math.log((1-alpha)/(1-q))
            else:
                raise ValueError(f"Unknown correction: {correction}")
            row["log_correction"] = log_kappa
            log_weight = log_kappa + row["reward"] / beta - stats["log_z"]
        log_weights.append(log_weight)
        if "logp_fast" in row and "logp_slow" in row:
            row["log_reference"] = log_mixture(row["logp_fast"], row["logp_slow"], alpha)
    if objective.get("normalize_groups", False):
        offset = logsumexp(log_weights) - math.log(len(log_weights))
        log_weights = [value - offset for value in log_weights]
    for row, log_weight in zip(candidates, log_weights):
        row["log_weight"] = log_weight
        row["weight"] = math.exp(log_weight)
    group["partition"] = stats
    group["all_wrong"] = not any(c["correct"] for c in candidates)
    return group


def preference_pairs(group: dict) -> list[tuple[int, int]]:
    candidates = group["candidates"]
    return [(i, j) for i, winner in enumerate(candidates) for j, loser in enumerate(candidates)
            if winner["reward"] > loser["reward"]]
