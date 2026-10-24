"""Reuse FAST's reward while substituting the shared benchmark correctness parser."""

import importlib.util
import json
import os
from pathlib import Path

from dualref.scoring import score_response


def compute_score(reward_inputs, **kwargs):
    if len(reward_inputs) % 8:
        raise ValueError("FAST requires complete groups of eight rollouts")
    path = Path(os.environ["DUALREF_FAST_ROOT"]) / "examples/reward_function/thinking_reward.py"
    spec = importlib.util.spec_from_file_location("_dualref_upstream_fast_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def accuracy(response, ground_truth):
        score = score_response(json.loads(ground_truth), response)
        return float(score["correct"] if isinstance(score, dict) else score.correct)

    module.accuracy_reward = accuracy
    return module.compute_score(reward_inputs, **kwargs)
