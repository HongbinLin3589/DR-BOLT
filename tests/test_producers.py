import copy
import json
import sys
from types import SimpleNamespace

import pytest

from dualref.bank import _claim, construct_weights, score_bank
from dualref.baselines import resources
from dualref.diagnostics import validate_groups
from dualref.training import _rows, candidate_tokens, matched_pairs


def group(example, answer="A"):
    return {**example, "candidates": [
        {"source": source, "response": answer, "token_count": 2, "candidate_id": source}
        for source in ("fast", "slow")]}


@pytest.mark.parametrize("candidate", [{}, {"token_count": 0}, {"token_count": True},
                                       {"token_ids": []}, {"token_ids": [1], "token_count": 2}])
def test_invalid_token_metadata_fails(candidate):
    with pytest.raises(ValueError):
        candidate_tokens(candidate)


def test_preference_budget_uses_observed_tokens(example):
    candidates = [{"response": "long " * 1000}, {"response": "long " * 1000}]
    row = ({**example, "candidates": candidates}, (0, 1))
    with pytest.raises(ValueError, match="observed"):
        matched_pairs([row], 1, 2, 42)
    candidates[0]["token_ids"] = [1, 2]
    candidates[1]["token_count"] = 3
    _, budget = matched_pairs([row], 1, 5, 42)
    assert budget["scheduled_response_tokens"] == 5


def test_training_rows_reject_unmeasured_lengths(write_records, example):
    bank = group(example)
    bank["candidates"][1]["response"] = "B"
    for candidate in bank["candidates"]:
        candidate.pop("token_count")
    cfg = {"bank": {"path": write_records("bank.jsonl", [bank])}, "training": {"method": "mr_dpo"}}
    with pytest.raises(ValueError, match="observed"):
        _rows(cfg)


def test_empty_scoring_does_not_commit_bank(tmp_path, write_records):
    source = write_records("empty.jsonl", [])
    output = tmp_path / "scored"
    with pytest.raises(ValueError, match="0 groups"):
        score_bank({"bank": {"path": source}, "output_dir": str(output)})
    assert not (output / "bank.jsonl").exists()


def test_all_filtered_groups_fail_before_commit(tmp_path, write_records, example):
    source = write_records("wrong.jsonl", [group(example, "B")])
    output = tmp_path / "weighted"
    cfg = {"bank": {"path": source}, "output_dir": str(output), "objective": {"drop_all_wrong": True}}
    with pytest.raises(ValueError, match="retained 0 of 1"):
        construct_weights(cfg)
    assert not (output / "bank.jsonl").exists()
    assert not (output / "coverage.json").exists()
    cfg["bank"]["path"] = write_records("right.jsonl", [group(example)])
    cfg["output_dir"] = str(tmp_path / "valid")
    result = construct_weights(cfg)
    assert json.loads((tmp_path / "valid" / "coverage.json").read_text())["prompts"] == 1
    assert result["bank"].endswith("bank.jsonl")


def test_orphan_bank_is_not_claimed(tmp_path):
    (tmp_path / "bank.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="no manifest"):
        _claim(tmp_path, {"source": "new"})
    assert not (tmp_path / "bank-manifest.json").exists()


def test_diagnostic_group_validation(example):
    valid = group(example)
    validate_groups([valid])
    for values in ([], [valid, copy.deepcopy(valid)], [{**valid, "candidates": valid["candidates"][:1]}]):
        with pytest.raises(ValueError):
            validate_groups(values)


@pytest.mark.parametrize("memory,iterations,expected", [(0, 1, "measured"), (128, 1, "measured"),
                                                        (-1, 1, "unavailable"), (128, 0, "unavailable")])
def test_nvml_unobserved_is_not_zero(monkeypatch, memory, iterations, expected):
    calls = iter([None] * iterations + [0])
    process = SimpleNamespace(pid=123, poll=lambda: next(calls), wait=lambda: 0)
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                           nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda index: index,
                           NVML_VALUE_NOT_AVAILABLE=-1,
                           nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=123, usedGpuMemory=memory)])
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: SimpleNamespace(children=lambda recursive: []), NoSuchProcess=ProcessLookupError))
    monkeypatch.setattr(resources.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(resources.time, "sleep", lambda seconds: None)
    result = resources.measured_command(["test"], cwd=".", env={})
    assert result["memory_status"] == expected
    if expected == "measured":
        assert result["peak_gpu_bytes"] == memory
        assert result["valid_memory_samples"] == [1]
    else:
        assert "peak_gpu_bytes" not in result
        assert result["per_gpu_peak_bytes"] == [None]
