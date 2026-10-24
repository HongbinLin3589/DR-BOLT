import copy
import json
from pathlib import Path

import pytest

from dualref.baselines.runner import SOURCES, prepare
from dualref.baselines.state import prepare_history, record_evaluation, run_identity


def test_identity_covers_training_and_ignores_resume_location(example):
    cfg = {"external": {"method": "grpo"}, "training": {"learning_rate": 1e-6, "steps": 100}}
    original = run_identity(cfg, [example], SOURCES["grpo"])
    for key, value in (("learning_rate", 2e-6), ("steps", 200)):
        modified = copy.deepcopy(cfg)
        modified["training"][key] = value
        assert run_identity(modified, [example], SOURCES["grpo"]) != original
    cfg["training"].update(output_dir="elsewhere", resume="checkpoint-50")
    assert run_identity(cfg, [example], SOURCES["grpo"]) == original


def test_prepare_rejects_changed_run_without_network(tmp_path, write_records, example):
    source = write_records("train.jsonl", [example])
    cfg = {"external": {"method": "grpo", "root": str(tmp_path / "third_party"), "fetch": False, "install": False},
           "training": {"train_file": source, "output_dir": str(tmp_path / "training"), "steps": 100, "learning_rate": 1e-6}}
    first = prepare(cfg)
    assert prepare(cfg)["fingerprint"] == first["fingerprint"]
    cfg["training"]["steps"] = 101
    with pytest.raises(ValueError, match="different data or settings"):
        prepare(cfg)


@pytest.mark.parametrize("method,filename,key", [("grpo", "trainer_state.json", "global_step"),
                                                ("ton", "trainer_state.json", "global_step"),
                                                ("dualmind", "progress.json", "step")])
def test_resume_backs_up_aligns_and_deduplicates_history(tmp_path, method, filename, key):
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / filename).write_text(json.dumps({key: 25}))
    (tmp_path / "external_manifest.json").write_text('{"fingerprint":"same"}')
    history = tmp_path / "history.jsonl"
    history.write_text('\n'.join(json.dumps({"step": step, "score": .5}) for step in (0, 25, 25, 50)) + '\n')
    original = history.read_bytes()
    prepare_history({"training": {"resume": str(checkpoint)}}, tmp_path, method, "same")
    assert (tmp_path / "history.before-resume.jsonl").read_bytes() == original
    record_evaluation(history, {"step": 25, "score": .8})
    record_evaluation(history, {"step": 50, "score": .9})
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 25, 50]
    assert rows[1]["score"] == .8


def test_external_history_cannot_be_silently_reused(tmp_path):
    (tmp_path / "history.jsonl").write_text('{"step":0}\n')
    with pytest.raises(ValueError, match="already exist"):
        prepare_history({}, tmp_path, "grpo", "same")
    with pytest.raises(ValueError, match="not supported"):
        prepare_history({"training": {"resume": "anything"}}, tmp_path, "fast", "same")
    with pytest.raises(ValueError, match="identity"):
        prepare_history({"training": {"resume": str(tmp_path / "checkpoint-1")}}, tmp_path, "grpo", "same")


def test_legacy_identity_is_verified_from_saved_configuration(tmp_path, write_records, example):
    cfg = {"external": {"method": "grpo", "root": str(tmp_path / "external"), "fetch": False},
           "training": {"train_file": write_records("input.jsonl", [example]), "output_dir": str(tmp_path / "out"), "steps": 10}}
    manifest = prepare(cfg)
    path = Path(manifest["output_dir"]) / "external_manifest.json"
    legacy = json.loads(path.read_text())
    legacy.pop("identity_version")
    legacy["fingerprint"] = "legacy-fingerprint"
    path.write_text(json.dumps(legacy))
    assert prepare(cfg)["fingerprint"] == manifest["fingerprint"]
