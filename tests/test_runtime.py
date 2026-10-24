import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dualref import cli
from dualref.analysis import REPORT_VERSION, run_analysis
from dualref.artifacts import validate_result
from dualref.experiments import list_experiments, merge, run_suite


def configuration(tmp_path, jobs, fail_fast=False):
    return {"output_dir": str(tmp_path / "runs"),
            "experiments": {"jobs": jobs, "fail_fast": fail_fast}}


def seed_cache(config, result, *, old_report=False):
    job = list_experiments(config)[0]
    cfg = merge(config, job["config"])
    cfg.get("experiments", {}).pop("jobs", None)
    payload = {"config": cfg, "dependencies": {}}
    if job["stage"] == "analyze" and not old_report:
        payload["report_version"] = REPORT_VERSION
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    path = Path(cfg["output_dir"]) / "job-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {**result, "_result": str(path), "_fingerprint": digest}
    path.write_text(json.dumps({"id": job["id"], "stage": job["stage"], "status": "completed",
                                "fingerprint": digest, "outputs": outputs}))
    return path


@pytest.mark.parametrize("status", ["failed", "partial", "missing", "incomplete", "not_reported", "unknown"])
def test_non_success_producer_blocks_dependencies(tmp_path, monkeypatch, status):
    jobs = [{"id": "a", "stage": "diagnostics", "config": {"diagnostics": {"kind": "coverage"}}},
            {"id": "b", "stage": "diagnostics", "depends_on": ["a"]}]
    calls = []
    def dispatch(stage, cfg):
        calls.append(stage)
        return {"status": status}
    monkeypatch.setattr(cli, "dispatch", dispatch)
    result = run_suite(configuration(tmp_path, jobs))
    assert result["status"] == "failed"
    assert [row["status"] for row in result["jobs"]] == ["failed", "blocked"]
    assert len(calls) == 1


def test_partial_analysis_can_feed_aggregate_but_not_strict_dependents(tmp_path, write_records, monkeypatch):
    path = write_records("input.jsonl", [{"correct": 1, "words": 2, "slow": False}, {"response": "A"}])
    jobs = [
        {"id": "part", "stage": "analyze", "config": {"analysis": {"kind": "benchmark_table", "sources": {"x": path}}}},
        {"id": "strict", "stage": "diagnostics", "depends_on": ["part"]},
        {"id": "report", "stage": "analyze", "config": {"analysis": {"kind": "report", "sources": {"part": "${part.summary}"}}}},
    ]
    monkeypatch.setattr(cli, "dispatch", lambda stage, cfg: run_analysis(cfg) if stage == "analyze" else pytest.fail("Blocked stage executed"))
    result = run_suite(configuration(tmp_path, jobs))
    assert result["status"] == "partial"
    assert {row["id"]: row["status"] for row in result["jobs"]} == {"part": "partial", "strict": "blocked", "report": "partial"}


@pytest.mark.parametrize("cached", [{"status": "incomplete"}, {"status": "not_reported"}, {"status": "complete"}])
def test_invalid_cache_is_not_reused(tmp_path, write_records, monkeypatch, cached):
    config = configuration(tmp_path, [{"id": "d", "stage": "diagnostics", "config": {"diagnostics": {"kind": "coverage"}}}])
    seed_cache(config, cached)
    records = write_records("records.jsonl", [{"prompts": 1, "candidates": 2, "oracle_coverage": 1}])
    calls = []
    monkeypatch.setattr(cli, "dispatch", lambda *args: calls.append(args) or {"records": records})
    result = run_suite(config)
    assert len(calls) == 1
    assert result["status"] == "completed"
    assert run_suite(config)["jobs"][0]["status"] == "cached"
    assert len(calls) == 1


def test_empty_or_invalid_artifact_cannot_validate(tmp_path):
    path = tmp_path / "records.jsonl"
    for text in ("", "\n", "not-json\n", "[]\n"):
        path.write_text(text)
        with pytest.raises(ValueError):
            validate_result("diagnostics", {"diagnostics": {"kind": "coverage"}}, {"records": str(path)})
    with pytest.raises(ValueError, match="required outputs"):
        validate_result("train", {}, {})


def test_prediction_metrics_must_agree(tmp_path, write_records):
    predictions = write_records("predictions.jsonl", [{"correct": 1, "words": 2, "slow": False}])
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"n": 1, "score": .5, "length": 2, "slow_ratio": 0}))
    outputs = {"predictions": predictions, "metrics": str(metrics)}
    with pytest.raises(ValueError, match="disagree"):
        validate_result("evaluate", {}, outputs)
    metrics.write_text(json.dumps({"n": 1, "score": 1, "length": 2, "slow_ratio": 0}))
    assert validate_result("evaluate", {}, outputs) == "completed"


def test_complete_summary_cannot_hide_partial_child(tmp_path):
    result = run_analysis({"output_dir": str(tmp_path), "analysis": {"kind": "protocol"}})
    path = Path(result["summary"])
    summary = json.loads(path.read_text())
    summary["reports"] = {"child": {"status": "partial"}}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete measurements"):
        validate_result("analyze", {"analysis": {"kind": "protocol"}}, result)


def test_readable_but_unusable_artifacts_are_rejected(tmp_path, write_records):
    path = write_records("not-a-dataset.jsonl", [{"irrelevant": 1}])
    with pytest.raises(ValueError):
        validate_result("prepare", {}, {"train": path})
    with pytest.raises(ValueError, match="measured counts"):
        validate_result("diagnostics", {"diagnostics": {"kind": "coverage"}}, {"records": path})
    checkpoint = tmp_path / "empty-model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="model_type"):
        validate_result("train", {}, {"checkpoint": str(checkpoint)})


def test_legacy_training_cache_survives_report_regeneration(tmp_path, write_records, monkeypatch):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"base_model_name_or_path":"test"}')
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test fixture weights")
    history = write_records("history.jsonl", [{"step": 25, "loss": 1., "score": .5, "length": 2}])
    jobs = [{"id": "train", "stage": "train"},
            {"id": "report", "stage": "analyze", "config": {"analysis": {"kind": "trajectory", "sources": {"train": "${train.history}"}}}}]
    config = configuration(tmp_path, jobs)
    seed_cache(config, {"checkpoint": str(checkpoint), "history": history})
    calls = []
    def dispatch(stage, cfg):
        assert stage == "analyze", "GPU task must remain cached"
        calls.append(stage)
        return run_analysis(cfg)
    monkeypatch.setattr(cli, "dispatch", dispatch)
    result = run_suite(config)
    assert [row["status"] for row in result["jobs"]] == ["cached", "completed"]
    report_path = Path(result["jobs"][1]["outputs"]["summary"])
    summary = json.loads(report_path.read_text())
    summary.pop("report_version")
    report_path.write_text(json.dumps(summary))
    again = run_suite(config)
    assert [row["status"] for row in again["jobs"]] == ["cached", "completed"]
    assert len(calls) == 2
    assert json.loads(report_path.read_text())["report_version"] == REPORT_VERSION


def test_old_analysis_fingerprint_is_invalidated(tmp_path, monkeypatch):
    config = configuration(tmp_path, [{"id": "report", "stage": "analyze", "config": {"analysis": {"kind": "protocol"}}}])
    job_cfg = merge(config, list_experiments(config)[0]["config"])
    outputs = run_analysis(job_cfg)
    seed_cache(config, outputs, old_report=True)
    calls = []
    monkeypatch.setattr(cli, "dispatch", lambda stage, cfg: calls.append(stage) or run_analysis(cfg))
    assert run_suite(config)["status"] == "completed"
    assert calls == ["analyze"]


def test_custom_job_paths_are_independent_and_collisions_rejected(tmp_path):
    jobs = [{"id": "a", "stage": "analyze"}, {"id": "b", "stage": "analyze"}]
    expanded = list_experiments(configuration(tmp_path, jobs))
    assert expanded[0]["config"]["output_dir"] != expanded[1]["config"]["output_dir"]
    for job in jobs:
        job["config"] = {"output_dir": str(tmp_path / "same")}
    with pytest.raises(ValueError, match="shared"):
        list_experiments(configuration(tmp_path, jobs))


def test_fail_fast_only_stops_independent_tasks_on_failure(tmp_path, write_records, monkeypatch):
    records = write_records("valid.jsonl", [{"prompts": 1, "candidates": 2, "oracle_coverage": 1}])
    jobs = [{"id": name, "stage": "diagnostics", "config": {"diagnostics": {"kind": "coverage"}}} for name in ("a", "b")]
    def dispatch(stage, cfg):
        return {"status": "failed"} if Path(cfg["output_dir"]).name == "a" else {"records": records}
    monkeypatch.setattr(cli, "dispatch", dispatch)
    assert run_suite(configuration(tmp_path, jobs, False))["jobs"][1]["status"] == "completed"
    assert run_suite(configuration(tmp_path, jobs, True))["jobs"][1]["status"] == "blocked"


def test_cli_analysis_exit_codes_and_dry_runs(tmp_path, write_records):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    good = write_records("good.jsonl", [{"id": "x", "correct": 1}])
    bad = write_records("bad.jsonl", [{"correct": 1}])
    for index, source in enumerate((good, bad)):
        config = tmp_path / f"config-{index}.yaml"
        config.write_text(yaml.safe_dump({"output_dir": str(tmp_path / f"cli-{index}"),
                                        "analysis": {"kind": "coverage", "sources": {"x": source}}}))
        process = subprocess.run([sys.executable, "-m", "dualref", "analyze", "--config", str(config)],
                                 cwd=tmp_path, env=env, capture_output=True, text=True)
        assert process.returncode == index, process.stderr
    for suite, expected in (("eval_scienceqa_dr_bolt", 7), ("all", 239)):
        code = "import sys; from dualref.cli import main; main(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        process = subprocess.run([sys.executable, "-c", code, "run", "--suite", suite, "--dry-run"],
                                 cwd=tmp_path, env=env, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert json.loads(process.stdout)["job_count"] == expected
    assert not (tmp_path / "outputs").exists()
