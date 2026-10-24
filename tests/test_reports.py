import json
from pathlib import Path
import sys

import pytest

from dualref import analysis
from dualref.experiments import list_experiments


def analyze(tmp_path, kind, sources, **settings):
    result = analysis.run_analysis({"output_dir": str(tmp_path / kind),
                                   "analysis": {"kind": kind, "sources": sources, **settings}})
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    return result, summary


def test_complete_summary_is_preserved(tmp_path, write_records):
    source = write_records("predictions.jsonl", [
        {"correct": 1., "words": 2, "slow": False},
        {"correct": 0., "words": 4, "slow": True}])
    child, summary = analyze(tmp_path, "benchmark_table", {"scienceqa/dr": source})
    parent, combined = analyze(tmp_path, "report", {"table": child["summary"]})
    assert parent["status"] == "complete"
    assert combined["reports"]["table"] == summary
    assert summary["rows"][0]["n"] == 2
    assert any(path.endswith(".svg") for path in child["plots"])
    assert any(path.endswith(".pdf") for path in child["plots"])
    assert all(Path(path).stat().st_size for path in child["plots"])


def test_aggregate_metric_summaries_and_reject_mixed_granularity(tmp_path):
    sources = {}
    for dataset, n, score in (("mathvision", 10, .8), ("dynamath", 30, .4)):
        path = tmp_path / (dataset + ".json")
        path.write_text(json.dumps({"n": n, "score": score, "length": 100, "slow_ratio": .4}))
        sources[dataset + "/dr"] = str(path)
    _, summary = analyze(tmp_path, "benchmark_table", sources)
    aggregate = summary["transfer_aggregate"][0]
    assert aggregate["n"] == 40
    assert aggregate["score"] == pytest.approx(.5)
    assert analysis.benchmark_metrics([{"score": None}])["status"] == "incomplete"
    with pytest.raises(ValueError, match="Cannot combine"):
        analysis.benchmark_metrics([{"score": .8, "n": 1}, {"correct": 1, "words": 1, "slow": False}])


@pytest.mark.parametrize("kind,rows", [
    ("benchmark_table", [{"response": "A"}]),
    ("calibration", [{"prompt_id": "x", "predicted_fast_probability": 2, "fast_count": 1, "total": 1}]),
    ("coverage", [{"correct": 1}]),
    ("difficulty", [{"id": "x", "correct": 1, "words": 2}]),
    ("trajectory", [{"step": 1}]),
])
def test_zero_effective_observations_are_not_complete(tmp_path, write_records, kind, rows):
    path = write_records("input.jsonl", rows)
    result, summary = analyze(tmp_path, kind, {"test": path})
    assert result["status"] == "missing"
    assert summary["valid_observations"] == 0


def test_partial_benchmark_and_single_source_matrix(tmp_path, write_records):
    path = write_records("benchmark.jsonl", [{"correct": 1, "words": 2, "slow": False}, {"response": "A"}])
    result, summary = analyze(tmp_path, "benchmark_table", {"test": path})
    assert result["status"] == "partial"
    assert summary["rows"][0]["status"] == "incomplete"
    path = write_records("likelihood.jsonl", [{"source": "fast", "logp_fast": -2, "logp_slow": -3, "token_count": 2}])
    result, _ = analyze(tmp_path, "cross_likelihood", {"test": path})
    assert result["status"] == "partial"


@pytest.mark.parametrize("count_key", ["total", "repeats", "n"])
def test_calibration_requires_observed_denominator(count_key):
    record = {"prompt_id": "x", "predicted_fast_probability": .5, "empirical_fast_rate": .5}
    with pytest.raises(ValueError, match="observed"):
        analysis.calibration_metrics([record])
    record[count_key] = 8
    assert analysis.calibration_metrics([record])["n_decodes"] == 8


def test_unrequested_combinations_are_not_synthesized():
    jobs = {job["id"]: job for job in list_experiments({})}
    settings = jobs["table_01_external"]["config"]["analysis"]
    data = {name: [{"correct": 1, "words": 2, "slow": False}] for name in settings["sources"]}
    table = analysis._benchmark_table(data, settings["sources"], settings)
    assert len(table["rows"]) == 18
    assert not any(row["dataset"] in {"dynamath", "mmmu_pro"} and row["method"] == "dualmind" for row in table["rows"])
    assert analysis._resources({}, {})["comparisons"] == []
    assert analysis._qualitative({}, {}, {})["cases"] == []


def test_trajectory_only_emits_observed_metrics():
    data = {"external": [{"step": 25, "score": .8, "length": 5}],
            "diagnostic": [{"step": 25, "loss": 2., "grad_norm": 1., "implicit_kl": .3}]}
    rows = {row["method"]: row for row in analysis._trajectory(data)["rows"]}
    assert set(rows["external"]) == {"method", "n_points"}
    assert rows["diagnostic"]["implicit_kl_step_25"] == .3
    assert "implicit_kl_step_50" not in rows["diagnostic"]


def test_qualitative_image_status_and_scope(tmp_path, write_records):
    from PIL import Image

    record = {"example_id": "x", "question": "Example", "images": ["picture.png"],
              "choices": [], "answer": "1", "response": "1"}
    path = write_records("examples.jsonl", [record])
    sources = {"mathvision/dr": {"path": path, "dataset": "mathvision"}}
    result, summary = analyze(tmp_path, "qualitative", sources, case_ids={"mathvision": "x"})
    assert result["status"] == "partial"
    assert len(summary["cases"]) == 1
    assert summary["cases"][0]["asset_issues"]
    assert "Image file not found" in Path(result["report"]).read_text(encoding="utf-8")
    Image.new("RGB", (4, 4)).save(tmp_path / "picture.png")
    result, summary = analyze(tmp_path, "qualitative", sources, case_ids={"mathvision": "x"})
    assert result["status"] == "complete"
    assert summary["cases"][0]["status"] == "measured"
    assert result["images"] and all(Path(image).is_file() for image in result["images"])


def test_legitimate_undefined_correlation():
    result = analysis.calibration_metrics([
        {"prompt_id": str(i), "predicted_fast_probability": .5, "fast_count": 1, "total": 2} for i in range(2)])
    assert result["n_prompts"] == 2
    assert result["pearson"] is None
    assert result["mae"] == 0


def test_resource_comparison_uses_only_requested_training_sources(tmp_path):
    identifiers = ["bank_scienceqa", "score_scienceqa", "warmup_scienceqa",
                   "weights_scienceqa_dr_bolt", "train_scienceqa_dr_bolt"]
    sources = {}
    for name in identifiers:
        path = tmp_path / name / "job-result.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"status": "completed", "stage": "train" if name.startswith("train_") else "bank",
                                    "elapsed_seconds": 1, "outputs": {}}))
        sources[name] = str(path)
    result, summary = analyze(tmp_path, "resources", sources)
    assert result["status"] == "complete"
    assert len(summary["comparisons"]) == 1
    assert summary["comparisons"][0]["end_to_end_hours"] == pytest.approx(5 / 3600)
    assert "speedup" not in summary["comparisons"][0]


def test_source_file_failures_remain_visible(tmp_path, write_records):
    valid = write_records("valid.jsonl", [{"id": "x", "correct": 1}])
    absent = str(tmp_path / "not-created.jsonl")
    result, summary = analyze(tmp_path, "coverage", {"valid": valid, "absent": absent})
    assert result["status"] == "partial"
    assert summary["missing"] == [{"source": "absent", "path": absent, "reason": "Artifact does not exist"}]


def test_old_summary_is_not_silently_unpacked(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"kind": "benchmark_table", "status": "complete", "rows": [{"score": 1}]}))
    result, summary = analyze(tmp_path, "report", {"old": str(path)})
    assert result["status"] == "missing"
    assert not summary["reports"]
    assert summary["missing"]


def test_underflow_measurement_does_not_fabricate_results(monkeypatch):
    assert analysis._measure_underflow([], {})["status"] == "missing"
    monkeypatch.setitem(sys.modules, "torch", None)
    result = analysis._measure_underflow([{"source": "fast", "logp_fast": -2}], {})
    assert result["status"] == "unavailable"
    assert result["rows"] == []
    assert result["error"]
