"""Experiment graph, artifact references, and a resumable suite runner.

Artifact references have the form ``${job_id.output_key}``.  Stages return a
mapping of named artifacts; references are resolved only after their producer
has completed.  A dry run never imports a model backend or writes artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

REFERENCE = re.compile(r"\$\{([^{}]+)\}")
INTERNAL = ("mix_sft", "reward_sft", "single_ref_fast", "single_ref_slow", "dr_bolt", "mr_dpo", "cond_rw")
EXTERNAL = ("grpo", "fast_grpo", "ton", "dualmind")
TRANSFER = ("mathvision", "dynamath", "mmmu_pro")


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _references(value: Any) -> set[str]:
    if isinstance(value, str):
        return {match.rsplit(".", 1)[0] for match in REFERENCE.findall(value)}
    if isinstance(value, dict):
        return set().union(*(_references(v) for v in value.values()), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(_references(v) for v in value), set())
    return set()


def resolve_references(value: Any, outputs: dict[str, dict]) -> Any:
    """Resolve exact references without stringifying non-string artifact values."""
    def lookup(expression: str) -> Any:
        job, key = expression.rsplit(".", 1)
        if job not in outputs or key not in outputs[job]:
            raise ValueError(f"Unresolved artifact reference: ${{{expression}}}")
        return outputs[job][key]
    if isinstance(value, str):
        match = REFERENCE.fullmatch(value)
        if match:
            return lookup(match.group(1))
        return REFERENCE.sub(lambda m: str(lookup(m.group(1))), value)
    if isinstance(value, dict):
        return {k: resolve_references(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, outputs) for v in value]
    return value


def list_experiments(config: dict) -> list[dict]:
    """Expand the experiment matrix, including diagnostics.

    ``experiments.jobs`` replaces the generated matrix with a user-defined DAG.
    Shared defaults are set in configs/experiments.yaml.
    """
    custom = config.get("experiments", {}).get("jobs")
    if custom is not None:
        jobs = copy.deepcopy(custom)
        root = Path(config.get("output_dir", "outputs/experiments"))
        for job in jobs:
            settings = job.setdefault("config", {})
            directory = settings.setdefault("output_dir", str(root / "runs" / job["id"]))
            if job["stage"] == "train":
                settings.setdefault("training", {}).setdefault("output_dir", directory)
            if job["stage"] == "evaluate":
                settings.setdefault("evaluation", {}).setdefault("output", directory)
        _validate(jobs)
        return jobs
    root = Path(config.get("output_dir", "outputs/experiments"))
    jobs: list[dict] = []
    datasets = config.get("datasets", {})
    main_steps = config.get("experiments", {}).get("main_steps", 1000)
    diagnostic_steps = config.get("experiments", {}).get("diagnostic_steps", 500)
    refs = config.get("model", {})
    fast = refs.get("fast", "Qwen/Qwen3-VL-8B-Instruct")
    slow = refs.get("slow", "Qwen/Qwen3-VL-8B-Thinking")

    def ref(job: str, key: str) -> str:
        return "${" + job + "." + key + "}"

    def add(identifier: str, stage: str, override: dict, suites: tuple[str, ...], depends: tuple[str, ...] = ()) -> str:
        directory = str(root / "runs" / identifier)
        override = merge({"output_dir": directory}, override)
        if stage == "train":
            override = merge({"training": {"output_dir": directory}}, override)
        if stage == "evaluate":
            override = merge({"evaluation": {"output": directory}}, override)
        jobs.append({"id": identifier, "stage": stage, "depends_on": sorted(set(depends) | _references(override)), "suites": list(suites), "config": override})
        return identifier

    prepared: dict[str, str] = {}
    for name in ("scienceqa", "virl39k", *TRANSFER, "3dsrbench"):
        prepared[name] = add(f"prepare_{name}", "prepare", {"data": merge({"dataset": name, "output_dir": str(root / "data" / name)}, datasets.get(name, {}))}, ("data",))

    scored: dict[str, str] = {}
    reference_scored: dict[str, str] = {}
    warmups: dict[str, str] = {}
    evaluations: dict[tuple[str, str], str] = {}
    trains: dict[tuple[str, str], str] = {}

    def make_bank(dataset: str, suffix: str = "", fast_count: int = 7, slow_count: int = 3) -> str:
        identifier = f"{dataset}{suffix}"
        bank = add(f"bank_{identifier}", "bank", {"data": {"dataset": dataset, "train_file": ref(prepared[dataset], "train")}, "generation": {"max_new_tokens": 2048 if dataset == "scienceqa" else 8192}, "bank": {"action": "generate", "input": ref(prepared[dataset], "train"), "num_fast": fast_count, "num_slow": slow_count}}, ("banks",))
        return add(f"score_{identifier}", "score", {"data": {"dataset": dataset}, "bank": {"path": ref(bank, "bank"), "score_references": False}}, ("banks",))

    for dataset in ("scienceqa", "virl39k", "3dsrbench"):
        scored[dataset] = make_bank(dataset)
        warmups[dataset] = add(f"warmup_{dataset}", "train", {"data": {"dataset": dataset, "train_file": ref(prepared[dataset], "train")}, "bank": {"path": ref(scored[dataset], "bank")}, "training": {"method": "mix_sft", "phase": "warmup", "steps": 100, "warmup_steps": 0, "balanced_sources": True}}, ("warmup",))

    def evaluate(identifier: str, dataset: str, checkpoint: str, suites: tuple[str, ...], extra: dict | None = None) -> str:
        cfg = {"generation": {"max_new_tokens": 2048 if dataset == "scienceqa" else 8192}, "evaluation": {"dataset": dataset, "input": ref(prepared[dataset], "test"), "checkpoint": checkpoint, "repeats": 1, "max_new_tokens": 2048 if dataset == "scienceqa" else 8192}}
        return add(identifier, "evaluate", merge(cfg, extra or {}), suites)

    def train_variant(dataset: str, label: str, method: str, suites: tuple[str, ...], overrides: dict | None = None, bank_id: str | None = None) -> str:
        bank_id = bank_id or scored[dataset]
        if method in ("single_ref_fast", "single_ref_slow", "mr_dpo") or (overrides or {}).get("objective", {}).get("correction") == "exact":
            if bank_id not in reference_scored:
                reference_scored[bank_id] = add(f"references_{bank_id}", "score", {"data": {"dataset": dataset}, "bank": {"path": ref(bank_id, "bank"), "score_references": True}}, ("references",))
            bank_id = reference_scored[bank_id]
        cfg = merge({"data": {"dataset": dataset, "train_file": ref(prepared[dataset], "train")}, "generation": {"max_new_tokens": 2048 if dataset == "scienceqa" else 8192}, "bank": {"path": ref(bank_id, "bank")}, "training": {"method": method, "phase": "policy", "steps": main_steps, "warmup_steps": 0, "initial_checkpoint": ref(warmups[dataset], "checkpoint"), "balanced_sources": False}}, overrides or {})
        if dataset == "scienceqa" and label in ("dr_bolt", "fast_grpo", "grpo"):
            cfg = merge(cfg, {"training": {"eval_every": 25}, "evaluation": {"dataset": dataset, "input": ref(prepared[dataset], "test")}})
        if method not in EXTERNAL:
            weighted = add(f"weights_{dataset}_{label}", "weights", cfg, suites)
            cfg["bank"]["path"] = ref(weighted, "bank")
        else:
            # External recipes own their SFT/rollout phases; no DR warm-up is injected.
            cfg["training"].pop("initial_checkpoint", None)
            cfg["bank"].pop("path", None)
            cfg["training"]["train_file"] = ref(prepared[dataset], "train")
            cfg["external"] = {"method": method}
        if method == "mr_dpo":
            cfg["training"]["response_token_budget"] = ref(f"train_{dataset}_dr_bolt", "response_tokens")
        trained = add(f"train_{dataset}_{label}", "train", cfg, suites)
        trains[(dataset, label)] = trained
        targets = ("scienceqa",) if dataset == "scienceqa" else (("3dsrbench",) if dataset == "3dsrbench" else TRANSFER)
        for target in targets:
            # This suite evaluates DualMindVLM on ScienceQA and MathVision.
            if method == "dualmind" and target in ("dynamath", "mmmu_pro"):
                continue
            evaluations[(target, label)] = evaluate(f"eval_{target}_{label}", target, ref(trained, "checkpoint"), suites)
        return trained

    for dataset in ("scienceqa", *TRANSFER, "3dsrbench"):
        for label, checkpoint in (("fast_reference", fast), ("slow_reference", slow)):
            evaluations[(dataset, label)] = evaluate(f"eval_{dataset}_{label}", dataset, checkpoint, ("main", "offline") if dataset != "3dsrbench" else ("3dsrbench",), {"evaluation": {"reference_role": "fast" if label == "fast_reference" else "slow"}})
    for dataset in ("scienceqa", "virl39k"):
        for method in INTERNAL:
            train_variant(dataset, method, method, ("main", "offline"))
        for method in EXTERNAL:
            train_variant(dataset, method, method, ("main", "external"))
        for method in ("two_policy_router", "thinking_prompt_router"):
            routed = add(f"router_{dataset}_{method}", "router", {"data": {"dataset": dataset, "train_file": ref(prepared[dataset], "train")}, "bank": {"path": ref(scored[dataset], "bank")}, "router": {"kind": method}}, ("routers",))
            for target in (("scienceqa",) if dataset == "scienceqa" else TRANSFER):
                evaluations[(target, method)] = evaluate(f"eval_{target}_{method}", target, slow, ("routers",), {"evaluation": {"router": ref(routed, "router"), "router_kind": method}})
    train_variant("3dsrbench", "dr_bolt", "dr_bolt", ("3dsrbench",))

    for correction in ("none", "exact"):
        train_variant("virl39k", f"correction_{correction}", "dr_bolt", ("correction",), {"objective": {"correction": correction}})

    sensitivity = [("alpha", .3), ("alpha", .7), ("q", .3), ("q", .5), ("q", .7), ("beta", .5), ("beta", 2.), ("length_penalty", 0.), ("length_penalty", .025), ("length_penalty", .1)]
    superset = make_bank("scienceqa", "_superset", 7, 7)
    test_bank = add("bank_scienceqa_test", "bank", {"data": {"dataset": "scienceqa", "train_file": ref(prepared["scienceqa"], "test")}, "generation": {"max_new_tokens": 2048}, "bank": {"action": "generate", "input": ref(prepared["scienceqa"], "test"), "num_fast": 7, "num_slow": 3}}, ("calibration", "figures"))
    test_bank = add("score_scienceqa_test", "score", {"data": {"dataset": "scienceqa"}, "bank": {"path": ref(test_bank, "bank")}}, ("calibration", "figures"))
    calibration_jobs: dict[str, str] = {}
    for label, trained in [("dr_bolt", trains[("scienceqa", "dr_bolt")])]:
        calibration_jobs[label] = add(f"calibration_scienceqa_{label}", "diagnostics", {"data": {"dataset": "scienceqa"}, "bank": {"path": ref(test_bank, "bank")}, "evaluation": {"dataset": "scienceqa", "input": ref(prepared["scienceqa"], "test"), "checkpoint": ref(trained, "checkpoint")}, "diagnostics": {"kind": "calibration", "prompts": 100, "repeats": 20, "bins": 5}}, ("calibration", "sensitivity"))
    for parameter, value in sensitivity:
        label = f"{parameter}_{str(value).replace('.', 'p')}"
        bank_id = scored["scienceqa"]
        if parameter == "q":
            subset = add(f"bank_scienceqa_{label}", "bank", {"data": {"dataset": "scienceqa"}, "bank": {"action": "subset", "path": ref(superset, "bank"), "num_fast": round(10 * value), "num_slow": 10 - round(10 * value)}}, ("sensitivity",))
            bank_id = add(f"score_scienceqa_{label}", "score", {"data": {"dataset": "scienceqa"}, "bank": {"path": ref(subset, "bank")}}, ("sensitivity",))
        overrides = {"objective": {parameter: value}}
        trained = train_variant("scienceqa", label, "dr_bolt", ("sensitivity",), overrides, bank_id)
        calibration_jobs[label] = add(f"calibration_scienceqa_{label}", "diagnostics", merge(overrides, {"data": {"dataset": "scienceqa"}, "bank": {"path": ref(test_bank, "bank")}, "evaluation": {"dataset": "scienceqa", "input": ref(prepared["scienceqa"], "test"), "checkpoint": ref(trained, "checkpoint")}, "diagnostics": {"kind": "calibration", "prompts": 100, "repeats": 20, "bins": 5}}), ("sensitivity",))
    for value in (0., .025, .1):
        train_variant("virl39k", f"length_penalty_{str(value).replace('.', 'p')}", "dr_bolt", ("sensitivity",), {"objective": {"length_penalty": value}})

    cross_jobs: dict[str, str] = {}
    diagnostic_trains: dict[str, str] = {}
    for path in ("main", "ablation"):
        cross_jobs[path] = add(f"cross_likelihood_{path}", "diagnostics", {"data": {"dataset": "scienceqa", "train_file": ref(prepared["scienceqa"], "train")}, "bank": {"path": ref(scored["scienceqa"], "bank"), "input": ref(prepared["scienceqa"], "train"), "limit": 1000}, "diagnostics": {"kind": "cross_likelihood", "path": path, "samples_per_mode": 1000, "replay_input": ref(prepared["scienceqa"], "validation")}}, ("diagnostics",))
        for bucket, correction in (("mode_wise", "modewise"), ("exact", "exact"), ("single", "single"), ("mode_wise_full", "modewise")):
            identifier = f"diagnostic_{path}_{bucket}"
            cfg = {"data": {"dataset": "scienceqa"}, "bank": {"path": ref(cross_jobs[path], "bank")}, "training": {"method": "dr_bolt", "phase": "diagnostic", "initial_checkpoint": ref(warmups["scienceqa"], "checkpoint"), "steps": diagnostic_steps, "warmup_steps": 0, "replay_file": ref(cross_jobs[path], "replay_bank"), "group_normalize": bucket != "mode_wise_full", "filter_all_wrong": bucket != "mode_wise_full", "diagnostics": {"enabled": True, "replay_every": 25, "replay_samples": 200}}, "objective": {"alpha": .5, "q": .7, "correction": correction, "normalize_groups": bucket != "mode_wise_full", "drop_all_wrong": bucket != "mode_wise_full"}}
            weights = add(f"weights_{identifier}", "weights", cfg, ("diagnostics",))
            cfg["bank"]["path"] = ref(weights, "bank")
            diagnostic_trains[identifier] = add(f"train_{identifier}", "train", cfg, ("diagnostics",))

    def source(job: str, key: str, **metadata: Any) -> dict:
        return {"path": ref(job, key), **metadata}

    def report(identifier: str, kind: str, sources: dict, suites: tuple[str, ...], **options: Any) -> str:
        return add(identifier, "analyze", {"analysis": {"kind": kind, "sources": sources, **options}}, suites)

    def evaluation_sources(methods: tuple[str, ...], targets: tuple[str, ...] = ("scienceqa", *TRANSFER)) -> dict:
        return {f"{dataset}/{method}": source(evaluations[(dataset, method)], "predictions", dataset=dataset, method=method) for dataset in targets for method in methods if (dataset, method) in evaluations}

    report("table_01_external", "benchmark_table", evaluation_sources((*EXTERNAL, "dr_bolt")), ("main", "external"))
    report("table_02_internal", "benchmark_table", evaluation_sources(("fast_reference", "slow_reference", *INTERNAL[:5])), ("main", "offline"))
    report("table_03_offline", "benchmark_table", evaluation_sources(("mr_dpo", "cond_rw", "dr_bolt")), ("offline",))
    report("table_13_correction", "benchmark_table", evaluation_sources(("correction_none", "correction_exact", "dr_bolt"), TRANSFER), ("correction",))
    sensitivity_sources = evaluation_sources(tuple(calibration_jobs), ("scienceqa",))
    sensitivity_sources.update({f"calibration/{label}": source(job, "records", kind="calibration", method=label) for label, job in calibration_jobs.items()})
    report("table_14_sensitivity", "sensitivity", sensitivity_sources, ("sensitivity",))
    report("table_15_transfer_length", "benchmark_table", evaluation_sources(("length_penalty_0p0", "length_penalty_0p025", "dr_bolt", "length_penalty_0p1"), TRANSFER), ("sensitivity",))
    report("table_16_routers", "benchmark_table", evaluation_sources(("two_policy_router", "thinking_prompt_router", "dr_bolt")), ("routers",))
    report("table_17_parsing", "parser_cases", evaluation_sources(("dr_bolt",)), ("qualitative", "protocol"))
    report("table_21_3dsrbench", "benchmark_table", evaluation_sources(("fast_reference", "slow_reference", "dr_bolt"), ("3dsrbench",)), ("3dsrbench",))
    report("figure_01_difficulty", "difficulty", {**evaluation_sources(("dr_bolt", "fast_grpo", "grpo"), ("scienceqa",)), "difficulty_bank": source(test_bank, "bank", kind="bank")}, ("figures",))
    report("figure_02_learning", "trajectory", {method: source(trains[("scienceqa", method)], "history", method=method) for method in ("dr_bolt", "fast_grpo", "grpo")}, ("figures",))
    report("figure_03_calibration", "calibration", {"default": source(calibration_jobs["dr_bolt"], "records")}, ("calibration",))
    for path, job in cross_jobs.items():
        report(f"separation_{path}", "cross_likelihood", {path: source(job, "records")}, ("diagnostics",))
        qs = (.5, .3, .7) if path == "main" else (.3,)
        for q in qs:
            report(f"correction_{path}_q{str(q).replace('.', 'p')}", "correction", {path: source(job, "records")}, ("diagnostics",), alpha=.5, q=q)
        report(f"dynamics_{path}", "trajectory", {bucket: source(diagnostic_trains[f"diagnostic_{path}_{bucket}"], "history", method=bucket) for bucket in ("mode_wise", "exact", "single", "mode_wise_full")}, ("diagnostics",))
    report("bank_coverage", "coverage", {"virl39k": source(scored["virl39k"], "bank")}, ("coverage", "diagnostics"))
    report("qualitative_cases", "qualitative", {"virl39k": source(scored["virl39k"], "bank", dataset="virl39k"), **evaluation_sources(("dr_bolt",), TRANSFER)}, ("qualitative",), case_ids=config.get("experiments", {}).get("qualitative_case_ids", {}))
    measured = {job["id"]: source(job["id"], "_result", stage=job["stage"]) for job in jobs if job["stage"] not in ("analyze", "prepare")}
    report("runtime_provenance", "resources", measured, ("resources",))
    report("protocol_tables", "protocol", {}, ("protocol",), resolved_config=config)
    report_ids = [job["id"] for job in jobs if job["stage"] == "analyze"]
    report("experiment_report", "report", {identifier: source(identifier, "summary") for identifier in report_ids}, ("report",))
    _validate(jobs)
    return jobs


def _validate(jobs: list[dict]) -> None:
    ids = [job["id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("Experiment IDs must be unique")
    directories = {}
    for job in jobs:
        cfg = job.get("config", {})
        paths = {cfg.get("output_dir")}
        if job["stage"] == "train":
            paths.add(cfg.get("training", {}).get("output_dir"))
        if job["stage"] == "evaluate":
            paths.add(cfg.get("evaluation", {}).get("output"))
        for path in paths - {None}:
            resolved = str(Path(path).resolve()).casefold()
            if resolved in directories and directories[resolved] != job["id"]:
                raise ValueError(f"Output directory shared by {directories[resolved]} and {job['id']}: {path}")
            directories[resolved] = job["id"]
    known = set(ids)
    remaining = {job["id"]: set(job.get("depends_on", [])) | _references(job.get("config", {})) for job in jobs}
    for identifier, dependencies in remaining.items():
        if dependencies - known:
            raise ValueError(f"{identifier}: unknown dependencies {sorted(dependencies - known)}")
    completed: set[str] = set()
    while remaining:
        ready = {identifier for identifier, deps in remaining.items() if deps <= completed}
        if not ready:
            raise ValueError(f"Cyclic experiment dependencies: {sorted(remaining)}")
        completed.update(ready)
        remaining = {identifier: deps for identifier, deps in remaining.items() if identifier not in ready}


def _select(jobs: list[dict], suite: str) -> list[dict]:
    by_id = {job["id"]: job for job in jobs}
    requested = set(suite.split(","))
    selected = set(by_id) if "all" in requested else {job["id"] for job in jobs if requested.intersection(job.get("suites", [])) or job["id"] in requested}
    if not selected:
        raise ValueError(f"Unknown or empty suite: {suite}")
    pending = list(selected)
    while pending:
        identifier = pending.pop()
        for dependency in set(by_id[identifier].get("depends_on", [])) | _references(by_id[identifier].get("config", {})):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    ordered: list[dict] = []
    completed: set[str] = set()
    while selected - completed:
        for job in jobs:
            deps = set(job.get("depends_on", [])) | _references(job.get("config", {}))
            if job["id"] in selected - completed and deps <= completed:
                ordered.append(job)
                completed.add(job["id"])
    return ordered


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_suite(config: dict, suite: str = "all", dry_run: bool = False) -> dict:
    """Execute a dependency-closed suite and validate its persisted results."""
    jobs = _select(list_experiments(config), suite)
    root = Path(config.get("output_dir", "outputs/experiments"))
    if dry_run:
        return {"status": "planned", "suite": suite, "job_count": len(jobs), "jobs": jobs}
    from .cli import dispatch
    from .analysis import REPORT_VERSION
    from .artifacts import validate_result
    outputs: dict[str, dict] = {}
    records: list[dict] = []
    failed: set[str] = set()
    partial: set[str] = set()
    blocked: set[str] = set()
    referenced = {job["id"]: set() for job in jobs}

    def collect(value):
        if isinstance(value, str):
            for expression in REFERENCE.findall(value):
                producer, key = expression.rsplit(".", 1)
                referenced[producer].add(key)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for job in jobs:
        collect(job.get("config", {}))
    fail_fast = config.get("experiments", {}).get("fail_fast", True)
    resume = config.get("experiments", {}).get("resume", True)
    ledger = root / "suite-results.json"
    for job in jobs:
        identifier = job["id"]
        dependencies = set(job.get("depends_on", [])) | _references(job.get("config", {}))
        aggregate = job["stage"] == "analyze" and job.get("config", {}).get("analysis", {}).get("kind") == "report"
        if (failed and fail_fast or dependencies & (failed | blocked)
                or dependencies & partial and not aggregate):
            blocked.add(identifier)
            records.append({"id": identifier, "status": "blocked", "reason": "A required stage is incomplete or execution stopped after a failure"})
            continue
        started = time.monotonic()
        result_path = root / "runs" / identifier / "job-result.json"
        try:
            stage_config = copy.deepcopy(config)
            # Future jobs' references belong to the graph, not this stage's inputs.
            stage_config.get("experiments", {}).pop("jobs", None)
            cfg = resolve_references(merge(stage_config, job.get("config", {})), outputs)
            payload = {"config": cfg, "dependencies": {dependency: outputs[dependency].get("_fingerprint") for dependency in sorted(dependencies)}}
            if job["stage"] == "analyze":
                payload["report_version"] = REPORT_VERSION
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
            result_path = Path(cfg.get("output_dir", root / "runs" / identifier)) / "job-result.json"
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8")) if resume and result_path.exists() else {}
            except (ValueError, OSError):
                cached = {}
            if not isinstance(cached, dict):
                cached = {}
            cache_outputs = cached.get("outputs", {})
            if cached.get("fingerprint") == digest and cached.get("status") == "completed":
                try:
                    valid = validate_result(job["stage"], cfg, cache_outputs, referenced[identifier])
                except (ValueError, OSError, TypeError, KeyError):
                    valid = "invalid"
                if valid == "completed":
                    outputs[identifier] = cache_outputs
                    records.append({"id": identifier, "status": "cached", "outputs": cache_outputs})
                    continue
            produced = dispatch(job["stage"], cfg)
            state = validate_result(job["stage"], cfg, produced, referenced[identifier])
            if state == "partial":
                partial.add(identifier)
            outputs[identifier] = {**produced, "_result": str(result_path), "_fingerprint": digest}
            record = {"id": identifier, "stage": job["stage"], "status": state, "fingerprint": digest, "elapsed_seconds": time.monotonic() - started, "outputs": outputs[identifier]}
            _write_json(result_path, record)
            records.append(record)
        except Exception as error:
            failed.add(identifier)
            record = {"id": identifier, "stage": job["stage"], "status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started}
            _write_json(result_path, record)
            records.append(record)
        _write_json(ledger, {"suite": suite, "status": "running", "jobs": records})
    result = {"suite": suite, "status": "failed" if failed else "partial" if partial or blocked else "completed", "jobs": records, "ledger": str(ledger)}
    _write_json(ledger, result)
    return result
