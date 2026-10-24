"""Public commands; expensive dependencies are imported only by their stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import config_path, load_config, merge, read_config, write_config


def dispatch(stage: str, cfg: dict) -> dict:
    if stage == "prepare":
        from .data import prepare_dataset

        settings = cfg.get("data", {})
        dataset = settings.get("dataset") or settings.get("name", "scienceqa")
        output = settings.get("output_dir") or str(Path(settings.get("root", "data")) / dataset)
        sources = read_config(config_path("datasets.yaml"))
        return prepare_dataset(dataset, output, merge(sources, {"datasets": {dataset: settings}}))
    if stage in ("bank", "score", "weights"):
        from .bank import build_bank, construct_weights, score_bank

        return {"bank": build_bank, "score": score_bank, "weights": construct_weights}[stage](cfg)
    if stage == "train":
        if cfg.get("training", {}).get("method") in ("grpo", "fast_grpo", "ton", "dualmind"):
            from .baselines import run

            return run(cfg)
        workers = cfg.get("training", {}).get("world_size", 4)
        if workers > 1 and "RANK" not in os.environ:
            output = Path(cfg.get("training", {}).get("output_dir") or cfg.get("output_dir", "outputs/train"))
            configuration = write_config(output / "launch-config.yaml", cfg)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc-per-node={workers}", "--module", "dualref", "train",
                       "--config", str(configuration)]
            subprocess.run(command, check=True)
            return json.loads((output / "result.json").read_text())
        from .training import train

        return train(cfg)
    if stage == "evaluate":
        if cfg.get("evaluation", {}).get("router"):
            from .routing import evaluate_router

            return evaluate_router(cfg)
        from .evaluation import evaluate

        return evaluate(cfg)
    if stage == "router":
        from .routing import train_router

        return train_router(cfg)
    if stage == "diagnostics":
        from .diagnostics import run_diagnostics

        return run_diagnostics(cfg)
    if stage == "analyze":
        from .analysis import run_analysis

        return run_analysis(cfg)
    raise ValueError(f"Unknown stage: {stage}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualref", description="Train and evaluate adaptive vision-language models")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", type=Path, help="YAML configuration merged over package defaults")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Dotted YAML override; repeatable")
    commands = parser.add_subparsers(dest="command", required=True)
    for stage in ("prepare", "bank", "score", "weights", "train", "evaluate", "router", "diagnostics", "analyze"):
        command = commands.add_parser(stage, parents=[common])
        if stage == "prepare":
            command.add_argument("--dataset", help="Dataset name")
            command.add_argument("--external", choices=["grpo", "fast_grpo", "ton", "dualmind", "all"], help="Prepare isolated upstream baseline environments")
    for name in ("run", "list"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("--suite", default="all", help="Experiment suite to run")
        if name == "run":
            command.add_argument("--dry-run", action="store_true", help="Expand jobs without writing files or loading models")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    path = args.config
    if path is None and args.command in ("run", "list"):
        path = config_path("experiments.yaml")
    cfg = load_config(path, args.overrides)
    if args.command == "prepare" and args.dataset:
        cfg.setdefault("data", {})["dataset"] = args.dataset
    if args.command == "prepare" and args.external:
        from .baselines import prepare

        methods = ["grpo", "fast_grpo", "ton", "dualmind"] if args.external == "all" else [args.external]
        results = {}
        for method in methods:
            results[method] = prepare(merge(cfg, {"external": {"method": method}}))
        result = results
    elif args.command in ("run", "list"):
        from .experiments import run_suite

        result = run_suite(cfg, args.suite, dry_run=args.command == "list" or args.dry_run)
    else:
        result = dispatch(args.command, cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result.get("status") in {"failed", "partial", "blocked", "missing", "unavailable", "incomplete", "not_reported"}:
        raise SystemExit(1)
