"""Prepare canonical data and launch upstream trainers in separate environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dualref.config import read_config, write_config
from .state import prepare_history, run_identity


SOURCES = {
    "fast": {
        "url": "https://github.com/Mr-Loevan/FAST.git",
        "revision": "233dff21128b678ce32c62cea8ae75d6d55e4dbf",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "requirements": ["transformers==4.57.0", "vllm==0.11.0", "scipy"],
    },
    "ton": {
        "url": "https://github.com/kokolerk/TON.git",
        "revision": "22102c281b004cd1c286c368c578ced5b37c4572",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "requirements": ["trl==0.14.0", "transformers==4.57.1", "math-verify"],
    },
    "grpo": {
        "url": "https://github.com/huggingface/trl.git",
        "revision": "04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "requirements": ["transformers==4.57.1", "peft==0.17.1"],
    },
    "dualmind": {
        "url": "https://github.com/maifoundations/DualMindVLM.git",
        "revision": "cfe00a8635c8ca12705d9a937d513c75a988808d",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "requirements": ["transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1"],
    },
}

ALIASES = {"fast-grpo": "fast", "fast_grpo": "fast", "dualmindvlm": "dualmind"}


def method_name(config: dict) -> str:
    name = config.get("external", {}).get("method")
    if name is None:
        candidate = config.get("training", {}).get("method", "grpo").lower()
        name = candidate if candidate in SOURCES or candidate in ALIASES else "grpo"
    name = name.lower()
    name = ALIASES.get(name, name)
    if name not in SOURCES:
        raise ValueError(f"Unknown external baseline: {name}")
    return name


def read_examples(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["id"]) in seen:
                raise ValueError(f"Duplicate example ID: {row['id']}")
            seen.add(str(row["id"]))
            row["images"] = [str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
                             for p in row.get("images", [])]
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
            rows.append(row)
    if not rows:
        raise ValueError("The training dataset is empty")
    return rows


def question_text(example: dict) -> str:
    choices = example.get("choices", [])
    text = example["question"]
    if choices:
        text += "\n" + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    return text


def fast_record(example: dict) -> dict:
    """The answer envelope carries benchmark metadata into upstream reward hooks."""
    return {
        "id": str(example["id"]),
        "problem": "<image>" * len(example.get("images", [])) + question_text(example),
        "answer": json.dumps(example, ensure_ascii=False),
        "images": example.get("images", []),
        "image_complexity": float(example.get("metadata", {}).get("image_complexity", 0.5)),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _checkout(source: dict, path: Path) -> None:
    fresh = not path.exists()
    if not path.exists():
        subprocess.run(["git", "clone", "--no-checkout", source["url"], str(path)], check=True)
    elif not (path / ".git").is_dir():
        raise ValueError(f"External source path is not a Git checkout: {path}")
    if not fresh:
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise ValueError(f"Refusing to change a modified external checkout: {path}")
    subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", "origin", source["revision"]], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["revision"]], check=True)


def _fast_config(config: dict, source_dir: Path, data: Path, output: Path, rows: list[dict]) -> dict:
    generation = config.get("generation", {})
    training = config.get("training", {})
    external = config.get("external", {})
    return {
        "data": {
            "train_files": str(data), "val_files": [], "val_names": [],
            "prompt_key": "problem", "answer_key": "answer", "image_key": "images",
            "format_prompt": None, "override_chat_template": str(output / "chat_template.jinja"),
            "max_prompt_length": int(training.get("max_prompt_length", 4096)),
            "max_response_length": int(generation.get("max_new_tokens", 8192)),
            "rollout_batch_size": int(external.get("rollout_batch_size", 32)),
            "mini_rollout_batch_size": int(external.get("rollout_batch_size", 32)),
            "seed": int(config.get("seed", 42)), "shuffle": True,
            "min_pixels": 3136, "max_pixels": 1048576, "filter_overlong_prompts": True,
        },
        "algorithm": {
            "adv_estimator": "grpo", "disable_kl": False, "use_kl_loss": True,
            "kl_penalty": "low_var_kl", "kl_coef": 0.01,
            "kl_type": "group_accuracy_based", "kl_min_coef": 0.001, "kl_max_coef": 0.01,
            "online_filtering": True, "filter_key": "accuracy", "filter_low": 0.01, "filter_high": 0.99,
            "dynamic_filter_schedule": [
                {"epoch_ratio": 0.5, "filter_low": 0.3, "filter_high": 0.99},
                {"epoch_ratio": 1.0, "filter_low": 0.01, "filter_high": 0.7},
            ],
        },
        "worker": {
            "actor": {
                "global_batch_size": 32, "micro_batch_size_per_device_for_update": 1,
                "micro_batch_size_per_device_for_experience": 1,
                "padding_free": True, "dynamic_batching": True, "ulysses_size": 1,
                "model": {"model_path": external["model"], "enable_gradient_checkpointing": True},
                "optim": {"lr": float(training.get("learning_rate", 1e-6)), "weight_decay": 0.01},
                "fsdp": {"enable_full_shard": True, "enable_cpu_offload": False},
                "offload": {"offload_params": True, "offload_optimizer": True},
            },
            "rollout": {
                "n": 8, "temperature": float(generation.get("temperature", 0.7)),
                "top_p": float(generation.get("top_p", 0.9)),
                "limit_images": max(1, max(len(x.get("images", [])) for x in rows)),
                "tensor_parallel_size": 1, "gpu_memory_utilization": 0.6,
            },
            "ref": {"fsdp": {"enable_full_shard": True, "enable_cpu_offload": True}},
            "reward": {
                "reward_type": "batch", "reward_function": str(Path(__file__).with_name("fast_reward.py")) + ":compute_score",
            },
        },
        "trainer": {
            "total_epochs": 1000, "max_steps": int(training.get("steps", 1000)),
            "project_name": "dualref", "experiment_name": "fast", "logger": ["file"],
            "nnodes": 1, "n_gpus_per_node": int(external.get("gpus", training.get("world_size", 4))),
            "val_freq": -1, "val_before_train": False, "save_freq": int(training.get("save_steps", 100)),
            "save_limit": 2, "save_checkpoint_path": str(output / "checkpoints"),
            "max_try_make_batch": 100,
        },
    }


def prepare(config: dict) -> dict:
    """Materialize inputs and pinned sources; install only with external.install=true."""
    config = json.loads(json.dumps(config))
    method = method_name(config)
    external = config.setdefault("external", {})
    source = SOURCES[method]
    external.setdefault("model", source["model"])
    root = Path(external.get("root", "third_party")).resolve()
    source_dir = root / method
    output = Path(config.get("training", {}).get("output_dir", f"outputs/{method}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    path = config.get("training", {}).get("train_file") or config.get("data", {}).get("train_file")
    if not path:
        raise ValueError("training.train_file or data.train_file is required")
    rows = read_examples(path)
    fingerprint = run_identity(config, rows, source)
    old_manifest = output / "external_manifest.json"
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        previous = old.get("fingerprint")
        if old.get("identity_version") != 2:
            saved = output / "external_config.yaml"
            if not saved.is_file() or old.get("revision") != source["revision"]:
                raise ValueError("Existing external output has insufficient run provenance; choose a new output directory")
            old_config = read_config(saved)
            old_rows = read_examples(old_config["training"]["train_file"])
            previous = run_identity(old_config, old_rows, source)
        if previous != fingerprint:
            raise ValueError("Existing external output belongs to different data or settings; choose a new output_dir")
    canonical = output / "train.jsonl"
    if Path(path).resolve() == canonical:
        canonical = output / "prepared_train.jsonl"
    _write_jsonl(canonical, rows)
    external["source_dir"] = str(source_dir)
    external["revision"] = source["revision"]
    config.setdefault("training", {}).update(train_file=str(canonical), output_dir=str(output))
    config_file = write_config(output / "external_config.yaml", config)
    environment = root / "envs" / method
    python = external.get("python", str(environment / "bin/python"))
    common = ["torch==2.8.0", "torchvision==0.23.0", "datasets==4.2.0", "accelerate==1.10.1", "peft==0.17.1",
              "Pillow", "numpy", "psutil", "nvidia-ml-py", "setuptools", "wheel", "ninja", "PyYAML>=6.0,<7"]
    installs = [[python, "-m", "pip", "install", *common, *source["requirements"]]]
    if method == "fast":
        # flash-attn builds need torch already installed in this environment.
        installs.append([python, "-m", "pip", "install", "--no-build-isolation", "-r", str(source_dir / "requirements.txt")])
    if method == "grpo":
        installs.append([python, "-m", "pip", "install", str(source_dir)])
    if external.get("fetch", True):
        _checkout(source, source_dir)
    if external.get("install", False):
        if not external.get("python") and not Path(python).is_file():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        for command in installs:
            subprocess.run(command, check=True)
    command = [python, "-m", "dualref.baselines.worker", "--config", str(config_file)]
    bridge = root / "bridge"
    shutil.copytree(Path(__file__).resolve().parent.parent, bridge / "dualref", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if method == "fast":
        fast_data = output / "fast_train.jsonl"
        _write_jsonl(fast_data, [fast_record(row) for row in rows])
        fast_config = _fast_config(config, source_dir, fast_data, output, rows)
        write_config(output / "fast_config.yaml", fast_config)
    manifest = {
        "method": method, "source": source["url"], "revision": source["revision"],
        "source_dir": str(source_dir), "config": str(config_file), "command": command,
        "install_commands": installs, "output_dir": str(output), "examples": len(rows),
        "fingerprint": fingerprint,
        "identity_version": 2,
        "bridge": str(bridge),
    }
    (output / "external_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(config: dict) -> dict:
    manifest = prepare(config)
    external = config.get("external", {})
    if external.get("prepare_only", False):
        return manifest
    env = os.environ.copy()
    # Never add the host environment's site-packages: it would override pinned dependencies.
    env["PYTHONPATH"] = manifest["bridge"]
    env["TOKENIZERS_PARALLELISM"] = "false"
    command = manifest["command"]
    if not Path(command[0]).is_file():
        raise FileNotFoundError("External Python environment not found. Run prepare with external.install=true, "
                                "or set external.python to a prepared isolated environment")
    actual_revision = subprocess.check_output(
        ["git", "-C", manifest["source_dir"], "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_revision != manifest["revision"]:
        raise ValueError("External checkout does not match the pinned revision; rerun prepare with external.fetch=true")
    gpus = int(external.get("gpus", config.get("training", {}).get("world_size", 4)))
    if manifest["method"] != "fast" and gpus > 1:
        command = [command[0], "-m", "torch.distributed.run", "--nproc_per_node", str(gpus),
                   "-m", "dualref.baselines.worker", "--config", manifest["config"]]
    prepare_history(config, Path(manifest["output_dir"]), manifest["method"], manifest["fingerprint"])
    subprocess.run(command, env=env, check=True)
    result_path = Path(manifest["output_dir"]) / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"External trainer did not produce {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))
