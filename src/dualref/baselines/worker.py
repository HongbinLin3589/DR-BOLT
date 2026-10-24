"""Subprocess entry points; heavy training dependencies are imported on demand."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from datetime import timedelta

from dualref.config import read_config, write_config

from .runner import SOURCES, method_name, read_examples
from .state import MODEL_REVISIONS, record_evaluation


PREFIXES = {"fast": "Short Thinking:\n", "slow": "Long Thinking:\n"}
DUAL_INSTRUCTION = (
    "Choose the thinking effort needed to answer this question. Begin your response with "
    "'Short Thinking:' for a concise direct answer, or 'Long Thinking:' for step-by-step reasoning. "
    "Finish with the final answer in \\boxed{}."
)
TON_INSTRUCTION = (
    "Write your reasoning inside <think>...</think>, which may be empty when no reasoning is needed. "
    "Put the final answer inside <answer>...</answer>."
)


def label_mode(lengths: list[int], correct: list[bool]) -> str | None:
    if len(lengths) != 8 or len(correct) != 8:
        raise ValueError("DualMind labeling requires eight completions")
    if not any(correct) or all(correct):
        return None
    average = sum(lengths) / len(lengths)
    return "fast" if average < 100 else "slow" if average > 200 else None


def prefix_reward(response: str, label: str) -> float:
    response = response.lstrip()
    if response.startswith(PREFIXES[label].strip()):
        return 1.0
    return 0.5 if any(response.startswith(p.strip()) for p in PREFIXES.values()) else 0.0


def centered_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        raise ValueError("An advantage group cannot be empty")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def ton_rewards(completions, example, **kwargs):
    values = []
    for completion, row in zip(completions, example):
        response = completion if isinstance(completion, str) else completion[0]["content"]
        fmt = bool(re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", response, re.DOTALL))
        values.append(float(_score(row, response)["correct"]) + float(fmt))
    return values


def _score(example: dict, response: str) -> dict:
    from dualref.scoring import score_response

    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer:
        response = answer.group(1)
    response = re.sub(r"^\s*(?:Short|Long) Thinking:\s*", "", response)
    return score_response(example, response)


def _instruction(example: dict, instruction: str) -> dict:
    row = copy.deepcopy(example)
    row["question"] += "\n\n" + instruction
    return row


def _rank() -> int:
    return int(os.environ.get("RANK", 0))


def _sync() -> None:
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _model_path(config: dict) -> str:
    from huggingface_hub import snapshot_download

    external = config["external"]
    model = external.get("model", SOURCES[method_name(config)]["model"])
    if Path(model).is_dir():
        return str(Path(model).resolve())
    revision = external.get("model_revision", MODEL_REVISIONS.get(model))
    if not revision:
        raise ValueError("Set external.model_revision when using a custom model repository")
    return snapshot_download(model, revision=revision)


def _backend(config: dict, checkpoint: str):
    from dualref.models import ModelBackend

    cfg = copy.deepcopy(config)
    cfg.setdefault("model", {})["gradient_checkpointing"] = True
    return ModelBackend(cfg, checkpoint=checkpoint, role="policy")


def _drop_empty_images(module, args, kwargs):
    """TRL slices mixed image/text batches into empty visual tensors for text rows."""
    pixels = kwargs.get("pixel_values")
    if pixels is not None and pixels.numel() == 0:
        kwargs = {key: value for key, value in kwargs.items() if key not in ("pixel_values", "image_grid_thw")}
    return args, kwargs


def _lora(config: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=8, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def _token_logps(model, backend, example: dict, response: str, skip_prefix: str = "", terminated: bool = True,
                 token_ids: list[int] | None = None):
    """Response-only log probabilities; inserted prefixes carry no policy gradient."""
    import torch

    inputs = backend.encode(example, response, token_ids=token_ids)
    labels = inputs.pop("labels")[:, 1:]
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[:, :-1]
    mask = labels != -100
    selected = logits[mask]
    targets = labels[mask]
    chunks = []
    for start in range(0, len(targets), 128):
        chunk = selected[start:start + 128].float().log_softmax(-1)
        chunks.append(chunk.gather(-1, targets[start:start + 128, None]).squeeze(-1))
    if not chunks:
        raise ValueError("A response has no trainable tokens")
    result = torch.cat(chunks)
    if not terminated and token_ids is None:
        result = result[:-1]  # encode appends EOS; truncation is not an observed EOS.
    if skip_prefix:
        count = len(backend.tokenizer.encode(skip_prefix, add_special_tokens=False))
        result = result[count:]
    if not result.numel():
        raise ValueError("Completion contains only a forced prefix")
    return result


def _training_args(config: dict, output: Path, *, steps: int | None = None):
    from transformers import TrainingArguments

    training = config.get("training", {})
    return TrainingArguments(
        output_dir=str(output), max_steps=int(steps if steps is not None else training.get("steps", 1000)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", training.get("gradient_accumulation", 8))),
        learning_rate=float(training.get("learning_rate", 1e-6)),
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
        lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True,
        logging_steps=1, save_steps=int(training.get("save_steps", 100)), save_total_limit=2,
        remove_unused_columns=False, report_to=[], seed=int(config.get("seed", 42)),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )


def portable_adapter(backend, config: dict) -> None:
    """Keep adapter checkpoints usable after moving them away from a local HF cache."""
    name = config["external"]["model"]
    revision = config["external"].get("model_revision", MODEL_REVISIONS.get(name))
    if Path(name).is_dir():
        return
    backend.base_name, backend.base_revision = name, revision
    for adapter in getattr(backend.model, "peft_config", {}).values():
        adapter.base_model_name_or_path = name
        adapter.revision = revision


def evaluate_history(config: dict, backend, output: Path, step: int, instruction: str = "") -> None:
    """Shard evaluation across ranks and preserve training random-number states."""
    path = config.get("evaluation", {}).get("input")
    if not path:
        return
    evaluation_start = time.perf_counter()
    import numpy as np
    import torch
    from dualref.scoring import score_response

    rows = read_examples(path)
    limit = config.get("evaluation", {}).get("limit")
    if limit is not None and len(rows) > int(limit):
        indices = sorted(random.Random(int(config.get("seed", 42))).sample(range(len(rows)), int(limit)))
        rows = [rows[index] for index in indices]
    world = int(os.environ.get("WORLD_SIZE", 1))
    python_state, numpy_state = random.getstate(), np.random.get_state()
    values = []
    devices = [backend.device.index] if backend.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(config.get("seed", 42)) + step + _rank())
            for row in rows[_rank()::world]:
                prompt = _instruction(row, instruction) if instruction else row
                response = backend.generate([prompt])[0][0]["response"]
                values.append(score_response(row, response))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    sums = torch.tensor([sum(float(value["correct"]) for value in values),
                         sum(value["words"] for value in values),
                         sum(bool(value["slow"]) for value in values), len(values)],
                        dtype=torch.float64, device=backend.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums)
    score, length, slow, count = sums.tolist()
    if not count:
        raise ValueError("Evaluation contains no examples")
    if _rank() == 0:
        record = {
            "step": step, "score": score / count, "length": length / count,
            "slow_ratio": slow / count, "examples": int(count),
        }
        record_evaluation(output / "history.jsonl", record)
    _sync()
    config["_evaluation_seconds"] = config.get("_evaluation_seconds", 0.0) + time.perf_counter() - evaluation_start


def _eval_callback(config: dict, backend, output: Path, instruction: str = ""):
    from transformers import TrainerCallback

    class EvaluationCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            backend.model = kwargs["model"]
            evaluate_history(config, backend, output, state.global_step, instruction)

        def on_step_end(self, args, state, control, **kwargs):
            interval = int(config.get("training", {}).get("eval_every", 25))
            if interval > 0 and (state.global_step % interval == 0 or state.global_step == state.max_steps):
                backend.model = kwargs["model"]
                evaluate_history(config, backend, output, state.global_step, instruction)

    return EvaluationCallback()


def run_fast(config: dict, checkpoint: str, output: Path) -> Path:
    from transformers import AutoProcessor
    from .resources import measured_command

    source = Path(config["external"]["source_dir"])
    processor = AutoProcessor.from_pretrained(checkpoint)
    template = processor.chat_template
    # Only the generation suffix is changed; FAST does not serialize SFT answers here.
    template = template.replace("assistant\\n<think>\\n", "assistant\\n")
    template = template.replace("assistant\n<think>\n", "assistant\n")
    (output / "chat_template.jinja").write_text(template, encoding="utf-8")
    fast_config_path = output / "fast_config.yaml"
    fast_config = read_config(fast_config_path)
    fast_config["worker"]["actor"]["model"]["model_path"] = checkpoint
    if config.get("evaluation", {}).get("input"):
        fast_config["trainer"]["save_freq"] = int(config.get("training", {}).get("eval_every", 25))
        fast_config["trainer"]["save_limit"] = -1
    write_config(fast_config_path, fast_config)
    env = os.environ.copy()
    env["DUALREF_FAST_ROOT"] = str(source)
    env["PYTHONPATH"] = os.pathsep.join([str(source), env.get("PYTHONPATH", "")])
    config["runtime"] = measured_command(
        [sys.executable, "-m", "verl.trainer.main", f"config={fast_config_path}"], cwd=source, env=env,
    )
    checkpoints = list((output / "checkpoints").glob("**/actor/huggingface"))
    if not checkpoints:
        raise RuntimeError("FAST did not save an actor checkpoint")
    def step(path):
        numbers = re.findall(r"(?:global_step|step)[_-]?(\d+)", str(path))
        return int(numbers[-1]) if numbers else 0
    evaluate_all = bool(config.get("evaluation", {}).get("input"))
    if evaluate_all:
        initial = _backend(config, checkpoint)
        evaluate_history(config, initial, output, 0)
        initial.close()
    actors = sorted(checkpoints, key=step) if evaluate_all else [max(checkpoints, key=step)]
    for path in actors:
        actor = path.parent
        subprocess.run([sys.executable, str(source / "scripts/model_merger.py"), "--local_dir", str(actor)],
                       cwd=source, env=env, check=True)
        if evaluate_all:
            backend = _backend(config, str(actor / "huggingface"))
            evaluate_history(config, backend, output, step(path))
            backend.close()
    return actor / "huggingface"


def run_grpo(config: dict, checkpoint: str, rows: list[dict], output: Path) -> Path:
    import datasets
    from trl import GRPOConfig, GRPOTrainer

    backend = _backend(config, checkpoint)
    # Register on the base model before PEFT wraps it, covering policy and disabled-adapter reference forwards.
    backend.model.register_forward_pre_hook(_drop_empty_images, with_kwargs=True)
    generation = config.get("generation", {})
    training = config.get("training", {})
    records = [{"prompt": backend.render_prompt(row), "images": row.get("images", []), "example": row} for row in rows]
    data = datasets.Dataset.from_list(records).cast_column("images", datasets.List(datasets.Image()))

    def reward(completions, example, **kwargs):
        values = []
        for completion, row in zip(completions, example):
            if not isinstance(completion, str):
                completion = completion[0]["content"]
            score = _score(row, completion)
            values.append(float(score["correct"]) + 0.1 * float(score["format"]))
        return values

    args = GRPOConfig(
        output_dir=str(output / "checkpoints"), max_steps=int(training.get("steps", 1000)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", training.get("gradient_accumulation", 8))),
        num_generations=8, learning_rate=float(training.get("learning_rate", 5e-6)),
        max_prompt_length=None, max_completion_length=int(generation.get("max_new_tokens", 8192)),
        temperature=float(generation.get("temperature", 0.7)), top_p=float(generation.get("top_p", 0.9)),
        beta=float(config["external"].get("kl_beta", 0.04)), loss_type="grpo", scale_rewards="group",
        use_vllm=False, bf16=True, remove_unused_columns=False, report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(config.get("seed", 42)), save_steps=int(training.get("save_steps", 100)), save_total_limit=2,
        logging_steps=1, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
    )
    trainer = GRPOTrainer(model=backend.model, processing_class=backend.processor, reward_funcs=reward,
                          train_dataset=data, peft_config=_lora(config), args=args,
                          callbacks=[_eval_callback(config, backend, output)])
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint") or training.get("resume"))
    result = output / "model"
    backend.model = trainer.model
    portable_adapter(backend, config)
    trainer.save_model(str(result))
    if _rank() == 0:
        backend.save(result)
    return result


def _ton_sft_data(config: dict, backend, rows: list[dict], output: Path) -> list[dict]:
    from transformers import set_seed

    cache = output / "ton_sft.jsonl"
    world = int(os.environ.get("WORLD_SIZE", 1))
    if not cache.exists():
        bank_file = config.get("training", {}).get("bank_file") or config.get("bank", {}).get("path")
        if bank_file and not Path(bank_file).is_file():
            raise FileNotFoundError(bank_file)
        candidates = {}
        if bank_file and Path(bank_file).is_file():
            with Path(bank_file).open() as handle:
                for line in handle:
                    item = json.loads(line)
                    pool = item.get("candidates", [item])
                    for candidate in pool:
                        if candidate.get("source") in ("slow", "s"):
                            candidates.setdefault(str(item.get("example_id", item.get("id"))), []).append(candidate)
        shard = output / f"ton_sft.rank-{_rank()}.jsonl"
        with shard.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                if index % world != _rank():
                    continue
                seed = int(config.get("seed", 42)) + index
                set_seed(seed)
                example = _instruction(row, TON_INSTRUCTION)
                pool = candidates.get(str(row["id"]), [])
                good = [item.get("response", "") for item in pool if _score(row, item.get("response", ""))["correct"]]
                if good:
                    text = good[0]
                elif bank_file:
                    text = ""
                else:
                    text = backend.generate([example])[0][0]["response"]
                thought = ""
                if text and _score(row, text)["correct"]:
                    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                    thought = match.group(1).strip() if match else text.split("</think>")[0].strip()
                if random.Random(seed).random() < float(config["external"].get("thought_dropout", 0.5)):
                    thought = ""
                target = f"<think>{thought}</think><answer>{row['answer']}</answer>"
                handle.write(json.dumps({"example": example, "response": target, "_index": index}, ensure_ascii=False) + "\n")
        _sync()
        if _rank() == 0:
            merged = [json.loads(line) for rank in range(world)
                      for line in (output / f"ton_sft.rank-{rank}.jsonl").read_text().splitlines() if line]
            temporary = cache.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in sorted(merged, key=lambda row: row["_index"]):
                    row.pop("_index")
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(cache)
    _sync()
    set_seed(int(config.get("seed", 42)) + _rank())
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]


def run_ton(config: dict, checkpoint: str, rows: list[dict], output: Path) -> Path:
    """Use TON's trainer lifecycle and HF rollout path with arbitrary image counts."""
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import Trainer
    from trl import GRPOConfig

    upstream = Path(config["external"]["source_dir"]) / "src/r1-v/src"
    sys.path.insert(0, str(upstream))
    # Import its single HF trainer directly: package __init__ also imports obsolete vLLM internals.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ton_hf_trainer", upstream / "open_r1/trainer/grpo_trainer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = _backend(config, checkpoint)
    backend.processor.pad_token_id = backend.tokenizer.pad_token_id
    backend.processor.eos_token_id = backend.tokenizer.eos_token_id
    sft = _ton_sft_data(config, backend, rows, output)
    backend.model = get_peft_model(backend.model, _lora(config))
    sft_steps = int(config["external"].get("sft_steps", 100))
    if sft_steps < 1:
        raise ValueError("TON requires a positive external.sft_steps budget")

    class ThoughtDropoutTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            losses = [-_token_logps(model, backend, row["example"], row["response"]).mean() for row in inputs]
            return torch.stack(losses).mean()

    trainer = ThoughtDropoutTrainer(
        model=backend.model, train_dataset=Dataset.from_list(sft), data_collator=lambda x: x,
        args=_training_args(config, output / "sft_checkpoints", steps=sft_steps),
    )
    trainer.train()
    backend.model = backend.model.merge_and_unload()
    del trainer
    _sync()
    generation = config.get("generation", {})
    training = config.get("training", {})

    class MultimodalTONTrainer(module.Qwen2VLGRPOTrainer):
        def completion_losses(self, model, inputs):
            for row in inputs:
                example = row["example"]
                was_training = backend.model.training
                backend.model.eval()
                with torch.no_grad():
                    samples = backend.generate([example], num_return_sequences=8)[0]
                backend.model.train(was_training)
                rewards = ton_rewards([sample["response"] for sample in samples], [example] * 8)
                values = torch.tensor(rewards, device=self.accelerator.device)
                advantages = (values - values.mean()) / (values.std() + 1e-4)
                for index, sample in enumerate(samples):
                    logps = _token_logps(model, backend, example, sample["response"],
                                        terminated=sample["finish_reason"] == "eos", token_ids=sample.get("token_ids"))
                    with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                        ref = _token_logps(self.accelerator.unwrap_model(model), backend, example, sample["response"],
                                          terminated=sample["finish_reason"] == "eos", token_ids=sample.get("token_ids"))
                    delta = ref - logps
                    kl = delta.exp() - delta - 1
                    # The released TON objective (its HF path) uses one update per rollout.
                    policy = (logps - logps.detach()).exp() * advantages[index]
                    yield -(policy - self.beta * kl).mean() / (8 * len(inputs))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return torch.stack(list(self.completion_losses(model, inputs))).sum()

        def training_step(self, model, inputs, num_items_in_batch=None):
            # Mirror Trainer's accumulation scaling while releasing each response graph immediately.
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            total = torch.zeros((), device=self.accelerator.device)
            accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            with self.compute_loss_context_manager():
                for loss in self.completion_losses(model, inputs):
                    scaled = loss / accumulation
                    self.accelerator.backward(scaled)
                    total += scaled.detach()
            return total

    args = GRPOConfig(
        output_dir=str(output / "checkpoints"), max_steps=int(training.get("steps", 1000)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", training.get("gradient_accumulation", 8))),
        num_generations=8, max_completion_length=int(generation.get("max_new_tokens", 8192)),
        learning_rate=float(training.get("learning_rate", 1e-6)), beta=float(config["external"].get("kl_beta", 0.04)),
        temperature=float(generation.get("temperature", 0.7)), bf16=True,
        remove_unused_columns=False, report_to=[], seed=int(config.get("seed", 42)),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=int(training.get("save_steps", 100)), save_total_limit=2,
    )
    trainer = MultimodalTONTrainer(
        model=backend.model, processing_class=None, peft_config=_lora(config), args=args,
        reward_funcs=[ton_rewards],
        train_dataset=Dataset.from_list([{"example": _instruction(row, TON_INSTRUCTION)} for row in rows]),
        attn_implementation="sdpa",
        callbacks=[_eval_callback(config, backend, output, TON_INSTRUCTION)],
    )
    backend.model = trainer.model
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint") or training.get("resume"))
    final = output / "model"
    _sync()
    if _rank() == 0:
        trainer.accelerator.unwrap_model(trainer.model).merge_and_unload().save_pretrained(final)
        backend.processor.save_pretrained(final)
    _sync()
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    method = method_name(config)
    output = Path(config["training"]["output_dir"])
    rows = read_examples(config["training"]["train_file"])
    import torch
    from transformers import set_seed

    set_seed(int(config.get("seed", 42)))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl", timeout=timedelta(days=7))
    checkpoint = _model_path(config)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if method == "fast":
        model = run_fast(config, checkpoint, output)
    elif method == "ton":
        model = run_ton(config, checkpoint, rows, output)
    elif method == "grpo":
        model = run_grpo(config, checkpoint, rows, output)
    else:
        from .dualmind import train
        model = train(config, checkpoint, rows, output)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    if peak is not None and torch.distributed.is_initialized():
        tensor = torch.tensor([peak], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        peak = int(tensor.item())
    if _rank() == 0:
        if method in ("ton", "dualmind"):
            metadata = {"method": method, "instruction": TON_INSTRUCTION if method == "ton" else DUAL_INSTRUCTION}
            (Path(model) / "dualref_external.json").write_text(json.dumps(metadata, indent=2) + "\n")
        result = {"method": method, "checkpoint": str(model),
                  "seconds": max(0.0, elapsed - config.get("_evaluation_seconds", 0.0)),
                  "source_revision": SOURCES[method]["revision"],
                  "examples": len(rows), "model": config["external"]["model"],
                  "model_revision": config["external"].get("model_revision", MODEL_REVISIONS.get(config["external"]["model"]))}
        gpu_count = int(config["external"].get("gpus", config.get("training", {}).get("world_size", 4))) if method == "fast" else int(os.environ.get("WORLD_SIZE", 1))
        gpu_count = min(gpu_count, torch.cuda.device_count())
        result["hardware"] = {"gpu_count": gpu_count,
                              "device_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)]}
        if peak is not None:
            result.update(peak_gpu_bytes=peak, memory_status="measured",
                          memory_measurement="torch.cuda.max_memory_allocated (maximum across workers)")
        else:
            result.update(memory_status="unavailable", memory_reason="CUDA memory measurement is not available")
        result.update(config.get("runtime", {}))
        result["wall_seconds_including_evaluation"] = elapsed
        if (output / "history.jsonl").exists():
            result["history"] = str(output / "history.jsonl")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    _sync()


if __name__ == "__main__":
    main()
