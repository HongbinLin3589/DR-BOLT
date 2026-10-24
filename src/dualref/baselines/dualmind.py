"""DualMind's auto-labeling and joint forced/free GRPO training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .worker import (
    DUAL_INSTRUCTION, PREFIXES, _backend, _instruction, _lora, _rank,
    _score, _sync, _token_logps, centered_advantages, label_mode, prefix_reward,
)


def _labels(config: dict, backend, rows: list[dict], output: Path) -> list[dict]:
    from transformers import set_seed

    path = output / "dualmind_labels.jsonl"
    world = int(os.environ.get("WORLD_SIZE", 1))
    if not path.exists():
        shard = output / f"dualmind_labels.rank-{_rank()}.jsonl"
        with shard.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(rows):
                if index % world != _rank():
                    continue
                set_seed(int(config.get("seed", 42)) + index)
                samples = backend.generate([example], num_return_sequences=8)[0]
                correct = [bool(_score(example, sample["response"])["correct"]) for sample in samples]
                lengths = [sample["token_count"] for sample in samples]
                mode = label_mode(lengths, correct)
                handle.write(json.dumps({"example": example, "mode": mode, "lengths": lengths,
                                         "correct": correct, "_index": index}, ensure_ascii=False) + "\n")
        _sync()
        if _rank() == 0:
            merged = [json.loads(line) for rank in range(world)
                      for line in (output / f"dualmind_labels.rank-{rank}.jsonl").read_text().splitlines() if line]
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in sorted(merged, key=lambda row: row["_index"]):
                    row.pop("_index")
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(path)
    _sync()
    set_seed(int(config.get("seed", 42)) + _rank())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if row["mode"] is not None]
    if not selected:
        raise ValueError("DualMind's labeling/filtering retained no training examples; inspect dualmind_labels.jsonl")
    return selected


def train(config: dict, checkpoint: str, rows: list[dict], output: Path) -> Path:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin
    from peft import get_peft_model
    from torch.utils.data import DataLoader, Sampler
    from transformers import get_cosine_schedule_with_warmup

    from .worker import evaluate_history, portable_adapter

    training = config.get("training", {})
    accumulation = int(training.get("gradient_accumulation_steps", training.get("gradient_accumulation", 8)))
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=accumulation, sync_with_dataloader=False),
        mixed_precision="bf16",
    )
    backend = _backend(config, checkpoint)
    selected = _labels(config, backend, rows, output)
    backend.model = get_peft_model(backend.model, _lora(config))
    backend.model.config.use_cache = False
    backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # A deterministic policy forward is required for the PPO likelihood ratio.
    for layer in backend.model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    optimizer = torch.optim.AdamW(
        [parameter for parameter in backend.model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 1e-6)), betas=(0.9, 0.95), weight_decay=0.1,
    )
    steps = int(training.get("steps", 1000))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    class EpochSampler(Sampler):
        def __init__(self):
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            generator = torch.Generator().manual_seed(int(config.get("seed", 42)) + self.epoch)
            return iter(torch.randperm(len(selected), generator=generator).tolist())

        def __len__(self):
            return len(selected)

    sampler = EpochSampler()
    loader = DataLoader(selected, batch_size=1, sampler=sampler, collate_fn=lambda x: x)
    model, optimizer, loader = accelerator.prepare(backend.model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    backend.model = accelerator.unwrap_model(model)
    optimizer.zero_grad()
    step = 0
    epoch = 0
    start_batch = 0
    resume = training.get("resume_from_checkpoint") or training.get("resume")
    if resume:
        accelerator.load_state(str(resume))
        progress = json.loads((Path(resume) / "progress.json").read_text())
        step = int(progress["step"])
        epoch = int(progress["epoch"])
        start_batch = int(progress["batch_in_epoch"])
    beta = float(config["external"].get("kl_beta", 0.001))
    epsilon = float(config["external"].get("clip_epsilon", 0.2))
    eval_every = int(training.get("eval_every", 25))
    save_every = int(training.get("save_steps", 100))
    evaluate_history(config, backend, output, step, DUAL_INSTRUCTION)
    while step < steps:
        sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < start_batch:
                continue
            row = batch[0]
            example = _instruction(row["example"], DUAL_INSTRUCTION)
            forced_prefix = PREFIXES[row["mode"]]
            with accelerator.accumulate(model):
                # Both halves belong to the same original prompt and reward group.
                free = backend.generate([example], num_return_sequences=4)[0]
                forced = backend.generate([example], num_return_sequences=4, prefix=forced_prefix)[0]
                samples = free + forced
                rewards = [float(_score(example, sample["response"])["correct"])
                           + prefix_reward(sample["response"], row["mode"]) for sample in samples]
                advantages = centered_advantages(rewards)
                backend.model.eval()
                with torch.no_grad():
                    old = [_token_logps(backend.model, backend, example, sample["response"],
                                       forced_prefix if index >= 4 else "", sample["finish_reason"] == "eos",
                                       sample.get("token_ids")).detach()
                           for index, sample in enumerate(samples)]
                    with backend.model.disable_adapter():
                        reference = [_token_logps(backend.model, backend, example, sample["response"],
                                                 forced_prefix if index >= 4 else "", sample["finish_reason"] == "eos",
                                                 sample.get("token_ids")).detach()
                                     for index, sample in enumerate(samples)]
                backend.model.train()
                loss_total = 0.0
                # Backpropagate one response at a time to avoid eight full VLM graphs in memory.
                for index, sample in enumerate(samples):
                    logps = _token_logps(model, backend, example, sample["response"],
                                        forced_prefix if index >= 4 else "", sample["finish_reason"] == "eos",
                                        sample.get("token_ids"))
                    ratio = (logps - old[index]).exp()
                    unclipped = ratio * advantages[index]
                    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[index]
                    delta = reference[index] - logps
                    kl = delta.exp() - delta - 1
                    loss = (-torch.minimum(unclipped, clipped) + beta * kl).mean() / 8
                    accelerator.backward(loss)
                    loss_total += float(loss.detach())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            step += 1
            if accelerator.is_main_process:
                with (output / "train_history.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, "loss": loss_total,
                                             "reward": sum(rewards) / 8, "mode": row["mode"]}) + "\n")
            if eval_every > 0 and (step % eval_every == 0 or step == steps):
                evaluate_history(config, backend, output, step, DUAL_INSTRUCTION)
            if step % save_every == 0 or step == steps:
                state = output / "checkpoints" / f"step-{step}"
                accelerator.save_state(str(state))
                if accelerator.is_main_process:
                    (state / "progress.json").write_text(json.dumps({
                        "step": step, "epoch": epoch, "batch_in_epoch": batch_index + 1,
                    }) + "\n")
            if step >= steps:
                break
        epoch += 1
        start_batch = 0
    final = output / "model"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        portable_adapter(backend, config)
        backend.save(final)
    accelerator.wait_for_everyone()
    return final
