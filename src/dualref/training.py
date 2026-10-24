"""LoRA sequence-weighted SFT and exact-mixture preference training."""

from __future__ import annotations

import copy
from bisect import bisect_left, bisect_right
from datetime import timedelta
import json
from pathlib import Path
import random

from .io import append_jsonl, file_hash, fingerprint, seed_everything, write_json
from .objectives import preference_pairs, weight_group


def _rows(cfg: dict):
    from .data import load_examples

    settings = cfg.get("training", {})
    method = settings.get("method", "dr_bolt")
    bank_file = cfg.get("bank", {}).get("path") or settings.get("bank_file")
    if not bank_file:
        raise ValueError("training requires bank.path")
    objective = copy.deepcopy(cfg.get("objective", {}))
    if settings.get("group_normalize"):
        objective["normalize_groups"] = True
    groups = []
    for group in load_examples(bank_file):
        weighted = weight_group(group, objective, method)
        if (settings.get("filter_all_wrong") or objective.get("drop_all_wrong")) and weighted["all_wrong"]:
            continue
        groups.append(weighted)
    if not groups:
        raise ValueError("No usable candidate groups remain")
    if method == "mr_dpo":
        for group in groups:
            for candidate in group["candidates"]:
                candidate_tokens(candidate)
    rows = []
    for group in groups:
        if method == "mr_dpo":
            for winner, loser in preference_pairs(group):
                rows.append((group, (winner, loser)))
        else:
            rows.extend((group, index) for index in range(len(group["candidates"])))
    if not rows:
        raise ValueError("No strictly reward-ordered preference pairs remain")
    if settings.get("balanced_sources"):
        fast = [row for row in rows if row[0]["candidates"][row[1]]["source"] == "fast"]
        slow = [row for row in rows if row[0]["candidates"][row[1]]["source"] == "slow"]
        rng = random.Random(cfg.get("seed", 42))
        rng.shuffle(fast)
        rng.shuffle(slow)
        rows = [item for i in range(max(len(fast), len(slow)))
                for item in (fast[i % len(fast)], slow[i % len(slow)])]
    return rows, groups


def candidate_tokens(candidate: dict) -> int:
    """Read observed response length without inventing a token budget."""
    ids = candidate.get("token_ids")
    count = candidate.get("token_count")
    if ids is not None:
        if not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Response token_ids must be a nonempty list of token IDs")
        if count is not None and (type(count) is not int or count != len(ids)):
            raise ValueError("Response token_count disagrees with token_ids")
        return len(ids)
    if type(count) is not int or count <= 0:
        raise ValueError("MR-DPO requires observed token_ids or a positive token_count for every response")
    return count


def matched_pairs(rows: list, count: int, target_tokens: int, seed: int) -> tuple[list, dict]:
    """Subsample whole preference pairs under a shared response-token budget.

    Preserve the requested response-sequence count. No preference completion is
    truncated to fit a budget. A final indivisible-pair remainder is reported.
    """
    buckets = {}
    for index, (group, indices) in enumerate(rows):
        length = sum(candidate_tokens(group["candidates"][i]) for i in indices)
        buckets.setdefault(length, []).append(index)
    lengths = sorted(buckets)
    minimum, maximum = lengths[0], lengths[-1]
    if not minimum * count <= target_tokens <= maximum * count:
        raise ValueError("The reference token budget cannot be represented by this many complete preference pairs")
    rng = random.Random(seed)
    selected, remaining = [], target_tokens
    for position in range(count):
        left = count - position
        lower = max(minimum, remaining - maximum * (left - 1))
        upper = min(maximum, remaining - minimum * (left - 1))
        low_index, high_index = bisect_left(lengths, lower), bisect_right(lengths, upper)
        if low_index >= high_index:
            # A discrete gap can occur only near the end; use a shorter complete pair.
            index = bisect_right(lengths, upper) - 1
            if index < 0:
                raise ValueError("Cannot fit a complete preference pair within the remaining budget")
            length = lengths[index]
        elif left > 128:
            original = rows[rng.randrange(len(rows))]
            sampled_length = sum(candidate_tokens(original[0]["candidates"][i]) for i in original[1])
            length = min(lengths[high_index - 1], max(lengths[low_index], sampled_length))
            if length not in buckets:
                length = lengths[bisect_left(lengths, length)]
        else:
            desired = remaining / left
            index = min(high_index - 1, max(low_index, bisect_left(lengths, desired)))
            alternatives = lengths[max(low_index, index - 1):min(high_index, index + 2)]
            length = min(alternatives, key=lambda value: abs(value - desired))
        selected.append(rows[rng.choice(buckets[length])])
        remaining -= length
    if remaining < 0 or remaining >= maximum:
        raise ValueError("Preference subsampling failed its complete-pair budget bound")
    return selected, {"target_response_tokens": target_tokens, "scheduled_response_tokens": target_tokens - remaining,
                      "unspent_response_tokens": remaining, "pairs": count,
                      "budget_rule": "fixed_sequence_count_whole_pairs_no_overshoot"}


def _trainable_backend(cfg: dict):
    from peft import LoraConfig, PeftModel, get_peft_model

    from .models import ModelBackend

    settings = cfg["training"]
    local = copy.deepcopy(cfg)
    local.setdefault("model", {})["trainable"] = True
    mode_tokens = settings.get("method") == "cond_rw"
    if mode_tokens:
        local["model"]["extra_tokens"] = ["<FAST>", "<SLOW>"]
    checkpoint = settings.get("initial_checkpoint") or settings.get("checkpoint")
    backend = ModelBackend(local, checkpoint=checkpoint)
    if not isinstance(backend.model, PeftModel):
        lora = LoraConfig(r=settings.get("lora_rank", 8), lora_alpha=settings.get("lora_alpha", 64),
                          lora_dropout=settings.get("lora_dropout", .05), bias="none", task_type="CAUSAL_LM",
                          target_modules=settings.get("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                          modules_to_save=["embed_tokens", "lm_head"] if mode_tokens else None)
        backend.model = get_peft_model(backend.model, lora)
    if settings.get("gradient_checkpointing", True):
        backend.model.enable_input_require_grads()
        backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backend.model.config.use_cache = False
    return backend


def _sequence_loss(backend, model, example, response, token_ids=None):
    """Sum response-token negative log probabilities; prompt tokens are masked."""
    from .models import response_token_logps

    encoded = backend.encode(example, response, token_ids=token_ids)
    labels = encoded.pop("labels")
    logits = model(**encoded, use_cache=False).logits
    values = response_token_logps(logits, labels)
    return -values.sum(), len(values)


def _batch_loss(backend, model, batch, method: str, dpo_beta: float):
    from torch.nn import functional as F

    losses, tokens, responses = [], 0, 0
    for example, index in batch:
        if method == "mr_dpo":
            winner, loser = (example["candidates"][i] for i in index)
            positive, n_positive = _sequence_loss(backend, model, example, winner["response"], winner.get("token_ids"))
            negative, n_negative = _sequence_loss(backend, model, example, loser["response"], loser.get("token_ids"))
            margin = -positive + negative - winner["log_reference"] + loser["log_reference"]
            losses.append(-F.logsigmoid(dpo_beta * margin))
            tokens += n_positive + n_negative
            responses += 2
        else:
            row = example["candidates"][index]
            response = row["response"]
            token_ids = row.get("token_ids")
            if method == "cond_rw":
                prefix = "<FAST>" if row["source"] == "fast" else "<SLOW>"
                response = prefix + response
                if token_ids is not None:
                    token_ids = backend.tokenizer.encode(prefix, add_special_tokens=False) + token_ids
            loss, count = _sequence_loss(backend, model, example, response, token_ids)
            losses.append(loss * row["weight"])
            tokens += count
            responses += 1
    return sum(losses) / len(losses), tokens, responses


def _replay_examples(cfg: dict):
    from .data import load_examples

    path = cfg["training"].get("replay_file")
    if not path:
        return []
    maximum = cfg["training"].get("diagnostics", {}).get("replay_samples", 200)
    rows = []
    for group in load_examples(path):
        for candidate in group["candidates"]:
            rows.append((group, candidate))
    random.Random(cfg.get("seed", 42)).shuffle(rows)
    return rows[:maximum]


def _replay(backend, rows, initial=None):
    values = [backend.sequence_log_probs(group, [candidate["response"]], response_token_ids=[candidate.get("token_ids")])[0]
              for group, candidate in rows]
    if initial is None:
        return values
    differences = [now["logp"] - old["logp"] for now, old in zip(values, initial)]
    return {"implicit_kl": sum(differences) / len(differences),
            "implicit_kl_per_token": sum(differences) / sum(v["token_count"] for v in values),
            "replay_samples": len(values),
            "replay_definition": "mean_fixed_bank_logp_policy_minus_logp_initial"}


def _evaluate_in_memory(backend, cfg: dict) -> dict:
    import numpy as np
    import torch

    from .data import load_examples
    from .scoring import score_response

    evaluation = cfg.get("evaluation", {})
    path = evaluation.get("input")
    if not path:
        return {}
    rows = load_examples(path)
    if evaluation.get("limit"):
        rows = rows[:evaluation["limit"]]
    results = []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [backend.device.index] if backend.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            seed_everything(cfg.get("seed", 42))
            for example in rows:
                result = backend.generate([example])[0][0]
                results.append(score_response(example, result["response"]))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if not results:
        return {}
    return {"score": sum(r["correct"] for r in results) / len(results),
            "length": sum(r["words"] for r in results) / len(results),
            "slow_ratio": sum(r["slow"] for r in results) / len(results), "evaluation_n": len(results)}


def train(config: dict) -> dict:
    import time

    import torch
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from .io import telemetry

    cfg = copy.deepcopy(config)
    settings = cfg.setdefault("training", {})
    method = settings.get("method", "dr_bolt")
    if method in ("grpo", "fast_grpo", "ton", "dualmind"):
        from .baselines import run

        return run(cfg)
    rows, groups = _rows(cfg)
    output = Path(settings.get("output_dir") or cfg.get("output_dir", "outputs/train"))
    output.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    # A direct CLI invocation supplies the same mixed warm-up as the experiment DAG.
    if (settings.get("warmup_steps", 100) and not settings.get("initial_checkpoint")
            and not settings.get("checkpoint") and method not in ("mix_sft", "warmup")
            and not settings.get("resume")):
        warm = copy.deepcopy(cfg)
        warm["training"].update(method="mix_sft", phase="warmup", balanced_sources=True,
                                steps=settings.get("warmup_steps", 100), warmup_steps=0,
                                output_dir=str(output / "warmup"), eval_every=0, replay_file=None)
        warm["output_dir"] = str(output / "warmup")
        settings["initial_checkpoint"] = train(warm)["checkpoint"]
    training_started = time.perf_counter()
    seed_everything(seed)
    accumulation = settings.get("gradient_accumulation", 8)
    # A DPO item has two responses; preserve the configured response-sequence batch size.
    if method == "mr_dpo":
        if accumulation % 2:
            raise ValueError("MR-DPO needs an even gradient_accumulation to match response batches")
        accumulation //= 2
    # Carry partial accumulation across epochs to keep every update equally sized.
    accelerator = Accelerator(gradient_accumulation_plugin=GradientAccumulationPlugin(
                                  num_steps=accumulation, sync_with_dataloader=False),
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(days=7))],
                              mixed_precision="bf16" if cfg.get("model", {}).get("precision", "bf16") == "bf16" and torch.cuda.is_available() else "no")
    resume = settings.get("resume")
    if resume:
        settings["initial_checkpoint"] = str(Path(resume) / "adapter")
    backend = _trainable_backend(cfg)
    model = backend.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The loaded adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=settings.get("learning_rate", 5e-6),
                                 weight_decay=settings.get("weight_decay", .1), betas=tuple(settings.get("betas", [.9, .95])))
    steps = settings.get("steps", 1000)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * settings.get("warmup_ratio", .1)), steps)
    generator = torch.Generator().manual_seed(seed)
    budget_info = None
    if method == "mr_dpo":
        count = steps * accumulation * settings.get("batch_size", 1) * accelerator.num_processes
        lengths = [candidate_tokens(c) for group in groups for c in group["candidates"]]
        target = settings.get("response_token_budget") or round(sum(lengths) / len(lengths) * count * 2)
        rows, budget_info = matched_pairs(rows, count, int(target), seed)
        if accelerator.is_main_process:
            write_json(output / "token-budget.json", budget_info)
    loader = DataLoader(rows, batch_size=settings.get("batch_size", 1), shuffle=method != "mr_dpo",
                        generator=generator, collate_fn=list, drop_last=True)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    backend.model = model
    history = output / "history.jsonl"
    identity = {"bank": file_hash(cfg["bank"]["path"]), "method": method,
                "objective": cfg.get("objective", {}), "seed": seed,
                "world_size": accelerator.num_processes, "batch": settings.get("batch_size", 1),
                "accumulation": accumulation, "steps": steps, "model": cfg.get("model", {}),
                "optimization": {key: value for key, value in settings.items()
                                 if key not in ("resume", "output_dir", "initial_checkpoint", "checkpoint")}}
    identity_hash = fingerprint(identity)
    start_step, epoch, skip_batches, seen_tokens, seen_responses = 0, 0, 0, 0, 0
    replay_rows = _replay_examples(cfg) if accelerator.is_main_process else []
    replay_initial = None
    if accelerator.is_main_process:
        write_json(output / "training-manifest.json", {"fingerprint": identity_hash, **identity})
        if history.exists() and not resume:
            raise ValueError(f"Training history exists at {history}; use training.resume or a new output")
    if resume:
        state = torch.load(Path(resume) / f"state-rank{accelerator.process_index}.pt", map_location="cpu", weights_only=False)
        if state["fingerprint"] != identity_hash:
            raise ValueError("Resume state does not match the data, objective, batch, or training budget")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, epoch, skip_batches = state["step"], state["epoch"], state["batch_offset"]
        seen_tokens, seen_responses = state["tokens"], state["responses"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        import numpy as np

        np.random.set_state(state["numpy_rng"])
    if replay_rows:
        backend.model = accelerator.unwrap_model(model)
        replay_cache = output / "replay-initial.json"
        replay_initial = json.loads(replay_cache.read_text()) if resume and replay_cache.exists() else _replay(backend, replay_rows)
        write_json(replay_cache, replay_initial)
        backend.model = model
    accelerator.wait_for_everyone()
    total_loss, total_tokens, total_responses, microsteps = 0.0, 0, 0, 0
    step = start_step
    checkpoint_every = settings.get("checkpoint_every", 100)
    replay_every = settings.get("diagnostics", {}).get("replay_every", settings.get("replay_every", 25))
    eval_every = settings.get("eval_every", 0)
    last_checkpoint = None
    context = telemetry(output / f"rank-{accelerator.process_index}", "train", cfg)
    with context as timing:
        started = time.perf_counter()
        while step < steps:
            generator.manual_seed(seed + epoch)
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                model.train()
                with accelerator.accumulate(model):
                    loss, token_count, response_count = _batch_loss(
                        backend, model, batch, method, cfg.get("objective", {}).get("dpo_beta", .1))
                    accelerator.backward(loss)
                    total_loss += loss.detach().float().item()
                    total_tokens += token_count
                    total_responses += response_count
                    microsteps += 1
                    gradient_norm = 0.0
                    if accelerator.sync_gradients:
                        gradient_norm = float(accelerator.clip_grad_norm_(trainable, settings.get("max_grad_norm", 1)))
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad()
                if not accelerator.sync_gradients:
                    continue
                step += 1
                counts = torch.tensor([total_loss, total_tokens, total_responses, microsteps], device=accelerator.device, dtype=torch.float64)
                counts = accelerator.reduce(counts, reduction="sum").tolist()
                seen_tokens += int(counts[1])
                seen_responses += int(counts[2])
                record = {"step": step, "loss": counts[0] / max(1, counts[3]), "grad_norm": gradient_norm,
                          "learning_rate": scheduler.get_last_lr()[0], "response_tokens": seen_tokens,
                          "response_sequences": seen_responses,
                          "seconds": time.perf_counter() - started - timing["excluded_seconds"]}
                evaluation_started = time.perf_counter()
                if accelerator.is_main_process:
                    backend.model = accelerator.unwrap_model(model)
                    if replay_rows and step % replay_every == 0:
                        record.update(_replay(backend, replay_rows, replay_initial))
                    if eval_every and step % eval_every == 0:
                        record.update(_evaluate_in_memory(backend, cfg))
                    backend.model = model
                    append_jsonl(history, record)
                accelerator.wait_for_everyone()
                if ((settings.get("replay_file") and step % replay_every == 0)
                        or (eval_every and step % eval_every == 0)):
                    timing["excluded_seconds"] += time.perf_counter() - evaluation_started
                if step % checkpoint_every == 0 or step == steps:
                    import numpy as np

                    last_checkpoint = output / "checkpoints" / f"step-{step:06d}"
                    last_checkpoint.mkdir(parents=True, exist_ok=True)
                    if accelerator.is_main_process:
                        backend.model = accelerator.unwrap_model(model)
                        backend.save(last_checkpoint / "adapter")
                        backend.model = model
                    state = {"fingerprint": identity_hash, "step": step, "epoch": epoch,
                             "batch_offset": batch_index + 1, "tokens": seen_tokens, "responses": seen_responses,
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                             "numpy_rng": np.random.get_state(),
                             "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
                    torch.save(state, last_checkpoint / f"state-rank{accelerator.process_index}.pt")
                    accelerator.wait_for_everyone()
                total_loss, total_tokens, total_responses, microsteps = 0.0, 0, 0, 0
                if step >= steps:
                    break
            epoch += 1
            skip_batches = 0
            if len(loader) == 0:
                raise ValueError("Not enough candidates for the configured distributed batch")
    if last_checkpoint is None:
        last_checkpoint = Path(resume)
    result = {"checkpoint": str(last_checkpoint / "adapter"), "resume": str(last_checkpoint),
              "history": str(history), "response_tokens": seen_tokens, "response_sequences": seen_responses,
              "seconds": max(0.0, time.perf_counter() - training_started - timing["excluded_seconds"]),
              "wall_seconds_including_evaluation": time.perf_counter() - training_started}
    if budget_info:
        if seen_tokens != budget_info["scheduled_response_tokens"]:
            raise ValueError("Actual preference token consumption differs from the scheduled budget")
        result["token_budget"] = str(output / "token-budget.json")
    if accelerator.is_main_process:
        write_json(output / "result.json", result)
    backend.close()
    accelerator.free_memory()
    return result
