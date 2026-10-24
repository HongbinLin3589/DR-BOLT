"""One multimodal boundary for generation, teacher forcing, and router features."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re


class ModelBackend:
    """Load a reference or an HF/PEFT checkpoint on one process-local device.

    Responses start immediately after the assistant header. In particular, the
    Thinking checkpoint's template must not put a forced <think> in the prompt.
    Ordinary model control tokens are removed from decoded completions, but
    reasoning and Cond-RW mode tokens are preserved.
    """

    def __init__(self, config: dict, checkpoint: str | None = None, role: str = "policy"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = config
        settings = config.get("model", {})
        alias = {"policy": "initial", "fast": "fast", "slow": "slow"}.get(role, role)
        name = checkpoint or settings.get(alias) or settings.get("initial")
        if name in ("fast", "slow", "initial"):
            alias, name = name, settings[name]
        if not name:
            raise ValueError("A model checkpoint is required")
        revision = settings.get(f"{alias}_revision") if name == settings.get(alias) else None
        for known in ("fast", "slow", "initial"):
            if name == settings.get(known):
                revision = settings.get(f"{known}_revision")
                break
        self.checkpoint = str(name)
        local = Path(name)
        external_metadata = local / "dualref_external.json"
        self.external_instruction = (json.loads(external_metadata.read_text()).get("instruction", "")
                                     if external_metadata.is_file() else "")
        adapter = local.is_dir() and (local / "adapter_config.json").exists()
        base_name = name
        if adapter:
            adapter_cfg = json.loads((local / "adapter_config.json").read_text())
            base_name = adapter_cfg["base_model_name_or_path"]
            meta_path = local / "dualref_model.json"
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            revision = metadata.get("base_revision", revision)
        self.base_name, self.base_revision = str(base_name), revision
        processor_name = str(local) if local.is_dir() and (local / "tokenizer_config.json").exists() else base_name
        processor_kw = {} if processor_name == str(local) and adapter else {"revision": revision}
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            min_pixels=settings.get("min_pixels", 4096),
            max_pixels=settings.get("max_pixels", 1048576),
            **processor_kw,
        )
        self.tokenizer = self.processor.tokenizer
        if settings.get("extra_tokens"):
            self.tokenizer.add_special_tokens({"additional_special_tokens": settings["extra_tokens"]})
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if settings.get("precision", "bf16") == "bf16" else torch.float32
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu" and dtype == torch.bfloat16:
            dtype = torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_name, revision=revision, torch_dtype=dtype,
            attn_implementation=settings.get("attention", "sdpa"),
        )
        # Qwen reserves padded vocabulary rows; preserve the frozen softmax support.
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if adapter:
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(str(local))
            if settings.get("extra_tokens"):
                adapter_config.modules_to_save = ["embed_tokens", "lm_head"]
            self.model = PeftModel.from_pretrained(
                self.model, str(local), config=adapter_config,
                is_trainable=settings.get("trainable", False),
            )
        self.model.to(self.device)
        self.model.eval()

    def messages(self, example: dict) -> list[dict]:
        content = []
        for i, path in enumerate(example.get("images", []), 1):
            content.extend([
                {"type": "text", "text": f"Image {i}:"},
                {"type": "image", "image": str(path)},
            ])
        question = re.sub(r"<image\s*(\d+)>", r"Image \1", example["question"])
        options = example.get("choices", [])
        if options:
            normalized_options = [re.sub(r"<image\s*(\d+)>", r"Image \1", option) for option in options]
            question += "\n" + "\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(normalized_options))
            question += "\nPut the final option letter in \\boxed{}."
        else:
            question += "\nPut the final answer in \\boxed{}."
        if example.get("metadata", {}).get("direct_prompt"):
            question += "\nAnswer directly. Do not include a reasoning trace."
        if example.get("metadata", {}).get("reasoning_prompt"):
            question += "\nReason step by step inside <think> and </think>, then answer."
        if self.external_instruction:
            question += "\n" + self.external_instruction
        content.append({"type": "text", "text": question})
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}]

    def render_prompt(self, example: dict) -> str:
        text = self.processor.apply_chat_template(
            self.messages(example), tokenize=False, add_generation_prompt=False,
        )
        return text + "<|im_start|>assistant\n"

    def encode(self, example: dict, response: str | None = None, token_ids: list[int] | None = None) -> dict:
        import torch
        from PIL import Image

        prompt = self.render_prompt(example)
        images = []
        for path in example.get("images", []):
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        kwargs = {"images": images} if images else {}
        encoded = self.processor(text=[prompt], return_tensors="pt", **kwargs)
        encoded = {key: value.to(self.device) if hasattr(value, "to") else value
                   for key, value in encoded.items()}
        if response is not None:
            response_ids = list(token_ids) if token_ids is not None else self.tokenizer.encode(response, add_special_tokens=False)
            if token_ids is None:
                eos = self.tokenizer.eos_token_id
                if not response_ids or response_ids[-1] != eos:
                    response_ids.append(eos)
            tail = torch.tensor([response_ids], device=self.device, dtype=torch.long)
            prompt_ids = encoded["input_ids"]
            encoded["input_ids"] = torch.cat([prompt_ids, tail], dim=1)
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            encoded["labels"] = torch.cat([torch.full_like(prompt_ids, -100), tail], dim=1)
            # Positions must be recomputed by the VLM after adding response tokens.
            encoded.pop("position_ids", None)
        return encoded

    def _decode(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
            text = text.replace(token, "")
        return text

    def generate(self, examples: list[dict], num_return_sequences: int = 1,
                 max_new_tokens: int | None = None, prefix: str = "") -> list[list[dict]]:
        import torch

        generation = self.config.get("generation", {})
        maximum = max_new_tokens or generation.get("max_new_tokens", 2048)
        temperature = generation.get("temperature", 0.7)
        was_training = self.model.training
        self.model.eval()
        results = []
        with torch.inference_mode():
            for example in examples:
                inputs = self.encode(example)
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                if prefix_ids:
                    tail = torch.tensor([prefix_ids], device=self.device)
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], tail], dim=1)
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                    inputs.pop("position_ids", None)
                prompt_length = inputs["input_ids"].shape[1]
                outputs = []
                for _ in range(num_return_sequences):
                    options = dict(max_new_tokens=maximum, do_sample=temperature > 0,
                                   use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                                   eos_token_id=self.tokenizer.eos_token_id)
                    if temperature > 0:
                        options.update(temperature=temperature, top_p=generation.get("top_p", 0.9))
                    generated = self.model.generate(**inputs, **options)
                    new_ids = generated[0, prompt_length:].tolist()
                    ended = self.tokenizer.eos_token_id in new_ids
                    if ended:
                        new_ids = new_ids[:new_ids.index(self.tokenizer.eos_token_id) + 1]
                    ids = prefix_ids + new_ids
                    outputs.append({"response": self._decode(ids), "token_count": len(ids),
                                    "token_ids": ids, "finish_reason": "eos" if ended else "length"})
                results.append(outputs)
        self.model.train(was_training)
        return results

    def sequence_log_probs(self, example: dict, responses: list[str],
                           return_tokens: bool = False,
                           response_token_ids: list[list[int] | None] | None = None) -> list[dict]:
        import torch

        was_training = self.model.training
        self.model.eval()
        records = []
        with torch.inference_mode():
            for index, response in enumerate(responses):
                ids = response_token_ids[index] if response_token_ids is not None else None
                encoded = self.encode(example, response, token_ids=ids)
                labels = encoded.pop("labels")
                logits = self.model(**encoded, use_cache=False).logits
                values = response_token_logps(logits, labels).detach().double().cpu()
                record = {"logp": values.sum().item(), "token_count": values.numel()}
                if return_tokens:
                    record["token_logps"] = values.tolist()
                records.append(record)
        self.model.train(was_training)
        return records

    def prompt_features(self, example: dict):
        import torch

        self.model.eval()
        with torch.inference_mode():
            result = self.model(**self.encode(example), output_hidden_states=True, use_cache=False)
            return result.hidden_states[-1][0, -1].detach().float().cpu()

    def save(self, directory: str | Path) -> str:
        from .io import write_json

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        write_json(path / "dualref_model.json", {
            "base_name": self.base_name, "base_revision": self.base_revision,
            "prompt_boundary": "<|im_start|>assistant\\n", "response_reduction": "sum",
        })
        return str(path)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def response_token_logps(logits, labels):
    """Chunk the fp32 softmax and return only response log probabilities."""
    import torch
    from torch.nn import functional as F

    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(-100)
    selected = logits[:, :-1][active]
    targets = shifted_labels[active]
    chunks = [
        -F.cross_entropy(selected[start:start+128].float(), targets[start:start+128], reduction="none")
        for start in range(0, len(targets), 128)
    ]
    return torch.cat(chunks) if chunks else logits.sum().reshape(1)[:0]
