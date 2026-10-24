"""Dataset adapters, local media, and reproducible split manifests."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .config import read_config
from .scoring import canonical_answer

_DEFAULTS = {
    "scienceqa": {"repo_id": "derek-thomas/ScienceQA", "splits": ["train", "validation", "test"]},
    "virl39k": {"repo_id": "TIGER-Lab/ViRL39K", "splits": ["train"], "annotation_file": "39Krelease.parquet", "image_archive": "images.zip"},
    "mathvision": {"repo_id": "MathLLMs/MathVision", "splits": ["test"]},
    "dynamath": {"repo_id": "DynaMath/DynaMath_Sample", "splits": [f"sample_variant{i}" for i in range(1, 11)]},
    "mmmu_pro": {"repo_id": "MMMU/MMMU_Pro", "config_name": "standard (10 options)", "splits": ["test"]},
    "3dsrbench": {"repo_id": "ccvl/3DSRBench", "config_name": "benchmark", "splits": ["test"], "train_fraction": 0.8, "seed": 42},
}


def dataset_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    names = {re.sub(r"[^a-z0-9]", "", item): item for item in _DEFAULTS}
    if key not in names:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {', '.join(_DEFAULTS)}")
    return names[key]


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            rows.append(row)
    return rows


def load_examples(path: str | Path) -> list[dict]:
    """Resolve prepared relative media paths for model inference."""
    path = Path(path).resolve()
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        _validate_example(row)
        if row["id"] in seen:
            raise ValueError(f"Duplicate example ID {row['id']!r} in {path}")
        seen.add(row["id"])
        row["images"] = [str((path.parent / image).resolve()) if not Path(image).is_absolute() else image for image in row["images"]]
        for image in row["images"]:
            if not Path(image).is_file():
                raise FileNotFoundError(f"Image file not found for {row['id']}: {image}")
    return rows


def _validate_example(row: dict) -> None:
    required = {"id", "dataset", "split", "question", "images", "choices", "answer", "metadata"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Example lacks fields: {sorted(missing)}")
    if not all(isinstance(row[key], str) for key in ("id", "dataset", "split", "question", "answer")):
        raise ValueError("Example identifiers, question, and answer must be strings")
    if not row["id"] or not row["question"].strip() or not row["answer"].strip():
        raise ValueError("Example ID, question, and answer must not be empty")
    if not isinstance(row["metadata"], dict):
        raise ValueError("metadata must be an object")
    for field in ("images", "choices"):
        if not isinstance(row[field], list) or not all(isinstance(value, str) for value in row[field]):
            raise ValueError(f"{field} must be a list of strings")


def _choice_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Options must be a list or a string encoding a list")
    return [str(item) for item in value]


class _MediaStore:
    def __init__(self, destination: Path, source_root: Path | None = None):
        self.root = destination
        self.source_root = source_root
        self.directory = destination / "images"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache: dict[tuple[str, bool], str] = {}

    def save(self, value: Any, *, flip: bool = False) -> str:
        from PIL import Image, ImageOps

        cache_key = (value, flip) if isinstance(value, str) else None
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        if isinstance(value, str) and flip:
            # Reuse the original local image instead of downloading each flip.
            value = self.root / self.save(value, flip=False)
        if isinstance(value, dict):
            value = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value.get("path")
        if isinstance(value, (bytes, bytearray)):
            value = io.BytesIO(value)
        if isinstance(value, str):
            if value.startswith(("https://", "http://")):
                request = urllib.request.Request(value, headers={"User-Agent": "dualref-dataset-preparation/1"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    value = io.BytesIO(response.read())
            else:
                source = Path(value)
                if not source.is_absolute() and self.source_root is not None:
                    source = self.source_root / source
                value = source
        image = value if isinstance(value, Image.Image) else Image.open(value)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if flip:
            image = ImageOps.mirror(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        filename = hashlib.sha256(content).hexdigest() + ".png"
        target = self.directory / filename
        if not target.exists():
            target.write_bytes(content)
        relative = target.relative_to(self.root).as_posix()
        if cache_key is not None:
            self.cache[cache_key] = relative
        return relative


def _adapt_record(name: str, raw: dict, split: str, index: int, media: _MediaStore) -> dict:
    metadata = {key: raw[key] for key in ("category", "subject", "grade", "level", "answer_type", "knowledge_level", "source") if raw.get(key) is not None}
    metadata.update({"source_split": split, "source_index": index})
    question = str(raw.get("question", "")).strip()
    identifier = str(raw.get("id", raw.get("qid", raw.get("index", index))))
    choices: list[str] = []
    images = []
    answer = canonical_answer(raw.get("answer", raw.get("ground_truth", "")))
    if name == "scienceqa":
        choices = _choice_list(raw["choices"])
        answer_index = int(raw["answer"])
        if not 0 <= answer_index < len(choices):
            raise ValueError(f"ScienceQA answer index out of bounds: {identifier}")
        answer = chr(65 + answer_index)
        if raw.get("hint"):
            question += "\nContext: " + str(raw["hint"]).strip()
        if raw.get("image") is not None:
            images.append(media.save(raw["image"]))
        identifier = f"{split}:{identifier}"
    elif name == "virl39k":
        values = raw.get("image", [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            images.append(media.save(value))
        question = re.sub(r"<image>\s*", "", question).strip()
    elif name in {"mathvision", "dynamath"}:
        choices = _choice_list(raw.get("options", []))
        value = raw.get("decoded_image")
        if value is None:
            value = raw.get("image")
        if value is not None:
            images.append(media.save(value))
        if name == "dynamath":
            metadata["seed_question_id"] = identifier
            identifier = f"{split}:{identifier}"
    elif name == "mmmu_pro":
        choices = _choice_list(raw["options"])
        order = []
        for text in [question, *choices]:
            for match in re.finditer(r"<image\s+(\d+)>", text):
                image_index = int(match[1])
                if image_index not in order:
                    order.append(image_index)
        present = sorted(int(key.split("_")[-1]) for key, value in raw.items() if re.fullmatch(r"image_\d+", key) and value is not None)
        for image_index in present:
            if image_index not in order:
                order.append(image_index)
        for image_index in order:
            if raw.get(f"image_{image_index}") is None:
                raise ValueError(f"MMMU-Pro {identifier} references unavailable image {image_index}")
            images.append(media.save(raw[f"image_{image_index}"]))
        mapping = {original: position + 1 for position, original in enumerate(order)}
        def renumber(text: str) -> str:
            return re.sub(r"<image\s+(\d+)>", lambda match: f"<image {mapping[int(match[1])]}>", text)
        question = renumber(question)
        choices = [renumber(choice) for choice in choices]
        metadata["source_image_order"] = order
    elif name == "3dsrbench":
        labels = [label for label in "ABCDEFGHIJ" if raw.get(label) is not None and str(raw[label]).strip() and str(raw[label]).lower() != "nan"]
        choices = [str(raw[label]) for label in labels]
        if answer not in labels:
            raise ValueError(f"3DSRBench answer not among supplied options: {identifier}")
        answer = chr(65 + labels.index(answer))
        source = str(raw["image_url"])
        flip = identifier.endswith("-flip")
        images.append(media.save(source, flip=flip))
        metadata.update({"image_url": source, "group_id": source, "flip": flip, "source_id": identifier})
    result = {"id": f"{name}:{identifier}", "dataset": name, "split": split, "question": question, "images": images, "choices": choices, "answer": answer, "metadata": metadata}
    _validate_example(result)
    return result


def _group_partition(rows: list[dict], fraction: float, seed: int) -> dict[str, list[dict]]:
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    groups = sorted({row["metadata"]["group_id"] for row in rows})
    if len(groups) < 2:
        raise ValueError("3DSRBench requires at least two source images for disjoint partitions")
    random.Random(seed).shuffle(groups)
    boundary = min(len(groups) - 1, max(1, int(len(groups) * fraction)))
    training = set(groups[:boundary])
    result: dict[str, list[dict]] = {"train": [], "test": []}
    for row in rows:
        split = "train" if row["metadata"]["group_id"] in training else "test"
        row = {**row, "split": split}
        result[split].append(row)
    return result


def _unpack_archive(archive: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Image archive contains an invalid path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _source_records(name: str, spec: dict, destination: Path) -> tuple[dict[str, Iterable[dict]], dict, Path | None]:
    # Native local source files permit offline preparation without Hub access.
    if spec.get("files"):
        records, hashes = {}, {}
        roots = []
        for split, filename in spec["files"].items():
            path = Path(filename).resolve()
            records[split] = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else read_jsonl(path)
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            roots.append(path.parent)
        root = Path(spec["media_root"]).resolve() if spec.get("media_root") else roots[0]
        return records, {"source": "local", "file_sha256": hashes, "revision": spec.get("revision", "local")}, root

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    repo = spec["repo_id"]
    revision = HfApi().dataset_info(repo, revision=spec.get("revision", "main")).sha
    source = {"repo_id": repo, "config_name": spec.get("config_name"), "revision": revision}
    if name == "virl39k":
        annotation = hf_hub_download(repo, spec["annotation_file"], repo_type="dataset", revision=revision)
        archive = hf_hub_download(repo, spec["image_archive"], repo_type="dataset", revision=revision)
        media_root = destination / "source_media"
        _unpack_archive(archive, media_root)
        records = {"train": load_dataset("parquet", data_files={"train": annotation}, split="train")}
        return records, source, media_root
    records = {
        split: load_dataset(repo, name=spec.get("config_name"), split=split, revision=revision)
        for split in spec["splits"]
    }
    return records, source, None


def prepare_dataset(name: str, output_dir: str, config: dict) -> dict[str, str]:
    """Prepare benchmark data as canonical JSONL with a split manifest.

    ``config`` may be a dataset spec, a name-to-spec map, or a complete config
    containing that map under ``datasets``. No downloading occurs at import time.
    """
    name = dataset_name(name)
    settings = config.get("datasets", config)
    if isinstance(settings, str):
        settings = read_config(settings)
        settings = settings.get("datasets", settings)
    if name in settings:
        supplied = settings[name]
    elif isinstance(settings.get("data"), dict):
        supplied = settings["data"]
    else:
        keys = {"repo_id", "revision", "config_name", "splits", "annotation_file", "image_archive", "train_fraction", "seed", "files", "media_root"}
        supplied = {key: value for key, value in settings.items() if key in keys}
    spec = {**_DEFAULTS[name], **supplied}
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_splits, source, media_root = _source_records(name, spec, destination)
    media = _MediaStore(destination, media_root)
    splits: dict[str, list[dict]] = {}
    for source_split, raw_rows in raw_splits.items():
        target_split = "test" if name == "dynamath" else source_split
        rows = splits.setdefault(target_split, [])
        for index, raw in enumerate(raw_rows):
            row = _adapt_record(name, raw, source_split, index, media)
            row["split"] = target_split
            rows.append(row)
    if name == "3dsrbench":
        splits = _group_partition([row for rows in splits.values() for row in rows], float(spec["train_fraction"]), int(spec["seed"]))
    outputs, split_manifest = {}, {}
    global_ids = set()
    for split, rows in splits.items():
        path = destination / f"{split}.jsonl"
        for row in rows:
            if row["id"] in global_ids:
                raise ValueError(f"Duplicate prepared ID: {row['id']}")
            global_ids.add(row["id"])
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[split] = str(path)
        split_manifest[split] = {"path": path.name, "count": len(rows), "ids": [row["id"] for row in rows], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema_version": 1, "dataset": name, "source": source, "preparation": spec, "splits": split_manifest}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs
