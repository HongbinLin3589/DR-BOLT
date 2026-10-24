import json
import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture
def write_records(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return str(path)
    return write


@pytest.fixture
def example():
    return {"id": "one", "dataset": "scienceqa", "split": "train", "question": "Pick A",
            "images": [], "choices": ["one", "two"], "answer": "A", "metadata": {}}
