"""Deterministic answer extraction and reward scoring."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_MODE = re.compile(r"^\s*(?:<FAST>|<SLOW>)\s*", re.IGNORECASE)
_SLOW = re.compile(r"</?think\b[^>]*>|<SLOW>|Long\s+Thinking\s*:", re.IGNORECASE)


def word_count(response: str) -> int:
    """Count whitespace-separated words, including the reasoning continuation."""
    return len(response.split())


def is_slow(response: str) -> bool:
    """Detect reasoning markers independently of response length."""
    return bool(_SLOW.search(response))


def answer_span(response: str) -> str:
    span = re.split(r"</think\s*>", response, flags=re.IGNORECASE)[-1].strip()
    # External methods may wrap the final answer; this only adapts formatting.
    tagged = re.findall(r"<answer\s*>(.*?)</answer\s*>", span, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        span = tagged[-1].strip()
    span = re.sub(r"</?answer\s*>", "", span, flags=re.IGNORECASE).strip()
    while _MODE.match(span):
        span = _MODE.sub("", span, count=1).strip()
    span = re.sub(r"^(?:Short|Long)\s+Thinking\s*:\s*", "", span, flags=re.IGNORECASE).strip()
    return span


def boxed_spans(text: str) -> list[str]:
    """Read balanced boxes, including nested LaTeX braces, without evaluation."""
    result = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth, pos = match.end(), 1, match.end()
        while pos < len(text) and depth:
            if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
                depth += 1
            elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
                depth -= 1
            pos += 1
        if depth == 0:
            result.append(text[start : pos - 1].strip())
    return result


def canonical_answer(value: Any) -> str:
    text = str(value).strip()
    boxes = boxed_spans(text)
    return boxes[-1] if boxes else text


def _option_letter(span: str, count: int) -> str | None:
    if not 1 <= count <= 26:
        return None
    valid = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count]
    # Explicit alternative lists are ambiguous, including boxed and bold forms.
    plain = span.replace("**", "")
    lists = re.finditer(
        r"\b([A-Z])\b\s*(?:,|/|\bor\b|\band\b)\s*\b([A-Z])\b",
        plain,
        flags=re.IGNORECASE,
    )
    if any(m[1].upper() in valid and m[2].upper() in valid and m[1].upper() != m[2].upper() for m in lists):
        return None
    for box in reversed(boxed_spans(span)):
        if box and box[0].upper() in valid:
            return box[0].upper()
    patterns = (
        r"^\s*(?:([A-Za-z])\.|\(([A-Za-z])\))",
        r"\banswer\s*(?:is|:|=|should\s+be)\s*(?:\*\*)?([A-Za-z])\b",
        r"^\s*\*\*([A-Za-z])\*\*",
        r"^\s*([A-Za-z])\s*$",
        r"^\s*([A-Za-z])\s+",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, span, flags=re.IGNORECASE))
        if matches:
            letter = next(group for group in matches[-1].groups() if group).upper()
            return letter if letter in valid else None
    for line in span.splitlines():
        letter = line.strip().strip(".()").strip().upper()
        if len(letter) == 1 and letter in valid:
            return letter
    return None


def _numeric_ground_truth(answer: str) -> float | None:
    try:
        value = float(answer)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _math_score(example: dict, span: str) -> tuple[bool, bool, str | None]:
    truth = canonical_answer(example["answer"])
    choices = example.get("choices", [])
    if choices and len(truth) == 1 and truth.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]:
        letter = _option_letter(span, len(choices))
        surface = str(choices[ord(truth.upper()) - ord("A")]).strip()
        # Accept the correct option's text when it appears in the answer.
        if surface and surface.casefold() in span.casefold():
            return True, True, letter or surface
        if letter is not None:
            return letter == truth.upper(), True, letter
        return False, False, None
    numeric_truth = _numeric_ground_truth(truth)
    if numeric_truth is not None:
        boxes = boxed_spans(span)
        numeric_span = boxes[-1] if boxes else span
        numbers = list(_NUMBER.finditer(numeric_span))
        if not numbers:
            return False, False, None
        prediction = numbers[-1].group()
        try:
            value = float(prediction)
        except (ValueError, OverflowError):
            return False, False, None
        if not math.isfinite(value):
            return False, False, None
        return abs(value - numeric_truth) <= 1e-3 * max(1.0, abs(numeric_truth)), True, prediction
    if not span.strip() or not truth:
        return False, False, None
    boxes = boxed_spans(span)
    prediction = boxes[-1] if boxes else span.strip()
    # Match short text answers by containment.
    return truth.casefold() in span.casefold(), True, prediction


def score_response(example: dict, response: str) -> dict:
    """Return correctness, format validity, extracted answer, and response cost."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    span = answer_span(response)
    dataset = re.sub(r"[^a-z0-9]", "", str(example.get("dataset", "")).lower())
    if dataset in {"scienceqa", "mmmupro", "3dsrbench"}:
        prediction = _option_letter(span, len(example.get("choices", [])))
        valid = prediction is not None
        correct = valid and prediction == str(example["answer"]).strip().upper()
    else:
        correct, valid, prediction = _math_score(example, span)
    return {
        "correct": float(correct),
        "format": float(valid),
        "prediction": prediction,
        "slow": is_slow(response),
        "words": word_count(response),
    }
