"""GSM8K data and scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLE = DATA / "gsm8k_test_50.jsonl"

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load(path: Path | str = SAMPLE, n: int | None = None) -> list[dict]:
    """Rows with id, question and gold (the number after '####')."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    return rows[:n] if n else rows


def _norm(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract_answer(text: str) -> str | None:
    """Final answer of a completion: the last \boxed{...}, else the last number after any
    </think>, else None. Text inside an unfinished think block gives None."""
    if "<think>" in text and "</think>" not in text:
        return None
    text = text.split("</think>")[-1]
    boxed = re.findall(r"\boxed\{([^{}]*)\}", text)
    if boxed:
        nums = _NUM.findall(boxed[-1])
        return _norm(nums[-1]) if nums else _norm(boxed[-1])
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def is_correct(text: str, gold: str) -> bool:
    ans = extract_answer(text)
    return ans is not None and ans == _norm(gold)
