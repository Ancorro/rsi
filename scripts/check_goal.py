"""Goal check: an edit to Qwen3.5-0.8B's weights that fixes its looping. CPU only, reads two files.

    python scripts/check_goal.py

Passes when the edit in replicate/edits/ours/ changes at most 131,072 weights (the paper's budget) and its GSM8K
accuracy in replicate/results/goal/summary.json is at least 50% (unedited: 13.3%), scored with the paper's settings.

The summary comes from a GPU run: papers/weight-space-programming/replicate/run.sh goal 50 2 ours
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPLICATE = Path(__file__).resolve().parents[1] / "papers" / "weight-space-programming" / "replicate"
EDIT = REPLICATE / "edits" / "ours" / "edit.npz"
SUMMARY = REPLICATE / "results" / "goal" / "summary.json"
BUDGET = 131_072
MIN_ACC = 0.50
SETTINGS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32_768,
            "n_problems_per_set": 50, "samples": 2, "thinking": True}


def check(edit: Path = EDIT, summary: Path = SUMMARY) -> list[str]:
    """Reasons the goal is not met; empty when it is."""
    if not edit.exists():
        return [f"no edit at {edit}"]
    if not summary.exists():
        return [f"no evaluation summary at {summary}"]
    with np.load(edit) as d:
        changed = sum(len(d[k]) for k in d.files if k.endswith("_idx"))
    s = json.loads(summary.read_text(encoding="utf-8"))
    errs = []
    if not 0 < changed <= BUDGET:
        errs.append(f"edit changes {changed:,} weights, need 1 to {BUDGET:,}")
    hp = s.get("hyperparameters", {})
    errs += [f"setting {k} is {hp.get(k)!r}, need {v!r}" for k, v in SETTINGS.items() if hp.get(k) != v]
    res = s.get("results", {}).get("ours", {}).get("gsm8k")
    if res is None:
        errs.append("no GSM8K result for 'ours' in the summary")
    elif res["acc"] < MIN_ACC:
        errs.append(f"GSM8K accuracy {res['acc']:.1%}, need {MIN_ACC:.0%}")
    return errs


def main() -> int:
    errs = check()
    print("PASS" if not errs else "FAIL")
    for e in errs:
        print(f"  - {e}")
    return int(bool(errs))


if __name__ == "__main__":
    sys.exit(main())
