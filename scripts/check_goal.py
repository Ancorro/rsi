"""Goal check for the looping-direction task. CPU only: reads files, runs no model.

    python scripts/check_goal.py

Passes when both hold:
1. A real edit that works: `edits/ours/` holds an edit (edit.npz + manifest.json, the released edits' format) that
   changes at most 131,072 weights (the paper's budget), whose manifest lists only training data as analysis input,
   and whose GSM8K result in `results/ours/summary.json` reaches 50% accuracy with at most 30% of answers cut at the
   token limit (unedited: 13.3% and 87%).
2. The paper's evaluation settings: 50 problems, 2 samples each, temperature 0.6, top-p 0.95, top-k 20, presence
   penalty 1.5, 32,768 tokens, thinking on.

The GPU evaluation that writes the summary: papers/weight-space-programming/replicate/run.sh ours 50 2 ours
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLICATE = ROOT / "papers" / "weight-space-programming" / "replicate"
BUDGET = 131_072
MIN_ACC, MAX_CUT = 0.50, 0.30
SETTINGS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32_768,
            "n_problems_per_set": 50, "samples": 2, "thinking": True}


def check_edit(edit_dir: Path, summary: dict, model: str) -> list[str]:
    """Condition 1: the edit exists, fits the budget, used training data only, and beats the unedited model."""
    errs = []
    npz, man = edit_dir / "edit.npz", edit_dir / "manifest.json"
    if not npz.exists() or not man.exists():
        return [f"no edit: need {npz} and {man}"]
    manifest = json.loads(man.read_text(encoding="utf-8"))
    with np.load(npz) as d:
        changed = sum(len(d[k]) for k in d.files if k.endswith("_idx"))
    if changed == 0:
        errs.append("edit changes no weights")
    if changed > BUDGET:
        errs.append(f"edit changes {changed:,} weights, budget {BUDGET:,}")
    data = manifest.get("analysis_data")
    if not data:
        errs.append("manifest.json has no analysis_data list (the datasets and splits the edit was found from)")
    else:
        bad = [s for s in data if "train" not in s.lower() or "test" in s.lower()]
        if bad:
            errs.append(f"analysis_data must be training splits only, found: {bad}")
    res = summary.get("results", {}).get(model, {}).get("gsm8k")
    if res is None:
        errs.append(f"no GSM8K result for '{model}' in the summary")
    else:
        if res["acc"] < MIN_ACC:
            errs.append(f"GSM8K accuracy {res['acc']:.1%}, need {MIN_ACC:.0%}")
        if res["cut_share"] > MAX_CUT:
            errs.append(f"{res['cut_share']:.1%} of answers hit the token limit, need at most {MAX_CUT:.0%}")
    return errs


def check_settings(summary: dict) -> list[str]:
    """Condition 2: the evaluation used the paper's settings."""
    hp = summary.get("hyperparameters", {})
    return [f"setting {k} is {hp.get(k)!r}, need {v!r}" for k, v in SETTINGS.items() if hp.get(k) != v]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", type=Path, default=REPLICATE / "edits" / "ours")
    ap.add_argument("--summary", type=Path, default=REPLICATE / "results" / "ours" / "summary.json")
    ap.add_argument("--model", default="ours", help="model name inside the summary")
    a = ap.parse_args(argv)
    if not a.summary.exists():
        print(f"FAIL: no evaluation summary at {a.summary}")
        return 1
    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    failed = False
    for name, errs in [("1. edit works", check_edit(a.edit, summary, a.model)),
                       ("2. paper settings", check_settings(summary))]:
        print(f"{'PASS' if not errs else 'FAIL'}: {name}")
        for e in errs:
            print(f"  - {e}")
        failed |= bool(errs)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
