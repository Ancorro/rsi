"""Goal check for the weight-editing task. CPU only: reads files, runs no model.

    python scripts/check_goal.py

Every model is evaluated once on a GPU into one summary, papers/weight-space-programming/replicate/results/goal/:
    replicate/run.sh goal 50 2 unedited,ours,round2,selfdiag,finetune

An edit is a folder under replicate/edits/ in the released edits' format (edit.npz + manifest.json). Every edit's
manifest lists `analysis_data` (training splits only) and `gpu_hours` (GPU time spent finding it).

Passes when all five hold:
1. Paper settings: 50 problems, 2 samples each, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
   32,768 tokens, thinking on.
2. First edit (`ours`): at most 131,072 weights changed, found from training data only, GSM8K accuracy at least 50%
   with at most 30% of answers cut at the token limit (unedited: 13.3% and 87%).
3. Repeated editing (`round2`): a second edit applied on top of `ours` (manifest `"base": "ours"`), same weight
   budget, raises GSM8K accuracy at least 5 points over `ours`.
4. Self-diagnosis (`selfdiag`): an edit made from Qwen3.5-0.8B's own diagnosis of its failed answers.
   `diagnosis.json` in the edit folder has `"diagnoser": "Qwen/Qwen3.5-0.8B"` and at least 20 entries with `id` and
   `diagnosis`; the edit raises GSM8K accuracy at least 10 points over `unedited`.
5. Edit against training (`finetune`): a fine-tune of the base model, `results/goal/finetune.json` recording
   `gpu_hours` no more than the `ours` edit's, evaluated in the same summary. Passes whichever scores higher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLICATE = ROOT / "papers" / "weight-space-programming" / "replicate"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BUDGET = 131_072
MIN_ACC, MAX_CUT = 0.50, 0.30
ROUND2_GAIN, SELFDIAG_GAIN = 0.05, 0.10
MIN_DIAGNOSES = 20
SETTINGS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32_768,
            "n_problems_per_set": 50, "samples": 2, "thinking": True}


def gsm8k(summary: dict, model: str) -> dict | None:
    return summary.get("results", {}).get(model, {}).get("gsm8k")


def read_edit(edit_dir: Path) -> tuple[dict | None, list[str]]:
    """Manifest of a valid edit, or the reasons it is not one: missing, empty, over budget, test data, no GPU time."""
    npz, man = edit_dir / "edit.npz", edit_dir / "manifest.json"
    if not npz.exists() or not man.exists():
        return None, [f"no edit: need {npz} and {man}"]
    manifest = json.loads(man.read_text(encoding="utf-8"))
    errs = []
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
    if not isinstance(manifest.get("gpu_hours"), (int, float)):
        errs.append("manifest.json has no gpu_hours number")
    return manifest, errs


def need(summary: dict, model: str, errs: list[str]) -> dict | None:
    res = gsm8k(summary, model)
    if res is None:
        errs.append(f"no GSM8K result for '{model}' in the summary")
    return res


def check_settings(summary: dict) -> list[str]:
    hp = summary.get("hyperparameters", {})
    return [f"setting {k} is {hp.get(k)!r}, need {v!r}" for k, v in SETTINGS.items() if hp.get(k) != v]


def check_first(edits: Path, summary: dict) -> list[str]:
    _, errs = read_edit(edits / "ours")
    res = need(summary, "ours", errs)
    if res:
        if res["acc"] < MIN_ACC:
            errs.append(f"GSM8K accuracy {res['acc']:.1%}, need {MIN_ACC:.0%}")
        if res["cut_share"] > MAX_CUT:
            errs.append(f"{res['cut_share']:.1%} of answers hit the token limit, need at most {MAX_CUT:.0%}")
    return errs


def check_round2(edits: Path, summary: dict) -> list[str]:
    manifest, errs = read_edit(edits / "round2")
    if manifest is not None and manifest.get("base") != "ours":
        errs.append('round2 manifest needs "base": "ours" (applied on top of the first edit)')
    first, res = need(summary, "ours", errs), need(summary, "round2", errs)
    if first and res and res["acc"] < first["acc"] + ROUND2_GAIN:
        errs.append(f"round2 GSM8K {res['acc']:.1%}, need at least {first['acc'] + ROUND2_GAIN:.1%} "
                    f"(ours {first['acc']:.1%} + {ROUND2_GAIN:.0%})")
    return errs


def check_selfdiag(edits: Path, summary: dict) -> list[str]:
    manifest, errs = read_edit(edits / "selfdiag")
    diag = edits / "selfdiag" / "diagnosis.json"
    if manifest is not None:
        if not diag.exists():
            errs.append(f"no {diag}")
        else:
            d = json.loads(diag.read_text(encoding="utf-8"))
            if d.get("diagnoser") != BASE_MODEL:
                errs.append(f'diagnosis.json needs "diagnoser": "{BASE_MODEL}" (the model diagnoses itself)')
            entries = [e for e in d.get("diagnoses", []) if e.get("id") and e.get("diagnosis")]
            if len(entries) < MIN_DIAGNOSES:
                errs.append(f"diagnosis.json has {len(entries)} diagnoses with id and diagnosis, "
                            f"need {MIN_DIAGNOSES}")
    base, res = need(summary, "unedited", errs), need(summary, "selfdiag", errs)
    if base and res and res["acc"] < base["acc"] + SELFDIAG_GAIN:
        errs.append(f"selfdiag GSM8K {res['acc']:.1%}, need at least {base['acc'] + SELFDIAG_GAIN:.1%} "
                    f"(unedited {base['acc']:.1%} + {SELFDIAG_GAIN:.0%})")
    return errs


def check_finetune(edits: Path, summary_path: Path, summary: dict) -> list[str]:
    errs = []
    ft_path = summary_path.parent / "finetune.json"
    manifest, _ = read_edit(edits / "ours")
    edit_hours = (manifest or {}).get("gpu_hours")
    if not ft_path.exists():
        errs.append(f"no {ft_path} with the fine-tune's gpu_hours")
    else:
        hours = json.loads(ft_path.read_text(encoding="utf-8")).get("gpu_hours")
        if not isinstance(hours, (int, float)):
            errs.append("finetune.json has no gpu_hours number")
        elif not isinstance(edit_hours, (int, float)):
            errs.append("the ours edit has no gpu_hours to compare against")
        elif hours > edit_hours:
            errs.append(f"fine-tune used {hours} GPU hours, the edit {edit_hours}; budget must match")
    need(summary, "finetune", errs)
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", type=Path, default=REPLICATE / "edits")
    ap.add_argument("--summary", type=Path, default=REPLICATE / "results" / "goal" / "summary.json")
    a = ap.parse_args(argv)
    if not a.summary.exists():
        print(f"FAIL: no evaluation summary at {a.summary}")
        return 1
    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    checks = [("1. paper settings", check_settings(summary)),
              ("2. first edit", check_first(a.edits, summary)),
              ("3. repeated editing", check_round2(a.edits, summary)),
              ("4. self-diagnosis", check_selfdiag(a.edits, summary)),
              ("5. edit against training", check_finetune(a.edits, a.summary, summary))]
    for name, errs in checks:
        print(f"{'PASS' if not errs else 'FAIL'}: {name}")
        for e in errs:
            print(f"  - {e}")
    rows = [(m, gsm8k(summary, m)) for m in ("unedited", "ours", "round2", "selfdiag", "finetune")]
    print("GSM8K: " + ", ".join(f"{m} {r['acc']:.1%}" for m, r in rows if r))
    return int(any(errs for _, errs in checks))


if __name__ == "__main__":
    sys.exit(main())
