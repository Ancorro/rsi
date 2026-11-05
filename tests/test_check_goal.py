import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_goal  # noqa: E402

ACC = {"unedited": 0.13, "ours": 0.60, "round2": 0.66, "selfdiag": 0.30, "finetune": 0.40}


def summary(acc=ACC, cut=0.17, **hp):
    return {"hyperparameters": {**check_goal.SETTINGS, **hp},
            "results": {m: {"gsm8k": {"acc": a, "cut_share": cut}} for m, a in acc.items()}}


def make_edit(edits, name, n=1000, data=("openai/gsm8k:train",), **extra):
    d = edits / name
    d.mkdir(parents=True)
    np.savez(d / "edit.npz", t0_idx=np.arange(n, dtype=np.int32), t0_val=np.zeros(n, np.uint16))
    (d / "manifest.json").write_text(json.dumps({"analysis_data": list(data), "gpu_hours": 2.0, **extra}))
    return d


def full_setup(tmp_path, finetune_hours=2.0):
    edits = tmp_path / "edits"
    make_edit(edits, "ours")
    make_edit(edits, "round2", base="ours")
    sd = make_edit(edits, "selfdiag")
    (sd / "diagnosis.json").write_text(json.dumps({
        "diagnoser": check_goal.BASE_MODEL,
        "diagnoses": [{"id": f"gsm8k-train-{i}", "diagnosis": "repeats the same step"} for i in range(20)]}))
    res = tmp_path / "results"
    res.mkdir()
    (res / "finetune.json").write_text(json.dumps({"gpu_hours": finetune_hours}))
    return edits, res / "summary.json"


def run(edits, path, summ):
    path.write_text(json.dumps(summ))
    return check_goal.main(["--edits", str(edits), "--summary", str(path)])


def test_everything_passes(tmp_path):
    assert run(*full_setup(tmp_path), summary()) == 0


def test_finetune_may_win(tmp_path):
    assert run(*full_setup(tmp_path), summary({**ACC, "finetune": 0.9})) == 0


def test_released_edit_fits_budget(tmp_path):
    released = check_goal.REPLICATE / "edits" / "p50c"
    if not released.exists():
        pytest.skip("released edits not included in this copy")
    d = tmp_path / "ours"
    shutil.copytree(released, d)
    m = json.loads((d / "manifest.json").read_text())
    (d / "manifest.json").write_text(json.dumps({**m, "analysis_data": ["MATH:train"], "gpu_hours": 1}))
    assert check_goal.read_edit(d)[1] == []


def test_over_budget_and_test_split_fail(tmp_path):
    make_edit(tmp_path, "ours", n=check_goal.BUDGET + 1, data=["openai/gsm8k:test"])
    errs = check_goal.read_edit(tmp_path / "ours")[1]
    assert any("budget" in e for e in errs) and any("training splits" in e for e in errs)


def test_weak_first_edit_fails(tmp_path):
    edits, _ = full_setup(tmp_path)
    assert len(check_goal.check_first(edits, summary({**ACC, "ours": 0.3}, cut=0.5))) == 2


def test_round2_needs_gain_and_base(tmp_path):
    edits, _ = full_setup(tmp_path)
    assert check_goal.check_round2(edits, summary({**ACC, "round2": 0.62}))
    (edits / "round2" / "manifest.json").write_text(
        json.dumps({"analysis_data": ["x:train"], "gpu_hours": 1}))
    assert any("base" in e for e in check_goal.check_round2(edits, summary()))


def test_selfdiag_needs_own_diagnosis(tmp_path):
    edits, _ = full_setup(tmp_path)
    (edits / "selfdiag" / "diagnosis.json").write_text(json.dumps({"diagnoser": "gpt", "diagnoses": []}))
    assert len(check_goal.check_selfdiag(edits, summary())) == 2
    assert check_goal.check_selfdiag(edits, summary({**ACC, "selfdiag": 0.2}))


def test_finetune_over_budget_fails(tmp_path):
    edits, path = full_setup(tmp_path, finetune_hours=5.0)
    assert check_goal.check_finetune(edits, path, summary())


def test_wrong_settings_fail(tmp_path):
    assert run(*full_setup(tmp_path), summary(samples=1, presence_penalty=0.0)) == 1


def test_missing_summary_fails(tmp_path):
    assert check_goal.main(["--edits", str(tmp_path), "--summary", str(tmp_path / "none.json")]) == 1
