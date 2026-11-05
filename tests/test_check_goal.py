import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_goal  # noqa: E402

GOOD = {"acc": 0.6, "cut_share": 0.17}


def summary(res=GOOD, **hp):
    return {"hyperparameters": {**check_goal.SETTINGS, **hp}, "results": {"ours": {"gsm8k": res}}}


def make_edit(tmp_path, n=1000, data=("openai/gsm8k:train",)):
    d = tmp_path / "edit"
    d.mkdir()
    np.savez(d / "edit.npz", t0_idx=np.arange(n, dtype=np.int32), t0_val=np.zeros(n, np.uint16))
    (d / "manifest.json").write_text(json.dumps({"analysis_data": list(data)}))
    return d


def run(tmp_path, edit, summ):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summ))
    return check_goal.main(["--edit", str(edit), "--summary", str(p)])


def test_good_edit_passes(tmp_path):
    assert run(tmp_path, make_edit(tmp_path), summary()) == 0


def test_released_edit_fits_budget(tmp_path):
    released = check_goal.REPLICATE / "edits" / "p50c"
    d = tmp_path / "edit"
    shutil.copytree(released, d)
    m = json.loads((d / "manifest.json").read_text())
    m["analysis_data"] = ["openai/gsm8k:train", "MATH:train"]
    (d / "manifest.json").write_text(json.dumps(m))
    assert check_goal.check_edit(d, summary(), "ours") == []


def test_over_budget_fails(tmp_path):
    errs = check_goal.check_edit(make_edit(tmp_path, n=check_goal.BUDGET + 1), summary(), "ours")
    assert any("budget" in e for e in errs)


def test_test_split_fails(tmp_path):
    errs = check_goal.check_edit(make_edit(tmp_path, data=["openai/gsm8k:test"]), summary(), "ours")
    assert any("training splits" in e for e in errs)


def test_weak_result_fails(tmp_path):
    errs = check_goal.check_edit(make_edit(tmp_path), summary({"acc": 0.3, "cut_share": 0.5}), "ours")
    assert len(errs) == 2


def test_missing_edit_fails(tmp_path):
    assert run(tmp_path, tmp_path / "none", summary()) == 1


def test_wrong_settings_fail(tmp_path):
    assert run(tmp_path, make_edit(tmp_path), summary(samples=1, presence_penalty=0.0)) == 1
