import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_goal  # noqa: E402


def setup(tmp_path, n=1000, acc=0.6, **hp):
    edit = tmp_path / "edit.npz"
    np.savez(edit, t0_idx=np.arange(n, dtype=np.int32), t0_val=np.zeros(n, np.uint16))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"hyperparameters": {**check_goal.SETTINGS, **hp},
                                   "results": {"ours": {"gsm8k": {"acc": acc}}}}))
    return edit, summary


def test_good_edit_passes(tmp_path):
    assert check_goal.check(*setup(tmp_path)) == []


def test_over_budget_fails(tmp_path):
    assert check_goal.check(*setup(tmp_path, n=check_goal.BUDGET + 1))


def test_low_accuracy_fails(tmp_path):
    assert check_goal.check(*setup(tmp_path, acc=0.3))


def test_easier_settings_fail(tmp_path):
    assert check_goal.check(*setup(tmp_path, samples=1))


def test_missing_files_fail(tmp_path):
    assert check_goal.check(tmp_path / "none.npz", tmp_path / "none.json")
