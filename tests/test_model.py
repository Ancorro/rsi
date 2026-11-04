import sys
from pathlib import Path

import torch

from wsp.model import mlp_down_proj, tiny_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_tiny_model_runs():
    model = tiny_model()
    logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert logits.shape == (1, 4, 256)


def test_down_proj_writes_residual():
    model = tiny_model()
    w = mlp_down_proj(model, 2).weight
    assert w.shape == (model.config.hidden_size, model.config.intermediate_size)


def test_eval_script_smoke(tmp_path):
    import eval_gsm8k

    out = tmp_path / "r.json"
    s = eval_gsm8k.main(["--tiny", "--n", "2", "--samples", "2", "--max-new-tokens", "6",
                         "--out", str(out)])
    assert s["n_problems"] == 2 and out.exists()
    assert 0.0 <= s["accuracy"] <= 1.0
    assert s["hit_limit_rate"] == 1.0  # random model with no end token always runs to the limit
