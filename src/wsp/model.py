"""Load Qwen3.5 text models, the real one or a tiny random one for CPU tests."""

from __future__ import annotations

import torch

MODEL = "Qwen/Qwen3.5-0.8B"


def device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = MODEL, dtype=torch.bfloat16, device: str | None = None):
    """Text-only causal LM and tokenizer. The vision tower of the checkpoint is not loaded."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return model.to(device or device_default()).eval(), tok


def tiny_model(seed: int = 0, vocab_size: int = 256):
    """Random Qwen3.5 text model with the same layer mix (3 linear-attention, 1 full), for tests."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        tie_word_embeddings=True,
    )
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = None
    return model


def mlp_down_proj(model, layer: int) -> torch.nn.Linear:
    """The MLP output matrix of one decoder layer: it writes into the residual stream."""
    return model.model.layers[layer].mlp.down_proj
