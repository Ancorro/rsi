"""Sampled generation with the paper's decoding settings, batched, with loop statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .loops import hit_limit, repeated_ngram_fraction

# Paper's settings (blog post, "Evaluation"): thinking on, temperature 0.6, top-p 0.95, top-k 20,
# presence penalty 1.5, 32,768 new tokens. transformers has no presence penalty; repetition_penalty
# is a different rule, so it is off by default and the difference is noted in the README.
DECODING = dict(temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=32_768)

PROMPT = "{question}\nPlease reason step by step, and put your final answer within \boxed{{}}."


@dataclass
class Completion:
    text: str
    num_tokens: int
    finished: bool
    hit_limit: bool
    repeat_frac: float


def chat_prompt(tok, question: str, thinking: bool = True) -> str:
    msgs = [{"role": "user", "content": PROMPT.format(question=question)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = DECODING["max_new_tokens"],
             temperature: float = DECODING["temperature"], top_p: float = DECODING["top_p"],
             top_k: int = DECODING["top_k"], seed: int = 0) -> list[Completion]:
    torch.manual_seed(seed)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, do_sample=temperature > 0, temperature=temperature, top_p=top_p,
                         top_k=top_k, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    gen_eos = model.generation_config.eos_token_id
    gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
    eos = {tok.eos_token_id, *gen_eos} - {None}
    res = []
    for row in out[:, enc["input_ids"].shape[1]:].tolist():
        cut = next((i for i, t in enumerate(row) if t in eos), None)
        toks = row if cut is None else row[:cut]
        res.append(Completion(
            text=tok.decode(toks, skip_special_tokens=True), num_tokens=len(toks),
            finished=cut is not None,
            hit_limit=hit_limit(len(toks), max_new_tokens, cut is not None),
            repeat_frac=repeated_ngram_fraction(toks)))
    return res
