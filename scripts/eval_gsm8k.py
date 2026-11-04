"""GSM8K accuracy and looping rate of a model, base or edited.

    python scripts/eval_gsm8k.py --n 50 --samples 2 --out results/base.json
    python scripts/eval_gsm8k.py --tiny --n 2 --max-new-tokens 8      # CPU smoke run, random model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from wsp import gsm8k  # noqa: E402
from wsp.generate import DECODING, chat_prompt, generate  # noqa: E402
from wsp.model import MODEL, load_model, tiny_model  # noqa: E402


class _Enc(dict):
    def to(self, _device):
        return self


class ByteTok:
    """Stand-in tokenizer for the tiny random model: UTF-8 bytes as ids, 0 as padding."""

    eos_token_id = None
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, msgs, **_):
        return msgs[-1]["content"]

    def __call__(self, texts, return_tensors="pt", padding=True):
        ids = [list(t.encode())[:64] for t in texts]
        w = max(map(len, ids))
        return _Enc(
            input_ids=torch.tensor([[0] * (w - len(i)) + i for i in ids]),
            attention_mask=torch.tensor([[0] * (w - len(i)) + [1] * len(i) for i in ids]),
        )

    def decode(self, toks, skip_special_tokens=True):
        return bytes(t % 256 for t in toks).decode(errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF id or local path (an edited model)")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on CPU (smoke test)")
    ap.add_argument("--data", default=str(gsm8k.SAMPLE))
    ap.add_argument("--n", type=int, default=None, help="first n problems")
    ap.add_argument("--samples", type=int, default=2, help="samples per problem (paper: 2)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=DECODING["max_new_tokens"])
    ap.add_argument("--temperature", type=float, default=DECODING["temperature"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON file: summary and every completion")
    a = ap.parse_args(argv)

    if a.tiny:
        model, tok = tiny_model(), ByteTok()
    else:
        model, tok = load_model(a.model)
    rows = gsm8k.load(a.data, a.n)
    jobs = [(r, s) for r in rows for s in range(a.samples)]
    t0 = time.time()
    records = []
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i:i + a.batch]
        outs = generate(model, tok, [chat_prompt(tok, r["question"]) for r, _ in chunk],
                        max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                        seed=a.seed + i)
        for (r, s), c in zip(chunk, outs, strict=True):
            records.append(dict(
                id=r["id"], sample=s, gold=r["gold"], answer=gsm8k.extract_answer(c.text),
                # A completion cut off at the token limit counts as wrong, whatever it last said.
                correct=c.finished and gsm8k.is_correct(c.text, r["gold"]), tokens=c.num_tokens,
                hit_limit=c.hit_limit, repeat_frac=round(c.repeat_frac, 4), text=c.text))
    k = len(records)
    summary = dict(
        model="tiny-random" if a.tiny else a.model, n_problems=len(rows), samples=a.samples,
        max_new_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed,
        accuracy=sum(r["correct"] for r in records) / k,
        hit_limit_rate=sum(r["hit_limit"] for r in records) / k,
        mean_tokens=sum(r["tokens"] for r in records) / k,
        seconds=round(time.time() - t0, 1))
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summary, records=records), indent=1))
    return summary


if __name__ == "__main__":
    main()
