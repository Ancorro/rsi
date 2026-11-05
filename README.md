# RSI

*AI-sourced: assisted by AI agents.*

Recursive self-improvement (RSI): work on models that improve themselves or their successors. The first thread is an
independent reproduction of one result from "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang,
Khattab; MIT CSAIL, November 2026; [blog post](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/RikiyaT/lm_understand-logs)). This repo is not affiliated with the authors.

Who it is for: anyone checking whether one direction in Qwen3.5-0.8B's weights causes its reasoning loops.

Built with: Python 3.12, PyTorch, Hugging Face transformers.

## The result being reproduced

The post reports that Qwen3.5-0.8B with thinking on often loops until it reaches the token limit. A model (GPT-6
Astra) working only from the weights traced the looping to one direction written by the MLP (multilayer perceptron)
outputs of layers 15 and 19. It subtracted that direction from 64 output columns in each layer, 131,072 weights
(0.016% of the model). Reported results, with 50 forward passes of analysis:

| Setting | MATH-500 | GSM8K | Hit token limit (%) |
| --- | --- | --- | --- |
| Base | 7.9 | 13.8 | 93.0 |
| Edited | 48.1 | 67.1 | 33.1 |

Evaluation settings in the post: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, 32,768
new tokens, 2 samples per GSM8K problem. The post does not say how the direction was computed or give the exact
projection; the logs may.

## What is here now

| Path | What |
| --- | --- |
| `src/wsp/model.py` | Load Qwen3.5-0.8B (text only), a tiny random Qwen3.5 for tests, the MLP output matrix of a layer. |
| `src/wsp/gsm8k.py` | GSM8K sample loader, final-answer extraction (`\boxed{}` or last number after `</think>`), scoring. |
| `src/wsp/generate.py` | Batched sampling with the post's decoding settings; per completion: tokens, finished, hit limit, repeats. |
| `src/wsp/loops.py` | Looping measures: share of repeated 20-token windows, hit-limit flag. |
| `scripts/eval_gsm8k.py` | GSM8K accuracy and hit-limit rate for a model (base or an edited checkpoint path), JSON output. |
| `data/gsm8k_test_50.jsonl` | 50 GSM8K test problems with numeric answers (MIT license, from `openai/gsm8k`). |
| `papers/weight-space-programming/` | Notes on the paper; `replicate/` holds the GPU probe (vLLM, one RTX 3090) that rebuilds the authors' released edits and scores them. |

Probe result (2026-11-02, `replicate/results/probe30/`): the authors' released edit (`p50c`) reproduces the gain on 30
evenly spaced problems per set, 1 sample, the post's decoding settings with presence penalty 1.5 (vLLM), 32,768 tokens:

| Model | MATH-500 | GSM8K | Hit token limit, MATH / GSM8K (%) |
| --- | --- | --- | --- |
| Unedited | 0.0 | 13.3 | 100 / 87 |
| Released edit `p50c` | 50.0 | 60.0 | 33 / 17 |

Not here yet: finding the looping direction ourselves, making our own edit, and evaluating it. The `src/wsp` package
is the CPU-testable base for that work; the `replicate/` probe stays the reference GPU evaluation.

Differences from the post in `src/wsp`: transformers has no presence penalty, so generation there runs without one (the
`replicate/` probe uses vLLM with it); a completion that reaches the token limit counts as wrong.

## How to run

```
bash setup.sh     # .venv/ with CPU torch 2.13.0 and transformers 5.19.0 (about 90 s)
bash check.sh     # unit tests on a tiny random Qwen3.5 and ruff (about 20 s), CPU only
```

Evaluation needs a GPU for the full 32,768-token limit; on CPU, Qwen3.5-0.8B makes about 0.7 tokens per second per
sequence, so only short smoke runs are practical there:

```
.venv/bin/python scripts/eval_gsm8k.py --n 50 --samples 2 --out results/base.json          # GPU
.venv/bin/python scripts/eval_gsm8k.py --n 2 --samples 1 --max-new-tokens 64              # CPU smoke run
.venv/bin/python scripts/eval_gsm8k.py --tiny --n 2 --max-new-tokens 8                     # random model
```

## Plans

See [ROADMAP.md](ROADMAP.md).

## Repo rules

This repo can be made public at any time, history included.

- No secrets: keys go in `.env` (untracked); `.env.example` lists the names only.
- No private information: cluster paths, run logs, costs and personal notes go in `private/`, which is untracked and
  backed up outside git.
- Keep the real history: no squashing or force-pushing published commits.
