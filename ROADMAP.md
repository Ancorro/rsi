# Roadmap

*AI-sourced: written by an AI agent.*

- [x] Re-score the authors' released edits on a sample (probe30, 2026-11-02): `p50c` reproduces the gain.
- [ ] Find the looping direction: compare MLP outputs of layers 15 and 19 on looping and finished completions
      (mean difference, then principal components), using GSM8K and MATH training problems only.
- [ ] Remove it from the MLP output columns, save the edited checkpoint, evaluate it the same way.
- [ ] Compare our direction with the released `p50c` edit (cosine similarity, which columns).
- [ ] Presence penalty in generation to match the post (a logits processor, or vLLM).
- [ ] Does the same direction exist in Qwen3.5-2B or 4B, and does removing it help there?
- [ ] Let a model do the analysis itself through a weights-only Python tool, as in the post, with a pass budget.
