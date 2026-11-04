# Roadmap

*AI-sourced: written by an AI agent.*

- [ ] Base model numbers on a GPU: GSM8K accuracy and hit-limit rate at the post's settings, then MATH-500.
- [ ] Find the looping direction: compare MLP outputs of layers 15 and 19 on looping and finished completions
      (mean difference, then principal components), using GSM8K and MATH training problems only.
- [ ] Remove it from the MLP output columns, save the edited checkpoint, evaluate it the same way.
- [ ] Compare against the authors' released edited model (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`).
- [ ] Presence penalty in generation to match the post (a logits processor, or vLLM).
- [ ] Does the same direction exist in Qwen3.5-2B or 4B, and does removing it help there?
- [ ] Let a model do the analysis itself through a weights-only Python tool, as in the post, with a pass budget.
