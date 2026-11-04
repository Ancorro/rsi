"""Measures of looping in a completion."""

from __future__ import annotations


def repeated_ngram_fraction(tokens: list[int], n: int = 20) -> float:
    """Share of length-n token windows that already appeared earlier in the sequence."""
    if len(tokens) < 2 * n:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    total = len(tokens) - n + 1
    for i in range(total):
        w = tuple(tokens[i:i + n])
        repeats += w in seen
        seen.add(w)
    return repeats / total


def hit_limit(num_new_tokens: int, max_new_tokens: int, finished: bool) -> bool:
    """True when generation stopped at the token limit instead of an end-of-sequence token."""
    return not finished and num_new_tokens >= max_new_tokens
