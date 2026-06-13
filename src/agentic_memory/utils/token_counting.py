"""Rough token-count estimation for context window budgeting."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the heuristic that 1 token is roughly 4 characters for English text.
    Good enough for budgeting; not a replacement for a real tokenizer.
    """
    return max(1, len(text) // 4)
