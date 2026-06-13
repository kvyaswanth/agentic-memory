"""Tests for exponential decay."""

from __future__ import annotations

from agentic_memory.graph.decay import DecayFunction


def test_decay_decreases_over_time():
    decay = DecayFunction(half_life_hours=168.0)
    score_fresh = decay.compute(1.0, 1.0)
    score_old = decay.compute(1.0, 168.0)
    assert score_fresh > score_old


def test_half_life():
    decay = DecayFunction(half_life_hours=100.0)
    score = decay.compute(1.0, 100.0)
    assert abs(score - 0.5) < 0.01


def test_access_boost():
    decay = DecayFunction(half_life_hours=100.0)
    no_access = decay.compute_with_access_boost(1.0, 100.0, 0)
    with_access = decay.compute_with_access_boost(1.0, 100.0, 100)
    assert with_access > no_access
