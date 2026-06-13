"""Exponential decay for memory importance.

Models memory forgetting: importance decays exponentially over time since
last access, with a configurable half-life. High-access memories decay slower.
"""

from __future__ import annotations

import math


class DecayFunction:
    """Exponential decay: ``importance * e^(-lambda * hours_since_access)``.

    Args:
        half_life_hours: Time for importance to drop to 50%. Default 168h (1 week).
    """

    def __init__(self, half_life_hours: float = 168.0) -> None:
        self.decay_rate = math.log(2) / half_life_hours

    def compute(self, base_importance: float, hours_since_access: float) -> float:
        """Return decayed importance score."""
        return base_importance * math.exp(-self.decay_rate * hours_since_access)

    def compute_with_access_boost(
        self,
        base_importance: float,
        hours_since_access: float,
        access_count: int,
        boost_factor: float = 0.1,
    ) -> float:
        """Return decayed importance, boosted by access frequency.

        Each access reduces the effective decay by ``boost_factor``.
        """
        import math as _math

        effective_rate = self.decay_rate / (1.0 + boost_factor * _math.log1p(access_count))
        return base_importance * math.exp(-effective_rate * hours_since_access)
