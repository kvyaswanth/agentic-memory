"""Deterministic prefixed ID generation."""

from __future__ import annotations

import uuid


def make_id(prefix: str) -> str:
    """Generate a prefixed unique identifier (e.g. ``node-a1b2c3...``)."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"
