"""Structured logging setup."""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger("agentic_memory")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``agentic_memory`` namespace."""
    if name:
        return _LOGGER.getChild(name)
    return _LOGGER
