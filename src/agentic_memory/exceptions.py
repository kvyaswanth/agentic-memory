"""Custom exception hierarchy for Agentic Memory."""


class AgenticMemoryError(Exception):
    """Base exception for all agentic-memory errors."""


# -- Storage errors --------------------------------------------------------


class StorageError(AgenticMemoryError):
    """General storage backend failure."""


class NodeNotFoundError(StorageError):
    """Requested memory node does not exist."""


class EdgeNotFoundError(StorageError):
    """Requested memory edge does not exist."""


class BlockNotFoundError(StorageError):
    """Requested core memory block does not exist."""


class PassageNotFoundError(StorageError):
    """Requested passage does not exist."""


# -- Memory errors ---------------------------------------------------------


class MemoryLimitExceededError(AgenticMemoryError):
    """Operation would exceed configured memory size limits."""


# -- Embedding errors ------------------------------------------------------


class EmbeddingError(AgenticMemoryError):
    """Failure during embedding generation."""


# -- Sleep-time errors -----------------------------------------------------


class ConsolidationError(AgenticMemoryError):
    """Failure during sleep-time consolidation."""
