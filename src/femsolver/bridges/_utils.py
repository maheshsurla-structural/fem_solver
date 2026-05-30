"""Internal helpers shared across the bridges submodules."""
from __future__ import annotations


def require_positive(**kwargs) -> None:
    """Raise ``ValueError`` if any keyword argument is not strictly positive.

    Used to consolidate the repetitive ``if x <= 0: raise ValueError(...)``
    blocks at the top of public bridge functions.
    """
    for name, val in kwargs.items():
        if val is None or val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")


def require_non_negative(**kwargs) -> None:
    """Raise ``ValueError`` if any keyword argument is negative."""
    for name, val in kwargs.items():
        if val is None or val < 0.0:
            raise ValueError(f"{name} must be >= 0, got {val}")
