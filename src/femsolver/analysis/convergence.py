"""Convergence tests for nonlinear iteration.

A :class:`ConvergenceTest` instance is consulted at the end of every
Newton iteration. It receives the latest residual ``R`` (the unbalanced
force vector) and displacement increment ``du``, and returns whether the
iteration is converged.

Two standard tests are provided here. Both close the loop by tolerance
on a vector norm of the relevant quantity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConvergenceTest(ABC):
    """Base class — subclasses implement :meth:`check`."""

    def __init__(self, tol: float = 1e-6, max_iter: int = 25, *, norm: str = "l2"):
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if norm not in ("l2", "linf"):
            raise ValueError(f"norm must be 'l2' or 'linf', got {norm!r}")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.norm = norm
        # populated each iteration for diagnostics
        self.last_value: float = float("nan")

    def _vec_norm(self, v: np.ndarray) -> float:
        if v.size == 0:
            return 0.0
        if self.norm == "l2":
            return float(np.linalg.norm(v))
        return float(np.max(np.abs(v)))

    @abstractmethod
    def check(self, R: np.ndarray, du: np.ndarray, iteration: int) -> bool:
        """Return True if converged."""


class NormDispIncr(ConvergenceTest):
    """Converged when ``||du|| < tol``.

    Cheap and intuitive but insensitive to residual force — best paired
    with a small ``tol`` and a sanity check on the residual after the
    fact. The very first iteration is never declared converged because
    ``du`` is zero before the first solve.
    """

    def check(self, R: np.ndarray, du: np.ndarray, iteration: int) -> bool:
        v = self._vec_norm(du)
        self.last_value = v
        if iteration == 0:
            return False
        return v < self.tol


class NormUnbalance(ConvergenceTest):
    """Converged when ``||R|| < tol``.

    The residual norm reflects the force imbalance directly. Recommended
    default for most analyses. ``tol`` is in force units (matching the
    units of the applied loads).
    """

    def check(self, R: np.ndarray, du: np.ndarray, iteration: int) -> bool:
        v = self._vec_norm(R)
        self.last_value = v
        return v < self.tol


class EnergyIncr(ConvergenceTest):
    """Converged when ``|du . R| < tol``.

    Energy increment = work done by the unbalanced force over the
    displacement step. Useful when force and displacement quantities have
    very different scales.
    """

    def check(self, R: np.ndarray, du: np.ndarray, iteration: int) -> bool:
        v = abs(float(du @ R))
        self.last_value = v
        if iteration == 0:
            return False
        return v < self.tol
