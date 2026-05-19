"""Constraint abstract base + the elementary constraint record."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BasicConstraint:
    """One scalar linear constraint equation.

    Form: ``u[c_node, c_dof] = sum_j(coeffs[j] * u[r_node[j], r_dof[j]]) + g``.

    Stored in node/dof terms (not equation numbers) — the handler resolves
    equation numbers after the model is numbered.
    """

    c_node: int
    c_dof: int
    r_terms: list[tuple[int, int, float]] = field(default_factory=list)
    g: float = 0.0


class Constraint(ABC):
    """Base class for multi-point constraints.

    Each concrete subclass decomposes itself into one or more
    :class:`BasicConstraint` rows via :meth:`basic_constraints`. The handler
    is responsible for assembling those rows into a transformation matrix or
    penalty term.
    """

    @abstractmethod
    def basic_constraints(self, model) -> list[BasicConstraint]:
        """Decompose this constraint into elementary scalar relations."""
