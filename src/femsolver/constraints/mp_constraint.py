"""General linear MP constraint."""
from __future__ import annotations

from typing import Iterable

from femsolver.constraints.base import BasicConstraint, Constraint


class MPConstraint(Constraint):
    """General linear constraint of the form

        u[c_node, c_dof] = sum_j (coeff_j * u[r_node_j, r_dof_j]) + g

    Use this for any tie that is not covered by :class:`EqualDOF`,
    :class:`RigidLink`, or :class:`RigidDiaphragm` — for example, an
    inclined roller, a prescribed nonzero displacement, or an averaging
    constraint over multiple retained DOFs.

    Parameters
    ----------
    constrained : (int, int)
        ``(node_tag, dof_index)`` of the constrained DOF.
    retained : iterable of (int, int, float)
        Each item is ``(node_tag, dof_index, coefficient)``.
    g : float, optional
        Constant term (defaults to 0). Set non-zero to impose a prescribed
        nonzero displacement.
    """

    def __init__(
        self,
        constrained: tuple[int, int],
        retained: Iterable[tuple[int, int, float]],
        g: float = 0.0,
    ):
        c_node, c_dof = constrained
        self.c_node = int(c_node)
        self.c_dof = int(c_dof)
        self.r_terms = [(int(n), int(d), float(c)) for (n, d, c) in retained]
        self.g = float(g)
        # An empty retained list is allowed (purely prescribed displacement).
        for n, d, _ in self.r_terms:
            if n == self.c_node and d == self.c_dof:
                raise ValueError(
                    "MPConstraint: a retained term refers to the constrained DOF itself"
                )

    def basic_constraints(self, model) -> list[BasicConstraint]:
        for tag in {self.c_node, *(t[0] for t in self.r_terms)}:
            if tag not in model.nodes:
                raise ValueError(f"MPConstraint references unknown node {tag}")
        if self.c_dof < 0 or self.c_dof >= model.node(self.c_node).ndf:
            raise ValueError(
                f"MPConstraint: c_dof {self.c_dof} out of range for node "
                f"{self.c_node} (ndf={model.node(self.c_node).ndf})"
            )
        for n, d, _ in self.r_terms:
            if d < 0 or d >= model.node(n).ndf:
                raise ValueError(
                    f"MPConstraint: retained DOF {d} out of range for node "
                    f"{n} (ndf={model.node(n).ndf})"
                )
        return [
            BasicConstraint(
                c_node=self.c_node,
                c_dof=self.c_dof,
                r_terms=list(self.r_terms),
                g=self.g,
            )
        ]

    def __repr__(self) -> str:
        return (
            f"MPConstraint(constrained=({self.c_node},{self.c_dof}), "
            f"retained={self.r_terms}, g={self.g})"
        )
