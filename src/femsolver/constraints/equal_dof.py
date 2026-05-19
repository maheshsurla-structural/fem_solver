"""EqualDOF — tie selected DOFs of two nodes together."""
from __future__ import annotations

from typing import Iterable

from femsolver.constraints.base import BasicConstraint, Constraint


class EqualDOF(Constraint):
    """Force ``u[constrained, dof] == u[retained, dof]`` for each listed DOF.

    Parameters
    ----------
    retained : int
        Tag of the retained (master) node.
    constrained : int
        Tag of the constrained (slave) node.
    dofs : iterable of int
        Zero-based DOF indices to tie. Both nodes must have at least
        ``max(dofs) + 1`` DOFs.
    """

    def __init__(self, retained: int, constrained: int, dofs: Iterable[int]):
        if retained == constrained:
            raise ValueError("EqualDOF: retained and constrained nodes must differ")
        self.retained = int(retained)
        self.constrained = int(constrained)
        self.dofs = [int(d) for d in dofs]
        if not self.dofs:
            raise ValueError("EqualDOF: dofs list is empty")
        if len(set(self.dofs)) != len(self.dofs):
            raise ValueError("EqualDOF: duplicate DOF index in dofs")

    def basic_constraints(self, model) -> list[BasicConstraint]:
        # validate node tags and DOF range against the model
        for tag in (self.retained, self.constrained):
            if tag not in model.nodes:
                raise ValueError(f"EqualDOF references unknown node {tag}")
        ndf_r = model.node(self.retained).ndf
        ndf_c = model.node(self.constrained).ndf
        for d in self.dofs:
            if d < 0 or d >= min(ndf_r, ndf_c):
                raise ValueError(
                    f"EqualDOF: dof index {d} out of range for nodes "
                    f"{self.retained}(ndf={ndf_r}) and {self.constrained}(ndf={ndf_c})"
                )
        return [
            BasicConstraint(
                c_node=self.constrained,
                c_dof=d,
                r_terms=[(self.retained, d, 1.0)],
                g=0.0,
            )
            for d in self.dofs
        ]

    def __repr__(self) -> str:
        return (
            f"EqualDOF(retained={self.retained}, constrained={self.constrained}, "
            f"dofs={self.dofs})"
        )
