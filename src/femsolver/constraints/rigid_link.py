"""Rigid link with offset — couples translations and rotations of two nodes
so the constrained node moves as if rigidly connected to the retained node."""
from __future__ import annotations

from femsolver.constraints.base import BasicConstraint, Constraint


class RigidLink(Constraint):
    """Rigid beam-type link between two nodes.

    The constrained node is forced to move as a rigid body anchored at the
    retained node. With offset vector ``d = x_c - x_r``:

    - **2D frame (ndf=3, DOFs = u, v, theta_z):**

      .. math::

          u_c = u_r - d_y \\cdot \\theta_{z,r} \\\\
          v_c = v_r + d_x \\cdot \\theta_{z,r} \\\\
          \\theta_{z,c} = \\theta_{z,r}

    - **3D frame (ndf=6, DOFs = u, v, w, theta_x, theta_y, theta_z):**

      Translational lever-arm coupling :math:`u_c = u_r + r \\times \\theta_r`
      plus identical rotations :math:`\\theta_c = \\theta_r`.

    Parameters
    ----------
    retained : int
        Tag of the retained (master) node.
    constrained : int
        Tag of the constrained (slave) node.
    kind : {"beam", "bar"}, optional
        ``"beam"`` (default) couples both translations and rotations through
        the offset; ``"bar"`` couples only translations (identical
        translation, no rotational tie). Both nodes must have rotational
        DOFs for ``"beam"``.
    """

    def __init__(self, retained: int, constrained: int, kind: str = "beam"):
        if retained == constrained:
            raise ValueError("RigidLink: retained and constrained nodes must differ")
        if kind not in ("beam", "bar"):
            raise ValueError(f"RigidLink: kind must be 'beam' or 'bar', got {kind!r}")
        self.retained = int(retained)
        self.constrained = int(constrained)
        self.kind = kind

    def basic_constraints(self, model) -> list[BasicConstraint]:
        for tag in (self.retained, self.constrained):
            if tag not in model.nodes:
                raise ValueError(f"RigidLink references unknown node {tag}")
        nr = model.node(self.retained)
        nc = model.node(self.constrained)
        if nr.ndf != nc.ndf:
            raise ValueError(
                f"RigidLink: retained and constrained nodes must have the same ndf "
                f"({nr.ndf} vs {nc.ndf})"
            )
        ndm = model.ndm
        ndf = nr.ndf
        d = nc.coords - nr.coords  # offset vector

        if self.kind == "bar":
            # translations only — identical, no offset effect
            return [
                BasicConstraint(
                    c_node=self.constrained,
                    c_dof=i,
                    r_terms=[(self.retained, i, 1.0)],
                )
                for i in range(ndm)
            ]

        # beam type: both translation and rotation tie
        if ndm == 2:
            if ndf < 3:
                raise ValueError(
                    "RigidLink (beam, 2D) requires nodes with ndf>=3 "
                    "(u, v, theta_z)"
                )
            dx, dy = float(d[0]), float(d[1])
            # u_c = u_r - dy * theta_r
            # v_c = v_r + dx * theta_r
            # theta_c = theta_r
            return [
                BasicConstraint(
                    c_node=self.constrained, c_dof=0,
                    r_terms=[(self.retained, 0, 1.0), (self.retained, 2, -dy)],
                ),
                BasicConstraint(
                    c_node=self.constrained, c_dof=1,
                    r_terms=[(self.retained, 1, 1.0), (self.retained, 2, dx)],
                ),
                BasicConstraint(
                    c_node=self.constrained, c_dof=2,
                    r_terms=[(self.retained, 2, 1.0)],
                ),
            ]

        # 3D
        if ndf < 6:
            raise ValueError(
                "RigidLink (beam, 3D) requires nodes with ndf>=6 "
                "(u, v, w, theta_x, theta_y, theta_z)"
            )
        dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
        # u_c     = u_r + dz*theta_y_r - dy*theta_z_r
        # v_c     = v_r - dz*theta_x_r + dx*theta_z_r
        # w_c     = w_r + dy*theta_x_r - dx*theta_y_r
        # theta_*_c = theta_*_r
        return [
            BasicConstraint(
                c_node=self.constrained, c_dof=0,
                r_terms=[
                    (self.retained, 0, 1.0),
                    (self.retained, 4, dz),
                    (self.retained, 5, -dy),
                ],
            ),
            BasicConstraint(
                c_node=self.constrained, c_dof=1,
                r_terms=[
                    (self.retained, 1, 1.0),
                    (self.retained, 3, -dz),
                    (self.retained, 5, dx),
                ],
            ),
            BasicConstraint(
                c_node=self.constrained, c_dof=2,
                r_terms=[
                    (self.retained, 2, 1.0),
                    (self.retained, 3, dy),
                    (self.retained, 4, -dx),
                ],
            ),
            BasicConstraint(
                c_node=self.constrained, c_dof=3,
                r_terms=[(self.retained, 3, 1.0)],
            ),
            BasicConstraint(
                c_node=self.constrained, c_dof=4,
                r_terms=[(self.retained, 4, 1.0)],
            ),
            BasicConstraint(
                c_node=self.constrained, c_dof=5,
                r_terms=[(self.retained, 5, 1.0)],
            ),
        ]

    def __repr__(self) -> str:
        return (
            f"RigidLink(retained={self.retained}, constrained={self.constrained}, "
            f"kind={self.kind!r})"
        )
