"""Mixed-NDF rigid offset coupling (beam-shell, beam-solid).

The :class:`~femsolver.constraints.rigid_link.RigidLink` primitive
couples two nodes that have the *same* number of DOFs (typical
beam-to-beam case). This module adds the **mixed-NDF** scenario
common in commercial modelling:

* **Beam-to-shell** -- a 3D beam carrying 6 DOFs (``u, v, w, theta_x,
  theta_y, theta_z``) connects to a shell mid-surface node carrying
  5 or 6 DOFs. If the offset is in the shell-normal direction the
  beam's rotation drives shell-node translation via the offset arm.
* **Beam-to-solid** -- a 3D beam node connects to a Hex8 / Hex20 /
  Tet4 corner node that has only 3 translational DOFs (``u, v, w``).
  The beam's translations *and* rotations together drive the solid
  node's translation::

      u_solid = u_beam + theta_beam x r,        r = x_solid - x_beam

  There is no rotational DOF on the solid side, so nothing to slave
  back; the beam's three rotational DOFs are master-only.

Both cases use the existing ``BasicConstraint`` row machinery, which
the standard constraint handler folds into the linear-system
transformation transparently. No solver changes required.

Use cases
---------

* Beam-on-shell deck (orthotropic steel bridge) where the longitudinal
  girder is modelled with a 3D beam below the deck mid-plane.
* Pile embedded in a soil block: pile cap node = beam, soil block
  nodes around the pile head = solid slaves arranged radially.
* Steel column embedded in a concrete foundation modelled with Hex8.
"""
from __future__ import annotations

import numpy as np

from femsolver.constraints.base import BasicConstraint, Constraint


class RigidOffset(Constraint):
    """Rigid kinematic link with arbitrary master / slave NDF combinations.

    The slave node's translational DOFs (the first :attr:`ndm` entries
    of its DOF vector) are constrained to follow the master via a
    rigid-body kinematic relation. Additional rotational DOFs on the
    slave (if any) can optionally be tied to the master's rotations.

    Parameters
    ----------
    master : int
        Tag of the retained master node (typically a beam node).
    slave : int
        Tag of the constrained slave node (typically a solid /
        shell node).
    couple_slave_rotations : bool, default False
        If ``True`` and the slave carries rotational DOFs, tie them
        1-to-1 to the master's rotations (full kinematic lock). If
        ``False`` (default), only translations are slaved -- the
        slave's rotations remain free.

    Notes
    -----
    The master node must have **at least the same number of
    translational DOFs as the slave**. The master must also have
    rotational DOFs to drive the offset arm.
    """

    def __init__(
        self,
        master: int,
        slave: int,
        *,
        couple_slave_rotations: bool = False,
    ):
        if master == slave:
            raise ValueError(
                "RigidOffset: master and slave nodes must differ"
            )
        self.master = int(master)
        self.slave = int(slave)
        self.couple_slave_rotations = bool(couple_slave_rotations)

    def basic_constraints(self, model) -> list[BasicConstraint]:
        if self.master not in model.nodes:
            raise ValueError(
                f"RigidOffset: unknown master node {self.master}"
            )
        if self.slave not in model.nodes:
            raise ValueError(
                f"RigidOffset: unknown slave node {self.slave}"
            )
        nm = model.node(self.master)
        ns = model.node(self.slave)
        ndm = int(model.ndm)
        if nm.ndf < ndm:
            raise ValueError(
                f"RigidOffset: master node {self.master} has ndf={nm.ndf} "
                f"< ndm={ndm}, cannot drive slave."
            )
        if ns.ndf < ndm:
            raise ValueError(
                f"RigidOffset: slave node {self.slave} has ndf={ns.ndf} "
                f"< ndm={ndm}; need at least the translational DOFs."
            )
        offset = ns.coords - nm.coords     # r = x_slave - x_master
        rows: list[BasicConstraint] = []

        if ndm == 2:
            dx, dy = float(offset[0]), float(offset[1])
            # 2D: master ndf must be >= 3 (u, v, theta_z) to drive offset
            if nm.ndf < 3:
                raise ValueError(
                    "RigidOffset (2D): master must carry theta_z (ndf>=3)."
                )
            # u_s = u_m - dy * theta_m
            rows.append(BasicConstraint(
                c_node=self.slave, c_dof=0,
                r_terms=[(self.master, 0, 1.0),
                          (self.master, 2, -dy)],
            ))
            # v_s = v_m + dx * theta_m
            rows.append(BasicConstraint(
                c_node=self.slave, c_dof=1,
                r_terms=[(self.master, 1, 1.0),
                          (self.master, 2, dx)],
            ))
            # Tie slave rotation to master if requested
            if self.couple_slave_rotations and ns.ndf >= 3:
                rows.append(BasicConstraint(
                    c_node=self.slave, c_dof=2,
                    r_terms=[(self.master, 2, 1.0)],
                ))
            return rows

        # 3D
        if nm.ndf < 6:
            raise ValueError(
                "RigidOffset (3D): master must carry full rotations "
                "(ndf>=6)."
            )
        dx, dy, dz = float(offset[0]), float(offset[1]), float(offset[2])
        # u_s = u_m + dz*theta_y_m - dy*theta_z_m
        rows.append(BasicConstraint(
            c_node=self.slave, c_dof=0,
            r_terms=[(self.master, 0, 1.0),
                      (self.master, 4, dz),
                      (self.master, 5, -dy)],
        ))
        # v_s = v_m - dz*theta_x_m + dx*theta_z_m
        rows.append(BasicConstraint(
            c_node=self.slave, c_dof=1,
            r_terms=[(self.master, 1, 1.0),
                      (self.master, 3, -dz),
                      (self.master, 5, dx)],
        ))
        # w_s = w_m + dy*theta_x_m - dx*theta_y_m
        rows.append(BasicConstraint(
            c_node=self.slave, c_dof=2,
            r_terms=[(self.master, 2, 1.0),
                      (self.master, 3, dy),
                      (self.master, 4, -dx)],
        ))
        if self.couple_slave_rotations and ns.ndf >= 6:
            for d in (3, 4, 5):
                rows.append(BasicConstraint(
                    c_node=self.slave, c_dof=d,
                    r_terms=[(self.master, d, 1.0)],
                ))
        return rows

    def __repr__(self) -> str:
        return (
            f"RigidOffset(master={self.master}, slave={self.slave}, "
            f"couple_slave_rotations={self.couple_slave_rotations})"
        )


# ============================================================ convenience

def beam_solid_coupling(
    model, beam_node: int, solid_nodes,
) -> list[RigidOffset]:
    """Couple a 3D beam-tip node rigidly to a set of solid (Hex/Tet)
    corner nodes. Returns the list of installed :class:`RigidOffset`
    constraints (also added to the model).

    Typical usage: pile head attached to surrounding soil-block nodes,
    or a column embedded in a concrete pier modelled with Hex8.
    """
    if not hasattr(solid_nodes, "__iter__"):
        solid_nodes = [solid_nodes]
    out: list[RigidOffset] = []
    for s in solid_nodes:
        if int(s) == int(beam_node):
            continue
        c = RigidOffset(master=int(beam_node), slave=int(s))
        model.add_mp_constraint(c)
        out.append(c)
    return out


def beam_shell_offset_coupling(
    model, beam_node: int, shell_node: int,
    couple_drilling: bool = True,
) -> RigidOffset:
    """Couple a beam node to an offset shell node. If the shell node
    carries a drilling rotation (ndf=3 in 2D, ndf=6 in 3D), tying
    rotations is recommended (``couple_drilling=True``)."""
    c = RigidOffset(
        master=int(beam_node), slave=int(shell_node),
        couple_slave_rotations=couple_drilling,
    )
    model.add_mp_constraint(c)
    return c
