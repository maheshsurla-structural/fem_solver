"""Rigid diaphragm — constrains in-plane motion of a set of slave nodes to
a master node, leaving out-of-plane DOFs free."""
from __future__ import annotations

from typing import Iterable

from femsolver.constraints.base import BasicConstraint, Constraint


class RigidDiaphragm(Constraint):
    """Rigid floor diaphragm in 3D.

    All slave nodes share the master node's two in-plane translations and
    its rotation about the diaphragm normal. Out-of-plane translation and
    the two in-plane rotations remain independent (uncoupled).

    For ``perp_dir = 2`` (XY plane, normal = Z):

    .. math::

        u_s = u_m - (y_s - y_m)\\,\\theta_{z,m} \\\\
        v_s = v_m + (x_s - x_m)\\,\\theta_{z,m} \\\\
        \\theta_{z,s} = \\theta_{z,m}

    Parameters
    ----------
    master : int
        Tag of the master node (typically the geometric centre / centre of
        mass of the diaphragm).
    slaves : iterable of int
        Tags of the slave nodes.
    perp_dir : {0, 1, 2}, optional
        Index of the global axis perpendicular to the diaphragm plane —
        0=X (YZ-plane), 1=Y (XZ-plane), 2=Z (XY-plane, default).

    Notes
    -----
    Only valid for 3D models with ndf>=6 (3 translations + 3 rotations).
    """

    def __init__(self, master: int, slaves: Iterable[int], perp_dir: int = 2):
        if perp_dir not in (0, 1, 2):
            raise ValueError(f"RigidDiaphragm: perp_dir must be 0, 1, or 2, got {perp_dir}")
        self.master = int(master)
        self.slaves = [int(s) for s in slaves]
        if not self.slaves:
            raise ValueError("RigidDiaphragm: slaves list is empty")
        if self.master in self.slaves:
            raise ValueError("RigidDiaphragm: master cannot also be a slave")
        if len(set(self.slaves)) != len(self.slaves):
            raise ValueError("RigidDiaphragm: duplicate slave tag")
        self.perp_dir = int(perp_dir)

    def basic_constraints(self, model) -> list[BasicConstraint]:
        if model.ndm != 3:
            raise ValueError("RigidDiaphragm requires a 3D model (ndm=3)")
        for tag in (self.master, *self.slaves):
            if tag not in model.nodes:
                raise ValueError(f"RigidDiaphragm references unknown node {tag}")
            if model.node(tag).ndf < 6:
                raise ValueError(
                    f"RigidDiaphragm requires ndf>=6 at node {tag} "
                    f"(got {model.node(tag).ndf})"
                )

        # In-plane translation indices and the rotational DOF about perp axis
        # convention: dof 0=u_x, 1=u_y, 2=u_z, 3=theta_x, 4=theta_y, 5=theta_z
        i, j = [k for k in range(3) if k != self.perp_dir]  # in-plane axes
        rot_perp = 3 + self.perp_dir  # rotation about the perpendicular axis

        master_coord = model.node(self.master).coords
        basics: list[BasicConstraint] = []
        for s in self.slaves:
            slave_coord = model.node(s).coords
            di = float(slave_coord[i] - master_coord[i])
            dj = float(slave_coord[j] - master_coord[j])
            # u_i_s = u_i_m - dj * theta_perp_m
            basics.append(
                BasicConstraint(
                    c_node=s, c_dof=i,
                    r_terms=[(self.master, i, 1.0), (self.master, rot_perp, -dj)],
                )
            )
            # u_j_s = u_j_m + di * theta_perp_m
            basics.append(
                BasicConstraint(
                    c_node=s, c_dof=j,
                    r_terms=[(self.master, j, 1.0), (self.master, rot_perp, di)],
                )
            )
            # theta_perp_s = theta_perp_m
            basics.append(
                BasicConstraint(
                    c_node=s, c_dof=rot_perp,
                    r_terms=[(self.master, rot_perp, 1.0)],
                )
            )
        return basics

    def __repr__(self) -> str:
        return (
            f"RigidDiaphragm(master={self.master}, slaves={self.slaves}, "
            f"perp_dir={self.perp_dir})"
        )
