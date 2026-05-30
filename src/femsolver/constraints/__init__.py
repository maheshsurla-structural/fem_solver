"""Multi-point constraints (MP constraints).

A multi-point constraint is a linear relation between DOFs of the form

    u_c = sum_j (c_j * u_r_j) + g

where `u_c` is a *constrained* (slave) DOF, `u_r_j` are *retained* (master)
DOFs, `c_j` are coefficients, and `g` is an optional constant (used for
prescribed nonzero displacements).

Concrete subclasses cover the common cases (equalDOF, rigid link with offset,
rigid diaphragm) plus a fully general form.
"""
from femsolver.constraints.base import BasicConstraint, Constraint
from femsolver.constraints.equal_dof import EqualDOF
from femsolver.constraints.rigid_link import RigidLink
from femsolver.constraints.rigid_offset import (
    RigidOffset,
    beam_shell_offset_coupling,
    beam_solid_coupling,
)
from femsolver.constraints.rigid_diaphragm import RigidDiaphragm
from femsolver.constraints.mp_constraint import MPConstraint

__all__ = [
    "BasicConstraint",
    "Constraint",
    "EqualDOF",
    "RigidLink",
    "RigidOffset",
    "RigidDiaphragm",
    "MPConstraint",
    "beam_shell_offset_coupling",
    "beam_solid_coupling",
]
