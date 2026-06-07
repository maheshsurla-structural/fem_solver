"""Shell / plate section properties -- the input for **surface (2-D)
elements** (``ShellMITC4``, ``ShellTri3``, ...), the 2-D counterpart of
:mod:`femsolver.sections` (which serves beam/column line elements).

Where a beam element consumes a cross-*section* mapping (strain, curvature)
-> (force, moment), a shell element consumes a through-thickness *section*
(a thickness + layer stack) mapping membrane strains + curvatures +
transverse shear to stress resultants per unit width. In MIDAS-style UX
this is the "Thickness" input as opposed to the "Section" input.

Contents
--------
* :mod:`base`        -- ``ShellSectionBase`` (the 2-D contract).
* :mod:`layered`     -- ``ElasticShellSection`` (single isotropic layer)
  and ``LayeredShellSection`` / ``ShellLayer`` (laminate stack, CLT, RC
  slab, sandwich).
* :mod:`ply_failure` -- composite laminate failure criteria
  (max-stress/strain, Tsai-Hill, Tsai-Wu).
* :mod:`clt`         -- cross-laminated-timber panel section.
"""
from femsolver.shell_sections.base import ShellSectionBase
from femsolver.shell_sections.layered import (
    ElasticShellSection,
    LayeredShellSection,
    ShellLayer,
)
from femsolver.shell_sections.ply_failure import (
    PlyStrength,
    evaluate_laminate,
    max_strain_index,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)
from femsolver.shell_sections.clt import CLTLayer, CLTSection

__all__ = [
    "ShellSectionBase",
    "ElasticShellSection",
    "LayeredShellSection",
    "ShellLayer",
    "PlyStrength",
    "evaluate_laminate",
    "max_strain_index",
    "max_stress_index",
    "tsai_hill_index",
    "tsai_wu_index",
    "tsai_wu_strength_ratio",
    "CLTLayer",
    "CLTSection",
]
