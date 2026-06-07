"""Section *response* -- the low-level, element-facing section layer.

These are the objects a **beam/column element** actually consumes during a
solve: each maps a vector of generalized strains (axial strain, curvatures)
to generalized stress resultants (axial force, moments) plus a tangent
stiffness, via ``get_response(e) -> (s, ks)``.

This is the engine room behind the user-facing
:class:`femsolver.sections.Section` (which you *define*); a ``Section``
produces these on demand through ``elastic_section_2d/3d()`` and
``fiber_section_2d/3d()``.

Contents
--------
* :mod:`base`       -- ``SectionBase`` (the 1-D section contract).
* :mod:`elastic`    -- ``ElasticSection2D/3D`` (constant EA/EI).
* :mod:`fiber`      -- ``Fiber`` + ``FiberSection2D/3D`` (the nonlinear
  workhorse: per-fiber uniaxial integration).
* :mod:`wall`       -- fiber wall sections (2-D + 3-D T/L/I/U shapes).
* :mod:`wall_shear` -- wall shear area / cracked-stiffness factors.
* :mod:`hinges`     -- lumped plastic-hinge / spring response.
"""
from femsolver.sections.response.base import SectionBase
from femsolver.sections.response.elastic import (
    ElasticSection2D,
    ElasticSection3D,
)
from femsolver.sections.response.fiber import (
    Fiber,
    FiberSection2D,
    FiberSection3D,
)
from femsolver.sections.response.wall import (
    WallRegion,
    i_wall_section_3d,
    l_wall_section_3d,
    t_wall_section_3d,
    u_wall_section_3d,
    wall_section_2d,
)
from femsolver.sections.response.wall_shear import (
    CrackedSectionFactors,
    aci318_cracked_factors,
    asce41_wall_factors,
    wall_base_shear_spring_stiffness,
    wall_lateral_stiffness,
    wall_shear_area,
)

__all__ = [
    "SectionBase",
    "ElasticSection2D",
    "ElasticSection3D",
    "Fiber",
    "FiberSection2D",
    "FiberSection3D",
    "WallRegion",
    "wall_section_2d",
    "t_wall_section_3d",
    "l_wall_section_3d",
    "i_wall_section_3d",
    "u_wall_section_3d",
    "CrackedSectionFactors",
    "aci318_cracked_factors",
    "asce41_wall_factors",
    "wall_base_shear_spring_stiffness",
    "wall_lateral_stiffness",
    "wall_shear_area",
]
