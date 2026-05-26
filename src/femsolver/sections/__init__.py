"""Cross-section response models for beam-column elements.

A *section* maps a vector of generalized strains (axial strain and
curvatures) to a vector of generalized stress resultants (axial force and
bending moments) plus a tangent stiffness matrix. This decouples the beam
element (which integrates section response along its length) from the
constitutive description of the cross-section, so the same element can host
elastic, fiber, or hinge-based sections without changes to its assembly
code.
"""
from femsolver.sections.base import SectionBase
from femsolver.sections.elastic import ElasticSection2D, ElasticSection3D
from femsolver.sections.fiber import Fiber, FiberSection2D, FiberSection3D
from femsolver.sections.shell import (
    ElasticShellSection,
    LayeredShellSection,
    ShellLayer,
    ShellSectionBase,
)
from femsolver.sections.ply_failure import (
    PlyStrength,
    evaluate_laminate,
    max_strain_index,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)

__all__ = [
    "SectionBase",
    "ElasticSection2D",
    "ElasticSection3D",
    "Fiber",
    "FiberSection2D",
    "FiberSection3D",
    "ShellSectionBase",
    "ElasticShellSection",
    "LayeredShellSection",
    "ShellLayer",
    "PlyStrength",
    "max_stress_index",
    "max_strain_index",
    "tsai_hill_index",
    "tsai_wu_index",
    "tsai_wu_strength_ratio",
    "evaluate_laminate",
]
