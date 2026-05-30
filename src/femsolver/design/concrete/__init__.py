"""Concrete design per ACI 318-19.

Phase 29 modules:

* :mod:`section` -- ConcreteSection / RebarLayout / ConcreteMaterial
  dataclasses + ACI material constants (β_1, φ factors, ε_cu).

Future Phase 29 modules (planned):

* ``flexure`` -- rectangular beam M_n per Ch. 22.2 (Phase 29.2)
* ``shear`` -- V_n per Ch. 22.5 (Phase 29.3)
* ``column`` -- P-M interaction per Ch. 22.4 (Phase 29.4)
* ``designer`` -- iterative member design driver (Phase 29.5)
"""
from femsolver.design.concrete.section import (
    EPSILON_CU,
    E_STEEL,
    ConcreteMaterial,
    ConcreteSection,
    PhiFactors,
    RebarLayout,
    beta_1_aci,
    phi_for_strain,
    rebar_area,
    rebar_diameter,
    standard_rebar_designations,
)
from femsolver.design.concrete.flexure import (
    FlexuralCheck,
    beam_flexural_strength,
)
from femsolver.design.concrete.shear import (
    PHI_SHEAR,
    ShearCheck,
    ShearDesign,
    beam_shear_strength,
    design_stirrup_spacing,
)
from femsolver.design.concrete.column import (
    InteractionPoint,
    InteractionSurface,
    column_interaction_point,
    column_interaction_surface,
)
from femsolver.design.concrete.designer import (
    BeamDesignDemand,
    BeamDesignResult,
    ColumnDesignDemand,
    ColumnDesignResult,
    RcMemberDesigner,
    design_beam,
    design_column,
)

__all__ = [
    "ConcreteMaterial",
    "ConcreteSection",
    "RebarLayout",
    "PhiFactors",
    "EPSILON_CU",
    "E_STEEL",
    "beta_1_aci",
    "phi_for_strain",
    "rebar_area",
    "rebar_diameter",
    "standard_rebar_designations",
    "FlexuralCheck",
    "beam_flexural_strength",
    "PHI_SHEAR",
    "ShearCheck",
    "ShearDesign",
    "beam_shear_strength",
    "design_stirrup_spacing",
    "InteractionPoint",
    "InteractionSurface",
    "column_interaction_point",
    "column_interaction_surface",
    "BeamDesignDemand",
    "BeamDesignResult",
    "ColumnDesignDemand",
    "ColumnDesignResult",
    "RcMemberDesigner",
    "design_beam",
    "design_column",
]
