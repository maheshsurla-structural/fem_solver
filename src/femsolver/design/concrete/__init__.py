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
from femsolver.design.concrete.biaxial import (
    BiaxialPMMPoint,
    BiaxialPMMSurface,
    StressBlockParams,
    aci_params,
    biaxial_pmm_point,
    biaxial_pmm_point_ec2,
    biaxial_pmm_point_is456,
    biaxial_pmm_surface,
    biaxial_pmm_surface_ec2,
    biaxial_pmm_surface_is456,
    ec2_params,
    is456_params,
)
from femsolver.design.concrete.moment_curvature import (
    MomentCurvaturePoint,
    MomentCurvatureResult,
    moment_curvature,
)
from femsolver.design.concrete.stress_field import (
    FiberState,
    SectionStressField,
    stress_at_point,
    stress_field,
    stress_field_to_svg,
)
from femsolver.design.concrete.cracked_section import (
    CrackedElasticConcrete,
    CrackedSectionProperties,
    branson_I_e,
    cracked_section_properties,
    ec2_mean_curvature,
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
    "BiaxialPMMPoint",
    "BiaxialPMMSurface",
    "StressBlockParams",
    "aci_params",
    "ec2_params",
    "is456_params",
    "biaxial_pmm_point",
    "biaxial_pmm_point_ec2",
    "biaxial_pmm_point_is456",
    "biaxial_pmm_surface",
    "biaxial_pmm_surface_ec2",
    "biaxial_pmm_surface_is456",
    "MomentCurvaturePoint",
    "MomentCurvatureResult",
    "moment_curvature",
    "FiberState",
    "SectionStressField",
    "stress_field",
    "stress_at_point",
    "stress_field_to_svg",
    "CrackedElasticConcrete",
    "CrackedSectionProperties",
    "cracked_section_properties",
    "branson_I_e",
    "ec2_mean_curvature",
    "BeamDesignDemand",
    "BeamDesignResult",
    "ColumnDesignDemand",
    "ColumnDesignResult",
    "RcMemberDesigner",
    "design_beam",
    "design_column",
]
