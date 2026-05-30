"""Bridge engineering: influence lines, PT tendons, creep/shrinkage,
composite sections.

Submodules:

* :mod:`influence`        -- influence lines + moving-load envelopes
  (AASHTO HL-93, IRC vehicles).
* :mod:`pt_tendon`        -- post-tensioning tendon profile, friction
  losses, anchorage slip, equivalent prestress load.
* :mod:`creep_shrinkage`  -- CEB-FIP 2010 creep + shrinkage,
  AASHTO LRFD prestress losses, steel relaxation.
* :mod:`composite_section`-- transformed composite girder + deck
  properties + fiber stresses.
"""
from femsolver.bridges.composite_section import (
    CompositeFiberStress,
    CompositeSectionProps,
    composite_fiber_stresses,
    composite_girder_deck,
)
from femsolver.bridges.creep_shrinkage import (
    CebFipCreepResult,
    CebFipShrinkageResult,
    PrestressLossBreakdown,
    cebfip_creep_coefficient,
    cebfip_shrinkage,
    prestress_long_term_loss,
    steel_relaxation_loss_ratio,
)
from femsolver.bridges.influence import (
    MovingLoad,
    aashto_hl93_lane_load_kN_per_m,
    aashto_hl93_tandem,
    aashto_hl93_truck,
    aashto_lane_moment_simple_span,
    evaluate_response_for_position,
    influence_line_simple_span_moment,
    influence_line_simple_span_shear,
    irc_class_70r_truck,
    irc_class_a,
    max_response_for_moving_load,
    max_truck_envelope_simple_span,
)
from femsolver.bridges.pt_tendon import (
    AnchorageSlipResult,
    FrictionLossResult,
    TendonProfile,
    anchorage_slip_loss,
    equivalent_uniform_load_parabolic,
    friction_loss,
    parabolic_drape_profile,
)
from femsolver.bridges.cable import (
    CableElement2D,
    CatenaryResult,
    catenary_max_tension,
    catenary_sag,
    ernst_equivalent_modulus,
)
from femsolver.bridges.staged_construction import (
    ConstructionStage,
    StagedConstructionAnalysis,
    StagedConstructionResult,
    effective_modulus_EMM,
)


__all__ = [
    # influence
    "MovingLoad",
    "influence_line_simple_span_moment",
    "influence_line_simple_span_shear",
    "evaluate_response_for_position",
    "max_response_for_moving_load",
    "aashto_hl93_truck",
    "aashto_hl93_tandem",
    "aashto_hl93_lane_load_kN_per_m",
    "aashto_lane_moment_simple_span",
    "max_truck_envelope_simple_span",
    "irc_class_a",
    "irc_class_70r_truck",
    # pt_tendon
    "TendonProfile",
    "parabolic_drape_profile",
    "FrictionLossResult",
    "friction_loss",
    "AnchorageSlipResult",
    "anchorage_slip_loss",
    "equivalent_uniform_load_parabolic",
    # creep_shrinkage
    "CebFipCreepResult",
    "cebfip_creep_coefficient",
    "CebFipShrinkageResult",
    "cebfip_shrinkage",
    "steel_relaxation_loss_ratio",
    "PrestressLossBreakdown",
    "prestress_long_term_loss",
    # composite
    "CompositeSectionProps",
    "composite_girder_deck",
    "CompositeFiberStress",
    "composite_fiber_stresses",
    # cable (Phase 45.1-45.2)
    "CableElement2D",
    "ernst_equivalent_modulus",
    "CatenaryResult",
    "catenary_sag",
    "catenary_max_tension",
    # staged construction (Phase 45.3-45.4)
    "ConstructionStage",
    "StagedConstructionAnalysis",
    "StagedConstructionResult",
    "effective_modulus_EMM",
]
