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
    AASHTO_MULTI_PRESENCE,
    DesignLane,
    MovingLoad,
    Vehicle2D,
    aashto_hl93_lane_load_kN_per_m,
    aashto_hl93_tandem,
    aashto_hl93_truck,
    aashto_lane_moment_simple_span,
    evaluate_response_for_position,
    generate_design_lanes,
    influence_line_simple_span_moment,
    influence_line_simple_span_shear,
    irc_class_70r_truck,
    irc_class_a,
    lane_load_surface_envelope,
    max_response_for_moving_load,
    max_truck_envelope_simple_span,
    moving_load_surface_envelope,
    multi_lane_envelope,
    multi_presence_factor,
    number_of_design_lanes,
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
from femsolver.bridges.tendon import (
    Tendon,
    tendon_secondary_forces,
    tendon_secondary_moment,
    tendon_secondary_shear,
)
from femsolver.bridges.cable import (
    CableElement2D,
    CableElement3D,
    CatenaryResult,
    catenary_max_tension,
    catenary_sag,
    ernst_equivalent_modulus,
)
from femsolver.bridges.form_finding import (
    FormFindingResult,
    force_density_form_find,
)
from femsolver.bridges.thermal_gradient import (
    SectionThermalActions,
    TemperatureGradient,
    aashto_gradient,
    apply_beam_thermal_actions,
    equivalent_thermal_actions,
    linear_gradient,
)
from femsolver.bridges.staged_construction import (
    ConstructionStage,
    ErectionStage,
    IncrementalStagedAnalysis,
    IncrementalStagedResult,
    StagedConstructionAnalysis,
    StagedConstructionResult,
    effective_modulus_EMM,
)
from femsolver.bridges.construction_stage import (
    StagedCamber,
    merge_stage_loads,
    staged_camber,
    tendon_stage_loads,
)
from femsolver.bridges.moving_force import (
    MovingForceAnalysis,
    VehicleAxles,
)
from femsolver.bridges.vbi import (
    SprungMassVehicle,
    VBIAnalysis,
)
from femsolver.bridges.cable_tuning import (
    Cable,
    CableTuningResult,
    apply_cable_tensions,
    unknown_load_factors,
)
from femsolver.bridges.moving_load import (
    BeamForce,
    DeckSurface,
    Displacement,
    InfluenceLine,
    InfluenceLineEngine,
    InfluenceSurface,
    Lane,
    Reaction,
    ResponseExtractor,
    aashto_hl93_envelope,
    lane_load_response,
    moving_load_envelope,
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
    # high-level tendon + equivalent-load apply (Phase B.4)
    "Tendon",
    "tendon_secondary_moment",
    "tendon_secondary_shear",
    "tendon_secondary_forces",
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
    # 3-D cable + form-finding (Phase B.2)
    "CableElement3D",
    "FormFindingResult",
    "force_density_form_find",
    # staged construction (Phase 45.3-45.4)
    "ConstructionStage",
    "StagedConstructionAnalysis",
    "StagedConstructionResult",
    "effective_modulus_EMM",
    # incremental staged erection w/ element birth + death (Phase B.3)
    "ErectionStage",
    "IncrementalStagedAnalysis",
    "IncrementalStagedResult",
    # general moving-load / influence-line engine (Phase B.1)
    "InfluenceLine",
    "InfluenceLineEngine",
    "Lane",
    "ResponseExtractor",
    "Displacement",
    "Reaction",
    "BeamForce",
    "moving_load_envelope",
    "lane_load_response",
    "aashto_hl93_envelope",
    # influence surfaces — 2-D moving load on decks / grillages (T1.1)
    "DeckSurface",
    "InfluenceSurface",
    "Vehicle2D",
    "moving_load_surface_envelope",
    # multi-lane placement + AASHTO multiple presence (T1.2)
    "DesignLane",
    "generate_design_lanes",
    "number_of_design_lanes",
    "multi_presence_factor",
    "AASHTO_MULTI_PRESENCE",
    "multi_lane_envelope",
    "lane_load_surface_envelope",
    # temperature gradient (T1.3)
    "TemperatureGradient",
    "linear_gradient",
    "aashto_gradient",
    "SectionThermalActions",
    "equivalent_thermal_actions",
    "apply_beam_thermal_actions",
    # construction-stage workflows: camber, tendon sequence, composite (T1.4)
    "StagedCamber",
    "staged_camber",
    "tendon_stage_loads",
    "merge_stage_loads",
    # moving-load time-history / VBI (T2.1)
    "VehicleAxles",
    "MovingForceAnalysis",
    "SprungMassVehicle",
    "VBIAnalysis",
    # cable-stayed initial-force optimisation — unknown load factor (T2.2)
    "Cable",
    "CableTuningResult",
    "unknown_load_factors",
    "apply_cable_tensions",
]
