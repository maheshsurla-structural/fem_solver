"""femsolver — a Python finite element solver for structural analysis."""

from femsolver import mesh, postproc                 # Phase 47 (Theme L)
from femsolver.core.model import Model
from femsolver.core.node import Node
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.materials.j2_plasticity import J2Plasticity3D
from femsolver.materials.orthotropic import OrthotropicLamina
from femsolver.materials.drucker_prager import DruckerPrager3D
from femsolver.materials.mohr_coulomb import MohrCoulomb3D
from femsolver.materials.cam_clay import ModifiedCamClay3D
from femsolver.materials.concrete_damage import ConcreteDamage3D
from femsolver.materials.concrete_damage_plasticity import (
    ConcreteDamagePlasticity3D,
)
from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.elements.truss_corot import Truss2DCorotational
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.beam_curved import CurvedBeam2D
from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.elements.beam_corot_3d import BeamColumn3DCorotational
from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.elements.beam_hinged import HingedBeamColumn2D
from femsolver.elements.plane import Quad4, Quad8
from femsolver.elements.membrane_drilling import MembraneQ4Drilling
from femsolver.elements.shell import ShellMITC4
from femsolver.elements.shell_mitc9 import ShellMITC9
from femsolver.elements.shell_dkmq4 import ShellDKMQ4
from femsolver.elements.shell_tri import ShellTri3
from femsolver.elements.shell_dkt3 import ShellDKT3
from femsolver.elements.shell_mesh import (
    cylindrical_shell_mesh,
    spherical_cap_mesh,
)
from femsolver.elements.solid import Hex8, Hex20, Tet4
from femsolver.elements.thermal import (
    ConvectionEdge2D,
    ThermalHex8,
    ThermalQuad4,
)
from femsolver.materials.thermal import ThermalMaterial
from femsolver.materials.hyperelastic import (
    MooneyRivlin3D,
    NeoHookean3D,
)
from femsolver.materials.finite_j2 import FiniteJ2Plasticity3D
from femsolver.elements.hex8_TL import Hex8TL
from femsolver.elements.contact import ContactNodeToPlane3D
from femsolver.thermal.heat_conduction import (
    SteadyHeatAnalysis,
    SteadyHeatResult,
    TransientHeatAnalysis,
    TransientHeatResult,
)
from femsolver.analysis.thermal_strain import (
    apply_thermal_load,
    beam_thermal_axial_force,
    beam_thermal_gradient_moment,
)
from femsolver.thermal.fire import (
    astm_e119_temperature,
    concrete_strength_reduction_ec2,
    ec1_parametric_temperature,
    hydrocarbon_temperature,
    iso_834_temperature,
    steel_critical_temperature,
    steel_modulus_reduction_ec3,
    steel_strength_reduction_ec3,
)
from femsolver.geotech.winkler import (
    BeamOnWinklerFoundation2D,
    HetenyiInfiniteBeamResult,
    hetenyi_characteristic_length,
    hetenyi_infinite_beam_point_load,
    subgrade_modulus_table,
)
from femsolver.geotech.pile_group import (
    GroupSettlementResult,
    group_efficiency_converse_labarre,
    group_p_multipliers,
    group_settlement_elastic,
    p_multiplier,
)
from femsolver.geotech.liquefaction import (
    LiquefactionTriggeringResult,
    CRR_from_N1_60cs,
    cyclic_stress_ratio,
    evaluate_liquefaction,
    fines_content_correction,
    K_sigma,
    magnitude_scaling_factor,
    stress_reduction_coefficient,
)
from femsolver.geotech.dynamic_gazetas import (
    DynamicFootingImpedance,
    DynamicImpedanceCoefficients,
    dimensionless_frequency,
    dynamic_footing_impedance,
    gazetas_dynamic_coefficients,
)
from femsolver.sections import (
    ElasticSection2D,
    ElasticSection3D,
    ElasticShellSection,
    Fiber,
    FiberSection2D,
    FiberSection3D,
    LayeredShellSection,
    PlyStrength,
    SectionBase,
    ShellLayer,
    ShellSectionBase,
    CrackedSectionFactors,
    WallRegion,
    aci318_cracked_factors,
    asce41_wall_factors,
    evaluate_laminate,
    i_wall_section_3d,
    l_wall_section_3d,
    max_strain_index,
    max_stress_index,
    t_wall_section_3d,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
    u_wall_section_3d,
    wall_base_shear_spring_stiffness,
    wall_lateral_stiffness,
    wall_section_2d,
    wall_shear_area,
)
from femsolver.sections.response.hinges import BilinearMomentRotationSpring
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    ConcreteMander,
    UniaxialBilinear,
    UniaxialBRB,
    UniaxialElastic,
    UniaxialGap,
    UniaxialHysteretic,
    UniaxialIMK,
    UniaxialIsotropicHardening,
    UniaxialMaterial,
    UniaxialMenegottoPinto,
    UniaxialPivot,
    UniaxialTakeda,
)
from femsolver.elements.zero_length import ZeroLengthElement
from femsolver.elements.coupling_beam import (
    CouplingBeamResult,
    add_coupling_beam_2d,
)
from femsolver.elements.isolators import (
    friction_pendulum,
    lead_rubber_bearing,
)
from femsolver.analysis.algorithm import (
    LineSearchNewton,
    ModifiedNewton,
    Newton,
    NotConvergedError,
)
from femsolver.analysis.convergence import (
    ConvergenceTest,
    EnergyIncr,
    NormDispIncr,
    NormUnbalance,
)
from femsolver.analysis.buckling import LinearBucklingAnalysis
from femsolver.analysis.damping import RayleighDamping
from femsolver.analysis.eigen import EigenAnalysis
from femsolver.analysis.integrator import (
    ArcLength,
    DisplacementControl,
    LoadControl,
    StaticIntegrator,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.analysis.loads import (
    LoadCombination,
    LoadPattern,
    apply_combination,
    asce7_lrfd_combinations,
    asce7_lrfd_seismic_combinations_per_direction,
)
from femsolver.analysis.envelope import (
    DispEnvelope,
    EnvelopeAnalysis,
    EnvelopeResult,
    ForceEnvelope,
)
from femsolver.performance.drift_check import (
    DriftCheck,
    drift_check,
    drift_check_worst_combo,
)
from femsolver.performance.ida import (
    IDADriver,
    IDAPoint,
    IDARecord,
    max_drift_edp,
    pga_scale_factor,
)
from femsolver.performance.ida_collapse import (
    CollapseResult,
    IDASummary,
    detect_collapse,
    multi_record_ida,
)
from femsolver.performance.fragility import (
    FragilityFit,
    fit_collapse_fragility,
    fit_lognormal_mle,
    fit_lognormal_method_of_moments,
)
from femsolver.performance.record_scaling import (
    SuiteScalingResult,
    amplitude_scale_factor,
    compute_sdof_response_spectrum,
    period_range_mask,
    record_response_spectrum,
    scale_record_suite,
)
from femsolver.performance.cms import (
    baker_jayaram_correlation,
    compute_epsilon,
    conditional_mean_spectrum,
    conditional_spectrum_variance,
)
from femsolver.geotech.ssi import (
    FootingImpedance,
    HalfspaceSoil,
    embedment_correction,
    gazetas_surface_footing,
)
from femsolver.geotech.soil_springs import (
    SoilSpringBackbone,
    py_curve_sand,
    py_curve_soft_clay,
    qz_curve,
    tz_curve_clay,
    tz_curve_sand,
)
from femsolver.performance.p58 import (
    ComponentDamageAssessment,
    ComponentFragility,
    ComponentGroup,
    DamageState,
    P58AssessmentResult,
)
from femsolver.analysis.solvers import (
    CachedFactorSolver,
    DirectSparseSolver,
    IterativeSolver,
    LinearSolver,
    PardisoSolver,
    pardiso_available,
)
from femsolver.analysis.substructure import (
    CraigBamptonResult,
    GuyanResult,
    craig_bampton,
    guyan_condensation,
    guyan_recover_full,
)
from femsolver.analysis.parallel_assembler import (
    assemble_stiffness_parallel,
)
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis
from femsolver.analysis.nonlinear_transient import NonlinearTransientAnalysis
from femsolver.analysis.response_spectrum import (
    ResponseSpectrum,
    ResponseSpectrumAnalysis,
    ground_motion_force,
    multi_support_ground_motion_force,
)
from femsolver.performance.capacity_design import (
    BilinearCurve,
    EquivalentSDOF,
    PushoverToTarget,
    bilinearize_capacity_curve,
    coefficient_method_target,
    equivalent_sdof,
    n2_target_displacement,
    seismic_combination,
    story_drifts,
)
from femsolver.analysis.modal_pushover import ModalPushoverAnalysis
from femsolver.analysis.transient import TransientAnalysis
from femsolver.analysis.transient_integrator import (
    CentralDifference,
    GeneralizedAlpha,
    HHTAlpha,
    Newmark,
    NewmarkNonlinear,
    TransientIntegrator,
)
from femsolver.analysis.constraint_handler import (
    PenaltyHandler,
    TransformationHandler,
)
from femsolver.constraints import (
    Constraint,
    EqualDOF,
    MPConstraint,
    RigidDiaphragm,
    RigidLink,
)
from femsolver import design   # design code modules (Phase 29+)
from femsolver import benchmarks   # V&V benchmark suite (Phase 35)
from femsolver import bridges      # bridge engineering (Phase 38, Theme E)
from femsolver import reliability  # FORM / SORM / MC (Phase 44, Theme D)

__version__ = "0.1.0"

__all__ = [
    "Model",
    "Node",
    "ElasticIsotropic",
    "J2Plasticity3D",
    "DruckerPrager3D",
    "OrthotropicLamina",
    "Truss2D",
    "Truss3D",
    "Truss2DCorotational",
    "BeamColumn2D",
    "BeamColumn3D",
    "BeamColumn2DCorotational",
    "BeamColumn3DCorotational",
    "ForceBeamColumn2DCorotational",
    "HingedBeamColumn2D",
    "Quad4",
    "ShellMITC4",
    "ShellMITC9",
    "ShellDKMQ4",
    "ShellTri3",
    "ShellDKT3",
    "cylindrical_shell_mesh",
    "spherical_cap_mesh",
    "Hex8",
    "Tet4",
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
    "BilinearMomentRotationSpring",
    "UniaxialMaterial",
    "UniaxialElastic",
    "UniaxialBilinear",
    "UniaxialBRB",
    "UniaxialIMK",
    "UniaxialIsotropicHardening",
    "UniaxialTakeda",
    "UniaxialPivot",
    "UniaxialMenegottoPinto",
    "UniaxialHysteretic",
    "UniaxialGap",
    "ConcreteKentPark",
    "ConcreteMander",
    "ZeroLengthElement",
    "lead_rubber_bearing",
    "friction_pendulum",
    "LinearStaticAnalysis",
    "LinearSolver",
    "DirectSparseSolver",
    "IterativeSolver",
    "EigenAnalysis",
    "LinearBucklingAnalysis",
    "NonlinearStaticAnalysis",
    "TransientAnalysis",
    "TransientIntegrator",
    "Newmark",
    "NewmarkNonlinear",
    "HHTAlpha",
    "GeneralizedAlpha",
    "CentralDifference",
    "NonlinearTransientAnalysis",
    "RayleighDamping",
    "ResponseSpectrum",
    "ResponseSpectrumAnalysis",
    "ground_motion_force",
    "multi_support_ground_motion_force",
    "BilinearCurve",
    "EquivalentSDOF",
    "PushoverToTarget",
    "bilinearize_capacity_curve",
    "coefficient_method_target",
    "equivalent_sdof",
    "n2_target_displacement",
    "seismic_combination",
    "story_drifts",
    "ModalPushoverAnalysis",
    "Newton",
    "ModifiedNewton",
    "LineSearchNewton",
    "NotConvergedError",
    "ConvergenceTest",
    "NormDispIncr",
    "NormUnbalance",
    "EnergyIncr",
    "LoadControl",
    "DisplacementControl",
    "ArcLength",
    "StaticIntegrator",
    "TransformationHandler",
    "PenaltyHandler",
    "Constraint",
    "EqualDOF",
    "RigidLink",
    "RigidDiaphragm",
    "MPConstraint",
    "SuiteScalingResult",
    "amplitude_scale_factor",
    "compute_sdof_response_spectrum",
    "period_range_mask",
    "record_response_spectrum",
    "scale_record_suite",
    "IDADriver",
    "IDAPoint",
    "IDARecord",
    "max_drift_edp",
    "pga_scale_factor",
    "CollapseResult",
    "IDASummary",
    "detect_collapse",
    "multi_record_ida",
    "FragilityFit",
    "fit_collapse_fragility",
    "fit_lognormal_mle",
    "fit_lognormal_method_of_moments",
    "baker_jayaram_correlation",
    "compute_epsilon",
    "conditional_mean_spectrum",
    "conditional_spectrum_variance",
    "FootingImpedance",
    "HalfspaceSoil",
    "embedment_correction",
    "gazetas_surface_footing",
    "SoilSpringBackbone",
    "py_curve_sand",
    "py_curve_soft_clay",
    "qz_curve",
    "tz_curve_clay",
    "tz_curve_sand",
    "ComponentDamageAssessment",
    "ComponentFragility",
    "ComponentGroup",
    "DamageState",
    "P58AssessmentResult",
    "WallRegion",
    "wall_section_2d",
    "t_wall_section_3d",
    "l_wall_section_3d",
    "u_wall_section_3d",
    "i_wall_section_3d",
    "CrackedSectionFactors",
    "aci318_cracked_factors",
    "asce41_wall_factors",
    "wall_base_shear_spring_stiffness",
    "wall_lateral_stiffness",
    "wall_shear_area",
    "CouplingBeamResult",
    "add_coupling_beam_2d",
]
