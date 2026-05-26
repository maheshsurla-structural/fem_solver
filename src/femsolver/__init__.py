"""femsolver — a Python finite element solver for structural analysis."""

from femsolver.core.model import Model
from femsolver.core.node import Node
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.materials.j2_plasticity import J2Plasticity3D
from femsolver.materials.orthotropic import OrthotropicLamina
from femsolver.materials.drucker_prager import DruckerPrager3D
from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.elements.truss_corot import Truss2DCorotational
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.elements.beam_corot_3d import BeamColumn3DCorotational
from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.elements.beam_hinged import HingedBeamColumn2D
from femsolver.elements.plane import Quad4
from femsolver.elements.shell import ShellMITC4
from femsolver.elements.shell_tri import ShellTri3
from femsolver.elements.shell_mesh import (
    cylindrical_shell_mesh,
    spherical_cap_mesh,
)
from femsolver.elements.solid import Hex8, Tet4
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
    evaluate_laminate,
    max_strain_index,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)
from femsolver.sections.hinges import BilinearMomentRotationSpring
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    ConcreteMander,
    UniaxialBilinear,
    UniaxialElastic,
    UniaxialGap,
    UniaxialHysteretic,
    UniaxialMaterial,
    UniaxialMenegottoPinto,
)
from femsolver.elements.zero_length import ZeroLengthElement
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
from femsolver.analysis.solvers import (
    DirectSparseSolver,
    IterativeSolver,
    LinearSolver,
)
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis
from femsolver.analysis.nonlinear_transient import NonlinearTransientAnalysis
from femsolver.analysis.response_spectrum import (
    ResponseSpectrum,
    ResponseSpectrumAnalysis,
    ground_motion_force,
    multi_support_ground_motion_force,
)
from femsolver.analysis.capacity_design import (
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
    "ShellTri3",
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
]
