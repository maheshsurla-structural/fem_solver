"""femsolver — a Python finite element solver for structural analysis."""

from femsolver.core.model import Model
from femsolver.core.node import Node
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.elements.truss_corot import Truss2DCorotational
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.elements.beam_corot_3d import BeamColumn3DCorotational
from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.elements.beam_hinged import HingedBeamColumn2D
from femsolver.elements.plane import Quad4
from femsolver.sections import (
    ElasticSection2D,
    ElasticSection3D,
    Fiber,
    FiberSection2D,
    FiberSection3D,
    SectionBase,
)
from femsolver.sections.hinges import BilinearMomentRotationSpring
from femsolver.materials.uniaxial import (
    UniaxialBilinear,
    UniaxialElastic,
    UniaxialMaterial,
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
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis
from femsolver.analysis.nonlinear_transient import NonlinearTransientAnalysis
from femsolver.analysis.response_spectrum import (
    ResponseSpectrum,
    ResponseSpectrumAnalysis,
    ground_motion_force,
)
from femsolver.analysis.transient import TransientAnalysis
from femsolver.analysis.transient_integrator import (
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
    "SectionBase",
    "ElasticSection2D",
    "ElasticSection3D",
    "Fiber",
    "FiberSection2D",
    "FiberSection3D",
    "BilinearMomentRotationSpring",
    "UniaxialMaterial",
    "UniaxialElastic",
    "UniaxialBilinear",
    "LinearStaticAnalysis",
    "EigenAnalysis",
    "LinearBucklingAnalysis",
    "NonlinearStaticAnalysis",
    "TransientAnalysis",
    "TransientIntegrator",
    "Newmark",
    "NewmarkNonlinear",
    "NonlinearTransientAnalysis",
    "RayleighDamping",
    "ResponseSpectrum",
    "ResponseSpectrumAnalysis",
    "ground_motion_force",
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
