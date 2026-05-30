"""Modal + buckling V&V benchmarks.

Modal:

* **Cantilever beam fundamental frequency** -- Euler-Bernoulli
  closed-form: ``omega_1 = (1.875)^2 sqrt(E I / (rho A L^4))``.

Buckling:

* **Pin-pin Euler column** -- ``P_cr = pi^2 E I / L^2``.
* **Cantilever Euler column** -- ``P_cr = pi^2 E I / (4 L^2)``.
* **Fixed-fixed column** -- ``P_cr = 4 pi^2 E I / L^2``.

Each benchmark builds a small model, runs the appropriate analysis,
and returns the scalar of interest.
"""
from __future__ import annotations

import math

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    ElasticIsotropic,
    LinearBucklingAnalysis,
    LinearStaticAnalysis,
    Model,
)
from femsolver.analysis.eigen import EigenAnalysis
from femsolver.benchmarks.harness import Benchmark


# ============================================================ modal

def _cantilever_beam_omega1_value() -> float:
    """Fundamental angular frequency (rad/s) of a cantilever beam."""
    E, A, Iz, L, rho = 2.0e11, 1.0e-2, 8.333e-7, 3.0, 7850.0
    n_elem = 12
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    eig = EigenAnalysis(m, num_modes=3)
    eig.run()
    return float(math.sqrt(eig.eigenvalues[0]))


def _cantilever_beam_omega1_reference() -> float:
    """omega_1 = (1.875)^2 sqrt(E I / (rho A L^4)).

    1.875 = first root of cos(kL) cosh(kL) + 1 = 0.
    """
    E, A, Iz, L, rho = 2.0e11, 1.0e-2, 8.333e-7, 3.0, 7850.0
    return (1.8751) ** 2 * math.sqrt(E * Iz / (rho * A * L ** 4))


# ============================================================ buckling

def _build_column(n_elem: int = 8, *, L: float = 3.0,
                   A: float = 0.01, Iz: float = 8.333e-7):
    """Vertical column built from corotational beam-columns (needed
    for linear-buckling, which requires geometric-stiffness-aware
    elements).
    """
    E = 2.0e11
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, 0.0, i * L / n_elem)
    for i in range(n_elem):
        m.add_element(BeamColumn2DCorotational(
            i + 1, (i + 1, i + 2), mat, A, Iz,
        ))
    return m, mat, E, A, Iz, L


def _pin_pin_buckling_value() -> float:
    m, mat, E, A, Iz, L = _build_column()
    m.fix(1, [1, 1, 0])
    m.fix(len(m.nodes), [1, 0, 0])
    # Apply a unit compressive load (axial, pointing down at top)
    m.add_nodal_load(len(m.nodes), [0.0, -1.0, 0.0])
    # Linearise about the pre-buckled state by running a static step
    LinearStaticAnalysis(m).run()
    res = LinearBucklingAnalysis(m, num_modes=1).run()
    return float(res["critical_load_factor"])


def _pin_pin_buckling_reference() -> float:
    E, Iz, L = 2.0e11, 8.333e-7, 3.0
    return math.pi ** 2 * E * Iz / L ** 2


def _cantilever_buckling_value() -> float:
    m, mat, E, A, Iz, L = _build_column()
    m.fix(1, [1, 1, 1])
    # Apply a unit compressive load (axial, pointing down at top)
    m.add_nodal_load(len(m.nodes), [0.0, -1.0, 0.0])
    # Linearise about the pre-buckled state by running a static step
    LinearStaticAnalysis(m).run()
    res = LinearBucklingAnalysis(m, num_modes=1).run()
    return float(res["critical_load_factor"])


def _cantilever_buckling_reference() -> float:
    E, Iz, L = 2.0e11, 8.333e-7, 3.0
    return math.pi ** 2 * E * Iz / (4.0 * L ** 2)


def _fixed_fixed_buckling_value() -> float:
    m, mat, E, A, Iz, L = _build_column(n_elem=12)
    m.fix(1, [1, 1, 1])
    # Top: lateral DOF fixed, axial free; rotation fixed
    m.fix(len(m.nodes), [1, 0, 1])
    # Apply a unit compressive load (axial, pointing down at top)
    m.add_nodal_load(len(m.nodes), [0.0, -1.0, 0.0])
    # Linearise about the pre-buckled state by running a static step
    LinearStaticAnalysis(m).run()
    res = LinearBucklingAnalysis(m, num_modes=1).run()
    return float(res["critical_load_factor"])


def _fixed_fixed_buckling_reference() -> float:
    E, Iz, L = 2.0e11, 8.333e-7, 3.0
    return 4.0 * math.pi ** 2 * E * Iz / L ** 2


# ============================================================ factory

def modal_buckling_benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="Cantilever omega_1 (Bernoulli)",
            category="modal",
            reference_value=_cantilever_beam_omega1_reference(),
            reference_source="(1.875)^2 sqrt(EI/(rho A L^4))",
            units="rad/s",
            tolerance=0.02,
            runner=_cantilever_beam_omega1_value,
            note="BeamColumn2D, 12 elem",
        ),
        Benchmark(
            name="Euler pin-pin P_cr",
            category="buckling",
            reference_value=_pin_pin_buckling_reference(),
            reference_source="pi^2 E I / L^2",
            units="N",
            tolerance=0.02,
            runner=_pin_pin_buckling_value,
            note="BeamColumn2D, 8 elem",
        ),
        Benchmark(
            name="Euler cantilever P_cr",
            category="buckling",
            reference_value=_cantilever_buckling_reference(),
            reference_source="pi^2 E I / (4 L^2)",
            units="N",
            tolerance=0.02,
            runner=_cantilever_buckling_value,
            note="BeamColumn2D, 8 elem",
        ),
        Benchmark(
            name="Euler fixed-fixed P_cr",
            category="buckling",
            reference_value=_fixed_fixed_buckling_reference(),
            reference_source="4 pi^2 E I / L^2",
            units="N",
            tolerance=0.05,
            runner=_fixed_fixed_buckling_value,
            note="BeamColumn2D, 12 elem",
        ),
    ]
