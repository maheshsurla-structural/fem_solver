"""Nonlinear V&V benchmarks.

* **EP cantilever shape factor** -- ratio of the full-plastic moment
  to the first-yield moment for a rectangular cross-section. From
  analytic plasticity: ``M_p / M_y = 1.5``.
* **Truss elasto-plastic axial** -- a simple bar should saturate at
  ``F = sigma_y * A`` regardless of strain demand.

These are deliberately simple, structure-level sanity tests rather
than the full-fledged hyperelastic / large-strain benchmarks (those
are deferred to a future Theme B).
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2D,
    DisplacementControl,
    ElasticIsotropic,
    FiberSection2D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)
import numpy as np
from femsolver.benchmarks.harness import Benchmark


# ============================================================ shape factor

def _ep_section_shape_factor_value() -> float:
    """Section-level rectangular shape factor M_p / M_y.

    Drives a FiberSection2D directly with monotonically increasing
    curvature; reports the ratio of the plateau moment to the
    first-yield moment. Pure bending, zero axial strain.

    For a rectangular EPP section: M_p / M_y = 1.5 exactly.
    """
    E = 2.0e11
    sigma_y = 300.0e6
    width, height = 0.10, 0.20
    n_fibers = 80     # fine discretization for accurate plateau

    My = sigma_y * width * height ** 2 / 6.0
    Mp = sigma_y * width * height ** 2 / 4.0
    I = width * height ** 3 / 12.0
    kappa_y = My / (E * I)

    # Build section, drive curvature directly through get_response.
    uniaxial = UniaxialBilinear(E=E, sigma_y=sigma_y, b=1.0e-8)
    section = FiberSection2D.rectangular(
        width=width, height=height,
        n_fibers=n_fibers, material=uniaxial,
    )
    # Push curvature to 20 kappa_y in 200 small steps so plateau is reached
    kappa_values = np.linspace(0.0, 20.0 * kappa_y, 201)
    M_values = np.zeros_like(kappa_values)
    for i, k in enumerate(kappa_values):
        e = np.array([0.0, float(k)])
        s, _ = section.get_response(e)
        M_values[i] = s[1]
        section.commit_state()
    M_plateau = float(np.max(np.abs(M_values)))
    return M_plateau / My


def _ep_section_shape_factor_reference() -> float:
    return 1.5     # M_p / M_y for rectangle (Z/S = 1.5)


# ============================================================ truss yield

def _truss_yield_force_value() -> float:
    """Peak axial force of a yielding bar under tip displacement.

    F_max = sigma_y * A (exact for EPP).
    """
    E = 2.0e11
    sigma_y = 400.0e6
    width, height = 0.01, 0.01
    L = 1.0
    eps_y = sigma_y / E
    u_target = 5.0 * eps_y * L   # push to 5x yield strain

    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
    uniaxial = UniaxialBilinear(E=E, sigma_y=sigma_y, b=1.0e-5)
    section = FiberSection2D.rectangular(
        width=width, height=height, n_fibers=4, material=uniaxial,
    )
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, section=section))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])    # axial only
    m.add_nodal_load(2, [1.0, 0.0, 0.0])    # unit ref load
    n_steps = 40
    integrator = DisplacementControl(
        node_tag=2, dof_index=0, du_step=u_target / n_steps,
    )
    ana = NonlinearStaticAnalysis(
        m, num_steps=n_steps, integrator=integrator,
        tol=1.0e-6, max_iter=40,
        convergence="disp_incr",
        track=(2, 0),
    )
    res = ana.run()
    lambdas = np.array(res["lambdas"])
    return float(np.max(np.abs(lambdas)))


def _truss_yield_force_reference() -> float:
    sigma_y, width, height = 400.0e6, 0.01, 0.01
    return sigma_y * width * height


# ============================================================ factory

def nonlinear_benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="Rectangular EPP shape factor M_p/M_y",
            category="nonlinear",
            reference_value=_ep_section_shape_factor_reference(),
            reference_source="Z/S = 1.5 (rectangular)",
            units="(dimensionless)",
            tolerance=0.02,
            runner=_ep_section_shape_factor_value,
            note="FiberSection2D, 80 fibers, M-kappa direct",
        ),
        Benchmark(
            name="Yielding bar peak axial",
            category="nonlinear",
            reference_value=_truss_yield_force_reference(),
            reference_source="F = sigma_y * A",
            units="N",
            tolerance=0.05,
            runner=_truss_yield_force_value,
            note="FiberSection2D, 4 fibers, EPP",
        ),
    ]
