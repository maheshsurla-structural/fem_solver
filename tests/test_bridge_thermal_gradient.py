"""Bridge temperature-gradient loads (bridge plan T1.3).

Covers the code gradient profiles, the section reduction (equivalent uniform
temperature, curvature, self-equilibrated stress), and the frame effects
(simple-span camber, continuity moments, restrained axial force).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       LinearStaticAnalysis, Model)
from femsolver.bridges import (TemperatureGradient,  # noqa: E402
                               aashto_gradient, apply_beam_thermal_actions,
                               equivalent_thermal_actions, linear_gradient)

E, ALPHA = 3.0e10, 1.0e-5
H, B = 1.0, 0.5
A, I = B * H, B * H ** 3 / 12.0


# ------------------------------------------------------------ profiles
def test_aashto_gradient_profile():
    g = aashto_gradient(3, 1.2, alpha=ALPHA)
    assert g.T(0.0) == pytest.approx(23.0)         # zone 3 T1
    assert g.T(0.1) == pytest.approx(6.0)          # zone 3 T2
    assert g.T(0.3) == pytest.approx(0.0)          # T3 = 0
    assert g.T(1.2) == pytest.approx(0.0)          # held to soffit


def test_aashto_rejects_bad_zone():
    with pytest.raises(ValueError):
        aashto_gradient(9, 1.0)


# ------------------------------------------------- section reduction
def test_linear_profile_has_zero_self_stress():
    a = equivalent_thermal_actions(linear_gradient(20.0, 4.0, H, alpha=ALPHA),
                                   height=H, width=B, E=E)
    # a linear temperature profile is fully accommodated by plane sections
    assert a.self_stress_top == pytest.approx(0.0, abs=1e-6)
    assert a.self_stress_bottom == pytest.approx(0.0, abs=1e-6)


def test_uniform_profile_reduces_to_axial_only():
    a = equivalent_thermal_actions(TemperatureGradient([0, H], [10.0, 10.0],
                                                       alpha=ALPHA),
                                   height=H, width=B, E=E)
    assert a.dT_uniform == pytest.approx(10.0)
    assert a.curvature == pytest.approx(0.0, abs=1e-12)
    assert a.eps0 == pytest.approx(ALPHA * 10.0)
    assert a.self_stress_top == pytest.approx(0.0, abs=1e-6)


def test_nonlinear_self_stress_is_self_equilibrated():
    g = aashto_gradient(2, H, alpha=ALPHA)
    a = equivalent_thermal_actions(g, height=H, width=B, E=E, n=2000)
    d = np.linspace(0, H, 2001)
    y = a.centroid_depth - d
    ss = a.self_stress(d)
    N = np.trapezoid(ss * B, d)
    M = np.trapezoid(ss * B * y, d)
    # integrate to zero axial force and zero moment (relative to stress scale)
    scale = E * ALPHA * 25.0 * A                    # ~ E·α·T·A
    assert abs(N) < 1e-6 * scale
    assert abs(M) < 1e-6 * scale * H
    assert a.self_stress_top < 0.0                  # top fibre restrained → compression


# -------------------------------------------------------- frame effects
def _beam(L, n, bc):
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.2)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    if bc == "ss":
        m.fix(1, [1, 1, 0]); m.fix(n + 1, [0, 1, 0])
    else:                                            # fixed–fixed
        m.fix(1, [1, 1, 1]); m.fix(n + 1, [1, 1, 1])
    return m


def test_simple_span_cambers_up_under_positive_gradient():
    a = equivalent_thermal_actions(aashto_gradient(3, H, alpha=ALPHA),
                                   height=H, width=B, E=E)
    L, n = 20.0, 10
    m = _beam(L, n, "ss")
    m.number_dofs()
    apply_beam_thermal_actions(m, eps0=0.0, kappa=a.curvature)
    LinearStaticAnalysis(m).run()
    v_mid = m.node(n // 2 + 1).disp[1]
    assert v_mid == pytest.approx(a.curvature * L ** 2 / 8.0, rel=2e-3)
    assert v_mid > 0.0                               # top hotter → upward


def test_fixed_fixed_develops_continuity_moment():
    a = equivalent_thermal_actions(aashto_gradient(3, H, alpha=ALPHA),
                                   height=H, width=B, E=E)
    L, n = 20.0, 10
    m = _beam(L, n, "ff")
    m.number_dofs()
    apply_beam_thermal_actions(m, eps0=0.0, kappa=a.curvature)
    LinearStaticAnalysis(m).run()
    # fully restrained curvature → uniform moment EIκ, no deflection
    max_v = max(abs(m.node(i + 1).disp[1]) for i in range(n + 1))
    assert max_v < 1e-9
    assert abs(m.node(1).reaction[2]) == pytest.approx(E * I * a.curvature,
                                                       rel=1e-6)


def test_fixed_fixed_uniform_temp_axial_force():
    a = equivalent_thermal_actions(TemperatureGradient([0, H], [20.0, 20.0],
                                                       alpha=ALPHA),
                                   height=H, width=B, E=E)
    m = _beam(20.0, 8, "ff")
    m.number_dofs()
    apply_beam_thermal_actions(m, eps0=a.eps0, kappa=0.0)
    LinearStaticAnalysis(m).run()
    assert abs(m.node(1).reaction[0]) == pytest.approx(E * A * a.eps0, rel=1e-6)


def test_apply_returns_element_count():
    m = _beam(10.0, 4, "ss")
    m.number_dofs()
    assert apply_beam_thermal_actions(m, eps0=1e-4, kappa=0.0) == 4
