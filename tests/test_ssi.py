"""Phase 27 tests -- Gazetas footing impedance + API soil-spring backbones.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    FootingImpedance,
    HalfspaceSoil,
    SoilSpringBackbone,
    embedment_correction,
    gazetas_surface_footing,
    py_curve_sand,
    py_curve_soft_clay,
    qz_curve,
    tz_curve_clay,
    tz_curve_sand,
)


# ============================================================ HalfspaceSoil

def test_halfspace_soil_basic():
    soil = HalfspaceSoil(G=50.0e6, nu=0.35, rho=1900.0)
    assert soil.Vs == pytest.approx(math.sqrt(50.0e6 / 1900.0), rel=1.0e-12)
    # Vla = 3.4/(pi*(1-nu)) * Vs > Vs (for nu < ~0.7)
    assert soil.Vla > soil.Vs


def test_halfspace_soil_validates():
    with pytest.raises(ValueError, match="G"):
        HalfspaceSoil(G=-1.0, nu=0.3, rho=1900.0)
    with pytest.raises(ValueError, match="nu"):
        HalfspaceSoil(G=1.0e6, nu=0.6, rho=1900.0)
    with pytest.raises(ValueError, match="rho"):
        HalfspaceSoil(G=1.0e6, nu=0.3, rho=-1.0)


# ============================================================ Gazetas

def test_gazetas_square_footing_stiffness_ordering():
    """For a square footing L=B, K_y == K_x; vertical > horizontal."""
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    imp = gazetas_surface_footing(soil, B=2.0, L=2.0)
    assert imp.K_x == pytest.approx(imp.K_y, rel=1.0e-10)
    assert imp.K_z > imp.K_x       # vertical stiffer than horizontal


def test_gazetas_rectangular_aspect():
    """For L > B, K_y > K_x (long-direction translation is stiffer)."""
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    imp = gazetas_surface_footing(soil, B=2.0, L=5.0)
    assert imp.K_y > imp.K_x
    # Rocking about long axis (K_rx) < rocking about short axis (K_ry)
    # because L/B^2.4 grows fast for K_ry
    assert imp.K_ry > imp.K_rx


def test_gazetas_scales_linearly_with_G():
    """Doubling G doubles all stiffnesses."""
    soil_1 = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    soil_2 = HalfspaceSoil(G=100.0e6, nu=0.30, rho=1900.0)
    imp_1 = gazetas_surface_footing(soil_1, B=2.0, L=3.0)
    imp_2 = gazetas_surface_footing(soil_2, B=2.0, L=3.0)
    for attr in ("K_z", "K_x", "K_y", "K_rx", "K_ry", "K_t"):
        v1 = getattr(imp_1, attr)
        v2 = getattr(imp_2, attr)
        assert v2 == pytest.approx(2.0 * v1, rel=1.0e-12), \
            f"{attr} should scale linearly with G"


def test_gazetas_validates():
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    with pytest.raises(ValueError, match="B"):
        gazetas_surface_footing(soil, B=-1.0, L=2.0)
    with pytest.raises(ValueError, match="L"):
        gazetas_surface_footing(soil, B=2.0, L=1.0)   # L<B


# ============================================================ Embedment

def test_embedment_correction_increases_stiffness():
    """All embedded stiffnesses >= surface."""
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    surface = gazetas_surface_footing(soil, B=2.0, L=3.0)
    embedded = embedment_correction(surface, soil, D=1.0)
    for attr in ("K_z", "K_x", "K_y", "K_rx", "K_ry", "K_t"):
        v_s = getattr(surface, attr)
        v_e = getattr(embedded, attr)
        assert v_e >= v_s, f"{attr} should be >= surface stiffness"


def test_embedment_D_zero_returns_same():
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    surface = gazetas_surface_footing(soil, B=2.0, L=3.0)
    embedded = embedment_correction(surface, soil, D=0.0)
    assert embedded.K_z == surface.K_z
    assert embedded.K_rx == surface.K_rx


def test_embedment_validates():
    soil = HalfspaceSoil(G=50.0e6, nu=0.30, rho=1900.0)
    surface = gazetas_surface_footing(soil, B=2.0, L=3.0)
    with pytest.raises(ValueError, match="D"):
        embedment_correction(surface, soil, D=-1.0)


# ============================================================ p-y curves

def test_py_curve_sand_monotonic():
    """p increases monotonically with y."""
    py = py_curve_sand(z=5.0, D=0.6, gamma_eff=9000.0, phi_deg=35.0)
    assert np.all(np.diff(py.p) >= -1.0e-9)
    assert py.p[0] == pytest.approx(0.0, abs=1.0e-9)
    assert py.p[-1] > 0.0


def test_py_curve_sand_initial_stiffness_positive():
    py = py_curve_sand(z=5.0, D=0.6, gamma_eff=9000.0, phi_deg=35.0)
    assert py.initial_stiffness() > 0.0


def test_py_curve_sand_evaluate_symmetric():
    """Backbone is antisymmetric in sign of y."""
    py = py_curve_sand(z=3.0, D=0.5, gamma_eff=10000.0, phi_deg=32.0)
    p_pos = py.evaluate(0.02)
    p_neg = py.evaluate(-0.02)
    assert p_pos == pytest.approx(-p_neg, rel=1.0e-9)


def test_py_curve_soft_clay_levels_off():
    """At large y, p reaches p_u and stops growing."""
    py = py_curve_soft_clay(z=5.0, D=0.6, c_u=50000.0, eps50=0.02)
    # At y = 8 y_c, p = p_u; at y = 16 y_c (end of backbone), still p_u
    p_end = py.p[-1]
    p_late = py.p[-5]
    assert p_end == pytest.approx(p_late, rel=1.0e-9)


def test_py_curve_validates():
    with pytest.raises(ValueError):
        py_curve_sand(z=-1.0, D=0.6, gamma_eff=9000.0, phi_deg=35.0)
    with pytest.raises(ValueError):
        py_curve_soft_clay(z=5.0, D=0.6, c_u=-1.0)


# ============================================================ t-z curve

def test_tz_curve_sand_plateau():
    """t-z sand: linear rise then constant plateau."""
    tz = tz_curve_sand(D=0.6, sigma_v_eff=80000.0, delta_deg=30.0)
    f_max = tz.p[-1]
    # Plateau plateau
    f_late_1 = tz.p[-1]
    f_late_2 = tz.p[-3]
    assert f_late_1 == pytest.approx(f_late_2, rel=1.0e-12)
    # f_max formula
    expected = 0.8 * 80000.0 * math.tan(math.radians(30.0)) * math.pi * 0.6
    assert f_max == pytest.approx(expected, rel=1.0e-9)


def test_tz_curve_clay_post_peak_degrades():
    """t-z clay: degradation -> residual < peak."""
    tz = tz_curve_clay(D=0.6, c_u=50000.0, alpha=0.8,
                       degrade_to_residual=0.85)
    f_max_analytic = 0.8 * 50000.0 * math.pi * 0.6
    f_res_analytic = 0.85 * f_max_analytic
    f_end = tz.p[-1]
    assert f_end < f_max_analytic                              # residual < peak
    assert f_end == pytest.approx(f_res_analytic, rel=1.0e-9)


def test_tz_curve_validates():
    with pytest.raises(ValueError):
        tz_curve_clay(D=0.6, c_u=50000.0, alpha=1.5)
    with pytest.raises(ValueError):
        tz_curve_sand(D=-0.5, sigma_v_eff=80000.0, delta_deg=30.0)


# ============================================================ q-z curve

def test_qz_curve_full_mobilization():
    """q-z: cube root rise to Q_ult at z_peak = 0.1 D, then constant."""
    qz = qz_curve(D=0.6, q_ult=5.0e6)
    A_tip = math.pi * 0.6 ** 2 / 4.0
    Q_ult = 5.0e6 * A_tip
    assert qz.p[-1] == pytest.approx(Q_ult, rel=1.0e-9)
    # At z_peak = 0.06 m, q should equal Q_ult
    Q_at_peak = qz.evaluate(0.06)
    assert Q_at_peak == pytest.approx(Q_ult, rel=0.05)


def test_qz_curve_validates():
    with pytest.raises(ValueError):
        qz_curve(D=-0.5, q_ult=5.0e6)
    with pytest.raises(ValueError):
        qz_curve(D=0.5, q_ult=-1.0)


# ============================================================ backbone helpers

def test_soil_spring_evaluate_clamps():
    """Out-of-range y_query clamps to endpoint value."""
    y = np.array([0.0, 0.01, 0.05])
    p = np.array([0.0, 100.0, 200.0])
    bb = SoilSpringBackbone(y=y, p=p)
    assert bb.evaluate(0.5) == pytest.approx(200.0, rel=1.0e-12)
    assert bb.evaluate(-0.5) == pytest.approx(-200.0, rel=1.0e-12)


def test_soil_spring_validates():
    with pytest.raises(ValueError, match="shape"):
        SoilSpringBackbone(y=np.array([0.0, 1.0]), p=np.array([1.0]))
    with pytest.raises(ValueError, match=">= 2"):
        SoilSpringBackbone(y=np.array([0.0]), p=np.array([0.0]))
