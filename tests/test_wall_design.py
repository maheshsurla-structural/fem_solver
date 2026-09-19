"""ACI 318-19 §18.10 special structural wall design (femsolver.design.walls)."""
from __future__ import annotations

import math

import pytest

from femsolver.design.walls import (WallDemand, WallGeometry, WallMaterial,
                                     WallReinforcement, boundary_element_check,
                                     design_wall_pier, shear_alpha_c,
                                     wall_min_web_reinforcement,
                                     wall_pm_capacity, wall_shear_strength)

FC = 30e6
FY = 420e6


def _geom(lw=3.0, t=0.25, hw=9.0):
    return WallGeometry(lw=lw, t=t, hw=hw)


def _mat():
    return WallMaterial(fc=FC, fy=FY)


def _reinf(**kw):
    base = dict(rho_l=0.0025, rho_t=0.0025, As_boundary=0.002, d_boundary=0.15)
    base.update(kw)
    return WallReinforcement(**base)


# ------------------------------------------------------------------ shear

def test_alpha_c_interpolation():
    assert shear_alpha_c(1.0) == 0.25          # squat
    assert shear_alpha_c(3.0) == 0.17          # slender
    assert shear_alpha_c(1.75) == pytest.approx(0.21)   # midpoint


def test_shear_strength_matches_hand_calc():
    g, m, r = _geom(), _mat(), _reinf()
    res = wall_shear_strength(g, m, r, Vu=500e3)
    sqrt_fc = math.sqrt(FC / 1e6) * 1e6         # ACI √f'c, MPa convention → Pa
    Vc = 0.75 * 0.17 * sqrt_fc                  # Acv=0.75, αc=0.17, λ=1
    Vs = 0.75 * 0.0025 * FY
    assert res.Vc == pytest.approx(Vc, rel=1e-6)
    assert res.Vs == pytest.approx(Vs, rel=1e-6)
    assert res.Vn == pytest.approx(Vc + Vs, rel=1e-6)
    assert res.phiVn == pytest.approx(0.75 * (Vc + Vs), rel=1e-6)
    assert res.dcr == pytest.approx(500e3 / res.phiVn, rel=1e-6)


def test_shear_upper_cap_applies():
    # a heavily reinforced squat wall hits the 0.83·Acv·√f'c cap
    g = _geom(lw=4.0, t=0.4, hw=4.0)
    res = wall_shear_strength(g, _mat(), _reinf(rho_t=0.02), Vu=0.0)
    assert res.Vn == pytest.approx(res.Vn_cap)


def test_two_curtains_flag():
    g, m = _geom(), _mat()
    lo = wall_shear_strength(g, m, _reinf(), Vu=1.0)
    hi = wall_shear_strength(g, m, _reinf(), Vu=5e6)
    assert not lo.two_curtains_required
    assert hi.two_curtains_required
    # a thick wall always needs two curtains
    assert wall_shear_strength(_geom(t=0.30), m, _reinf(),
                               Vu=1.0).two_curtains_required


# --------------------------------------------------------- boundary elements

def test_boundary_element_stress_trigger():
    g, m = _geom(), _mat()
    be = boundary_element_check(g, m, WallDemand(Pu=3e6, Mu=2e6))
    # σ = 3e6/0.75 + 2e6·1.5/0.5625 = 9.333 MPa > 0.2·30 = 6 MPa → required
    assert be.sigma_max == pytest.approx(9.3333e6, rel=1e-4)
    assert be.required
    assert be.ratio == pytest.approx(9.3333e6 / FC, rel=1e-4)
    assert be.lbe_min == pytest.approx(0.15 * 3.0)


def test_boundary_element_not_required_low_stress():
    g, m = _geom(), _mat()
    be = boundary_element_check(g, m, WallDemand(Pu=1e6, Mu=0.2e6))
    assert not be.required
    assert be.can_discontinue          # σ well below 0.15 f'c


# ------------------------------------------------------- minimum reinforcement

def test_min_web_reinforcement_thresholds():
    g, m = _geom(), _mat()
    thr = 0.083 * g.Acv * (math.sqrt(FC / 1e6) * 1e6)
    low = wall_min_web_reinforcement(g, m, _reinf(), Vu=0.5 * thr)
    high = wall_min_web_reinforcement(g, m, _reinf(), Vu=2.0 * thr)
    assert not low.high_shear and low.rho_t_min == 0.0020
    assert high.high_shear and high.rho_t_min == 0.0025
    # a wall with ρt = 0.0025 satisfies both regimes; 0.0015 fails the high one
    assert wall_min_web_reinforcement(g, m, _reinf(rho_t=0.0015),
                                      Vu=2.0 * thr).rho_t_ok is False


# ------------------------------------------------------------------ P-M

def test_pure_flexure_is_tension_controlled():
    g, m, r = _geom(), _mat(), _reinf()
    res = wall_pm_capacity(g, m, r, WallDemand(Pu=0.0, Mu=1.0))
    assert res.phiMn > 0
    assert res.phi == pytest.approx(0.90, abs=1e-6)      # tension-controlled
    assert res.eps_t > FY / 200e9 + 0.003


def test_moderate_compression_raises_moment_capacity():
    g, m, r = _geom(), _mat(), _reinf()
    m0 = wall_pm_capacity(g, m, r, WallDemand(Pu=0.0, Mu=1.0)).phiMn
    m1 = wall_pm_capacity(g, m, r, WallDemand(Pu=1.5e6, Mu=1.0)).phiMn
    assert m1 > m0                                        # axial helps flexure


def test_dcr_scales_with_moment_demand():
    g, m, r = _geom(), _mat(), _reinf()
    d1 = wall_pm_capacity(g, m, r, WallDemand(Pu=1e6, Mu=1e6)).dcr
    d2 = wall_pm_capacity(g, m, r, WallDemand(Pu=1e6, Mu=2e6)).dcr
    assert d2 == pytest.approx(2 * d1, rel=1e-3)


def test_excess_axial_governs_dcr_above_one():
    g, m, r = _geom(), _mat(), _reinf()
    Ast = 2 * r.As_boundary + r.rho_l * g.Ag
    Pn_max = 0.80 * (0.85 * FC * (g.Ag - Ast) + FY * Ast)
    res = wall_pm_capacity(g, m, r, WallDemand(Pu=0.9 * Pn_max, Mu=1.0))
    assert res.dcr > 1.0


# --------------------------------------------------------- assembled check

def test_design_wall_pier_ok_case():
    g, m, r = _geom(), _mat(), _reinf()
    res = design_wall_pier(g, m, r, WallDemand(Pu=1e6, Mu=0.5e6, Vu=300e3))
    assert res.dcr == max(res.pm.dcr, res.shear.dcr)
    assert res.ok
    assert res.notes == [] or all(isinstance(n, str) for n in res.notes)


def test_design_wall_pier_flags_boundary_and_min_reinf():
    g, m = _geom(), _mat()
    r = _reinf(rho_t=0.0015)                              # below high-shear min
    res = design_wall_pier(g, m, r, WallDemand(Pu=4e6, Mu=3e6, Vu=1.2e6))
    assert res.boundary.required
    assert any("boundary" in n.lower() for n in res.notes)
    assert not res.min_reinf.rho_t_ok
    assert not res.ok


# ------------------------------------------ W3 polish: drift trigger + codes

from femsolver.design.walls import (WALL_CODES, boundary_element_check,  # noqa: E402
                                    wall_detailing)


def test_displacement_boundary_trigger():
    g, m = _geom(lw=4.0, t=0.30, hw=24.0), _mat()
    d = WallDemand(Pu=1e6, Mu=0.2e6)             # low stress → stress trigger off
    # large neutral axis + high drift → drift trigger fires
    be = boundary_element_check(g, m, d, c=1.5, drift=0.02)
    # c_limit = lw/(600*0.02) = 4/12 = 0.333 m; c=1.5 ≥ that → required
    assert be.disp_required and be.required
    assert not be.stress_required
    # ℓbe = max(c − 0.1ℓw, c/2) = max(1.5−0.4, 0.75) = 1.1 m
    assert be.lbe_min == pytest.approx(1.1)


def test_displacement_trigger_off_for_small_c():
    g, m = _geom(lw=4.0, t=0.30, hw=24.0), _mat()
    be = boundary_element_check(g, m, WallDemand(Pu=1e6, Mu=0.2e6),
                                c=0.2, drift=0.01)
    # c_limit = 4/(600*0.01)=0.667; c=0.2 < that → no drift trigger
    assert not be.disp_required and not be.required


def test_drift_floor_half_percent():
    g, m = _geom(lw=4.0, t=0.30, hw=24.0), _mat()
    # tiny drift is floored at 0.005 → c_limit = 4/(600*0.005)=1.333 m
    be = boundary_element_check(g, m, WallDemand(Pu=1e6, Mu=0.2e6),
                               c=1.4, drift=1e-6)
    assert be.disp_required            # 1.4 ≥ 1.333 using the 0.005 floor


def test_detailing_codes_boundary_minimums():
    g, m, r = _geom(), _mat(), _reinf(As_boundary=0.0)   # no boundary bars
    be = boundary_element_check(g, m, WallDemand(Pu=4e6, Mu=3e6))  # BE required
    assert be.required
    aci = wall_detailing("ACI 318-19", g, m, r, be)
    is13920 = wall_detailing("IS 13920", g, m, r, be)
    ec8 = wall_detailing("EC8", g, m, r, be)
    # ACI keys boundary detailing off ties (no ρ floor) → ok even with no bars
    assert aci.boundary_rho_min == 0.0 and aci.boundary_rho_ok
    # IS 13920 needs 0.8% boundary vertical → fails with no boundary bars
    assert is13920.boundary_rho_min == pytest.approx(0.008)
    assert not is13920.boundary_rho_ok
    # EC8 needs 0.5%
    assert ec8.boundary_rho_min == pytest.approx(0.005)
    assert not ec8.boundary_rho_ok
    assert set(WALL_CODES) == {"ACI 318-19", "IS 13920", "EC8"}


def test_design_wall_pier_code_and_drift_wired():
    g, m, r = _geom(lw=4.0, t=0.30, hw=24.0), _mat(), _reinf()
    d = WallDemand(Pu=1e6, Mu=0.2e6, Vu=200e3)
    res = design_wall_pier(g, m, r, d, code="IS 13920", drift=0.02)
    assert res.detailing.code == "IS 13920"
    assert res.boundary.drift == 0.02
    # a drift-triggered boundary element shows up in the notes
    if res.boundary.disp_required:
        assert any("drift" in n.lower() for n in res.notes)
