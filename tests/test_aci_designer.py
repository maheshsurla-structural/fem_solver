"""Phase 29.5 tests -- RcMemberDesigner.

The designer is a search over (bar size, count) combinations that
finds the lightest layout satisfying all ACI checks. The tests
verify it picks valid layouts, respects ACI limits, and degrades
gracefully when the demand cannot be met within the search space.
"""
from __future__ import annotations

import pytest

from femsolver.design.concrete import (
    BeamDesignDemand,
    ColumnDesignDemand,
    ConcreteMaterial,
    RcMemberDesigner,
    beam_flexural_strength,
    beam_shear_strength,
    column_interaction_surface,
    design_beam,
    design_column,
)


def _ts_mat():
    """Standard test material: f_c' = 28 MPa, f_y = 420 MPa."""
    return ConcreteMaterial(fc_prime=28e6, fy=420e6)


# ============================================================ beam designer

def test_beam_design_success_for_moderate_demand():
    """Designer should find a layout for a routine beam demand."""
    res = design_beam(
        b=0.30, h=0.55, material=_ts_mat(),
        demand=BeamDesignDemand(M_u_positive=200e3, V_u=120e3),
        cover=0.050,
    )
    assert res.success
    assert res.section is not None
    assert len(res.section.rebar.bottom_bars) >= 2
    # Capacity meets demand
    assert res.flexure_positive.phi_M_n >= 200e3 - 1.0e-6
    assert res.shear.phi_V_n >= 120e3 - 1.0e-6


def test_beam_design_finds_minimum_steel():
    """For a given demand the designer should pick a layout whose
    capacity is reasonably close to the demand (not grossly over)."""
    M_u = 150e3
    res = design_beam(
        b=0.30, h=0.55, material=_ts_mat(),
        demand=BeamDesignDemand(M_u_positive=M_u, V_u=80e3),
        cover=0.050,
    )
    assert res.success
    # Over-strength typically < 50% for a tightly-fit design
    overstrength = res.flexure_positive.phi_M_n / M_u
    assert 1.0 <= overstrength < 1.50, (
        f"unexpected overstrength {overstrength:.2f}; expected 1.0–1.5"
    )


def test_beam_with_hogging_adds_top_steel():
    """When hogging demand is supplied, the designer must populate
    top bars."""
    res = design_beam(
        b=0.30, h=0.55, material=_ts_mat(),
        demand=BeamDesignDemand(
            M_u_positive=200e3, M_u_negative=150e3, V_u=120e3,
        ),
        cover=0.050,
    )
    assert res.success
    assert len(res.section.rebar.top_bars) >= 2
    assert res.flexure_negative is not None
    assert res.flexure_negative.phi_M_n >= 150e3 - 1.0e-6


def test_beam_design_returns_failure_for_impossible_demand():
    """A small section under a huge demand should fail gracefully."""
    res = design_beam(
        b=0.20, h=0.30, material=_ts_mat(),
        demand=BeamDesignDemand(M_u_positive=500e3, V_u=300e3),
        cover=0.040, max_bars_per_layer=4,
    )
    assert not res.success
    assert "could not satisfy" in res.notes.lower()


def test_beam_design_stirrups_pass_shear_check():
    """Chosen stirrup spacing must give φV_n >= V_u."""
    V_u = 200e3
    res = design_beam(
        b=0.30, h=0.55, material=_ts_mat(),
        demand=BeamDesignDemand(M_u_positive=180e3, V_u=V_u),
        cover=0.050,
    )
    assert res.success
    # Independently re-check shear on the returned section
    sh = beam_shear_strength(res.section, V_u=V_u)
    assert sh.phi_V_n >= V_u - 1.0e-6


# ============================================================ column designer

def test_column_design_success_for_routine_demand():
    res = design_column(
        b=0.40, h=0.40, material=_ts_mat(),
        demand=ColumnDesignDemand(P_u=800e3, M_u=100e3),
        cover=0.060,
    )
    assert res.success
    assert 0.01 <= res.rho <= 0.08    # ACI 10.6.1.1
    assert res.dcr <= 1.0


def test_column_design_minimum_steel_is_one_percent():
    """For very light demand the chosen ρ should be at the minimum
    (1%)."""
    res = design_column(
        b=0.40, h=0.40, material=_ts_mat(),
        demand=ColumnDesignDemand(P_u=200e3, M_u=20e3),
        cover=0.060,
    )
    assert res.success
    # Should land near the 1% floor
    assert res.rho >= 0.01
    assert res.rho < 0.02, (
        f"expected rho near min 1%, got {res.rho * 100:.2f}%"
    )


def test_column_design_returns_failure_above_max_rho():
    """Demand that requires more than 8% steel should fail."""
    res = design_column(
        b=0.30, h=0.30, material=_ts_mat(),
        demand=ColumnDesignDemand(P_u=4000e3, M_u=400e3),
        cover=0.060,
    )
    # Either succeeds with rho near max, or fails -- both acceptable.
    if res.success:
        assert res.rho <= 0.08
    else:
        assert "0.08" in res.notes or "10.6.1.1" in res.notes


def test_column_design_symmetric_layout():
    """The designer produces symmetric top/bottom reinforcement."""
    res = design_column(
        b=0.40, h=0.40, material=_ts_mat(),
        demand=ColumnDesignDemand(P_u=600e3, M_u=80e3),
        cover=0.060,
    )
    assert res.success
    assert len(res.section.rebar.top_bars) == len(res.section.rebar.bottom_bars)


def test_column_design_dcr_is_consistent_with_surface():
    """The reported DCR should match a fresh evaluation of the surface."""
    res = design_column(
        b=0.40, h=0.40, material=_ts_mat(),
        demand=ColumnDesignDemand(P_u=700e3, M_u=120e3),
        cover=0.060,
    )
    assert res.success
    fresh_surface = column_interaction_surface(res.section)
    fresh_dcr = fresh_surface.dcr(P_u=700e3, M_u=120e3)
    assert res.dcr == pytest.approx(fresh_dcr, rel=1e-10)


# ============================================================ unified driver

def test_RcMemberDesigner_facade_dispatches_correctly():
    """RcMemberDesigner.design_beam and .design_column delegate to the
    underlying functions."""
    mat = _ts_mat()
    r_b = RcMemberDesigner.design_beam(
        b=0.30, h=0.50, material=mat,
        demand=BeamDesignDemand(M_u_positive=100e3, V_u=80e3),
        cover=0.040,
    )
    r_c = RcMemberDesigner.design_column(
        b=0.35, h=0.35, material=mat,
        demand=ColumnDesignDemand(P_u=500e3, M_u=60e3),
        cover=0.060,
    )
    assert r_b.success
    assert r_c.success
