"""Phase 30.6 tests -- SteelMemberDesigner: aggregated checks +
auto-sizing.
"""
from __future__ import annotations

import math

import pytest

from femsolver.design.steel import (
    SteelMemberCheck,
    SteelMemberDemand,
    SteelMemberDesigner,
    astm_a992,
    auto_size,
    check_member,
    combined_force_check,
    get_section,
    shear_strength,
    w_series,
)


IN = 0.0254
FT = 12 * IN
KIP = 4448.222
KIPFT = KIP * FT


# ============================================================ check_member

def test_check_member_governing_DCR_is_max_of_combined_and_shear():
    """When both combined and shear are active, governing DCR is the larger."""
    sec = get_section("W14x90"); mat = astm_a992()
    demand = SteelMemberDemand(P_u=300 * KIP, M_ux=100 * KIPFT,
                                  V_u=50 * KIP)
    chk = check_member(sec, mat, demand, L=14 * FT)
    # Re-compute the underlying terms independently
    cfc = combined_force_check(
        sec, mat,
        P_r=demand.P_u, M_rx=demand.M_ux, L=14 * FT, L_b=14 * FT,
    )
    shr = shear_strength(sec, mat)
    shear_dcr = demand.V_u / shr.phi_V_n
    expected = max(cfc.DCR, shear_dcr)
    assert chk.governing_DCR == pytest.approx(expected, rel=1e-10)


def test_check_member_pure_axial_no_shear():
    """Demand with only P_u: no shear check, governing = combined."""
    sec = get_section("W14x90"); mat = astm_a992()
    demand = SteelMemberDemand(P_u=300 * KIP)
    chk = check_member(sec, mat, demand, L=14 * FT)
    assert chk.shear is None
    assert chk.combined is not None
    assert chk.governing_limit_state == "combined"


def test_check_member_pure_shear():
    """Demand with only V_u: no combined check, governing = shear."""
    sec = get_section("W14x90"); mat = astm_a992()
    demand = SteelMemberDemand(V_u=100 * KIP)
    chk = check_member(sec, mat, demand, L=14 * FT)
    assert chk.combined is None
    assert chk.shear is not None
    assert chk.governing_limit_state == "shear"


def test_check_member_passes_for_low_demand():
    """A heavy section under light demand: governing DCR << 1, passes."""
    sec = get_section("W36x150"); mat = astm_a992()
    demand = SteelMemberDemand(P_u=100 * KIP, M_ux=50 * KIPFT)
    chk = check_member(sec, mat, demand, L=10 * FT)
    assert chk.governing_DCR < 0.5
    assert chk.passes


def test_check_member_fails_for_overload():
    """A small section under huge demand: DCR > 1, fails."""
    sec = get_section("W6x9"); mat = astm_a992()
    demand = SteelMemberDemand(P_u=500 * KIP, M_ux=200 * KIPFT)
    chk = check_member(sec, mat, demand, L=14 * FT)
    assert chk.governing_DCR > 1.0
    assert not chk.passes


def test_check_member_returns_section_and_weight():
    sec = get_section("W14x90")
    chk = check_member(sec, astm_a992(),
                        SteelMemberDemand(P_u=200 * KIP), L=14 * FT)
    assert chk.section is sec
    assert chk.weight_per_length == sec.weight_per_length


# ============================================================ auto_size

def test_auto_size_picks_lightest_passing_section():
    """The returned 'best' section is the lightest among all passing
    candidates."""
    demand = SteelMemberDemand(P_u=300 * KIP, M_ux=100 * KIPFT,
                                  V_u=50 * KIP)
    res = auto_size(astm_a992(), demand, L=14 * FT)
    assert res.success
    assert res.best is not None
    # 'best' has the lowest weight per length among all passing
    for c in res.all_passing:
        assert c.weight_per_length >= res.best.weight_per_length


def test_auto_size_all_passing_sorted_by_weight():
    """all_passing is sorted lightest-to-heaviest."""
    demand = SteelMemberDemand(P_u=300 * KIP, M_ux=100 * KIPFT)
    res = auto_size(astm_a992(), demand, L=14 * FT)
    weights = [c.weight_per_length for c in res.all_passing]
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1]


def test_auto_size_with_w_series_constraint():
    """Restricting to a W-series limits the search."""
    demand = SteelMemberDemand(P_u=600 * KIP, M_ux=200 * KIPFT)
    res = auto_size(
        astm_a992(), demand, L=14 * FT,
        candidates=w_series("W14"),
    )
    assert res.success
    # All passing candidates are W14
    for c in res.all_passing:
        assert c.section.designation.startswith("W14")


def test_auto_size_fails_for_impossible_demand():
    """Far-beyond-catalog demand fails gracefully."""
    demand = SteelMemberDemand(P_u=10000 * KIP, M_ux=2000 * KIPFT)
    res = auto_size(astm_a992(), demand, L=14 * FT)
    assert not res.success
    assert res.best is None
    assert "no section" in res.notes.lower()


def test_auto_size_n_candidates_reflects_catalog():
    """n_candidates_tested matches the input list size."""
    catalog = w_series("W14")
    res = auto_size(
        astm_a992(),
        SteelMemberDemand(P_u=100 * KIP),
        L=14 * FT,
        candidates=catalog,
    )
    assert res.n_candidates_tested == len(catalog)


def test_auto_size_empty_candidates_returns_failure():
    res = auto_size(
        astm_a992(),
        SteelMemberDemand(P_u=100 * KIP),
        L=14 * FT,
        candidates=[],
    )
    assert not res.success
    assert "empty" in res.notes.lower()


# ============================================================ best section near DCR=1

def test_lightest_passing_has_DCR_close_to_one():
    """For a 'tight' demand, the lightest passing section should have
    a DCR not much below 1.0 -- otherwise it could be lighter."""
    # Pick a demand calibrated against a known section, then look for
    # the lightest. The lightest passing should have DCR > 0.5 (not
    # grossly over-designed) for most reasonable demands.
    demand = SteelMemberDemand(M_ux=300 * KIPFT, V_u=60 * KIP)
    res = auto_size(astm_a992(), demand, L=20 * FT, L_b=20 * FT)
    assert res.success
    # If there were a lighter section in the catalog, it would have
    # been picked. So the 'best' should be close to DCR=1 (not 0.3).
    assert res.best.governing_DCR > 0.5


# ============================================================ facade

def test_facade_methods_dispatch():
    res = SteelMemberDesigner.auto_size(
        material=astm_a992(),
        demand=SteelMemberDemand(P_u=300 * KIP, M_ux=100 * KIPFT),
        L=14 * FT,
    )
    assert res.success
    chk = SteelMemberDesigner.check_member(
        res.best.section, astm_a992(),
        SteelMemberDemand(P_u=300 * KIP, M_ux=100 * KIPFT),
        L=14 * FT,
    )
    assert chk.governing_DCR == pytest.approx(res.best.governing_DCR,
                                                  rel=1e-10)


# ============================================================ demand helpers

def test_demand_has_axial_or_flexure_flag():
    d = SteelMemberDemand(P_u=1.0)
    assert d.has_axial_or_flexure
    assert not d.has_shear
    d2 = SteelMemberDemand(V_u=1.0)
    assert not d2.has_axial_or_flexure
    assert d2.has_shear
    d3 = SteelMemberDemand()
    assert not d3.has_axial_or_flexure
    assert not d3.has_shear
