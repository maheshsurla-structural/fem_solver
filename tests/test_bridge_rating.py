"""AASHTO LRFR load rating — MBE §6A (bridge plan T2.3).

Validates the rating-factor arithmetic against closed-form hand calculations
(Strength-I inventory/operating), the load-/condition-/system-factor tables,
the capacity floor, the tons conversion, and the tie-in to the moving-load
influence-line engine.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       Model)
from femsolver.bridges import (BridgeRating, condition_factor,  # noqa: E402
                               legal_live_load_factor, legal_load,
                               live_load_effect, permit_load,
                               rate_from_influence_line, rate_member,
                               rating_factor, strength_i_inventory,
                               strength_i_operating, system_factor)
from femsolver.bridges.moving_load import (BeamForce,  # noqa: E402
                                           InfluenceLineEngine, Lane,
                                           aashto_hl93_envelope)


# --------------------------------------------------------------- hand calc
# Rn=1000, DC=200, DW=50, LL+IM=300 (consistent units)
_RN, _DC, _DW, _LL = 1000.0, 200.0, 50.0, 300.0
# numerator C - 1.25*DC - 1.50*DW = 1000 - 250 - 75 = 675
_NUM = 675.0


def test_inventory_rating_factor_hand_calc():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                      factors=strength_i_inventory())
    assert r.level == "inventory"
    assert r.rf == pytest.approx(_NUM / (1.75 * _LL))       # 675/525
    assert r.rf == pytest.approx(1.285714, rel=1e-5)
    assert r.capacity == pytest.approx(_RN)
    assert r.dead_load_effect == pytest.approx(1.25 * _DC + 1.50 * _DW)
    assert r.live_load_effect == pytest.approx(1.75 * _LL)
    assert r.adequate                                       # RF > 1


def test_operating_rating_factor_hand_calc():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                      factors=strength_i_operating())
    assert r.rf == pytest.approx(_NUM / (1.35 * _LL))       # 675/405
    assert r.rf == pytest.approx(1.666667, rel=1e-5)
    # operating always rates higher than inventory (smaller gamma_LL)
    inv = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                        factors=strength_i_inventory())
    assert r.rf > inv.rf


def test_deficient_when_rf_below_one():
    # small capacity → deficient
    r = rating_factor(Rn=400.0, DC=_DC, DW=_DW, LL_IM=_LL,
                      factors=strength_i_inventory())
    assert not r.adequate
    assert 0.0 <= r.rf < 1.0


def test_phi_factors_scale_capacity():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL, phi=0.9,
                      phi_c=0.95, phi_s=1.0,
                      factors=strength_i_inventory())
    C = 0.95 * 1.0 * 0.9 * _RN
    assert r.capacity == pytest.approx(C)
    assert not r.capacity_floored
    assert r.rf == pytest.approx((C - 325.0) / (1.75 * _LL))


def test_capacity_floor_phi_c_phi_s():
    # phi_c*phi_s = 0.85*0.90 = 0.765 < 0.85 → floored to 0.85
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL, phi=1.0,
                      phi_c=0.85, phi_s=0.90,
                      factors=strength_i_inventory())
    assert r.capacity_floored
    assert r.capacity == pytest.approx(0.85 * _RN)


def test_zero_live_load_is_infinite_rating():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=0.0,
                      factors=strength_i_inventory())
    assert math.isinf(r.rf)
    assert r.adequate


def test_negative_live_effect_uses_magnitude():
    # a sagging (max) and hogging (min) effect of equal magnitude rate equally
    pos = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                        factors=strength_i_inventory())
    neg = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=-_LL,
                        factors=strength_i_inventory())
    assert neg.rf == pytest.approx(pos.rf)


# ------------------------------------------------------ permanent load P
def test_permanent_load_reduces_capacity_demand():
    base = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                         factors=strength_i_inventory())
    withP = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL, P=40.0,
                          factors=strength_i_inventory())
    # unfavourable P (positive) lowers the RF by gamma_P*P/(gamma_LL*LL)
    assert withP.rf < base.rf
    assert withP.dead_load_effect == pytest.approx(
        base.dead_load_effect + 1.0 * 40.0)


# ------------------------------------------------------ factor tables
def test_legal_live_load_factor_interpolation():
    assert legal_live_load_factor(5000) == pytest.approx(1.80)
    assert legal_live_load_factor(1000) == pytest.approx(1.65)
    assert legal_live_load_factor(100) == pytest.approx(1.40)
    assert legal_live_load_factor(50) == pytest.approx(1.40)     # clamp low
    assert legal_live_load_factor(20000) == pytest.approx(1.80)  # clamp high
    assert legal_live_load_factor(None) == pytest.approx(1.80)   # unknown ADTT
    # midway 100→1000: 1.40 + 0.25*(550-100)/900 = 1.525
    assert legal_live_load_factor(550) == pytest.approx(1.525)
    # midway 1000→5000: 1.65 + 0.15*(3000-1000)/4000 = 1.725
    assert legal_live_load_factor(3000) == pytest.approx(1.725)


def test_legal_and_permit_load_factor_sets():
    lg = legal_load(adtt=1000)
    assert lg.level == "legal" and lg.gamma_LL == pytest.approx(1.65)
    assert lg.gamma_DC == 1.25 and lg.gamma_DW == 1.50
    pm = permit_load(1.15)
    assert pm.level == "permit" and pm.gamma_LL == pytest.approx(1.15)


def test_condition_factor_table():
    assert condition_factor(nbi_rating=7) == 1.00
    assert condition_factor(nbi_rating=5) == 0.95
    assert condition_factor(nbi_rating=3) == 0.85
    assert condition_factor(condition="fair") == 0.95
    assert condition_factor(condition="Poor") == 0.85


def test_system_factor_table():
    assert system_factor("girder") == 1.00
    assert system_factor("welded_two_girder") == 0.85
    assert system_factor("four_girder") == 0.95
    assert system_factor("welded_two_girder", shear=True) == 1.00  # shear
    with pytest.raises(ValueError):
        system_factor("nonsense")


def test_field_measured_wearing_surface_uses_1p25():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                      factors=strength_i_inventory(gamma_DW=1.25))
    assert r.dead_load_effect == pytest.approx(1.25 * _DC + 1.25 * _DW)


# ------------------------------------------------------ tons + multi-level
def test_rating_in_tons():
    r = rating_factor(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL,
                      factors=strength_i_operating())
    # RT = RF * W ; HS-20 truck ≈ 36 tons
    assert r.rating_tons(36.0) == pytest.approx(r.rf * 36.0)


def test_rate_member_all_levels_and_controlling():
    br = rate_member(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL, phi=1.0,
                     adtt=1000, permit_gamma_LL=1.15)
    assert isinstance(br, BridgeRating)
    assert set(br.results) == {"inventory", "operating", "legal", "permit"}
    # inventory (gamma_LL 1.75) is the most demanding → lowest RF → controls
    assert br.controlling.level == "inventory"
    assert br["operating"].rf > br["inventory"].rf
    # permit (gamma_LL 1.15) rates highest
    assert br["permit"].rf > br["operating"].rf


def test_rate_member_defaults_to_design_levels_only():
    br = rate_member(Rn=_RN, DC=_DC, DW=_DW, LL_IM=_LL)
    assert set(br.results) == {"inventory", "operating"}


# ------------------------------------------------ tie-in to moving-load engine
def _ss_beam(L=20.0, nel=20, A=0.5, I=0.2, E=30e9):
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    return m


def test_rate_from_influence_line_matches_direct():
    """The IL-driven rating uses the HL-93 envelope for LL+IM and then the
    same rating arithmetic — so it must equal a direct rating_factor call fed
    the envelope's governing magnitude."""
    nel = 20
    m = _ss_beam(nel=nel)
    eng = InfluenceLineEngine(m)
    lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
    il = eng.influence_line(
        lane, BeamForce(element_tag=nel // 2, component="M", end="j"))

    env = aashto_hl93_envelope(il)
    ll_im = max(abs(env["max"]), abs(env["min"]))
    assert ll_im > 0.0
    assert live_load_effect(il) == pytest.approx(ll_im)

    # capacity chosen so the bridge is comfortably adequate at inventory
    Rn = 4.0 * ll_im
    DC, DW = 0.5 * ll_im, 0.1 * ll_im
    br = rate_from_influence_line(il, Rn=Rn, DC=DC, DW=DW, adtt=5000)
    direct = rate_member(Rn=Rn, DC=DC, DW=DW, LL_IM=ll_im, adtt=5000)
    for lvl in ("inventory", "operating", "legal"):
        assert br[lvl].rf == pytest.approx(direct[lvl].rf)
    assert br["inventory"].adequate


def test_live_load_effect_sense_selects_envelope_end():
    nel = 20
    m = _ss_beam(nel=nel)
    eng = InfluenceLineEngine(m)
    lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
    il = eng.influence_line(
        lane, BeamForce(element_tag=nel // 2, component="M", end="j"))
    env = aashto_hl93_envelope(il)
    assert live_load_effect(il, sense="max") == pytest.approx(abs(env["max"]))
    assert live_load_effect(il, sense="min") == pytest.approx(abs(env["min"]))
    assert live_load_effect(il, sense="governing") == pytest.approx(
        max(abs(env["max"]), abs(env["min"])))
