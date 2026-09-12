"""Engine tests for the lifted section-analysis core (plan §15 U1).

These exercise `femsolver.sections.analysis` directly — the exact-integration
moment-curvature / P-M slice and the material-from-spec + Mander confinement
helpers that were relocated out of the Section Designer GUI file. Before U1
this code lived only in `section_gui_core.py` and had no engine tests.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.benchmarks.section_designer import ALL_SECTION_BUILDERS
from femsolver.sections.analysis import (
    concrete_uniaxial_from,
    exact_mphi,
    mander_confinement,
    section_pm_slice,
    steel_uniaxial_from,
)


def _rc_case():
    """First standard verification section (an RC section with rebar)."""
    for build in ALL_SECTION_BUILDERS:
        case = build()
        if case.section.reinforcement and case.section.reinforcement.bars:
            return case
    raise AssertionError("no reinforced case in ALL_SECTION_BUILDERS")


# ------------------------------------------------------ exact_mphi

def test_exact_mphi_milestones_ordered():
    case = _rc_case()
    r = exact_mphi(case, 0.0)
    assert len(r["kappa"]) > 5
    assert r["M_cr"] > 0.0
    assert r["M_u"] > 0.0
    # cracking precedes yield precedes ultimate (curvatures increasing)
    assert 0.0 < r["kappa_cr"]
    if r["kappa_y"] is not None:
        assert r["kappa_cr"] < r["kappa_y"] <= r["kappa_u"]
        assert r["M_cr"] < r["M_y"] <= r["M_u"] * 1.0001
    assert r["failure_mode"]
    assert all(math.isfinite(m) for m in r["M"])


def test_exact_mphi_axial_raises_moment_capacity():
    """Modest axial compression should raise the ultimate moment of a
    column-type RC section (P-M interaction on the compression-controlled
    side is not in play at low axial load)."""
    case = _rc_case()
    m0 = exact_mphi(case, 0.0)["M_u"]
    m_axial = exact_mphi(case, 500.0)["M_u"]   # 500 kN compression
    assert m_axial > m0


# ------------------------------------------------------ section_pm_slice

def test_pm_slice_shape():
    case = _rc_case()
    pm = section_pm_slice(case)
    assert len(pm["P"]) == len(pm["M"]) > 3
    assert max(pm["P"]) > 0.0            # squash/compression capacity
    assert max(pm["M"]) > 0.0            # balanced-ish bending capacity
    assert all(math.isfinite(x) for x in pm["P"] + pm["M"])


# ------------------------------------------------------ mander_confinement

def test_mander_confinement_circular():
    r = mander_confinement({
        "fc": 35e6, "eps_c0": 0.002, "conf_shape": "Circular",
        "conf_fyh": 400e6, "conf_Asp": 1.0e-4, "conf_s": 0.1,
        "conf_ds": 0.4, "conf_hooptype": "Spiral", "conf_rho_cc": 0.02,
    })
    assert r["fcc"] > 35e6                 # confinement strengthens
    assert r["eps_cc"] > 0.002             # and raises the peak strain
    assert r["eps_cu"] > r["eps_cc"]
    assert 0.0 < r["ke"] <= 1.0
    assert r["fl"] > 0.0


def test_mander_no_confinement_returns_base():
    r = mander_confinement({
        "fc": 30e6, "eps_c0": 0.002, "conf_shape": "Circular",
        "conf_fyh": 400e6, "conf_Asp": 0.0, "conf_s": 0.1, "conf_ds": 0.4,
    })
    assert r["fcc"] == pytest.approx(30e6, rel=1e-9)   # no ties -> unconfined


# ------------------------------------------------------ material factories

def test_concrete_factory_compression_negative():
    c = concrete_uniaxial_from({"fc": 30e6, "conc_model": "Kent-Park",
                                "eps_c0": 0.002, "eps_cu": 0.0035})
    sig, Et = c.get_response(-0.001)       # compression
    assert sig < 0.0
    assert c.get_response(0.0)[0] == pytest.approx(0.0, abs=1.0)


def test_steel_factory_yields():
    s = steel_uniaxial_from({"fy": 500e6, "Es": 200e9,
                             "steel_model": "Elastic - perfectly plastic"})
    sig, Et = s.get_response(0.01)         # well past yield
    assert sig == pytest.approx(500e6, rel=1e-6)
    assert Et == pytest.approx(0.0, abs=1e3)
