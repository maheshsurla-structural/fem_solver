"""P2 — Mander confinement calculator, verified against the Midas benchmark.

The core `mander_confinement` was lifted into the engine in U1; P2 verifies it
against the Midas MANDER card for the benchmark 84 in Caltrans column (plan
§2.2) and adds the typed, unit-agnostic `mander_confined_circular` wrapper the
fiber-hinge stream uses (§5.2 / §15 U4 — one confinement calc for the section
tool and the hinge).

Midas card (line 304 of the axial-load MCT), f'c = 5 ksi, core dia 79 in,
#8 hoop @ 6 in, f_yh = 68 ksi:
    fcc' = 6.35571 ksi   eps_cc = 0.00522731   ke = 0.962451   fl = 0.218155 ksi
"""
from __future__ import annotations

import math

import pytest

from femsolver.materials.uniaxial import ConcreteMander
from femsolver.sections.analysis import (
    ConfinedCircular,
    mander_confined_circular,
    mander_confinement,
)

# Midas benchmark targets (kip, in)
FCC_MIDAS = 6.35571
ECC_MIDAS = 0.00522731
KE_MIDAS = 0.962451
FL_MIDAS = 0.218155


def _benchmark():
    """84 in Caltrans column core in kip-in units (circular hoops)."""
    return mander_confined_circular(
        fco=5.0, eps_co=0.002219,
        D_core=79.0, hoop_area=0.79, hoop_spacing=6.0, fyh=68.0,
        rho_long=0.0257055, hoop_type="hoop", clear_spacing=5.0,
        eps_su_hoop=0.09,
    )


# ------------------------------------------------------ benchmark verification

def test_confined_strength_matches_midas():
    r = _benchmark()
    assert r.fcc == pytest.approx(FCC_MIDAS, rel=0.02)      # 0.24% actual
    assert r.eps_cc == pytest.approx(ECC_MIDAS, rel=0.02)   # 0.64% actual


def test_ke_and_fl_match_midas_exactly():
    r = _benchmark()
    assert r.ke == pytest.approx(KE_MIDAS, rel=1e-4)
    assert r.fl == pytest.approx(FL_MIDAS, rel=1e-3)


def test_confinement_strengthens():
    r = _benchmark()
    assert r.fcc > 5.0                    # confinement raises peak stress
    assert r.eps_cc > 0.002219           # and peak strain
    assert r.eps_cu > r.eps_cc
    assert 0.0 < r.ke <= 1.0


# ------------------------------------------------------ wrapper behaviour

def test_to_material_peak_is_fcc():
    r = _benchmark()
    mat = r.to_material(Ec=3605.0)
    assert isinstance(mat, ConcreteMander)
    sig, _ = mat.get_response(-r.eps_cc)      # at the confined peak strain
    assert -sig == pytest.approx(r.fcc, rel=0.02)


def test_unit_agnostic_kipin_vs_si():
    """Ratios/strains are unit-free: SI inputs give the same fcc/f'c ratio."""
    KSI, IN = 6.894757e6, 0.0254
    si = mander_confined_circular(
        fco=5 * KSI, eps_co=0.002219, D_core=79 * IN, hoop_area=0.79 * IN * IN,
        hoop_spacing=6 * IN, fyh=68 * KSI, rho_long=0.0257055,
        hoop_type="hoop", clear_spacing=5 * IN,
    )
    kipin = _benchmark()
    assert si.fcc / (5 * KSI) == pytest.approx(kipin.fcc / 5.0, rel=1e-9)
    assert si.eps_cc == pytest.approx(kipin.eps_cc, rel=1e-9)
    assert si.ke == pytest.approx(kipin.ke, rel=1e-9)


def test_hoop_vs_spiral():
    common = dict(fco=5.0, eps_co=0.002219, D_core=79.0, hoop_area=0.79,
                  hoop_spacing=6.0, fyh=68.0, rho_long=0.0257055,
                  clear_spacing=5.0)
    hoop = mander_confined_circular(hoop_type="hoop", **common)
    spiral = mander_confined_circular(hoop_type="spiral", **common)
    # spiral is more effective (exponent 1 vs 2) -> higher ke, fcc
    assert spiral.ke > hoop.ke
    assert spiral.fcc > hoop.fcc


def test_negligible_hoop_gives_unconfined():
    r = mander_confined_circular(
        fco=5.0, eps_co=0.002219, D_core=79.0, hoop_area=1e-9,
        hoop_spacing=6.0, fyh=68.0,
    )
    assert r.fcc == pytest.approx(5.0, rel=1e-3)   # no confinement -> f'c


# ------------------------------------------------------ dict API still works

def test_dict_api_rectangular_still_runs():
    """The GUI-facing dict API (used by the Section Designer) still computes
    a rectangular core -- guards the shared calculator."""
    r = mander_confinement({
        "fc": 5.0, "eps_c0": 0.002219, "conf_shape": "Rectangular",
        "conf_fyh": 68.0, "conf_Asp": 0.79, "conf_s": 6.0, "conf_sp": 5.0,
        "conf_bc": 30.0, "conf_dc": 40.0, "conf_ny": 3, "conf_nz": 3,
        "conf_nlong": 12, "conf_rho_cc": 0.02, "conf_ecu_method": "experiment",
    })
    assert r["fcc"] > 5.0
    assert math.isfinite(r["eps_cu"])
