"""Engine tests for the unified section-analysis API (plan §15 U3).

One result type + backend selector wrapping the four capacity models the
codebase already computes (exact / fibre M-phi; fibre / nominal / design P-M),
so the Section Designer tool, the tests, and the fiber-hinge stream consume the
*same* output instead of forking a parallel section engine.
"""
from __future__ import annotations

import math

import pytest

import section_gui_core
from femsolver.benchmarks.section_designer import ALL_SECTION_BUILDERS
from femsolver.sections.analysis import (
    C_DESIGN,
    C_EXACT,
    C_FIBRE,
    C_NOMINAL,
    MPHI_BACKENDS,
    PM_BACKENDS,
    MomentCurvatureResult,
    PMInteractionResult,
    exact_mphi,
    moment_curvature_analysis,
    mphi_data,
    pm_interaction,
    pmm_slice,
    section_pm_slice,
)


def _rc_case():
    """First standard verification section (an RC section with rebar)."""
    for build in ALL_SECTION_BUILDERS:
        case = build()
        if case.section.reinforcement and case.section.reinforcement.bars:
            return case
    raise AssertionError("no reinforced case in ALL_SECTION_BUILDERS")


# ------------------------------------------------------ backend tokens

def test_backend_tokens():
    assert (C_EXACT, C_FIBRE, C_NOMINAL, C_DESIGN) == (
        "exact", "fibre", "nominal", "design")
    assert MPHI_BACKENDS == (C_EXACT, C_FIBRE)
    assert PM_BACKENDS == (C_FIBRE, C_NOMINAL, C_DESIGN)


# ------------------------------------------------------ moment_curvature_analysis

@pytest.mark.parametrize("backend, fn", [(C_EXACT, exact_mphi),
                                         (C_FIBRE, mphi_data)])
def test_mphi_dispatch_matches_backend(backend, fn):
    case = _rc_case()
    r = moment_curvature_analysis(case, 0.0, backend=backend)
    d = fn(case, 0.0)
    assert isinstance(r, MomentCurvatureResult)
    assert r.backend == backend
    assert r.kappa == d["kappa"]
    assert r.M == d["M"]
    assert r.M_u == d["M_u"]
    assert r.raw is not None and r.raw["kappa"] == d["kappa"]
    # the unified result carries the milestone/failure metadata through
    assert r.failure_mode
    assert all(math.isfinite(m) for m in r.M)


def test_mphi_forwards_kwargs():
    """Axial compression is forwarded and raises the ultimate moment, just as
    the bare backend does."""
    case = _rc_case()
    m0 = moment_curvature_analysis(case, 0.0, backend=C_EXACT).M_u
    m_axial = moment_curvature_analysis(case, 500.0, backend=C_EXACT).M_u
    assert m_axial > m0


def test_mphi_rejects_pm_backend():
    case = _rc_case()
    with pytest.raises(ValueError):
        moment_curvature_analysis(case, 0.0, backend=C_NOMINAL)


# ------------------------------------------------------ pm_interaction

def test_pm_fibre_matches_section_pm_slice():
    case = _rc_case()
    r = pm_interaction(case, backend=C_FIBRE, n_points=40)
    d = section_pm_slice(case, n_points=40)
    assert isinstance(r, PMInteractionResult)
    assert r.backend == C_FIBRE
    assert r.code is None
    assert r.has_design is False
    assert r.landmarks is None
    assert r.P == d["P"] and r.M == d["M"]


@pytest.mark.parametrize("backend, pkey, mkey",
                         [(C_NOMINAL, "P_nom", "M_nom"),
                          (C_DESIGN, "P_des", "M_des")])
def test_pm_code_matches_pmm_slice(backend, pkey, mkey):
    case = _rc_case()
    code = "AASHTO LRFD 2024"
    r = pm_interaction(case, backend=backend, code=code, n_points=40)
    curve, lm = pmm_slice(case, code, n=40)
    assert r.backend == backend
    assert r.code == code
    assert r.P == curve[pkey]
    assert r.M == curve[mkey]
    assert r.has_design == curve["has_design"]
    assert r.landmarks == lm


def test_pm_rejects_mphi_backend():
    case = _rc_case()
    with pytest.raises(ValueError):
        pm_interaction(case, backend=C_EXACT)


# ------------------------------------------------------ GUI shim identity (U3)

def test_gui_reexports_lifted_pmm_slice():
    """The Section Designer GUI now re-exports the lifted engine function
    rather than owning a private copy (same U1 shim pattern)."""
    assert section_gui_core.pmm_slice is pmm_slice
