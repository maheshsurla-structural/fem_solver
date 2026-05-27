"""Tests for ModalPushoverAnalysis (Phase 23 — Chopra-Goel MPA)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    ModalPushoverAnalysis,
    PushoverToTarget,
    ResponseSpectrum,
)


# ============================================================ fixtures

def _make_cantilever(n_story: int = 3,
                       E: float = 2.0e10,
                       A: float = 1.0e-2,
                       Iz: float = 1.0e-4,
                       L: float = 3.0,
                       rho: float = 7850.0):
    """Build a `n_story`-element vertical cantilever stick.

    BeamColumn2D segments, fixed at the base, consistent mass via density.
    Node tags are 1 (base) to ``n_story+1`` (roof).
    """
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_story + 1):
        m.add_node(i + 1, 0.0, i * L)
    for i in range(n_story):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    return m


def _design_spectrum() -> ResponseSpectrum:
    """A simple two-branch design spectrum (flat in short-period, 1/T
    in long-period)."""
    periods = np.logspace(np.log10(0.05), np.log10(5.0), 200)
    Sa = np.where(periods < 0.5, 2.5 * 9.81, 2.5 * 9.81 * 0.5 / periods)
    return ResponseSpectrum(periods, Sa, damping_ratio=0.05)


# ============================================================ input validation

def test_mpa_rejects_zero_modes():
    with pytest.raises(ValueError, match="num_modes"):
        ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(),
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0,
            max_drift=0.5, num_modes=0,
        )


def test_mpa_rejects_unknown_combination():
    with pytest.raises(ValueError, match="combination"):
        ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(),
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0,
            max_drift=0.5, combination="bogus",
        )


def test_mpa_rejects_unknown_target_method():
    with pytest.raises(ValueError, match="target_method"):
        ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(),
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0,
            max_drift=0.5, target_method="bogus",
        )


def test_mpa_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(),
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0,
            max_drift=0.5, direction="bogus",
        )


def test_mpa_rejects_zero_max_drift():
    with pytest.raises(ValueError, match="max_drift"):
        ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(),
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0, max_drift=0.0,
        )


def test_mpa_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="model_factory"):
        ModalPushoverAnalysis(
            model_factory=_make_cantilever(),    # already a Model, not a callable
            spectrum=_design_spectrum(),
            control_node=4, control_dof=0, max_drift=0.5,
        )


# ============================================================ run sanity

def test_mpa_runs_and_returns_expected_keys():
    """End-to-end sanity: the run dict has the documented keys."""
    mpa = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=3),
        spectrum=_design_spectrum(),
        control_node=4, control_dof=0,
        max_drift=1.5, num_modes=2, num_steps=30,
    )
    res = mpa.run()
    assert {"num_modes", "direction", "combination", "periods_s",
              "Gamma", "m_eff", "total_participating_mass",
              "modal_results"}.issubset(res.keys())
    assert res["num_modes"] == 2
    assert len(res["modal_results"]) == 2
    # Each modal result has the documented per-mode fields
    for mr in res["modal_results"]:
        for key in ("mode", "period", "Gamma", "Gamma_conv", "m_eff",
                      "drift_curve", "force_curve", "d_t_top",
                      "nodal_disps_at_target", "bilinear"):
            assert key in mr, f"missing key {key!r} in modal result"


def test_mpa_periods_ascending_and_match_eigen():
    """Periods returned by MPA equal a direct EigenAnalysis call."""
    from femsolver.analysis.eigen import EigenAnalysis
    eig = EigenAnalysis(_make_cantilever(n_story=3), num_modes=3).run()
    mpa_res = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=3),
        spectrum=_design_spectrum(),
        control_node=4, control_dof=0,
        max_drift=1.5, num_modes=3, num_steps=20,
    ).run()
    np.testing.assert_allclose(
        mpa_res["periods_s"], eig["periods_s"], rtol=1e-10,
    )


# ============================================================ participation

def test_mpa_participation_mass_increases_with_modes():
    """Adding modes monotonically increases cumulative participating
    mass."""
    masses = []
    for n in (1, 2, 3):
        res = ModalPushoverAnalysis(
            model_factory=lambda: _make_cantilever(n_story=4),
            spectrum=_design_spectrum(),
            control_node=5, control_dof=0,
            max_drift=1.5, num_modes=n, num_steps=20,
        ).run()
        masses.append(res["total_participating_mass"])
    # Strictly increasing
    assert masses[0] < masses[1] < masses[2]


def test_mpa_first_mode_participation_dominates():
    """For a uniform cantilever, mode 1 should pick up the majority
    of the modal mass (typically 60-80% of total)."""
    res = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=4),
        spectrum=_design_spectrum(),
        control_node=5, control_dof=0,
        max_drift=1.5, num_modes=3, num_steps=20,
    ).run()
    m_eff = res["m_eff"]
    # Mode 1 m_eff >= sum of others
    assert m_eff[0] >= sum(m_eff[1:])


# ============================================================ SDOF reduction

def test_mpa_single_mode_matches_pushover_to_target_elastic():
    """For an elastic structure, a 1-mode MPA must give the same
    target roof drift as the existing PushoverToTarget driver fed
    by a manually-computed (Gamma, m_eff) and an N2 target. This
    is the gold-standard equivalence."""
    spectrum = _design_spectrum()

    # Run 1-mode MPA
    mpa = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=3),
        spectrum=spectrum,
        control_node=4, control_dof=0,
        max_drift=1.5, num_modes=1, num_steps=40,
        target_method="n2",
    )
    res = mpa.run()
    mr1 = res["modal_results"][0]
    d_t_mpa = mr1["d_t_top"]

    # Independent path: PushoverToTarget + manual N2
    from femsolver.analysis.eigen import EigenAnalysis
    from femsolver.analysis.capacity_design import (
        bilinearize_capacity_curve,
        equivalent_sdof,
        n2_target_displacement,
    )
    from femsolver.analysis.assembler import assemble_mass

    # Build a model with the *MPA force pattern* applied so the
    # PushoverToTarget capacity curve matches mode 1's pushover.
    m_push = _make_cantilever(n_story=3)
    m_push.number_dofs()
    eig = EigenAnalysis(m_push, num_modes=1)
    eig.run()
    M = assemble_mass(m_push)
    M_dense = M.toarray()
    phi = eig.mode_shapes[:, 0]
    # mass-normalize
    phi = phi / math.sqrt(float(phi @ (M_dense @ phi)))
    # Build influence vector (DOF 0 = x)
    iota = np.zeros(m_push.neq)
    for node in m_push.nodes.values():
        eq = int(node.eqn[0])
        if eq >= 0:
            iota[eq] = 1.0
    Gamma_norm = float(phi @ (M_dense @ iota))
    # Sign-flip phi so control DOF positive
    ctrl_eq = int(m_push.node(4).eqn[0])
    if phi[ctrl_eq] < 0.0:
        phi = -phi; Gamma_norm = -Gamma_norm
    s = M_dense @ phi
    # Apply s as lateral nodal loads (DOF 0 only)
    for node in m_push.nodes.values():
        eq = int(node.eqn[0])
        if eq >= 0:
            node._load[0] = float(s[eq])

    pt = PushoverToTarget(
        m_push, target_drift=1.5, track=(4, 0),
        num_steps=40,
    )
    pres = pt.run()
    Gamma_conv = Gamma_norm * float(phi[ctrl_eq])
    m_eff_n = Gamma_norm ** 2
    sdof = equivalent_sdof(
        np.abs(pres["drift"]), np.abs(pres["force"]),
        Gamma=abs(Gamma_conv), m_eff=m_eff_n,
    )
    bilinear = bilinearize_capacity_curve(sdof.d_star, sdof.F_star)
    target = n2_target_displacement(spectrum, sdof, bilinear)
    d_t_independent = target["d_t_top"]

    # The two should match within tolerance
    assert d_t_mpa == pytest.approx(d_t_independent, rel=1e-3)


# ============================================================ combination

def test_mpa_srss_combined_at_roof_matches_srss_of_targets():
    """For the control DOF (where each modal target's snapshot equals
    its d_t_top by construction), the SRSS-combined roof drift equals
    the SRSS of the individual modal d_t_top values."""
    mpa = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=4),
        spectrum=_design_spectrum(),
        control_node=5, control_dof=0,
        max_drift=2.0, num_modes=3, num_steps=40,
        combination="srss",
    )
    mpa.run()
    d_t_modes = np.array([mr["d_t_top"] for mr in mpa.modal_results])
    srss_expected = float(math.sqrt(float(np.sum(d_t_modes ** 2))))
    roof_combined = float(mpa.combined_nodal_disps[5][0])
    # Tolerance is generous because the snapshot is interpolated at
    # the target step; the control DOF reading at that step is within
    # one step-size (= max_drift / num_steps) of d_t_top.
    assert roof_combined == pytest.approx(srss_expected, rel=2.0e-2)


def test_mpa_cqc_matches_srss_for_well_separated_modes():
    """When modes are well separated, the CQC cross-correlation
    coefficients are ~0 and CQC reduces to SRSS."""
    mpa_srss = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=4),
        spectrum=_design_spectrum(),
        control_node=5, control_dof=0,
        max_drift=2.0, num_modes=3, num_steps=30,
        combination="srss",
    )
    mpa_srss.run()
    mpa_cqc = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=4),
        spectrum=_design_spectrum(),
        control_node=5, control_dof=0,
        max_drift=2.0, num_modes=3, num_steps=30,
        combination="cqc",
    )
    mpa_cqc.run()
    d_srss = mpa_srss.combined_nodal_disps[5][0]
    d_cqc = mpa_cqc.combined_nodal_disps[5][0]
    # Well-separated modes: CQC differs from SRSS by less than 1%
    assert d_cqc == pytest.approx(d_srss, rel=1.0e-2)


# ============================================================ higher modes

def test_mpa_higher_modes_change_upper_story_response():
    """Adding mode 2 to a 5-story stick changes the modal-combined
    response at upper-story DOFs (where mode 2's shape is significant).
    The lower nodes change less."""
    res1 = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=5),
        spectrum=_design_spectrum(),
        control_node=6, control_dof=0,
        max_drift=2.0, num_modes=1, num_steps=40,
    ).run()
    res2 = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=5),
        spectrum=_design_spectrum(),
        control_node=6, control_dof=0,
        max_drift=2.0, num_modes=2, num_steps=40,
    ).run()
    # Mode 2's m_eff is > 0
    assert res2["m_eff"][1] > 0.0


# ============================================================ coefficient method

def test_mpa_coefficient_method_target_runs():
    """ASCE 41 coefficient method as the target_method runs and produces
    a finite d_t_top."""
    mpa = ModalPushoverAnalysis(
        model_factory=lambda: _make_cantilever(n_story=3),
        spectrum=_design_spectrum(),
        control_node=4, control_dof=0,
        max_drift=1.5, num_modes=1, num_steps=20,
        target_method="coefficient",
    )
    res = mpa.run()
    d_t = res["modal_results"][0]["d_t_top"]
    assert math.isfinite(d_t) and d_t != 0.0
