"""Tests for Phase 19 capacity-design utilities."""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BilinearCurve,
    ElasticIsotropic,
    EquivalentSDOF,
    Model,
    PushoverToTarget,
    ResponseSpectrum,
    bilinearize_capacity_curve,
    coefficient_method_target,
    equivalent_sdof,
    n2_target_displacement,
    seismic_combination,
    story_drifts,
)


# ====================================================== bilinearization

def test_bilinearize_perfectly_plastic_curve():
    """A perfectly elastic-plastic capacity curve recovers (K_i, d_y, F_y)
    cleanly: yield at d=0.01 with F=100, plateau out to d=0.05."""
    drift = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
    force = np.array([0.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    bl = bilinearize_capacity_curve(drift, force)
    assert bl.K_i == pytest.approx(100.0 / 0.01, rel=1e-3)    # 10000
    assert bl.F_y == pytest.approx(100.0, rel=0.1)
    assert bl.d_y == pytest.approx(0.01, rel=0.2)
    assert abs(bl.alpha) < 0.05


def test_bilinearize_returns_bilinear_curve_dataclass():
    drift = np.array([0.0, 0.01, 0.02, 0.03])
    force = np.array([0.0, 100.0, 100.0, 100.0])
    bl = bilinearize_capacity_curve(drift, force)
    assert isinstance(bl, BilinearCurve)
    # force_at uses the bilinear envelope
    f_below = bl.force_at(0.005)
    assert f_below == pytest.approx(0.5 * bl.K_i * 0.01, rel=0.2)


def test_bilinearize_rejects_unsorted_drift():
    drift = np.array([0.0, 0.02, 0.01, 0.03])
    force = np.array([0.0, 100.0, 50.0, 120.0])
    with pytest.raises(ValueError, match="sorted ascending"):
        bilinearize_capacity_curve(drift, force)


def test_bilinearize_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        bilinearize_capacity_curve([0.0, 0.01], [0.0, 50.0, 100.0])


def test_bilinearize_descending_branch():
    """Curve with strength loss: descending branch past peak. The
    bilinearization should still return a sensible result."""
    drift = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    force = np.array([0.0, 60.0, 100.0, 95.0, 85.0, 70.0])
    bl = bilinearize_capacity_curve(drift, force)
    # Yield point is somewhere between zero and the peak
    assert 0.0 < bl.d_y < 0.04
    assert bl.F_y > 0.0
    # alpha can be negative (strength loss)
    assert bl.alpha < 0.0


# ====================================================== equivalent SDOF

def test_equivalent_sdof_scales_by_gamma():
    """d* = d / Gamma, F* = F / Gamma."""
    drift = np.array([0.0, 0.01, 0.02, 0.03])
    force = np.array([0.0, 100.0, 200.0, 300.0])
    sdof = equivalent_sdof(drift, force, Gamma=2.0, m_eff=1000.0)
    np.testing.assert_allclose(sdof.d_star, drift / 2.0)
    np.testing.assert_allclose(sdof.F_star, force / 2.0)
    assert isinstance(sdof, EquivalentSDOF)


def test_equivalent_sdof_period():
    """T* = 2 pi sqrt(m* / K_i*) where K_i* is the initial slope."""
    drift = np.array([0.0, 0.01, 0.02, 0.03])
    force = np.array([0.0, 100.0, 200.0, 300.0])
    # K_i (MDOF) = 100/0.01 = 1e4 ; K_i* = 1e4 / Gamma**2 ? No:
    # K_i* = (F*[1] - F*[0]) / (d*[1] - d*[0]) = (50)/0.005 = 1e4
    sdof = equivalent_sdof(drift, force, Gamma=2.0, m_eff=1000.0)
    # T* = 2 pi sqrt(1000 / 10000) = 2 pi sqrt(0.1) ~ 1.987
    assert sdof.T_eff == pytest.approx(2.0 * math.pi * math.sqrt(0.1), rel=1e-3)


def test_equivalent_sdof_rejects_zero_gamma():
    with pytest.raises(ValueError, match="Gamma"):
        equivalent_sdof([0.0, 1.0], [0.0, 1.0], Gamma=0.0, m_eff=1.0)


# ====================================================== N2 method

def test_n2_long_period_equal_displacement():
    """For T* >= Tc, the equal-displacement rule gives d_t* = d_e*."""
    drift = np.linspace(0.0, 0.1, 11)
    force = drift * 1000.0       # linear, no yielding
    sdof = equivalent_sdof(drift, force, Gamma=1.0, m_eff=1000.0)
    bl = bilinearize_capacity_curve(sdof.d_star, sdof.F_star)
    # Flat spectrum at Sa = 2.0
    spec = ResponseSpectrum(periods=[0.01, 5.0],
                              accelerations=[2.0, 2.0])
    out = n2_target_displacement(spec, sdof, bl, Tc=0.5)
    # Since T* > Tc and the structure is linear (R <= 1), d_t* = d_e*
    assert out["d_t_star"] == pytest.approx(out["d_e_star"], rel=1e-12)


def test_n2_short_period_amplification():
    """For T* < Tc and R > 1 (yielded), the short-period correction
    inflates the inelastic demand above the elastic value."""
    # Capacity curve with a low yield force so R > 1
    drift = np.array([0.0, 0.005, 0.01, 0.02, 0.05])
    force = np.array([0.0, 250.0, 500.0, 500.0, 500.0])    # plateau at 500
    sdof = equivalent_sdof(drift, force, Gamma=1.0, m_eff=10000.0)
    bl = bilinearize_capacity_curve(sdof.d_star, sdof.F_star)
    spec = ResponseSpectrum(periods=[0.01, 5.0],
                              accelerations=[20.0, 20.0])   # high demand
    out = n2_target_displacement(spec, sdof, bl, Tc=0.5)
    if out["R"] > 1.0 and out["T_star"] < 0.5:
        assert out["d_t_star"] > out["d_e_star"]


def test_n2_target_displacement_returns_all_fields():
    drift = np.array([0.0, 0.01, 0.02])
    force = np.array([0.0, 100.0, 100.0])
    sdof = equivalent_sdof(drift, force, Gamma=1.5, m_eff=1000.0)
    bl = bilinearize_capacity_curve(sdof.d_star, sdof.F_star)
    spec = ResponseSpectrum(periods=[0.01, 5.0], accelerations=[1.0, 1.0])
    out = n2_target_displacement(spec, sdof, bl)
    for key in ("T_star", "d_y_star", "F_y_star", "d_e_star",
                "d_t_star", "d_t_top", "R", "Sa_T_star"):
        assert key in out


# ====================================================== coefficient method

def test_coefficient_method_classic_formula():
    """d_t = C0 * C1 * C2 * Sa * (T / 2pi)^2 (for unit g)."""
    spec = ResponseSpectrum(periods=[0.01, 5.0], accelerations=[2.0, 2.0])
    T = 1.0
    out = coefficient_method_target(spec, T_eff=T, C0=1.0, C1=1.0, C2=1.0)
    expected = 2.0 * (T / (2.0 * math.pi)) ** 2
    assert out["d_t_top"] == pytest.approx(expected, rel=1e-12)


def test_coefficient_method_coefficients_scale_target():
    """Doubling C0 doubles the target."""
    spec = ResponseSpectrum(periods=[0.01, 5.0], accelerations=[2.0, 2.0])
    out_1 = coefficient_method_target(spec, T_eff=1.0, C0=1.0)
    out_2 = coefficient_method_target(spec, T_eff=1.0, C0=2.0)
    assert out_2["d_t_top"] == pytest.approx(2.0 * out_1["d_t_top"], rel=1e-12)


# ====================================================== seismic combination

def test_seismic_combination_100_30():
    """100-30: pick the worst permutation of 100% one direction + 30% of others."""
    responses = {"x": 100.0, "y": 50.0}
    out = seismic_combination(responses, rule="100-30")
    # Permutation 1: 100 in x + 30% of y = 100 + 15 = 115
    # Permutation 2: 50 in y + 30% of x = 50 + 30 = 80
    # Worst = 115
    assert out == pytest.approx(115.0, rel=1e-12)


def test_seismic_combination_srss():
    """SRSS: sqrt(sum of squares)."""
    out = seismic_combination({"x": 3.0, "y": 4.0}, rule="SRSS")
    assert out == pytest.approx(5.0, rel=1e-12)


def test_seismic_combination_3d_100_30():
    """In 3D, 100-30 considers 100% in any one direction + 30% of the others."""
    out = seismic_combination({"x": 100.0, "y": 50.0, "z": 20.0},
                                rule="100-30")
    # Best: 100 + 0.3*50 + 0.3*20 = 100 + 15 + 6 = 121
    assert out == pytest.approx(121.0, rel=1e-12)


def test_seismic_combination_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown combination"):
        seismic_combination({"x": 1.0}, rule="some_made_up_rule")


# ====================================================== story drifts

def _build_3_story_frame():
    """3-story shear-stick frame."""
    mat = ElasticIsotropic(1, E=2e10, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    L = 3.0
    for i in range(4):
        m.add_node(i + 1, 0.0, i * L)
    for i in range(3):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, 1e-2, 1e-4))
    m.fix(1, [1, 1, 1])
    return m


def test_story_drifts_returns_structured_arrays():
    m = _build_3_story_frame()
    # Impose known displacement state
    m.number_dofs()
    m.node(2).disp[0] = 0.005
    m.node(3).disp[0] = 0.015
    m.node(4).disp[0] = 0.030
    sd = story_drifts(m, [2, 3, 4], direction=0)
    assert "story" in sd and "interstory_drift" in sd and "drift_ratio" in sd
    np.testing.assert_array_equal(sd["story"], [1, 2, 3])
    np.testing.assert_allclose(sd["absolute_disp"], [0.005, 0.015, 0.030])
    np.testing.assert_allclose(sd["interstory_drift"],
                                  [0.005, 0.010, 0.015])
    # 3 m story heights -> drift ratios 5/3000, 10/3000, 15/3000
    np.testing.assert_allclose(sd["drift_ratio"],
                                  [5e-3 / 3, 10e-3 / 3, 15e-3 / 3],
                                  rtol=1e-12)


def test_story_drifts_with_explicit_base_reference():
    m = _build_3_story_frame()
    m.number_dofs()
    # Settling base
    m.node(1).disp[0] = 0.002
    m.node(2).disp[0] = 0.005
    m.node(3).disp[0] = 0.015
    m.node(4).disp[0] = 0.030
    sd = story_drifts(m, [2, 3, 4], direction=0, base_node_tag=1)
    # absolute_disp is relative to base
    np.testing.assert_allclose(sd["absolute_disp"],
                                  [0.003, 0.013, 0.028])


def test_story_drifts_rejects_empty_list():
    m = _build_3_story_frame()
    with pytest.raises(ValueError, match="non-empty"):
        story_drifts(m, [], direction=0)


# ====================================================== pushover-to-target

def test_pushover_to_target_reaches_drift():
    """The driver runs DispControl pushover until the tracked DOF
    reaches the target drift exactly (within roundoff)."""
    mat = ElasticIsotropic(1, E=2e10, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 3); m.add_node(3, 0, 6); m.add_node(4, 0, 9)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1e-2, 1e-4))
    m.add_element(BeamColumn2D(2, (2, 3), mat, 1e-2, 1e-4))
    m.add_element(BeamColumn2D(3, (3, 4), mat, 1e-2, 1e-4))
    m.fix(1, [1, 1, 1])
    for i in range(1, 4):
        m.add_nodal_load(i + 1, [i * 1.0e3, 0, 0])

    pt = PushoverToTarget(m, target_drift=0.030, track=(4, 0),
                            num_steps=10, tol=1e-6)
    res = pt.run()
    assert res["drift"].size == 10
    # Final drift hits the target
    assert res["drift"][-1] == pytest.approx(0.030, rel=1e-6)
    # Capacity-curve forces are monotonic for an elastic frame
    assert np.all(np.diff(res["force"]) >= -1e-6)


def test_pushover_to_target_rejects_zero_target():
    mat = ElasticIsotropic(1, E=2e10, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 3)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1e-2, 1e-4))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [1.0e3, 0, 0])
    with pytest.raises(ValueError, match="target_drift"):
        PushoverToTarget(m, target_drift=0.0, track=(2, 0))
