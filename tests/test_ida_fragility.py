"""Phase 25 tests -- IDA driver + collapse detection + fragility fitting.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    CollapseResult,
    ElasticIsotropic,
    FragilityFit,
    IDADriver,
    IDAPoint,
    IDARecord,
    IDASummary,
    Model,
    RayleighDamping,
    detect_collapse,
    fit_collapse_fragility,
    fit_lognormal_method_of_moments,
    fit_lognormal_mle,
    max_drift_edp,
    multi_record_ida,
    pga_scale_factor,
)
from femsolver.analysis.eigen import EigenAnalysis


# ============================================================ fixtures

def _stick_model_factory():
    """3-story stick cantilever, fresh each call."""
    def factory():
        mat = ElasticIsotropic(1, E=2.0e10, nu=0.3, rho=7850.0)
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        for i, y in enumerate([0.0, 3.0, 6.0, 9.0]):
            m.add_node(i + 1, 0.0, y)
        A = 1.0e-2; Iz = 1.0e-4
        for i in range(3):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        return m
    return factory


def _ricker(t, *, t0=2.0, fp=1.0, amp=0.5):
    """Centered Ricker pulse."""
    tau = math.pi * fp * (t - t0)
    return amp * (1.0 - 2.0 * tau ** 2) * math.exp(-tau ** 2)


# ============================================================ scaling helper

def test_pga_scale_factor_returns_correct_multiplier():
    """sf · max|a_g| = target_PGA."""
    def ag(t):
        return 1.0 * math.sin(t)        # max amplitude = 1
    sf_fn = pga_scale_factor(ag, t_end=10.0, dt=0.01)
    sf = sf_fn(2.5)
    # target PGA = 2.5, record PGA = 1 -> sf = 2.5
    assert sf == pytest.approx(2.5, rel=1.0e-2)


def test_pga_scale_factor_raises_on_zero_PGA():
    def zero_ag(t):
        return 0.0
    with pytest.raises(ValueError, match="zero PGA"):
        pga_scale_factor(zero_ag, t_end=1.0, dt=0.01)


# ============================================================ max-drift EDP

def test_max_drift_edp_returns_required_keys():
    factory = _stick_model_factory()
    m = factory()
    extractor = max_drift_edp([2, 3, 4], direction=0)
    edps = extractor(m)
    assert "max_drift_ratio" in edps
    assert "max_roof_drift" in edps
    # At rest, all displacements zero
    assert edps["max_drift_ratio"] == pytest.approx(0.0)


def test_max_drift_edp_computes_ratio_correctly():
    """Apply known displacements and verify drift ratio."""
    factory = _stick_model_factory()
    m = factory()
    # Set displacements manually: 1mm at story 1, 4mm at story 2, 6mm at roof
    m.node(2).disp[0] = 0.001     # story 1 (y=3): interstory = 1 mm / 3 m
    m.node(3).disp[0] = 0.004     # story 2 (y=6): interstory = 3 mm / 3 m
    m.node(4).disp[0] = 0.006     # roof (y=9): interstory = 2 mm / 3 m
    extractor = max_drift_edp([2, 3, 4], direction=0, base_node_tag=1)
    edps = extractor(m)
    # Largest interstory ratio = 3 mm / 3 m = 1e-3 (story 2)
    assert edps["max_drift_ratio"] == pytest.approx(1.0e-3, rel=1.0e-3)
    assert edps["max_roof_drift"] == pytest.approx(0.006)


# ============================================================ IDA driver

def test_IDA_driver_runs_and_returns_record():
    """Single-record IDA on a simple stick produces an IM-EDP curve."""
    factory = _stick_model_factory()
    # Light damping to keep convergence reliable
    m_temp = factory()
    eig = EigenAnalysis(m_temp, num_modes=1).run()
    omega_1 = 2 * math.pi / eig["periods_s"][0]
    damping = RayleighDamping.from_modes(
        omega_1=omega_1, omega_2=2 * omega_1, zeta_1=0.05, zeta_2=0.05,
    )
    extractor = max_drift_edp([2, 3, 4], direction=0)
    scale_fn = pga_scale_factor(_ricker, t_end=6.0, dt=0.01)
    driver = IDADriver(
        model_factory=factory,
        accel_function=_ricker,
        t_end=6.0, dt=0.01,
        IM_levels=[0.2, 0.5, 1.0],
        scale_fn=scale_fn,
        edp_extractor=extractor,
        damping=damping,
    )
    record = driver.run()
    assert isinstance(record, IDARecord)
    assert len(record.points) == 3
    # IMs preserved
    np.testing.assert_allclose(record.IMs(), [0.2, 0.5, 1.0])


def test_IDA_EDP_array_helper():
    """IDARecord.EDP_array returns a numpy array of EDP values."""
    rec = IDARecord(
        record_name="test",
        points=[
            IDAPoint(IM=0.1, scale_factor=0.1, EDPs={"max_drift_ratio": 0.001},
                      converged=True),
            IDAPoint(IM=0.5, scale_factor=0.5, EDPs={"max_drift_ratio": 0.01},
                      converged=True),
        ],
    )
    arr = rec.EDP_array("max_drift_ratio")
    np.testing.assert_allclose(arr, [0.001, 0.01])


def test_IDA_validates_IM_levels():
    factory = _stick_model_factory()
    extractor = max_drift_edp([2, 3, 4], direction=0)
    sf = pga_scale_factor(_ricker, t_end=6.0, dt=0.01)
    with pytest.raises(ValueError, match="IM_levels"):
        IDADriver(model_factory=factory, accel_function=_ricker,
                   t_end=6.0, dt=0.01, IM_levels=[],
                   scale_fn=sf, edp_extractor=extractor)
    with pytest.raises(ValueError, match="must all be positive"):
        IDADriver(model_factory=factory, accel_function=_ricker,
                   t_end=6.0, dt=0.01, IM_levels=[0.1, -0.5],
                   scale_fn=sf, edp_extractor=extractor)


# ============================================================ collapse detection

def test_detect_collapse_drift_limit():
    """Drift exceeding the limit -> collapse, cause = drift_limit."""
    rec = IDARecord(
        record_name="r1",
        points=[
            IDAPoint(IM=0.1, scale_factor=0.1, EDPs={"max_drift_ratio": 0.005},
                      converged=True),
            IDAPoint(IM=0.5, scale_factor=0.5, EDPs={"max_drift_ratio": 0.02},
                      converged=True),
            IDAPoint(IM=1.0, scale_factor=1.0, EDPs={"max_drift_ratio": 0.15},
                      converged=True),
        ],
    )
    coll = detect_collapse(rec, drift_limit=0.10)
    assert coll.collapse_IM == pytest.approx(1.0)
    assert coll.cause == "drift_limit"
    assert coll.collapse_point_index == 2


def test_detect_collapse_non_convergence():
    """A non-converged point is a collapse, cause = non_convergence."""
    rec = IDARecord(
        record_name="r2",
        points=[
            IDAPoint(IM=0.1, scale_factor=0.1, EDPs={"max_drift_ratio": 0.001},
                      converged=True),
            IDAPoint(IM=0.5, scale_factor=0.5, EDPs={"max_drift_ratio": 0.005},
                      converged=False),
        ],
    )
    coll = detect_collapse(rec)
    assert coll.collapse_IM == pytest.approx(0.5)
    assert coll.cause == "non_convergence"


def test_detect_collapse_no_collapse():
    """Within sweep range, no collapse -> inf."""
    rec = IDARecord(
        record_name="r3",
        points=[
            IDAPoint(IM=0.1, scale_factor=0.1, EDPs={"max_drift_ratio": 0.001},
                      converged=True),
            IDAPoint(IM=0.5, scale_factor=0.5, EDPs={"max_drift_ratio": 0.005},
                      converged=True),
        ],
    )
    coll = detect_collapse(rec, drift_limit=0.10)
    assert math.isinf(coll.collapse_IM)
    assert coll.cause == "no_collapse"


# ============================================================ fragility moments

def test_fragility_method_of_moments_recovers_theta_and_beta():
    """Generate samples from a known lognormal; verify estimates."""
    np.random.seed(42)
    theta_true = 1.5
    beta_true = 0.40
    samples = theta_true * np.exp(np.random.normal(0, beta_true, 200))
    fit = fit_lognormal_method_of_moments(samples)
    assert fit.theta == pytest.approx(theta_true, rel=0.1)
    assert fit.beta == pytest.approx(beta_true, rel=0.15)
    assert fit.method == "moments"


def test_fragility_method_of_moments_rejects_too_few_samples():
    with pytest.raises(ValueError, match="finite"):
        fit_lognormal_method_of_moments([float("inf")])


def test_fragility_P_collapse_at_theta_is_0p5():
    """At IM = θ, the CDF gives 0.5 by definition."""
    fit = FragilityFit(theta=1.0, beta=0.3, n_records=10,
                         n_collapsed=10, method="moments")
    assert fit.P_collapse(1.0) == pytest.approx(0.5, abs=1.0e-12)


def test_fragility_P_collapse_monotonically_increases():
    fit = FragilityFit(theta=1.0, beta=0.3, n_records=10,
                         n_collapsed=10, method="moments")
    IMs = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
    Ps = [fit.P_collapse(im) for im in IMs]
    for i in range(len(Ps) - 1):
        assert Ps[i + 1] >= Ps[i]
    assert Ps[0] < 0.05
    assert Ps[-1] > 0.95


# ============================================================ fragility MLE

def test_fragility_mle_handles_censored_records():
    """Mix of collapsed + censored records -> MLE gives reasonable
    estimates and respects the censored data (theta isn't pulled
    artificially low)."""
    np.random.seed(0)
    theta_true = 2.0
    beta_true = 0.30
    n = 50
    samples = theta_true * np.exp(np.random.normal(0, beta_true, n))
    # Censor: any sample > 3.0 is treated as not-collapsed at IM=3.0
    cutoff = 3.0
    collapsed = samples[samples <= cutoff]
    n_censored = int(np.sum(samples > cutoff))
    no_coll_max = np.full(n_censored, cutoff)
    fit = fit_lognormal_mle(collapsed, no_coll_max)
    # MLE should give theta close to truth (not biased by ignoring
    # the censored records). With censoring, the method-of-moments
    # on just the collapsed records would severely underestimate
    # theta; MLE corrects for this.
    assert fit.theta == pytest.approx(theta_true, rel=0.40)
    assert fit.method == "mle"


def test_fit_collapse_fragility_dispatches_correctly():
    """fit_collapse_fragility uses moments when no censored data,
    MLE when censored data is present."""
    samples = [0.5, 1.0, 1.5, 2.0, 2.5]
    fit1 = fit_collapse_fragility(samples)
    assert fit1.method == "moments"
    fit2 = fit_collapse_fragility(samples, no_collapse_IM_max=[3.0, 3.0])
    assert fit2.method == "mle"


# ============================================================ multi-record

def test_multi_record_ida_aggregates_results():
    """Multi-record IDA on two synthetic records."""
    factory = _stick_model_factory()
    m_temp = factory()
    eig = EigenAnalysis(m_temp, num_modes=1).run()
    omega_1 = 2 * math.pi / eig["periods_s"][0]
    damping = RayleighDamping.from_modes(
        omega_1=omega_1, omega_2=2 * omega_1, zeta_1=0.05, zeta_2=0.05,
    )
    extractor = max_drift_edp([2, 3, 4], direction=0)

    def ag1(t): return _ricker(t, t0=2.0, fp=1.0, amp=0.5)
    def ag2(t): return _ricker(t, t0=3.0, fp=0.8, amp=0.7)

    records = [
        {"name": "R1", "accel_function": ag1, "t_end": 6.0, "dt": 0.02,
         "scale_fn": pga_scale_factor(ag1, t_end=6.0, dt=0.02)},
        {"name": "R2", "accel_function": ag2, "t_end": 6.0, "dt": 0.02,
         "scale_fn": pga_scale_factor(ag2, t_end=6.0, dt=0.02)},
    ]
    summary = multi_record_ida(
        model_factory=factory, records=records,
        IM_levels=[0.5, 1.0],
        edp_extractor=extractor, damping=damping,
        drift_limit=0.1,
    )
    assert isinstance(summary, IDASummary)
    assert len(summary.records) == 2
    assert len(summary.collapse_results) == 2
    # Each per-record result is a CollapseResult
    for cr in summary.collapse_results:
        assert isinstance(cr, CollapseResult)
