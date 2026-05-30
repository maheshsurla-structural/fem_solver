"""Phase 31 tests -- load combinations + envelope + drift checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    DriftCheck,
    ElasticIsotropic,
    EnvelopeAnalysis,
    EnvelopeResult,
    ForceEnvelope,
    LinearStaticAnalysis,
    LoadCombination,
    LoadPattern,
    Model,
    apply_combination,
    asce7_lrfd_combinations,
    asce7_lrfd_seismic_combinations_per_direction,
    drift_check,
    drift_check_worst_combo,
)


# ============================================================ fixtures

def _simple_cantilever():
    """2-node cantilever for trivial loading tests."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 5.0, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 0.01, 1.0e-4))
    m.fix(1, [1, 1, 1])
    return m


def _multistory_frame():
    """3-story 2-bay frame for envelope + drift tests."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    H = 3.5; L = 6.0
    n_col = 3
    for j in range(4):
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_node(tag, i * L, j * H)
    etag = 1
    for j in range(3):
        for i in range(n_col):
            n_b = j * n_col + i + 1
            n_t = (j + 1) * n_col + i + 1
            m.add_element(BeamColumn2D(etag, (n_b, n_t), mat, 0.02, 0.001))
            etag += 1
    for j in range(1, 4):
        for i in range(2):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            m.add_element(BeamColumn2D(etag, (n_L, n_R), mat, 0.02, 0.001))
            etag += 1
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])
    return m


# ============================================================ ASCE 7 combos

def test_asce7_combinations_first_is_1p4D():
    combos = asce7_lrfd_combinations()
    assert combos[0].name == "1.4D"
    assert combos[0].factor("D") == 1.4


def test_asce7_combinations_no_seismic_excludes_E():
    combos = asce7_lrfd_combinations(include_seismic=False)
    for c in combos:
        assert "E" not in c.factors


def test_asce7_combinations_no_wind_excludes_W():
    combos = asce7_lrfd_combinations(include_wind=False)
    for c in combos:
        assert "W" not in c.factors


def test_asce7_combinations_strip_snow_collapses():
    """Without Lr/S/R, combination 2 collapses to just 1.2D+1.6L."""
    combos = asce7_lrfd_combinations(include_snow_rain_roof_live=False)
    # Find the 1.2D+1.6L combination
    found = [c for c in combos if c.factors == {"D": 1.2, "L": 1.6}]
    assert len(found) == 1


def test_seismic_direction_combos_have_E_x_and_E_y():
    """Direction-resolved combos include both E_x and E_y."""
    combos = asce7_lrfd_seismic_combinations_per_direction()
    for c in combos:
        assert "E_x" in c.factors
        assert "E_y" in c.factors


def test_load_combination_factor_default_zero():
    combo = LoadCombination("1.4D", {"D": 1.4})
    assert combo.factor("D") == 1.4
    assert combo.factor("L") == 0.0
    assert combo.factor("nonexistent") == 0.0


# ============================================================ apply_combination

def test_apply_combination_clears_then_applies():
    m = _simple_cantilever()
    # Stale load that should be cleared first
    m.add_nodal_load(2, [99.0, 99.0, 99.0])

    def dead(model, factor=1.0):
        model.add_nodal_load(2, [0.0, -1000.0 * factor, 0.0])

    patterns = {"D": LoadPattern("D", dead)}
    combo = LoadCombination("1.4D", {"D": 1.4})
    apply_combination(m, patterns, combo)
    np.testing.assert_allclose(
        m.node(2)._load, [0.0, -1400.0, 0.0],
    )


def test_apply_combination_skips_missing_patterns():
    """Pattern referenced by combo but absent from patterns dict --
    silently skipped (other terms still apply)."""
    m = _simple_cantilever()

    def dead(model, factor=1.0):
        model.add_nodal_load(2, [0.0, -1000.0 * factor, 0.0])

    patterns = {"D": LoadPattern("D", dead)}    # no "L" pattern
    combo = LoadCombination("1.2D + 1.6L", {"D": 1.2, "L": 1.6})
    apply_combination(m, patterns, combo)
    # Only 1.2D applied; L silently skipped
    np.testing.assert_allclose(m.node(2)._load, [0.0, -1200.0, 0.0])


def test_apply_combination_zero_factor_skipped():
    """A pattern with factor 0 in the combo is not applied."""
    m = _simple_cantilever()
    called = [False]

    def dead(model, factor=1.0):
        called[0] = True

    patterns = {"D": LoadPattern("D", dead)}
    combo = LoadCombination("0D", {"D": 0.0})
    apply_combination(m, patterns, combo)
    assert not called[0]


# ============================================================ EnvelopeAnalysis

def test_envelope_runs_each_combo_and_collects_forces():
    m = _multistory_frame()

    def dead(model, factor=1.0):
        for etag in (10, 11, 12, 13, 14, 15):    # beams
            if etag in model.elements:
                model.elements[etag].add_uniform_load(-10e3 * factor)

    def wind(model, factor=1.0):
        for ntag in (4, 7, 10):    # floor nodes
            model.add_nodal_load(ntag, [20e3 * factor, 0, 0])

    patterns = {
        "D": LoadPattern("D", dead),
        "W": LoadPattern("W", wind),
    }
    combos = [
        LoadCombination("1.4D", {"D": 1.4}),
        LoadCombination("1.2D + 1.0W", {"D": 1.2, "W": 1.0}),
        LoadCombination("0.9D + 1.0W", {"D": 0.9, "W": 1.0}),
    ]
    env = EnvelopeAnalysis(m, patterns, combos).run()
    assert isinstance(env, EnvelopeResult)
    assert len(env.combinations) == 3
    # Member envelopes recorded for every BeamColumn2D
    assert len(env.member_envelopes) == len(m.elements)
    # Each member envelope has 6 components (BeamColumn2D end forces)
    sample = next(iter(env.member_envelopes.values()))
    assert sample.n_components == 6
    assert sample.max_values.shape == (6,)


def test_envelope_max_min_consistent_with_raw():
    """For any element, env.max_values >= raw values from any combo."""
    m = _multistory_frame()

    def dead(model, factor=1.0):
        for etag in m.elements:
            if etag >= 10:
                model.elements[etag].add_uniform_load(-10e3 * factor)

    patterns = {"D": LoadPattern("D", dead)}
    combos = [
        LoadCombination("0.9D", {"D": 0.9}),
        LoadCombination("1.4D", {"D": 1.4}),
    ]
    env = EnvelopeAnalysis(m, patterns, combos).run()
    for etag, fe in env.member_envelopes.items():
        for combo_name, ef in env.raw_member_forces[etag].items():
            assert np.all(fe.max_values >= ef - 1e-6)
            assert np.all(fe.min_values <= ef + 1e-6)


def test_envelope_governing_combo_in_combinations_list():
    """The governing-combo names must come from the runs."""
    m = _multistory_frame()

    def dead(model, factor=1.0):
        for etag in m.elements:
            if etag >= 10:
                model.elements[etag].add_uniform_load(-10e3 * factor)

    patterns = {"D": LoadPattern("D", dead)}
    combos = [
        LoadCombination("0.9D", {"D": 0.9}),
        LoadCombination("1.4D", {"D": 1.4}),
    ]
    env = EnvelopeAnalysis(m, patterns, combos).run()
    valid_names = {c.name for c in combos}
    for fe in env.member_envelopes.values():
        for cn in fe.max_combos:
            assert cn in valid_names
        for cn in fe.min_combos:
            assert cn in valid_names


# ============================================================ drift check

def test_drift_check_under_lateral_load():
    """Single-state drift check on a multi-story frame with lateral load."""
    m = _multistory_frame()
    # Apply lateral nodal loads (single combo)
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [50e3, 0, 0])
    LinearStaticAnalysis(m).run()
    story_tags = [4, 7, 10]      # representative node per story (col 1)
    dc = drift_check(
        m, story_tags, direction=0,
        C_d=5.5, I_e=1.0, risk_category="II",
    )
    assert isinstance(dc, DriftCheck)
    assert dc.story_index.shape == (3,)
    assert dc.delta_amplified.shape == (3,)
    assert dc.drift_limit == pytest.approx(0.020)     # Risk Cat I/II


def test_drift_check_amplifies_by_C_d_over_I_e():
    """Δ_amplified = C_d · Δ_elastic / I_e."""
    m = _multistory_frame()
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [50e3, 0, 0])
    LinearStaticAnalysis(m).run()
    story_tags = [4, 7, 10]
    dc1 = drift_check(m, story_tags, direction=0,
                       C_d=4.0, I_e=1.0)
    dc2 = drift_check(m, story_tags, direction=0,
                       C_d=4.0, I_e=2.0)
    # Doubling I_e halves the amplified drift
    np.testing.assert_allclose(
        dc2.delta_amplified, 0.5 * dc1.delta_amplified,
    )


def test_drift_check_limits_by_risk_category():
    """ASCE 7-22 Table 12.12-1: I/II -> 0.020, III -> 0.015, IV -> 0.010."""
    m = _multistory_frame()
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [50e3, 0, 0])
    LinearStaticAnalysis(m).run()
    story_tags = [4, 7, 10]
    assert drift_check(m, story_tags, risk_category="II").drift_limit == pytest.approx(0.020)
    assert drift_check(m, story_tags, risk_category="III").drift_limit == pytest.approx(0.015)
    assert drift_check(m, story_tags, risk_category="IV").drift_limit == pytest.approx(0.010)


def test_drift_check_passes_for_small_drift():
    m = _multistory_frame()
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [5e3, 0, 0])    # tiny load
    LinearStaticAnalysis(m).run()
    dc = drift_check(m, [4, 7, 10])
    assert dc.passes
    assert all(dc.passes_per_story)


def test_drift_check_fails_for_huge_drift():
    m = _multistory_frame()
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [10e6, 0, 0])    # huge
    LinearStaticAnalysis(m).run()
    dc = drift_check(m, [4, 7, 10])
    assert not dc.passes
    assert not all(dc.passes_per_story)


def test_drift_check_rejects_unknown_risk_category():
    m = _multistory_frame()
    for ntag in (4, 7, 10):
        m.add_nodal_load(ntag, [50e3, 0, 0])
    LinearStaticAnalysis(m).run()
    with pytest.raises(ValueError, match="risk_category"):
        drift_check(m, [4, 7, 10], risk_category="bogus")


# ============================================================ multi-combo drift

def test_drift_check_worst_combo_tracks_governing():
    """Run multiple combos; each story's worst-case Δ is tracked
    along with which combo produced it."""
    m = _multistory_frame()

    def wind_small(model, factor=1.0):
        for ntag in (4, 7, 10):
            model.add_nodal_load(ntag, [10e3 * factor, 0, 0])

    def wind_large(model, factor=1.0):
        for ntag in (4, 7, 10):
            model.add_nodal_load(ntag, [50e3 * factor, 0, 0])

    patterns = {
        "W_small": LoadPattern("W_small", wind_small),
        "W_large": LoadPattern("W_large", wind_large),
    }
    combos = [
        LoadCombination("Small wind", {"W_small": 1.0}),
        LoadCombination("Large wind", {"W_large": 1.0}),
    ]
    dc = drift_check_worst_combo(
        m, patterns, combos, [4, 7, 10],
        direction=0, C_d=4.0,
    )
    # Large wind governs every story
    assert dc.governing_combo == ["Large wind"] * 3
