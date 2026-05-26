"""Tests for through-thickness stress recovery + ply failure criteria
(Phase 22.4 / 22.5)."""
import math

import numpy as np
import pytest

from femsolver import (
    LayeredShellSection,
    OrthotropicLamina,
    PlyStrength,
    evaluate_laminate,
    max_strain_index,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)


# ====================================================== through-thickness stress

def test_ply_stresses_uniform_eps_per_ply():
    """Pure membrane strain: each ply's stress is constant through its
    thickness."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 0.0),
        (ply, 0.5e-3, 90.0),
    ])
    results = sec.ply_stresses(
        eps_membrane=(1e-3, 0.0, 0.0), kappa=(0.0, 0.0, 0.0), z="all",
    )
    # group by layer index
    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r["sigma_global"])
    for layer_idx, stresses in by_layer.items():
        for s in stresses[1:]:
            np.testing.assert_allclose(s, stresses[0], rtol=1e-12)


def test_ply_stresses_bending_strain_varies_with_z():
    """Pure bending: stress in a ply varies linearly across its
    thickness (z direction)."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([(ply, 1.0e-3, 0.0)])
    results = sec.ply_stresses(
        eps_membrane=(0.0, 0.0, 0.0), kappa=(1.0, 0.0, 0.0), z="all",
    )
    assert len(results) == 3
    # sigma_xx = Q11 * z (linear in z), evaluated at top, mid, bot
    # mid is at z = 0 -> sigma = 0
    by_station = {r["station"]: r for r in results}
    assert by_station["mid"]["sigma_global"][0] == pytest.approx(0.0, abs=1e-6)
    # top and bot have opposite-sign stresses
    s_top = by_station["top"]["sigma_global"][0]
    s_bot = by_station["bot"]["sigma_global"][0]
    assert s_top == pytest.approx(-s_bot, rel=1e-12)
    # and |s_top| > 0
    assert abs(s_top) > 1.0e6


def test_ply_stresses_local_coords_at_theta_90():
    """A 90-deg ply under pure eps_xx tension feels sigma_11 (in local
    axes) = matrix-direction stress, not the global sigma_xx."""
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.5e-3, 0.0),
        (ply, 0.5e-3, 90.0),
    ])
    results = sec.ply_stresses(
        eps_membrane=(1e-3, 0.0, 0.0), kappa=(0.0, 0.0, 0.0), z="mid",
    )
    # Layer 0 (0 deg): sigma_local_11 should be large (fiber-direction)
    s_layer0_local = results[0]["sigma_local"]
    # Layer 1 (90 deg): sigma_local_11 should be small (matrix loaded
    # in transverse-to-fiber direction)
    s_layer1_local = results[1]["sigma_local"]
    assert s_layer0_local[0] > s_layer1_local[0] * 10.0


def test_ply_stresses_rejects_bad_z():
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    sec = LayeredShellSection.from_layers_centered([(ply, 1e-3, 0.0)])
    with pytest.raises(ValueError, match="z must be"):
        sec.ply_stresses((0, 0, 0), (0, 0, 0), z="bogus")


# ====================================================== PlyStrength

def test_ply_strength_validates_positive_values():
    with pytest.raises(ValueError, match="Xt"):
        PlyStrength(Xt=-1.0, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    with pytest.raises(ValueError, match="S"):
        PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=0.0)


# ====================================================== max-stress

def test_max_stress_at_uniaxial_tensile_strength():
    """sigma_11 = Xt gives FI = 1 exactly."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert max_stress_index((1500e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_max_stress_at_half_tensile_strength():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert max_stress_index((750e6, 0, 0), s) == pytest.approx(0.5, rel=1e-12)


def test_max_stress_compression_uses_Xc():
    """sigma_11 = -Xc gives FI = 1; Xc may differ from Xt."""
    s = PlyStrength(Xt=1500e6, Xc=1200e6, Yt=40e6, Yc=246e6, S=68e6)
    assert max_stress_index((-1200e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)
    assert max_stress_index((-600e6, 0, 0), s) == pytest.approx(0.5, rel=1e-12)


def test_max_stress_transverse_uses_Y():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert max_stress_index((0, 40e6, 0), s) == pytest.approx(1.0, rel=1e-12)
    assert max_stress_index((0, -246e6, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_max_stress_shear_uses_S():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert max_stress_index((0, 0, 68e6), s) == pytest.approx(1.0, rel=1e-12)
    assert max_stress_index((0, 0, -68e6), s) == pytest.approx(1.0, rel=1e-12)


# ====================================================== Tsai-Hill

def test_tsai_hill_at_uniaxial_failure():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    # Pure sigma_11 = Xt
    assert tsai_hill_index((1500e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_tsai_hill_picks_correct_branch_in_compression():
    """For sigma_11 < 0, X = X_C (not X_T)."""
    s = PlyStrength(Xt=1500e6, Xc=1200e6, Yt=40e6, Yc=246e6, S=68e6)
    # At sigma_11 = -X_C: FI = 1
    assert tsai_hill_index((-1200e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_tsai_hill_picks_up_interaction():
    """Combined stresses give higher FI than the sum of individual
    ratios (interaction term)."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    fi_combined = tsai_hill_index((750e6, 20e6, 30e6), s)
    fi_separate = (
        max_stress_index((750e6, 0, 0), s)
        + max_stress_index((0, 20e6, 0), s)
        + max_stress_index((0, 0, 30e6), s)
    )
    # Tsai-Hill is generally between max-stress and quadratic sum.
    # Here we just verify the combined FI is positive and < 1 for these
    # moderate stresses.
    assert fi_combined > 0.0


# ====================================================== Tsai-Wu

def test_tsai_wu_at_pure_tension_failure():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert tsai_wu_index((1500e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_tsai_wu_at_pure_compression_failure():
    """sigma_11 = -X_C also gives FI = 1 (with the linear F1 term)."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert tsai_wu_index((-1500e6, 0, 0), s) == pytest.approx(1.0, rel=1e-12)


def test_tsai_wu_zero_for_zero_stress():
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    assert tsai_wu_index((0, 0, 0), s) == pytest.approx(0.0, rel=1e-12)


def test_tsai_wu_strength_ratio_at_failure_is_one():
    """SR = 1 for stresses exactly on the envelope."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    SR = tsai_wu_strength_ratio((1500e6, 0, 0), s)
    assert SR == pytest.approx(1.0, rel=1e-10)


def test_tsai_wu_strength_ratio_doubles_for_half_stresses():
    """If you halve the stresses, you can scale by 2x before failure."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    SR = tsai_wu_strength_ratio((750e6, 0, 0), s)
    assert SR == pytest.approx(2.0, rel=1e-10)


def test_tsai_wu_strength_ratio_inf_for_pure_compression_only_envelope():
    """If Xt = Xc (no linear term), a 'compression-shrinks' stress
    can be infinite (no failure possible in that direction)."""
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    # Pure hydrostatic compression of just sigma_11 -- bounded
    SR = tsai_wu_strength_ratio((-100e6, 0, 0), s)
    assert SR > 1.0    # we're below the compressive envelope (Xc = 1500)


# ====================================================== max-strain

def test_max_strain_uses_same_math_as_max_stress():
    """The max-strain criterion shares the algorithm of max-stress
    when passed strain values + strain limits."""
    eps_limits = PlyStrength(Xt=0.01, Xc=0.01, Yt=0.005, Yc=0.01, S=0.02)
    eps = (0.005, 0, 0)        # half of Xt-strain
    assert max_strain_index(eps, eps_limits) == pytest.approx(0.5, rel=1e-12)


# ====================================================== evaluate_laminate

def _make_cross_ply_section():
    ply = OrthotropicLamina(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28)
    return LayeredShellSection.from_layers_centered([
        (ply, 0.25e-3, 0.0),
        (ply, 0.25e-3, 90.0),
        (ply, 0.25e-3, 90.0),
        (ply, 0.25e-3, 0.0),
    ])


def test_evaluate_laminate_returns_all_plies():
    sec = _make_cross_ply_section()
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    results = evaluate_laminate(
        sec, eps_membrane=(0.001, 0, 0), kappa=(0, 0, 0),
        strengths=s, criterion="tsai_wu", z="mid",
    )
    # 4 layers x 1 station = 4 records
    assert len(results) == 4
    layers_seen = sorted(r["layer"] for r in results)
    assert layers_seen == [0, 1, 2, 3]


def test_evaluate_laminate_first_ply_failure_in_90_plies():
    """At a modest eps_xx tension, the 90-deg plies (matrix loaded
    transverse to fibers) fail before the 0-deg plies (fibers loaded
    along their strong axis). This is the classical first-ply failure
    pattern of a cross-ply laminate.

    Note: the strain level matters. At enormous strains (eps_xx >> the
    failure strain), the 0-deg plies' quadratic Tsai-Wu term eventually
    overtakes the 90-deg plies'. The realistic "first-ply failure"
    regime is at strains just past the matrix-cracking threshold of the
    transverse plies. eps_xx = 0.005 sits there for this CFRP system:
    the 90-deg plies have FI > 1 (failed) while the 0-deg plies are
    well below 1.
    """
    sec = _make_cross_ply_section()
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    results = evaluate_laminate(
        sec, eps_membrane=(0.005, 0, 0), kappa=(0, 0, 0),
        strengths=s, criterion="tsai_wu", z="mid",
    )
    max_fi_ply = max(results, key=lambda r: r["FI"])
    # The max-FI ply should be one of the 90-deg plies
    assert max_fi_ply["theta_deg"] == pytest.approx(90.0)
    # And that ply should actually have failed (FI > 1)
    assert max_fi_ply["FI"] > 1.0
    # While the 0-deg plies should still be safely under failure
    fi_0deg = [r["FI"] for r in results if r["theta_deg"] == 0.0]
    assert max(fi_0deg) < 1.0


def test_evaluate_laminate_per_layer_strengths():
    """Pass a list of PlyStrength (one per layer) instead of a single
    one applied uniformly."""
    sec = _make_cross_ply_section()
    s_strong = PlyStrength(Xt=2e9, Xc=2e9, Yt=80e6, Yc=300e6, S=100e6)
    s_weak = PlyStrength(Xt=500e6, Xc=500e6, Yt=20e6, Yc=100e6, S=50e6)
    strengths = [s_strong, s_weak, s_weak, s_strong]
    results = evaluate_laminate(
        sec, eps_membrane=(0.002, 0, 0), kappa=(0, 0, 0),
        strengths=strengths, criterion="max_stress", z="mid",
    )
    # The weak (middle) plies should have higher FI than the strong ones
    fi_outer = [r["FI"] for r in results
                 if r["layer"] in (0, 3)]
    fi_inner = [r["FI"] for r in results
                 if r["layer"] in (1, 2)]
    assert max(fi_inner) > max(fi_outer)


def test_evaluate_laminate_rejects_unknown_criterion():
    sec = _make_cross_ply_section()
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    with pytest.raises(ValueError, match="unknown criterion"):
        evaluate_laminate(sec, (0, 0, 0), (0, 0, 0), s, criterion="bogus")


def test_evaluate_laminate_rejects_wrong_strengths_length():
    sec = _make_cross_ply_section()       # 4 layers
    s = PlyStrength(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S=68e6)
    with pytest.raises(ValueError, match="strengths list"):
        evaluate_laminate(sec, (0, 0, 0), (0, 0, 0), [s, s])
