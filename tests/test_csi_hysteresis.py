"""Tests for the Phase 24 CSI-style hysteresis material catalog.

Five new uniaxial materials complement the existing library to mirror
the hysteresis types offered by CSI codes (SAP2000, PERFORM-3D) for
fiber-hinge analysis:

* :class:`UniaxialIsotropicHardening` -- isotropic-hardening bilinear
* :class:`UniaxialTakeda` -- Modified Takeda for RC (stiffness-degrading
  unloading)
* :class:`UniaxialPivot` -- Dowell-Seible-Hines pivot model
* :class:`UniaxialIMK` -- Ibarra-Medina-Krawinkler with capping +
  cyclic deterioration
* :class:`UniaxialBRB` -- buckling-restrained-brace asymmetric
  hardening

Each material is exercised on:
1. construction / validation
2. monotonic backbone correctness
3. cyclic-loop sanity (key signature unique to the model)
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    UniaxialBilinear,
    UniaxialBRB,
    UniaxialIMK,
    UniaxialIsotropicHardening,
    UniaxialPivot,
    UniaxialTakeda,
)


# ============================================================ Isotropic

class TestUniaxialIsotropicHardening:
    """Isotropic hardening: yield surface expands with cumulative
    plastic strain. Differs from kinematic on reversal (no Bauschinger)."""

    def test_rejects_negative_E(self):
        with pytest.raises(ValueError, match="E must be positive"):
            UniaxialIsotropicHardening(E=-1.0, sigma_y=1.0)

    def test_rejects_negative_sigma_y(self):
        with pytest.raises(ValueError, match="sigma_y must be positive"):
            UniaxialIsotropicHardening(E=1.0, sigma_y=-1.0)

    def test_rejects_b_out_of_range(self):
        with pytest.raises(ValueError, match="b must be"):
            UniaxialIsotropicHardening(E=1.0, sigma_y=1.0, b=1.0)
        with pytest.raises(ValueError, match="b must be"):
            UniaxialIsotropicHardening(E=1.0, sigma_y=1.0, b=-0.1)

    def test_elastic_response_below_yield(self):
        mat = UniaxialIsotropicHardening(E=2.0e11, sigma_y=4.0e8, b=0.05)
        sigma, Et = mat.get_response(1.0e-3)    # well below yield (~2e-3)
        assert sigma == pytest.approx(2.0e11 * 1.0e-3)
        assert Et == pytest.approx(2.0e11)
        mat.commit_state()
        # Cumulative plastic strain still zero
        assert mat.p_committed == pytest.approx(0.0)

    def test_monotonic_matches_kinematic(self):
        """For monotonic loading, isotropic and kinematic hardening
        give identical stress-strain response."""
        E, sigma_y, b = 2.0e11, 4.0e8, 0.05
        iso = UniaxialIsotropicHardening(E=E, sigma_y=sigma_y, b=b)
        kin = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b)
        for eps in np.linspace(0.0, 0.01, 50):
            s_iso, _ = iso.get_response(float(eps))
            s_kin, _ = kin.get_response(float(eps))
            iso.commit_state(); kin.commit_state()
            assert s_iso == pytest.approx(s_kin, rel=1e-10)

    def test_reverse_yield_after_preload_is_larger_than_kinematic(self):
        """Loading to eps_max past yield, then reversing: isotropic
        sees yield at the *expanded* surface (larger |sigma|), while
        kinematic sees it at sigma_y - 2*q (Bauschinger, smaller |sigma|)."""
        E, sigma_y, b = 2.0e11, 4.0e8, 0.05
        iso = UniaxialIsotropicHardening(E=E, sigma_y=sigma_y, b=b)
        kin = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b)
        # Load both to eps = +0.005
        for mat in (iso, kin):
            mat.get_response(0.005)
            mat.commit_state()
        # Now reverse to eps = -0.005
        s_iso, _ = iso.get_response(-0.005)
        s_kin, _ = kin.get_response(-0.005)
        # Both negative; isotropic should be MORE negative
        assert abs(s_iso) > abs(s_kin), (
            f"Isotropic reverse stress |{s_iso/1e6:.1f}| should exceed "
            f"kinematic |{s_kin/1e6:.1f}|"
        )

    def test_cumulative_plastic_strain_grows_monotonically(self):
        """The equivalent plastic strain p is monotonically increasing,
        even on reversal -- defining property of isotropic hardening."""
        mat = UniaxialIsotropicHardening(E=2.0e11, sigma_y=4.0e8, b=0.05)
        # Cycle: +0.005 -> -0.005 -> +0.005
        ps = []
        for eps in (0.005, -0.005, 0.005):
            mat.get_response(eps)
            mat.commit_state()
            ps.append(mat.p_committed)
        assert ps[0] > 0.0
        assert ps[1] > ps[0]
        assert ps[2] > ps[1]

    def test_commit_revert_roundtrip(self):
        """Trial state thrown away by revert; committed state survives."""
        mat = UniaxialIsotropicHardening(E=2.0e11, sigma_y=4.0e8, b=0.05)
        mat.get_response(0.005)
        eps_p_t = mat.eps_p_trial; p_t = mat.p_trial
        mat.revert_state()
        # Trial reset to committed (which is zero, since no commit)
        assert mat.eps_p_trial == mat.eps_p_committed
        assert mat.p_trial == mat.p_committed
        # Re-run, then commit
        mat.get_response(0.005)
        mat.commit_state()
        assert mat.eps_p_committed == pytest.approx(eps_p_t)
        assert mat.p_committed == pytest.approx(p_t)


# ============================================================ Takeda

class TestUniaxialTakeda:
    """Modified Takeda: bilinear backbone, stiffness-degrading
    unloading with α-rule, no pinching."""

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            UniaxialTakeda(E=1.0, sigma_y=1.0, alpha=-0.1)

    def test_monotonic_matches_bilinear_envelope(self):
        """Monotonic loading follows the bilinear backbone."""
        E, sigma_y, b = 2.0e10, 4.0e6, 0.02
        tk = UniaxialTakeda(E=E, sigma_y=sigma_y, b=b, alpha=0.5)
        eps_y = sigma_y / E
        # Below yield
        s, Et = tk.get_response(0.5 * eps_y)
        assert s == pytest.approx(E * 0.5 * eps_y, rel=1e-12)
        assert Et == pytest.approx(E)
        tk.commit_state()
        # Past yield
        s, Et = tk.get_response(5.0 * eps_y)
        s_theory = sigma_y + b * E * (5.0 * eps_y - eps_y)
        assert s == pytest.approx(s_theory, rel=1e-12)
        assert Et == pytest.approx(b * E)

    def test_unloading_uses_degraded_stiffness(self):
        """Unloading from peak ε_max past yield: slope = E·(ε_y/ε_max)^α."""
        E, sigma_y, b, alpha = 2.0e10, 4.0e6, 0.02, 0.5
        tk = UniaxialTakeda(E=E, sigma_y=sigma_y, b=b, alpha=alpha)
        eps_max = 0.005           # 25 × eps_y
        eps_y = sigma_y / E
        K_u_theory = E * (eps_y / eps_max) ** alpha
        # Load to peak then unload one small step
        tk.get_response(eps_max)
        tk.commit_state()
        s_peak = tk.sigma_trial
        deps = -0.0001
        s_after, Et = tk.get_response(eps_max + deps)
        # Slope check
        slope_observed = (s_after - s_peak) / deps
        assert slope_observed == pytest.approx(K_u_theory, rel=1e-6)
        assert Et == pytest.approx(K_u_theory, rel=1e-6)

    def test_cyclic_loop_returns_to_starting_point_after_full_cycle(self):
        """After a full +/-/+ cycle the trace closes the loop --
        the path is well-defined and reversible at the same eps_max."""
        E, sigma_y, b, alpha = 2.0e10, 4.0e6, 0.02, 0.5
        tk = UniaxialTakeda(E=E, sigma_y=sigma_y, b=b, alpha=alpha)
        # +0.005
        tk.get_response(0.005); tk.commit_state()
        s_peak_pos_1 = tk.sigma_trial
        # -0.005
        tk.get_response(-0.005); tk.commit_state()
        # +0.005 (second cycle)
        tk.get_response(0.005); tk.commit_state()
        s_peak_pos_2 = tk.sigma_trial
        # Backbone-only: both peaks should equal the bilinear envelope
        # at eps=0.005 (no strength deterioration in this simplified
        # Takeda).
        assert s_peak_pos_2 == pytest.approx(s_peak_pos_1, rel=1e-10)

    def test_alpha_zero_gives_elastic_unloading(self):
        """α = 0: K_u = E always (no degradation)."""
        E, sigma_y = 2.0e10, 4.0e6
        tk = UniaxialTakeda(E=E, sigma_y=sigma_y, b=0.02, alpha=0.0)
        tk.get_response(0.005)
        tk.commit_state()
        s_peak = tk.sigma_trial
        deps = -0.0001
        s_after, Et = tk.get_response(0.005 + deps)
        slope_observed = (s_after - s_peak) / deps
        assert slope_observed == pytest.approx(E, rel=1e-6)

    def test_larger_alpha_gives_more_stiffness_degradation(self):
        """Higher α -> lower K_u after a given excursion."""
        E, sigma_y, eps_max = 2.0e10, 4.0e6, 0.005
        slopes = []
        for alpha in (0.0, 0.3, 0.7):
            tk = UniaxialTakeda(E=E, sigma_y=sigma_y, b=0.02, alpha=alpha)
            tk.get_response(eps_max); tk.commit_state()
            s_peak = tk.sigma_trial
            s_after, _ = tk.get_response(eps_max - 1.0e-5)
            slopes.append(s_peak - s_after)
        # Slopes should be monotonically decreasing with alpha
        assert slopes[0] > slopes[1] > slopes[2]


# ============================================================ Pivot

class TestUniaxialPivot:
    """Single-pivot Dowell-Seible-Hines model: load-reversal
    trajectories aimed at pivot points outside the yield envelope."""

    def test_rejects_alpha_less_than_one(self):
        with pytest.raises(ValueError, match="alpha"):
            UniaxialPivot(E=1.0, sigma_y=1.0, alpha=0.5)

    def test_monotonic_follows_bilinear_envelope(self):
        E, sigma_y, b = 2.0e10, 4.0e6, 0.02
        pv = UniaxialPivot(E=E, sigma_y=sigma_y, b=b, alpha=5.0)
        eps_y = sigma_y / E
        s, Et = pv.get_response(5.0 * eps_y)
        s_theory = sigma_y + b * E * 4.0 * eps_y
        assert s == pytest.approx(s_theory, rel=1e-12)
        assert Et == pytest.approx(b * E)

    def test_reversal_trajectory_aims_at_opposite_pivot(self):
        """After loading to ε_max_pos, the unloading line should pass
        through the negative pivot P_neg = (-α·ε_y, -α·σ_y)."""
        E, sigma_y, b, alpha = 2.0e10, 4.0e6, 0.02, 5.0
        pv = UniaxialPivot(E=E, sigma_y=sigma_y, b=b, alpha=alpha)
        eps_y = sigma_y / E
        # Load to +max
        eps_max = 5.0 * eps_y
        pv.get_response(eps_max); pv.commit_state()
        sg_pos = pv.sigma_trial
        # The unloading line goes from (eps_max, sg_pos) toward
        # P_neg = (-α·eps_y, -α·sigma_y).  At any interior strain eps
        # the response should sit exactly on that line.
        P_neg_x = -alpha * eps_y
        P_neg_y = -alpha * sigma_y
        slope = (P_neg_y - sg_pos) / (P_neg_x - eps_max)
        eps_test = 2.0 * eps_y     # interior point between -y and +max
        s_test, _ = pv.get_response(eps_test)
        s_expected = sg_pos + slope * (eps_test - eps_max)
        assert s_test == pytest.approx(s_expected, rel=1e-12)

    def test_higher_alpha_gives_steeper_interior_slope(self):
        """Larger α moves the pivot farther out -> the interior
        reload line becomes steeper (closer to elastic E)."""
        E, sigma_y = 2.0e10, 4.0e6
        eps_y = sigma_y / E
        eps_max = 5.0 * eps_y
        slopes = []
        for alpha in (2.0, 5.0, 20.0):
            pv = UniaxialPivot(E=E, sigma_y=sigma_y, b=0.02, alpha=alpha)
            pv.get_response(eps_max); pv.commit_state()
            sg_pos = pv.sigma_trial
            deps = -1.0e-5
            s_after, Et = pv.get_response(eps_max + deps)
            slopes.append(Et)
        # Monotonic: slope grows with alpha (toward elastic E)
        assert slopes[0] < slopes[1] < slopes[2]

    def test_pivot_loops_are_symmetric(self):
        """For a symmetric +/-/+ history, the loop closes on itself."""
        pv = UniaxialPivot(E=2.0e10, sigma_y=4.0e6, b=0.02, alpha=5.0)
        s_history = []
        for eps in (0.005, -0.005, 0.005):
            pv.get_response(eps); pv.commit_state()
            s_history.append(pv.sigma_trial)
        # Backbone stresses are symmetric and stable
        assert s_history[0] == pytest.approx(-s_history[1], rel=1e-12)
        assert s_history[2] == pytest.approx(s_history[0], rel=1e-12)


# ============================================================ IMK

class TestUniaxialIMK:
    """Ibarra-Medina-Krawinkler trilinear-backbone collapse model."""

    def test_rejects_non_negative_alpha_pc(self):
        with pytest.raises(ValueError, match="alpha_pc"):
            UniaxialIMK(E=1.0, sigma_y=1.0, alpha_pc=0.0)
        with pytest.raises(ValueError, match="alpha_pc"):
            UniaxialIMK(E=1.0, sigma_y=1.0, alpha_pc=0.1)

    def test_rejects_eps_cap_below_yield(self):
        with pytest.raises(ValueError, match="eps_cap"):
            UniaxialIMK(E=1.0, sigma_y=1.0, eps_cap=0.5)

    def test_rejects_invalid_sigma_res_ratio(self):
        with pytest.raises(ValueError, match="sigma_res_ratio"):
            UniaxialIMK(E=1.0, sigma_y=1.0, sigma_res_ratio=1.5)

    def test_monotonic_backbone_elastic_region(self):
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y)
        s, Et = mat.get_response(mat.eps_y * 0.5)
        assert s == pytest.approx(E * mat.eps_y * 0.5, rel=1e-12)
        assert Et == pytest.approx(E)

    def test_monotonic_backbone_hardening_to_cap(self):
        """Past yield to capping point: slope = b*E."""
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y, b=0.03)
        # At capping
        s, Et = mat.get_response(mat.eps_cap)
        assert s == pytest.approx(mat.sigma_cap, rel=1e-12)
        assert Et == pytest.approx(0.03 * E)

    def test_monotonic_backbone_post_cap_negative(self):
        """Past capping: slope = alpha_pc * E (negative)."""
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y, alpha_pc=-0.1)
        # Midway between cap and res
        eps_mid = 0.5 * (mat.eps_cap + mat.eps_res)
        s, Et = mat.get_response(eps_mid)
        assert Et < 0.0
        assert Et == pytest.approx(-0.1 * E, rel=1e-12)
        # Below sigma_cap (since we're past it)
        assert s < mat.sigma_cap

    def test_monotonic_backbone_residual_plateau(self):
        """Past eps_res: sigma = sigma_res, slope = 0."""
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y, sigma_res_ratio=0.4)
        eps_mid = 0.5 * (mat.eps_res + mat.eps_ult)
        s, Et = mat.get_response(eps_mid)
        assert s == pytest.approx(mat.sigma_res, rel=1e-12)
        assert Et == pytest.approx(0.0, abs=1e-12)

    def test_monotonic_backbone_fracture(self):
        """Past eps_ult: sigma drops to zero."""
        mat = UniaxialIMK(E=2.0e11, sigma_y=4.0e8)
        s, Et = mat.get_response(mat.eps_ult * 1.1)
        assert s == pytest.approx(0.0, abs=1e-12)
        assert Et == pytest.approx(0.0, abs=1e-12)

    def test_unloading_from_peak_is_elastic(self):
        """The first unloading after a backbone excursion uses
        elastic slope E (peak-oriented memory)."""
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y)
        eps_peak = mat.eps_cap * 0.7    # in hardening branch
        mat.get_response(eps_peak); mat.commit_state()
        s_peak = mat.sigma_trial
        deps = -1.0e-5
        s_after, Et = mat.get_response(eps_peak + deps)
        slope = (s_after - s_peak) / deps
        assert slope == pytest.approx(E, rel=1e-6)
        assert Et == pytest.approx(E)

    def test_post_cap_excursion_loses_strength_on_return(self):
        """After a post-cap excursion, the recorded peak σ is below
        the original σ_cap. The next loading toward + sees a reduced
        effective peak."""
        E, sigma_y = 2.0e11, 4.0e8
        mat = UniaxialIMK(E=E, sigma_y=sigma_y, b=0.03, alpha_pc=-0.1,
                            sigma_res_ratio=0.4)
        # Push past capping into post-cap region
        eps_post_cap = mat.eps_cap + 0.3 * (mat.eps_res - mat.eps_cap)
        mat.get_response(eps_post_cap); mat.commit_state()
        sigma_at_peak = mat.sigma_trial
        assert sigma_at_peak < mat.sigma_cap
        # Unload to 0, reload
        mat.get_response(0.0); mat.commit_state()
        mat.get_response(eps_post_cap); mat.commit_state()
        # Should reach approximately the same backbone point
        assert mat.sigma_trial == pytest.approx(sigma_at_peak, rel=1e-6)


# ============================================================ BRB

class TestUniaxialBRB:
    """Buckling-restrained-brace material: asymmetric backbone
    (compression overstrength β) + combined kinematic/isotropic
    hardening."""

    def test_rejects_beta_below_one(self):
        with pytest.raises(ValueError, match="beta"):
            UniaxialBRB(E=1.0, sigma_y=1.0, beta=0.9)

    def test_rejects_negative_a_iso(self):
        with pytest.raises(ValueError, match="a_iso"):
            UniaxialBRB(E=1.0, sigma_y=1.0, a_iso=-1.0)

    def test_first_tension_yield_at_sigma_y(self):
        """First yield in tension occurs at +σ_y exactly."""
        E, sigma_y = 2.0e11, 4.0e8
        br = UniaxialBRB(E=E, sigma_y=sigma_y, b=0.0, beta=1.1)
        # Right at yield
        s, _ = br.get_response(sigma_y / E)
        assert s == pytest.approx(sigma_y, rel=1e-12)

    def test_first_compression_yield_at_beta_sigma_y(self):
        """First yield in pure compression (no preload) occurs at
        -β·σ_y exactly."""
        E, sigma_y, beta = 2.0e11, 4.0e8, 1.10
        br = UniaxialBRB(E=E, sigma_y=sigma_y, b=0.0, beta=beta)
        # Drive into compression past first elastic limit
        eps_test = -beta * sigma_y / E
        s, _ = br.get_response(eps_test)
        # Right at the asymmetric yield
        assert s == pytest.approx(-beta * sigma_y, rel=1e-12)

    def test_symmetric_brb_recovers_kinematic_bilinear(self):
        """β = 1, a_iso = 0 -> identical to UniaxialBilinear."""
        E, sigma_y = 2.0e11, 4.0e8
        br = UniaxialBRB(E=E, sigma_y=sigma_y, b=0.05, beta=1.0,
                          a_iso=0.0)
        kin = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.05)
        # Cyclic strain history
        eps_history = [0.001, 0.005, 0.0, -0.005, 0.005]
        for eps in eps_history:
            s_br, _ = br.get_response(eps); br.commit_state()
            s_kin, _ = kin.get_response(eps); kin.commit_state()
            assert s_br == pytest.approx(s_kin, rel=1e-10)

    def test_isotropic_growth_expands_envelope_over_cycles(self):
        """With a_iso > 0, successive +max stresses grow."""
        E, sigma_y = 2.0e11, 4.0e8
        br = UniaxialBRB(E=E, sigma_y=sigma_y, b=0.02, beta=1.1,
                          a_iso=30.0)
        s_pos = []
        for _ in range(3):
            br.get_response(+0.005); br.commit_state()
            s_pos.append(br.sigma_trial)
            br.get_response(-0.005); br.commit_state()
        # +max stress should grow each cycle
        assert s_pos[1] > s_pos[0]
        assert s_pos[2] > s_pos[1]

    def test_cumulative_plastic_strain_grows(self):
        """Each plastic excursion contributes to cumulative p."""
        br = UniaxialBRB(E=2.0e11, sigma_y=4.0e8)
        ps = []
        for eps in (0.005, -0.005, 0.005):
            br.get_response(eps); br.commit_state()
            ps.append(br.p_committed)
        assert ps[0] > 0.0
        assert ps[1] > ps[0]
        assert ps[2] > ps[1]
