"""Phase 38 tests -- bridge engineering (influence lines, PT tendons,
creep/shrinkage, composite section).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.bridges import (
    aashto_hl93_tandem,
    aashto_hl93_truck,
    aashto_lane_moment_simple_span,
    anchorage_slip_loss,
    cebfip_creep_coefficient,
    cebfip_shrinkage,
    composite_fiber_stresses,
    composite_girder_deck,
    equivalent_uniform_load_parabolic,
    evaluate_response_for_position,
    friction_loss,
    influence_line_simple_span_moment,
    influence_line_simple_span_shear,
    irc_class_a,
    irc_class_70r_truck,
    max_response_for_moving_load,
    max_truck_envelope_simple_span,
    parabolic_drape_profile,
    steel_relaxation_loss_ratio,
    MovingLoad,
)


# ============================================================ influence lines

class TestInfluenceLines:
    def test_moment_at_midspan_max_is_L_over_4(self):
        """For SS beam, max IL moment at midspan = L/4."""
        L = 20.0
        xi = np.linspace(0, L, 21)
        eta = influence_line_simple_span_moment(L=L, x=L/2, xi=xi)
        assert eta.max() == pytest.approx(L / 4, rel=1.0e-12)

    def test_moment_zero_at_supports(self):
        L = 20.0
        xi = np.array([0.0, L])
        eta = influence_line_simple_span_moment(L=L, x=L/2, xi=xi)
        np.testing.assert_allclose(eta, 0.0, atol=1e-12)

    def test_shear_at_midspan_sign(self):
        """At x = L/2: IL_V(xi=0) = 0, IL_V(xi=L) = 0,
        IL_V(xi just right of midspan) = +0.5."""
        L = 20.0
        xi = np.array([0.0, L/2 + 0.001, L])
        eta = influence_line_simple_span_shear(L=L, x=L/2, xi=xi)
        assert eta[0] == pytest.approx(0.0, abs=1e-12)
        assert eta[1] == pytest.approx(0.5, abs=0.001)
        assert eta[2] == pytest.approx(0.0, abs=1e-12)

    def test_il_validates_x_range(self):
        with pytest.raises(ValueError, match=r"x"):
            influence_line_simple_span_moment(L=10.0, x=15.0,
                                                xi=np.array([0.0, 5.0]))


# ============================================================ moving loads

class TestMovingLoads:
    def test_aashto_truck_total_weight(self):
        t = aashto_hl93_truck()
        # 35 + 145 + 145 = 325 kN
        assert t.total_load == pytest.approx(325e3, rel=1.0e-12)

    def test_aashto_tandem_axles(self):
        t = aashto_hl93_tandem()
        assert t.axle_loads.size == 2
        assert t.total_length == pytest.approx(1.2, rel=1.0e-12)

    def test_moving_load_drops_off_span(self):
        L = 10.0
        load = MovingLoad(
            axle_loads=np.array([100.0e3]),
            axle_offsets=np.array([0.0]),
        )
        def il(xi):
            return influence_line_simple_span_moment(L=L, x=L/2, xi=xi)
        # Head at 5 (midspan): response = 100e3 * L/4 = 250e3
        r = evaluate_response_for_position(
            head_position=5.0, moving_load=load, influence_line=il, L=L,
        )
        assert r == pytest.approx(100e3 * L / 4, rel=1e-12)
        # Head past the end (e.g. 12): off-span, expect 0
        r = evaluate_response_for_position(
            head_position=12.0, moving_load=load, influence_line=il, L=L,
        )
        assert r == 0.0

    def test_max_search_finds_peak(self):
        L = 10.0
        load = MovingLoad(
            axle_loads=np.array([100.0e3]),
            axle_offsets=np.array([0.0]),
        )
        def il(xi):
            return influence_line_simple_span_moment(L=L, x=L/2, xi=xi)
        m_max, pos = max_response_for_moving_load(
            moving_load=load, influence_line=il, L=L,
        )
        # Peak when single axle is at midspan
        assert m_max == pytest.approx(100e3 * L / 4, rel=0.01)
        assert pos == pytest.approx(L / 2, rel=0.02)

    def test_hl93_truck_envelope_textbook(self):
        """For a 30 m SS span, the HL-93 truck max moment is
        approximately 2050-2100 kN.m (textbook example)."""
        env = max_truck_envelope_simple_span(L=30.0, x=15.0)
        # Truck envelope at midspan
        assert 1950e3 < env["M_truck"] < 2150e3
        # Lane: 9.34 * 15 * 15 / 2 = 1051
        assert env["M_lane"] == pytest.approx(
            aashto_lane_moment_simple_span(w=9.34e3, L=30.0, x=15.0),
            rel=1e-12,
        )

    def test_irc_vehicles_loadable(self):
        a = irc_class_a()
        seven = irc_class_70r_truck()
        assert a.total_load > 0.0
        assert seven.total_load == pytest.approx(700e3, rel=0.01)

    def test_preset_lookup(self):
        """MovingLoad.preset dispatches to the right factory."""
        truck = MovingLoad.preset("hl93_truck")
        assert truck.total_load == pytest.approx(325e3, rel=1e-12)
        tandem = MovingLoad.preset("hl93_tandem")
        assert tandem.axle_loads.size == 2
        irc_a = MovingLoad.preset("irc_class_a")
        assert irc_a.total_length > 0.0

    def test_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown moving-load"):
            MovingLoad.preset("does_not_exist")


# ============================================================ PT tendon

class TestPTTendon:
    def test_parabolic_drape_length_approx_L(self):
        p = parabolic_drape_profile(L=30.0, drape=0.6, n_segments=50)
        # Total chord length should be slightly > L due to draping
        assert p.total_length > 30.0
        assert p.total_length < 30.1

    def test_friction_loss_monotonic_decrease(self):
        p = parabolic_drape_profile(L=20.0, drape=0.4, n_segments=20)
        fric = friction_loss(p, mu=0.20, k=0.0066)
        assert np.all(np.diff(fric.P_over_P0) <= 1e-12)

    def test_friction_starts_at_one(self):
        p = parabolic_drape_profile(L=20.0, drape=0.4, n_segments=20)
        fric = friction_loss(p, mu=0.20, k=0.0066)
        assert fric.P_over_P0[0] == pytest.approx(1.0, abs=1.0e-12)

    def test_no_friction_no_loss(self):
        p = parabolic_drape_profile(L=20.0, drape=0.4, n_segments=20)
        fric = friction_loss(p, mu=0.0, k=0.0)
        np.testing.assert_allclose(fric.P_over_P0, 1.0)

    def test_anchorage_slip_reduces_anchor_force(self):
        p = parabolic_drape_profile(L=30.0, drape=0.6, n_segments=40)
        ank = anchorage_slip_loss(
            p, P_0=2000e3, mu=0.20, k=0.0066,
            slip=0.006, A_ps=1.0e-3,
        )
        # After seating, P at anchor < P_0
        assert ank.P0_after_seating < 2000e3
        assert ank.l_a > 0.0

    def test_equivalent_UDL_formula(self):
        # w_eq = 8 P e / L^2
        w = equivalent_uniform_load_parabolic(P=1000e3, drape=0.5, L=20.0)
        assert w == pytest.approx(8 * 1000e3 * 0.5 / 20.0 ** 2, rel=1e-12)


# ============================================================ creep / shrinkage

class TestCreepShrinkage:
    def test_creep_at_long_time(self):
        """phi at 50 years for C30 / 28-day loading is about 1.5-2.5."""
        creep = cebfip_creep_coefficient(
            t_days=18250, t0_days=28,
            f_cm=38e6, RH=70.0, h_0=0.20,
        )
        assert 1.4 < creep.phi < 2.5

    def test_creep_zero_at_loading(self):
        creep = cebfip_creep_coefficient(
            t_days=29, t0_days=28,
            f_cm=38e6, RH=70.0,
        )
        assert 0.0 < creep.phi < 1.0     # at very small dt

    def test_shrinkage_long_term_magnitude(self):
        """50-year shrinkage for normal-weight concrete at 70% RH is
        in the range 300-700 microstrain."""
        shr = cebfip_shrinkage(
            t_days=18250, t_s_days=3,
            f_cm=38e6, RH=70.0, h_0=0.20,
        )
        assert -700.0e-6 < shr.eps_cs < -300.0e-6

    def test_relaxation_low_at_short_time(self):
        # 1 hr: log(t/t0) = 0 -> relaxation = 0
        rel = steel_relaxation_loss_ratio(
            t_hours=1.0, t_initial_hours=1.0,
            fpi_over_fpy=0.75,
        )
        assert rel == 0.0

    def test_relaxation_50_yr_low_relax(self):
        rel = steel_relaxation_loss_ratio(
            t_hours=18250 * 24, fpi_over_fpy=0.75,
            relaxation_class="low",
        )
        # log10(50yr/1hr) ~ log10(438000) ~ 5.6
        # rel = 5.6 / 45 * (0.75 - 0.55) = 0.025 = 2.5%
        assert 0.020 < rel < 0.035


# ============================================================ composite section

class TestCompositeSection:
    def test_n_correctly_computed(self):
        props = composite_girder_deck(
            girder_area=0.45, girder_I=0.04,
            girder_y_centroid=0.50, girder_height=1.20,
            deck_width=2.40, deck_thickness=0.20,
            E_girder=34e9, E_deck=28e9,
        )
        assert props.n == pytest.approx(28 / 34, rel=1e-12)

    def test_composite_I_exceeds_girder_I(self):
        """Adding the deck should increase the moment of inertia."""
        props = composite_girder_deck(
            girder_area=0.45, girder_I=0.04,
            girder_y_centroid=0.50, girder_height=1.20,
            deck_width=2.40, deck_thickness=0.20,
            E_girder=34e9, E_deck=28e9,
        )
        assert props.I_t > 0.04

    def test_centroid_moves_upward_with_deck(self):
        """Adding the deck on top moves y_bar up from girder centroid."""
        props = composite_girder_deck(
            girder_area=0.45, girder_I=0.04,
            girder_y_centroid=0.50, girder_height=1.20,
            deck_width=2.40, deck_thickness=0.20,
            E_girder=34e9, E_deck=28e9,
        )
        assert props.y_bar > 0.50

    def test_fiber_stress_signs_for_prestress(self):
        """Under pure prestress P<0 (compression), all fibres should
        be in compression (negative)."""
        props = composite_girder_deck(
            girder_area=0.45, girder_I=0.04,
            girder_y_centroid=0.50, girder_height=1.20,
            deck_width=2.40, deck_thickness=0.20,
            E_girder=34e9, E_deck=28e9,
        )
        sigma = composite_fiber_stresses(
            props=props, P=-1.0e6, M=0.0,
            strand_y_from_bottom=0.10,
        )
        assert sigma.sigma_top_girder < 0
        assert sigma.sigma_bot_girder < 0

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            composite_girder_deck(
                girder_area=-1, girder_I=0.04,
                girder_y_centroid=0.5, girder_height=1.2,
                deck_width=2.4, deck_thickness=0.2,
                E_girder=34e9, E_deck=28e9,
            )
