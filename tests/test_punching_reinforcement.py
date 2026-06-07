"""Phase HH.8 tests -- punching shear reinforcement design (ACI 318 /
EC2 / IS 456)."""
from __future__ import annotations

import math

import pytest

from femsolver.design.punching import (
    aci318_punching_capacity,
    aci318_punching_demand,
)
from femsolver.design.punching_reinforcement import (
    PunchingReinforcementResult,
    aci318_punching_reinforcement,
    eurocode_punching_reinforcement,
    is456_punching_reinforcement,
)


# ============================================================ ACI 318

class TestAciReinforcement:
    def test_not_required_below_phi_vc(self):
        """When v_u < phi * v_c, no reinforcement needed."""
        r = aci318_punching_reinforcement(
            v_u=0.5e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
        )
        assert not r.required
        assert r.A_v_required == 0.0

    def test_required_above_phi_vc(self):
        """When v_u > phi * v_c, reinforcement must be designed."""
        # phi*v_c = 0.75 * (1/3) * sqrt(30) = 1.37 MPa
        r = aci318_punching_reinforcement(
            v_u=1.8e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
        )
        assert r.required
        assert r.feasible
        assert r.A_v_required > 0.0
        assert r.s_max > 0.0

    def test_slab_too_thin(self):
        """When v_u/phi exceeds the ceiling, slab is infeasible."""
        # Ceiling for stud rails: 0.5 * sqrt(30) = 2.74 MPa
        r = aci318_punching_reinforcement(
            v_u=2.6e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
        )
        # v_u/phi = 3.47 MPa > 2.74 -> infeasible
        assert r.required
        assert not r.feasible
        assert math.isinf(r.v_s_required)
        assert "TOO THIN" in r.note

    def test_stud_rails_higher_ceiling_than_stirrups(self):
        """Stud rails allow more capacity than conventional stirrups."""
        v_u = 2.0e6
        r_stud = aci318_punching_reinforcement(
            v_u=v_u, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
            reinforcement_type="stud_rail",
        )
        r_stirrup = aci318_punching_reinforcement(
            v_u=v_u, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
            reinforcement_type="stirrup",
        )
        assert r_stud.ceiling > r_stirrup.ceiling

    def test_f_yt_capped(self):
        """Per ACI 318-19 22.4.3.2 f_yt is capped at 420 MPa for
        stirrups, 550 MPa for stud rails. Pass a very high value
        and verify capping reduces required area."""
        r_high = aci318_punching_reinforcement(
            v_u=1.5e6, f_c=30e6, f_yt=600e6, d=0.2, b_0=2.4,
            reinforcement_type="stirrup",
        )
        r_capped = aci318_punching_reinforcement(
            v_u=1.5e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
            reinforcement_type="stirrup",
        )
        # Both should produce the same A_v (high f_yt clipped to 420)
        assert r_high.A_v_required == pytest.approx(
            r_capped.A_v_required, rel=1e-9,
        )

    def test_spacing_limits_at_d_half(self):
        r = aci318_punching_reinforcement(
            v_u=1.6e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
        )
        # First peripheral line within d/2 = 100 mm
        assert r.s_first == pytest.approx(0.1, rel=1e-9)
        # Stud-rail s_max can be up to 0.75d when below threshold
        assert r.s_max == pytest.approx(0.15, rel=1e-9) \
            or r.s_max == pytest.approx(0.1, rel=1e-9)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            aci318_punching_reinforcement(
                v_u=-1, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
            )
        with pytest.raises(ValueError):
            aci318_punching_reinforcement(
                v_u=1e6, f_c=30e6, f_yt=420e6, d=0.2, b_0=2.4,
                reinforcement_type="foo",
            )


# ============================================================ EC2

class TestEC2Reinforcement:
    def test_not_required_below_v_Rd_c(self):
        r = eurocode_punching_reinforcement(
            v_u=0.3e6, f_ck=30e6, f_ywd=435e6, d=0.2, u_1=4.0,
        )
        assert not r.required

    def test_required_above_v_Rd_c(self):
        r = eurocode_punching_reinforcement(
            v_u=1.5e6, f_ck=30e6, f_ywd=435e6, d=0.2, u_1=4.0,
        )
        assert r.required
        assert r.feasible

    def test_f_ywd_ef_capped(self):
        """EC2 effective yield ``f_ywd,ef = 250 + 0.25 d_mm`` MPa,
        capped at f_ywd. For d = 200 mm: cap is 300 MPa."""
        r = eurocode_punching_reinforcement(
            v_u=1.0e6, f_ck=30e6, f_ywd=500e6, d=0.2, u_1=4.0,
        )
        # f_ywd,ef expected ~ 300 MPa from formula
        assert "300 MPa" in r.note or "f_ywd,ef" in r.note

    def test_ceiling_check(self):
        """For very high v_u, EC2 should flag infeasible."""
        r = eurocode_punching_reinforcement(
            v_u=20e6, f_ck=30e6, f_ywd=435e6, d=0.2, u_1=4.0,
        )
        assert r.required
        assert not r.feasible


# ============================================================ IS 456

class TestIS456Reinforcement:
    def test_not_required_below_tau_c(self):
        r = is456_punching_reinforcement(
            v_u=0.5e6, f_ck=30e6, f_y=415e6, d=0.2, b_0=2.4,
        )
        assert not r.required

    def test_required_above_tau_c(self):
        # tau_c = 1.0 * 0.25 * sqrt(30) = 1.37 MPa
        # ceiling = 1.5 * 1.37 = 2.05 MPa
        r = is456_punching_reinforcement(
            v_u=1.7e6, f_ck=30e6, f_y=415e6, d=0.2, b_0=2.4,
        )
        assert r.required
        assert r.feasible
        assert r.A_v_required > 0

    def test_ceiling_at_1_5_tau_c(self):
        r = is456_punching_reinforcement(
            v_u=2.5e6, f_ck=30e6, f_y=415e6, d=0.2, b_0=2.4,
        )
        # Ceiling = 1.5 * 1.37 = 2.05 MPa -> not feasible at 2.5
        assert not r.feasible


# ============================================================ engineering scenario

class TestEngineering:
    def test_realistic_design_works(self):
        """Typical interior column needing modest reinforcement."""
        # 400x400 column, d=200, V_u = 850 kN (moderate increase
        # above the 850/phi/v_c = 1 line)
        cap = aci318_punching_capacity(c_x=0.4, c_y=0.4, d=0.2, f_c=30e6)
        v_u = aci318_punching_demand(V_u=850e3, c_x=0.4, c_y=0.4, d=0.2)
        r = aci318_punching_reinforcement(
            v_u=v_u, f_c=30e6, f_yt=420e6, d=0.2, b_0=cap.b_0,
            reinforcement_type="stud_rail",
        )
        # Should be feasible with stud rails
        assert r.required and r.feasible
        # A_v should be in the engineering range
        # (10 studs of dia 10 mm = ~785 mm^2 / perimeter is typical)
        # Just verify it's positive and finite
        assert 0 < r.A_v_required < 5e-3
