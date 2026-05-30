"""Phase 37 tests -- steel + RC connections.

Covers Krawinkler panel zone, AISC 358 RBS, Richard-Abbott PR,
and bolt/weld strength (AISC + IS 800).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design.connections import (
    PanelZoneProperties,
    Pr_preset,
    RBSGeometry,
    RichardAbbottParams,
    aisc358_recommended_RBS,
    block_shear_aisc,
    bolt_bearing_aisc,
    bolt_bearing_is800,
    bolt_shear_aisc,
    bolt_shear_is800,
    build_panel_zone_material,
    fillet_weld_aisc,
    fillet_weld_is800,
    krawinkler_panel_zone,
    reduced_beam_section,
)
from femsolver.materials.uniaxial.bilinear import UniaxialBilinear


# ============================================================ Panel zone

class TestKrawinklerPanelZone:
    def test_basic_W14x90_W24x84(self):
        """A typical AISC W14x90 / W24x84 joint gives V_y around 1 MN."""
        pz = krawinkler_panel_zone(
            f_y=345e6, d_c=0.355, t_p=0.0112,
            d_b=0.612, b_cf=0.368, t_cf=0.018,
        )
        assert isinstance(pz, PanelZoneProperties)
        assert 8.0e5 < pz.V_y < 1.5e6      # 800 kN - 1500 kN
        assert pz.K_p == pytest.approx(0.06 * pz.K_e, rel=1e-12)
        assert pz.gamma_y == pytest.approx(pz.V_y / pz.K_e, rel=1e-12)

    def test_boundary_term_positive(self):
        pz = krawinkler_panel_zone(
            f_y=345e6, d_c=0.355, t_p=0.0112,
            d_b=0.612, b_cf=0.368, t_cf=0.018,
        )
        # Boundary contribution should be a positive enhancement
        # from the column flanges (0 < term < 1 typically)
        assert 0.0 < pz.b_over_a < 1.0

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            krawinkler_panel_zone(
                f_y=-1.0, d_c=0.1, t_p=0.01,
                d_b=0.5, b_cf=0.3, t_cf=0.02,
            )
        with pytest.raises(ValueError, match="K_p_ratio"):
            krawinkler_panel_zone(
                f_y=345e6, d_c=0.355, t_p=0.0112,
                d_b=0.612, b_cf=0.368, t_cf=0.018,
                K_p_ratio=2.0,
            )

    def test_build_uniaxial_material(self):
        pz = krawinkler_panel_zone(
            f_y=345e6, d_c=0.355, t_p=0.0112,
            d_b=0.612, b_cf=0.368, t_cf=0.018,
        )
        mat = build_panel_zone_material(pz)
        assert isinstance(mat, UniaxialBilinear)
        assert mat.E == pytest.approx(pz.K_e_rot, rel=1e-12)
        assert mat.sigma_y == pytest.approx(pz.M_y_joint, rel=1e-12)


# ============================================================ RBS

class TestRBS:
    def test_recommended_dimensions_are_within_aisc(self):
        rbs = aisc358_recommended_RBS(
            d=0.612, b_f=0.229, t_f=0.0236,
            f_y=345e6, Z_x=3.99e-3, L_clear=6.0,
        )
        assert rbs.aisc_a_ok and rbs.aisc_b_ok and rbs.aisc_c_ok

    def test_Z_RBS_reduces_Z_x(self):
        rbs = aisc358_recommended_RBS(
            d=0.612, b_f=0.229, t_f=0.0236,
            f_y=345e6, Z_x=3.99e-3, L_clear=6.0,
        )
        assert rbs.Z_RBS < 3.99e-3
        # Typical reduction 60-75% of original; check a wide range
        assert 0.50 < rbs.Z_RBS / 3.99e-3 < 0.80

    def test_M_pr_face_amplified_by_Cpr_and_geometry(self):
        rbs = aisc358_recommended_RBS(
            d=0.612, b_f=0.229, t_f=0.0236,
            f_y=345e6, Z_x=3.99e-3, L_clear=6.0,
            Cpr=1.15,
        )
        # M_pr at face > M_p_RBS (Cpr * L_clear / L_eff factor)
        assert rbs.M_pr_face > rbs.M_p_RBS

    def test_invalid_cut_depth_raises(self):
        # c too large -> negative b_f_reduced
        with pytest.raises(ValueError, match="cut depth"):
            reduced_beam_section(
                f_y=345e6, Z_x=3.99e-3,
                d=0.612, b_f=0.229, t_f=0.0236,
                a=0.15, b=0.45, c=0.20,    # c=0.20 vs b_f=0.229
                L_clear=6.0,
            )

    def test_recommended_picks_midpoints(self):
        rbs = aisc358_recommended_RBS(
            d=0.612, b_f=0.229, t_f=0.0236,
            f_y=345e6, Z_x=3.99e-3, L_clear=6.0,
        )
        assert rbs.a == pytest.approx(0.625 * 0.229, rel=1e-12)
        assert rbs.b == pytest.approx(0.75 * 0.612, rel=1e-12)
        assert rbs.c == pytest.approx(0.20 * 0.229, rel=1e-12)


# ============================================================ Richard-Abbott PR

class TestRichardAbbott:
    def test_M_at_zero_rotation_is_zero(self):
        ra = Pr_preset("end_plate_4_bolts")
        assert ra.M(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_initial_stiffness_at_origin(self):
        """Tangent at theta=0 should equal R_ki."""
        ra = RichardAbbottParams(
            R_ki=1.0e8, R_kp=1.0e6, M_0=200.0e3, n=2.0,
        )
        # Tangent at theta=0
        t = ra.tangent(0.0)
        assert t == pytest.approx(ra.R_ki, rel=1e-9)

    def test_asymptote_R_kp(self):
        """At very large theta, M ~ M_0 + R_kp · theta (asymptote)."""
        ra = RichardAbbottParams(
            R_ki=1.0e8, R_kp=1.0e6, M_0=200.0e3, n=3.0,
        )
        theta_big = 1.0
        # Asymptotically: M / theta -> R_kp
        M_at = ra.M(theta_big)
        slope = M_at / theta_big
        assert abs(slope - ra.R_kp) / ra.R_kp < 0.20    # within 20%

    def test_antisymmetric(self):
        ra = Pr_preset("tee_stub")
        for t in [0.001, 0.01, 0.05]:
            assert ra.M(t) == pytest.approx(-ra.M(-t), rel=1e-9)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="R_ki"):
            RichardAbbottParams(R_ki=-1, R_kp=0.5, M_0=100, n=2)
        with pytest.raises(ValueError, match="R_kp"):
            RichardAbbottParams(R_ki=1e6, R_kp=2e6, M_0=100, n=2)
        with pytest.raises(ValueError, match="M_0"):
            RichardAbbottParams(R_ki=1e6, R_kp=1e5, M_0=-1, n=2)
        with pytest.raises(ValueError, match="n"):
            RichardAbbottParams(R_ki=1e6, R_kp=1e5, M_0=100, n=0)

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="unknown PR preset"):
            Pr_preset("does_not_exist")


# ============================================================ Bolts (AISC + IS 800)

class TestBoltShear:
    def test_aisc_basic(self):
        # 4-M22 A325-X: A_b=3.80e-4 m^2, F_nv=0.563*825 MPa = 464.5 MPa
        res = bolt_shear_aisc(
            n_bolts=4, A_b=3.80e-4, F_nv=0.563 * 825e6,
        )
        # phi*Rn per bolt = 0.75 * 464.5e6 * 3.80e-4 = 132.4 kN
        # Total = 4 * 132.4 = 529.6 kN
        assert res.V_d_total == pytest.approx(529.6e3, rel=0.02)

    def test_is800_basic(self):
        res = bolt_shear_is800(
            n_bolts=4, f_ub=800e6, A_nb=303e-6, A_sb=380e-6,
            n_shear_planes_thread=0, n_shear_planes_shank=1,
        )
        # Per bolt: 1*380e-6*800e6 / (sqrt(3)*1.25) = 140.4 kN
        # Total: 4 * 140.4 = 561.7 kN
        assert res.V_d_total == pytest.approx(561.7e3, rel=0.02)


class TestBoltBearing:
    def test_aisc(self):
        # M22 bolt (d_b=22mm) on 10mm plate, L_c=40mm, F_u=400 MPa
        res = bolt_bearing_aisc(
            n_bolts=1, d_b=0.022, t=0.010,
            F_u=400e6, L_c=0.040,
        )
        # min(1.2*0.04*0.01*400e6, 3.0*0.022*0.01*400e6) = min(192e3, 264e3) = 192e3
        # phi*Rn = 0.75 * 192e3 = 144 kN
        assert res.V_d_single == pytest.approx(144e3, rel=0.02)

    def test_is800(self):
        # M22 on 10mm plate, e=40 mm, f_u=400, f_ub=800
        res = bolt_bearing_is800(
            n_bolts=1, d_b=0.022, t=0.010,
            f_u=400e6, f_ub=800e6, e=0.040,
        )
        # k_b = min(e/3d0, f_ub/f_u, 1) = min(40/(3*23.5), 2, 1) = min(0.567, ..) = 0.567
        # V_dpb = 2.5 * 0.567 * 0.022 * 0.010 * 400e6 / 1.25 = 99.8 kN
        assert res.V_d_single == pytest.approx(99.8e3, rel=0.05)


class TestBlockShear:
    def test_aisc(self):
        # Idealized values
        res = block_shear_aisc(
            A_gv=600e-6, A_nv=400e-6,
            A_nt=200e-6,
            F_y=350e6, F_u=450e6,
        )
        # 0.6 * 450e6 * 400e-6 + 1.0 * 450e6 * 200e-6 = 108000 + 90000 = 198 kN
        # vs 0.6 * 350e6 * 600e-6 + 90000 = 126e3 + 90e3 = 216e3
        # min = 198 kN; R_d = 0.75 * 198 = 148.5 kN
        assert res.R_d == pytest.approx(148.5e3, rel=0.02)


# ============================================================ Welds

class TestWelds:
    def test_fillet_aisc_8mm_E70(self):
        weld = fillet_weld_aisc(leg_size=0.008, F_EXX=480e6)
        # 0.6 * 480e6 * 0.707 * 0.008 = 1629 N/mm = 1.63 kN/mm nominal
        # 0.75 * 1629 = 1221 N/mm design
        assert weld.R_d_per_length == pytest.approx(1221.0e3, rel=0.02)

    def test_fillet_is800_8mm(self):
        weld = fillet_weld_is800(leg_size=0.008, f_u_weld=410e6)
        # f_u * 0.7 * 0.008 / (sqrt(3) * 1.25) = 1060 N/mm
        assert weld.R_d_per_length == pytest.approx(1060e3, rel=0.02)

    def test_invalid_weld_size(self):
        with pytest.raises(ValueError):
            fillet_weld_aisc(leg_size=-0.005, F_EXX=480e6)
