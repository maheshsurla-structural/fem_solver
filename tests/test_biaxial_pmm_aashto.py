"""AASHTO LRFD 2024 biaxial P-M-M tests.

The AASHTO normal-strength stress block equals the ACI 318 Whitney
block, so the *nominal* surface must match ACI to round-off; only the
resistance factors phi differ (Art. 5.5.4.2: 0.75 compression-
controlled tied/spiral, 0.90 tension-controlled RC, 1.00 PC).
"""
from __future__ import annotations

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    aashto_params,
    alpha_1_aashto,
    biaxial_pmm_point,
    biaxial_pmm_point_aashto,
    biaxial_pmm_surface,
    biaxial_pmm_surface_aashto,
    phi_for_strain_aashto,
)
from femsolver.sections import ReinforcementLayout, rc_rectangular_section


def _make_rc(b=0.4, h=0.6):
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h,
        bottom_bars=[(510e-6, "#8")] * 4,
        top_bars=[(510e-6, "#8")] * 4,
        bottom_cover=0.05, top_cover=0.05,
    )
    return rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)


# ============================================================ alpha_1

class TestAlpha1:
    def test_normal_strength(self):
        # f'c <= 10 ksi (69 MPa) -> 0.85
        assert alpha_1_aashto(30e6) == pytest.approx(0.85)
        assert alpha_1_aashto(55e6) == pytest.approx(0.85)

    def test_high_strength_reduction(self):
        # 80 MPa = 11.603 ksi -> 0.85 - 0.02*(11.603-10) = 0.8179
        assert alpha_1_aashto(80e6) == pytest.approx(0.8179, abs=1e-3)

    def test_floor(self):
        # very high strength floors at 0.75
        assert alpha_1_aashto(150e6) == pytest.approx(0.75)


# ============================================================ phi function

class TestPhiAashto:
    def test_compression_controlled(self):
        eps_y = 420e6 / 200e9
        assert phi_for_strain_aashto(0.0, eps_y) == pytest.approx(0.75)
        assert phi_for_strain_aashto(eps_y, eps_y) == pytest.approx(0.75)

    def test_tension_controlled_rc(self):
        eps_y = 420e6 / 200e9
        assert phi_for_strain_aashto(0.005, eps_y) == pytest.approx(0.90)
        assert phi_for_strain_aashto(0.02, eps_y) == pytest.approx(0.90)

    def test_tension_controlled_pc(self):
        eps_y = 420e6 / 200e9
        assert phi_for_strain_aashto(
            0.02, eps_y, prestressed=True) == pytest.approx(1.00)

    def test_transition_is_linear(self):
        eps_y = 0.002
        # midpoint between eps_y=0.002 and 0.005 -> phi halfway 0.75..0.90
        mid = phi_for_strain_aashto(0.0035, eps_y)
        assert mid == pytest.approx(0.75 + 0.5 * (0.90 - 0.75))


# ============================================================ params

class TestAashtoParams:
    def test_normal_strength_matches_aci_block(self):
        p = aashto_params(f_c_prime=30e6, f_y=420e6)
        assert p.code == "AASHTO"
        assert p.sigma_block == pytest.approx(0.85 * 30e6)
        assert p.eps_cu == pytest.approx(0.003)
        assert p.f_yd == pytest.approx(420e6)   # phi applied separately
        assert p.apply_phi_table is True
        assert p.phi_func is not None


# ============================================================ nominal == ACI

class TestNominalMatchesAci:
    def test_point_nominal_matches_aci(self):
        sec = _make_rc()
        for c in (0.15, 0.3, 0.5):
            pa = biaxial_pmm_point(sec, 0.0, c, f_c_prime=30e6, f_y=420e6)
            px = biaxial_pmm_point_aashto(
                sec, 0.0, c, f_c_prime=30e6, f_y=420e6)
            assert px.P_n == pytest.approx(pa.P_n, rel=1e-9, abs=1.0)
            assert px.M_nz == pytest.approx(pa.M_nz, rel=1e-9, abs=1.0)

    def test_surface_P_o_matches_aci(self):
        sec = _make_rc()
        sa = biaxial_pmm_surface(sec, f_c_prime=30e6, f_y=420e6,
                                 n_angles=4, n_depths=6)
        sx = biaxial_pmm_surface_aashto(sec, f_c_prime=30e6, f_y=420e6,
                                        n_angles=4, n_depths=6)
        assert sx.P_o == pytest.approx(sa.P_o, rel=1e-9)
        assert sx.P_n_max == pytest.approx(sa.P_n_max, rel=1e-9)


# ============================================================ phi differs

class TestPhiDiffersFromAci:
    def test_compression_controlled_phi(self):
        """At a deep NA (compression-controlled) the tied ACI phi is
        0.65 while AASHTO is 0.75."""
        sec = _make_rc()
        pa = biaxial_pmm_point(sec, 0.0, 0.55, f_c_prime=30e6, f_y=420e6)
        px = biaxial_pmm_point_aashto(
            sec, 0.0, 0.55, f_c_prime=30e6, f_y=420e6)
        assert pa.phi == pytest.approx(0.65)
        assert px.phi == pytest.approx(0.75)

    def test_tension_controlled_phi_matches(self):
        """At pure flexure (tension-controlled) both give 0.90 RC."""
        sec = _make_rc()
        pa = biaxial_pmm_point(sec, 0.0, 0.08, f_c_prime=30e6, f_y=420e6)
        px = biaxial_pmm_point_aashto(
            sec, 0.0, 0.08, f_c_prime=30e6, f_y=420e6)
        if pa.epsilon_t >= 0.005:
            assert pa.phi == pytest.approx(0.90)
            assert px.phi == pytest.approx(0.90)


# ============================================================ surface + errors

class TestAashtoSurface:
    def test_builds_surface(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_aashto(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=8, n_depths=8)
        assert len(surf.points) == 64

    def test_spiral_cap(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_aashto(
            sec, f_c_prime=30e6, f_y=420e6, spiral=True,
            n_angles=4, n_depths=4)
        assert surf.P_n_max == pytest.approx(0.85 * surf.P_o)

    def test_rejects_invalid_strength(self):
        sec = _make_rc()
        with pytest.raises(ValueError, match="positive"):
            biaxial_pmm_surface_aashto(sec, f_c_prime=-1, f_y=420e6)
