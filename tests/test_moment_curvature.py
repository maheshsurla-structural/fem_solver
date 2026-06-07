"""Phase II.12 tests -- moment-curvature for RC sections."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    MomentCurvatureResult,
    moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    UniaxialBilinear,
    UniaxialElastic,
)
from femsolver.sections import (
    ReinforcementLayout,
    rc_rectangular_section,
)
from femsolver.sections.section import RebarBar, ReinforcementLayout as RL2


# ============================================================ fixtures

def _make_beam(b=0.3, h=0.6):
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h,
        bottom_bars=[(510e-6, "#8")] * 3,
        top_bars=[(285e-6, "#6")] * 2,
        bottom_cover=0.04, top_cover=0.04,
    )
    return rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)


def _kent_park():
    return ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=12e6, eps_cu=0.005)


def _grade60_steel():
    return UniaxialBilinear(E=200e9, sigma_y=420e6, b=0.01)


# ============================================================ basic behavior

class TestBasicMonotonic:
    def test_zero_kappa_zero_moment(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=20,
        )
        # Curve must start at the origin
        assert res.points[0].kappa == pytest.approx(0.0)
        assert res.points[0].M == pytest.approx(0.0, abs=10.0)

    def test_returns_result_with_points(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=20,
        )
        assert isinstance(res, MomentCurvatureResult)
        assert len(res.points) > 5
        assert res.M_u > 0
        assert res.kappa_u > 0

    def test_moment_increases_then_plateaus(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        Ms = [p.M for p in res.points]
        # Strictly increasing for the first few steps (elastic-cracked)
        assert Ms[1] > Ms[0]
        assert Ms[3] > Ms[1]
        # Ultimate is achieved (didn't fail to find a peak)
        assert res.M_u > 100e3


# ============================================================ matches Whitney to ~10%

class TestVsWhitneyHandCalc:
    """Kent-Park parabolic concrete yields a few % more moment than
    Whitney's rectangular block (which is conservative by design).
    M_u should land within +/- 12% of the Whitney closed-form M_n."""

    def test_singly_reinforced_beam_within_15pct(self):
        # As = 3 * 510e-6 = 1530 mm^2
        # d = h - cover = 0.56 m
        # a = As*fy / (0.85*fc*b)
        # M_n = As*fy*(d - a/2)
        b, h = 0.3, 0.6
        d = h - 0.04
        A_s = 3 * 510e-6
        a = A_s * 420e6 / (0.85 * 30e6 * b)
        M_n_whitney = A_s * 420e6 * (d - a / 2)

        res = moment_curvature(
            _make_beam(b=b, h=h),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        # Kent-Park gives ~5-12% above Whitney (Kent-Park area > Whitney area)
        assert 1.00 <= res.M_u / M_n_whitney <= 1.15


# ============================================================ first yield + ultimate

class TestYieldAndUltimate:
    def test_first_yield_detected(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        assert res.M_y is not None
        assert res.kappa_y is not None
        # Yield happens before ultimate
        assert res.kappa_y < res.kappa_u
        assert res.M_y > 0

    def test_curvature_ductility_finite_and_realistic(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.10, n_steps=50,
        )
        # Tension-controlled RC beam: mu_phi typically 3-10
        assert res.mu_phi is not None
        assert 2.0 < res.mu_phi < 15.0

    def test_concrete_crushing_termination(self):
        """For a lightly-reinforced beam, concrete crushes -- the loop
        terminates at the crushing step and reports failure_mode."""
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.10, n_steps=50,
            eps_cu_crush=0.003,
        )
        assert res.failure_mode == "concrete_crushing"
        # Last point's eps_top should exceed crushing strain
        assert abs(res.points[-1].eps_top_concrete) >= 0.003 * 0.99


# ============================================================ axial load effect

class TestAxialLoadEffect:
    def test_compression_increases_ultimate_moment(self):
        """Moderate axial compression increases M_u up to the balanced
        point. Verify M_u at P=500 kN > M_u at P=0."""
        beam = _make_beam()
        res_0 = moment_curvature(
            beam, P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        res_compress = moment_curvature(
            beam, P_target=500e3,    # 500 kN compression
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        assert res_compress.M_u > res_0.M_u

    def test_axial_force_satisfied_at_each_step(self):
        """Newton iteration should achieve P_target at every converged
        step within tolerance."""
        res = moment_curvature(
            _make_beam(), P_target=300e3,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.03, n_steps=20,
        )
        for p in res.points:
            if p.converged:
                assert p.P == pytest.approx(300e3, abs=1e4)

    def test_tension_convention(self):
        """P_convention='tension' should produce a section in net
        tension when P_target > 0."""
        beam = _make_beam()
        # P_target=200kN tension means N_target=200kN (tension positive)
        res = moment_curvature(
            beam, P_target=200e3,
            P_convention="tension",
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.02, n_steps=10,
        )
        # P reported back is in tension convention
        for p in res.points[:5]:
            if p.converged:
                assert p.P == pytest.approx(200e3, abs=1e4)


# ============================================================ cracking moment

class TestCrackingMoment:
    def test_M_cr_when_f_rupture_given(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=20,
            f_rupture=3.4e6,    # ~0.62*sqrt(30) MPa
        )
        assert res.M_cr is not None
        # M_cr for 300x600 beam: f_r * I_g / y = 3.4e6 * 0.3*0.6^3/12 / 0.3
        # = 3.4e6 * 0.0054 / 0.3 = 61.2 kN.m
        assert res.M_cr == pytest.approx(61.2e3, rel=0.05)
        assert res.kappa_cr is not None
        assert res.kappa_cr > 0

    def test_M_cr_none_when_no_f_rupture(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=20,
        )
        assert res.M_cr is None
        assert res.kappa_cr is None

    def test_compression_increases_M_cr(self):
        """Axial compression delays cracking -> larger M_cr."""
        beam = _make_beam()
        res_0 = moment_curvature(
            beam, P_target=0.0, f_rupture=3.4e6,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.01, n_steps=5,
        )
        res_c = moment_curvature(
            beam, P_target=500e3, f_rupture=3.4e6,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.01, n_steps=5,
        )
        assert res_c.M_cr > res_0.M_cr


# ============================================================ bilinear

class TestBilinear:
    def test_bilinear_returns_two_points(self):
        res = moment_curvature(
            _make_beam(),
            P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.05, n_steps=30,
        )
        yp, up = res.bilinear()
        kappa_y_bi, M_y_bi = yp
        kappa_u, M_u = up
        # Yield point should be between origin and ultimate
        assert 0 < kappa_y_bi < kappa_u
        # M_y_bi should be near M_u (idealized as plateau through ultimate level)
        assert M_y_bi == pytest.approx(M_u, rel=0.01)


# ============================================================ validation

class TestValidation:
    def test_unreinforced_raises(self):
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        sec = rc_rectangular_section(b=0.3, h=0.6, concrete=cm)
        with pytest.raises(ValueError, match="reinforcement"):
            moment_curvature(
                sec, P_target=0.0,
                concrete_uniaxial=_kent_park(),
                steel_uniaxial=_grade60_steel(),
            )

    def test_negative_kappa_max_raises(self):
        with pytest.raises(ValueError, match="kappa_max"):
            moment_curvature(
                _make_beam(), P_target=0.0,
                concrete_uniaxial=_kent_park(),
                steel_uniaxial=_grade60_steel(),
                kappa_max=-0.01,
            )

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="P_convention"):
            moment_curvature(
                _make_beam(), P_target=0.0,
                concrete_uniaxial=_kent_park(),
                steel_uniaxial=_grade60_steel(),
                P_convention="bogus",
            )


# ============================================================ arrays

class TestResultArrays:
    def test_arrays_match_points(self):
        res = moment_curvature(
            _make_beam(), P_target=0.0,
            concrete_uniaxial=_kent_park(),
            steel_uniaxial=_grade60_steel(),
            kappa_max=0.02, n_steps=10,
        )
        n = len(res.points)
        assert len(res.kappa_array) == n
        assert len(res.M_array) == n
        assert res.kappa_array[0] == res.points[0].kappa
        assert res.M_array[-1] == res.points[-1].M
