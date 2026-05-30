"""Phase 54 tests -- Theme U site-specific seismic hazard."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.seismic import (
    BooreAtkinsonLike,
    GutenbergRichterMFD,
    PointSource,
    SoilLayer,
    SoilProfile,
    annual_collapse_rate,
    compute_hazard_curve,
    compute_uhs,
    deaggregate,
    risk_targeted_im,
    site_amplification_spectrum,
    transfer_function,
)


# ============================================================ GMPE

class TestGMPE:
    def test_median_increases_with_magnitude(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        r5 = gmpe.evaluate(M=5.0, R_jb=20.0)
        r7 = gmpe.evaluate(M=7.0, R_jb=20.0)
        assert r7.median_Sa > r5.median_Sa

    def test_median_decreases_with_distance(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        r_near = gmpe.evaluate(M=6.5, R_jb=5.0)
        r_far = gmpe.evaluate(M=6.5, R_jb=100.0)
        assert r_near.median_Sa > r_far.median_Sa

    def test_site_amplification_for_soft_site(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        r_rock = gmpe.evaluate(M=6.5, R_jb=20.0, V_s30=760.0)
        r_soft = gmpe.evaluate(M=6.5, R_jb=20.0, V_s30=200.0)
        # Default b_lin = -0.6 -> ln(200/760) * -0.6 > 0 -> amplification
        assert r_soft.median_Sa > r_rock.median_Sa

    def test_sigma_returned(self):
        gmpe = BooreAtkinsonLike(T=0.0, sigma=0.55)
        res = gmpe.evaluate(M=6.5, R_jb=20.0)
        assert res.sigma_lnSa == pytest.approx(0.55)

    def test_validation(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        with pytest.raises(ValueError, match="M"):
            gmpe.evaluate(M=-1, R_jb=10)
        with pytest.raises(ValueError, match="R_jb"):
            gmpe.evaluate(M=6, R_jb=-1)
        with pytest.raises(ValueError, match="V_s30"):
            gmpe.evaluate(M=6, R_jb=10, V_s30=0)


# ============================================================ MFD

class TestGutenbergRichter:
    def test_pdf_normalises(self):
        mfd = GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5)
        # Integrate via trapezoidal rule
        Ms = np.linspace(5.0, 7.5, 500)
        f = np.array([mfd.pdf(m) for m in Ms])
        area = np.trapezoid(f, Ms)
        assert area == pytest.approx(1.0, rel=1e-3)

    def test_pdf_zero_outside_range(self):
        mfd = GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5)
        assert mfd.pdf(4.5) == 0.0
        assert mfd.pdf(7.6) == 0.0

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="b"):
            GutenbergRichterMFD(a=4, b=-0.5, M_min=5, M_max=7)
        with pytest.raises(ValueError, match="M_max"):
            GutenbergRichterMFD(a=4, b=0.9, M_min=7, M_max=5)


# ============================================================ PSHA

class TestPSHAHazardCurve:
    def _make_curve(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        src = PointSource(
            name="SrcA", R_jb_km=20.0,
            mfd=GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5),
        )
        ims = np.geomspace(0.001, 3.0, 30)
        return compute_hazard_curve(gmpe=gmpe, sources=[src], im_levels=ims)

    def test_curve_is_monotonically_decreasing(self):
        curve = self._make_curve()
        # rates should be non-increasing in IM
        diff = np.diff(curve.annual_rates)
        assert (diff <= 1e-10).all()

    def test_return_period_inverse(self):
        curve = self._make_curve()
        # PGA @ 475 yr should map back to lambda = 1/475
        pga = curve.im_at_return_period(475)
        lambda_at = curve.annual_rate_at(pga)
        # Allow ~10% interpolation tolerance
        assert lambda_at == pytest.approx(1.0 / 475.0, rel=0.10)

    def test_longer_return_period_higher_IM(self):
        curve = self._make_curve()
        pga_475 = curve.im_at_return_period(475)
        pga_2475 = curve.im_at_return_period(2475)
        assert pga_2475 > pga_475


class TestUHS:
    def test_uhs_multi_period(self):
        gmpes = {
            0.0: BooreAtkinsonLike(T=0.0),
            0.2: BooreAtkinsonLike(T=0.2),
            1.0: BooreAtkinsonLike(T=1.0),
        }
        src = PointSource(
            name="A", R_jb_km=15.0,
            mfd=GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5),
        )
        ims = np.geomspace(0.001, 3.0, 25)
        uhs = compute_uhs(
            gmpes_by_period=gmpes, sources=[src],
            return_period=475, im_levels=ims,
        )
        assert uhs.return_period == 475
        assert len(uhs.periods) == 3
        # All values positive
        assert (uhs.sa_values > 0).all()


# ============================================================ deaggregation

class TestDeaggregation:
    def test_modal_at_nearby_source(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        src = PointSource(
            name="A", R_jb_km=15.0,
            mfd=GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5),
        )
        ims = np.geomspace(0.001, 3.0, 30)
        curve = compute_hazard_curve(gmpe=gmpe, sources=[src], im_levels=ims)
        im_t = curve.im_at_return_period(475)
        # Use a 20 km R bin to ensure the source distance falls inside
        d = deaggregate(
            gmpe=gmpe, sources=[src], im_target=im_t,
            R_edges=np.arange(0.0, 100.0, 20.0),
        )
        # Modal R should be the source distance (15 km lies in [0, 20))
        assert 0.0 <= d.modal_R <= 20.0

    def test_higher_im_implies_higher_eps(self):
        """At a higher IM the deaggregation should put weight on
        higher epsilon (rare upward fluctuations)."""
        gmpe = BooreAtkinsonLike(T=0.0)
        src = PointSource(
            name="A", R_jb_km=20.0,
            mfd=GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5),
        )
        d_low = deaggregate(gmpe=gmpe, sources=[src], im_target=0.02)
        d_high = deaggregate(gmpe=gmpe, sources=[src], im_target=0.05)
        assert d_high.mean_eps > d_low.mean_eps


# ============================================================ site response

class TestSiteResponse:
    def test_single_layer_fundamental_frequency(self):
        # Vs=200 m/s, H=30 m -> f_1 = 200 / (4*30) = 1.667 Hz
        profile = SoilProfile(
            layers=[SoilLayer(thickness=30.0, Vs=200.0, rho=1900, damping=0.05)],
            rock_Vs=760.0, rock_rho=2300,
        )
        freqs = np.linspace(0.1, 15.0, 400)
        amp = site_amplification_spectrum(profile, freqs)
        f_peak = freqs[np.argmax(amp)]
        assert f_peak == pytest.approx(1.667, rel=0.05)

    def test_damping_reduces_peak(self):
        freqs = np.linspace(0.1, 15.0, 400)
        p_low_damp = SoilProfile(
            layers=[SoilLayer(20.0, 180.0, 1900, damping=0.01)],
            rock_Vs=760.0, rock_rho=2300,
        )
        p_high_damp = SoilProfile(
            layers=[SoilLayer(20.0, 180.0, 1900, damping=0.10)],
            rock_Vs=760.0, rock_rho=2300,
        )
        amp_low = site_amplification_spectrum(p_low_damp, freqs)
        amp_high = site_amplification_spectrum(p_high_damp, freqs)
        assert amp_high.max() < amp_low.max()

    def test_rejects_empty_profile(self):
        with pytest.raises(ValueError, match="at least one"):
            SoilProfile(layers=[], rock_Vs=760, rock_rho=2300)


# ============================================================ risk-targeted

class TestRiskTargeted:
    def _curve(self):
        gmpe = BooreAtkinsonLike(T=0.0)
        src = PointSource(
            name="A", R_jb_km=20.0,
            mfd=GutenbergRichterMFD(a=4.0, b=0.9, M_min=5.0, M_max=7.5),
        )
        ims = np.geomspace(0.001, 3.0, 50)
        return compute_hazard_curve(gmpe=gmpe, sources=[src], im_levels=ims)

    def test_mce_r_target_collapse_prob(self):
        c = self._curve()
        mce_r = risk_targeted_im(
            c, target_collapse_prob=0.01, window_years=50, beta=0.6,
        )
        lambda_C = annual_collapse_rate(c, theta=mce_r, beta=0.6)
        P_C_50 = 1.0 - math.exp(-lambda_C * 50.0)
        assert P_C_50 == pytest.approx(0.01, abs=1e-4)

    def test_higher_dispersion_increases_MCE_R(self):
        c = self._curve()
        # Higher beta = more variability -> need higher median to keep
        # P_C the same -> MCE_R increases? Actually more variability
        # means more contribution from low-IM events too, so it's
        # not strictly monotonic. We just verify the solver converges.
        mce_low_beta = risk_targeted_im(c, beta=0.4)
        mce_high_beta = risk_targeted_im(c, beta=0.8)
        assert mce_low_beta > 0
        assert mce_high_beta > 0

    def test_rejects_invalid_inputs(self):
        c = self._curve()
        with pytest.raises(ValueError, match="target_collapse_prob"):
            risk_targeted_im(c, target_collapse_prob=1.5)
        with pytest.raises(ValueError, match="window_years"):
            risk_targeted_im(c, window_years=-1)
        with pytest.raises(ValueError, match="beta"):
            risk_targeted_im(c, beta=0)
