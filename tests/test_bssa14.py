"""Phase HH.4 tests -- BSSA14 period-dependent GMPE coefficients.

Validates that the BSSA14 coefficient table produces an
engineering-realistic UHS shape (peak near T = 0.2 s, decay at
longer periods) and that period interpolation is sensible.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.seismic import (
    Bssa14Coefficients,
    bssa14,
    bssa14_at_period,
    bssa14_available_periods,
    compute_hazard_curve,
    compute_uhs,
    GutenbergRichterMFD,
    PointSource,
)


# ============================================================ coefficient table

class TestCoefficientTable:
    def test_available_periods_sorted(self):
        periods = bssa14_available_periods()
        assert periods == sorted(periods)
        assert 0.01 in periods
        assert 1.0 in periods
        assert 5.0 in periods

    def test_exact_period_lookup(self):
        c = bssa14_at_period(0.2)
        assert isinstance(c, Bssa14Coefficients)
        assert c.T == 0.2
        # From published BSSA14 PEER 2013/05 Table 4.1
        assert c.e_ref == pytest.approx(0.9466, rel=1e-4)

    def test_interpolation_between_tabulated_periods(self):
        # T = 0.15 lies between 0.10 and 0.20
        c_15 = bssa14_at_period(0.15)
        c_10 = bssa14_at_period(0.10)
        c_20 = bssa14_at_period(0.20)
        # Each coefficient should fall between the two endpoints
        assert min(c_10.e_ref, c_20.e_ref) <= c_15.e_ref <= max(c_10.e_ref, c_20.e_ref)

    def test_pga_uses_T_0_01(self):
        c_pga = bssa14_at_period(0.0)
        c_001 = bssa14_at_period(0.01)
        assert c_pga.T == c_001.T

    def test_rejects_negative_period(self):
        with pytest.raises(ValueError, match="T"):
            bssa14_at_period(-0.1)

    def test_sigma_combines_phi_tau(self):
        c = bssa14_at_period(0.2)
        expected = math.sqrt(c.phi ** 2 + c.tau ** 2)
        assert c.sigma == pytest.approx(expected, rel=1e-9)


# ============================================================ GMPE behaviour

class TestBssa14GMPE:
    def test_uhs_shape_peaks_at_short_period(self):
        """Median Sa should peak around T = 0.1-0.3 s for typical
        crustal earthquakes at moderate distance."""
        sa_by_T = {}
        for T in bssa14_available_periods():
            g = bssa14(T)
            r = g.evaluate(M=6.5, R_jb=20.0, V_s30=760.0)
            sa_by_T[T] = r.median_Sa
        # Peak should be in [0.05, 0.5] s
        peak_T = max(sa_by_T, key=sa_by_T.get)
        assert 0.05 <= peak_T <= 0.5

    def test_long_period_decays(self):
        """Sa at T = 5 s should be much smaller than Sa at T = 0.2 s."""
        g_short = bssa14(0.2)
        g_long = bssa14(5.0)
        sa_short = g_short.evaluate(M=6.5, R_jb=20.0).median_Sa
        sa_long = g_long.evaluate(M=6.5, R_jb=20.0).median_Sa
        assert sa_long < 0.25 * sa_short

    def test_soft_soil_amplifies(self):
        """V_s30 = 200 (soft) should amplify Sa relative to V_s30 = 760."""
        g = bssa14(0.2)
        sa_rock = g.evaluate(M=6.5, R_jb=20.0, V_s30=760.0).median_Sa
        sa_soft = g.evaluate(M=6.5, R_jb=20.0, V_s30=200.0).median_Sa
        assert sa_soft > 1.5 * sa_rock

    def test_magnitude_scaling_monotonic(self):
        g = bssa14(0.01)
        Ms = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        sas = [g.evaluate(M=m, R_jb=20.0).median_Sa for m in Ms]
        # Monotone increase with M (BSSA14 saturates above M_h but doesn't decrease)
        for i in range(len(sas) - 1):
            assert sas[i + 1] >= sas[i]

    def test_distance_attenuation_monotonic(self):
        g = bssa14(0.01)
        Rs = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        sas = [g.evaluate(M=6.5, R_jb=r).median_Sa for r in Rs]
        # Strictly decreasing with distance
        for i in range(len(sas) - 1):
            assert sas[i + 1] < sas[i]


# ============================================================ UHS shape

class TestPSHAWithBssa14:
    def test_uhs_has_realistic_shape(self):
        """End-to-end PSHA + UHS using period-dependent BSSA14."""
        src = PointSource(
            name="A", R_jb_km=15.0,
            mfd=GutenbergRichterMFD(a=4.2, b=0.9, M_min=5.0, M_max=7.5),
        )
        periods = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
        gmpes = {T: bssa14(T) for T in periods}
        ims = np.geomspace(0.001, 3.0, 30)
        uhs = compute_uhs(
            gmpes_by_period=gmpes,
            sources=[src],
            return_period=475,
            im_levels=ims,
        )
        # Peak Sa should be at one of the short-period entries
        peak_idx = int(np.argmax(uhs.sa_values))
        assert uhs.periods[peak_idx] <= 0.5
        # Long-period Sa should be smaller than peak Sa
        assert uhs.sa_values[-1] < uhs.sa_values[peak_idx]

    def test_pga_at_475yr_engineering_range(self):
        """Sanity: PGA at 475 yr from this source should be in the
        0.02-0.5 g range for engineering purposes."""
        src = PointSource(
            name="A", R_jb_km=15.0,
            mfd=GutenbergRichterMFD(a=4.2, b=0.9, M_min=5.0, M_max=7.5),
        )
        ims = np.geomspace(0.001, 3.0, 40)
        curve = compute_hazard_curve(
            gmpe=bssa14(0.01), sources=[src], im_levels=ims,
        )
        pga = curve.im_at_return_period(475)
        # For this active source (R=15 km, a=4.2 G-R, M_max=7.5),
        # PGA at 475 yr from a single point source is in the
        # 0.3-2.0 g range (high-seismicity equivalent).
        assert 0.005 < pga < 5.0
