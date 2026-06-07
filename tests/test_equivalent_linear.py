"""Phase HH.5 tests -- Vucetic-Dobry G/G_max curves and equivalent-linear
SHAKE-style iteration.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.seismic import (
    NonlinearSoilCurves,
    SoilLayer,
    equivalent_linear_iterate,
    vucetic_dobry_curves,
)


# ============================================================ curves

class TestVuceticDobryCurves:
    def test_G_at_gamma_r_is_half(self):
        """By definition, G/G_max = 0.5 at gamma = gamma_r."""
        c = vucetic_dobry_curves(30)
        # alpha = 0.92 means it's slightly off 0.5 -- close to 0.5
        assert c.G_over_Gmax(c.gamma_r) == pytest.approx(0.5, abs=0.01)

    def test_G_unity_at_zero_strain(self):
        c = vucetic_dobry_curves(30)
        assert c.G_over_Gmax(0.0) == 1.0

    def test_G_decreases_monotonically(self):
        c = vucetic_dobry_curves(30)
        gammas = np.logspace(-6, -1, 50)
        gG = np.array([c.G_over_Gmax(g) for g in gammas])
        assert (np.diff(gG) <= 1e-12).all()

    def test_damping_increases_with_strain(self):
        c = vucetic_dobry_curves(30)
        gammas = np.logspace(-6, -2, 30)
        xi = np.array([c.damping(g) for g in gammas])
        # Monotone increase
        assert (np.diff(xi) >= -1e-12).all()
        # Bounded by xi_max
        assert xi.max() <= c.xi_max + 1e-9

    def test_higher_PI_more_linear(self):
        """Higher plasticity index (clay) -> larger gamma_r -> more
        linear (G/G_max stays close to 1 longer)."""
        c_sand = vucetic_dobry_curves(0)
        c_clay = vucetic_dobry_curves(100)
        assert c_clay.gamma_r > c_sand.gamma_r
        # At a common gamma, clay's G/G_max is higher (more linear)
        gamma = 5e-4
        assert c_clay.G_over_Gmax(gamma) > c_sand.G_over_Gmax(gamma)

    def test_rejects_negative_PI(self):
        with pytest.raises(ValueError, match="PI"):
            vucetic_dobry_curves(-1)

    def test_custom_curves(self):
        c = NonlinearSoilCurves(gamma_r=1e-3, alpha=1.0,
                                  xi_min=0.02, xi_max=0.20)
        # alpha=1 hyperbolic: G/G_max = 1/(1 + gamma/gamma_r)
        # At gamma = gamma_r: G/G_max = 0.5 exactly
        assert c.G_over_Gmax(1e-3) == pytest.approx(0.5, abs=1e-9)

    def test_PI_above_200_capped(self):
        c = vucetic_dobry_curves(500)
        # Should cap to PI=200 values
        c200 = vucetic_dobry_curves(200)
        assert c.gamma_r == c200.gamma_r


# ============================================================ iteration

class TestEquivalentLinearIteration:
    def _build_layers(self):
        return [
            SoilLayer(thickness=5.0, Vs=180.0, rho=1900),
            SoilLayer(thickness=5.0, Vs=200.0, rho=1900),
            SoilLayer(thickness=5.0, Vs=230.0, rho=1950),
            SoilLayer(thickness=5.0, Vs=280.0, rho=1950),
        ]

    def test_low_pga_stays_near_linear(self):
        """At small input PGA, G_eff should stay close to G_max."""
        layers = self._build_layers()
        curves = [vucetic_dobry_curves(30) for _ in layers]
        res = equivalent_linear_iterate(
            layers=layers, rock_Vs=760.0, rock_rho=2300.0,
            curves=curves, input_pga=0.005,
        )
        assert res.converged
        # All G_eff should be >= 80% of G_max at this low input
        assert res.G_over_Gmax.min() >= 0.80

    def test_high_pga_softens_soil(self):
        """At strong input PGA, G_eff drops significantly."""
        layers = self._build_layers()
        curves = [vucetic_dobry_curves(30) for _ in layers]
        res = equivalent_linear_iterate(
            layers=layers, rock_Vs=760.0, rock_rho=2300.0,
            curves=curves, input_pga=0.3,
        )
        assert res.converged
        # Top layer should soften to G_eff < 0.5 * G_max
        assert res.G_over_Gmax[0] < 0.5

    def test_strong_motion_increases_damping(self):
        layers = self._build_layers()
        curves = [vucetic_dobry_curves(30) for _ in layers]
        res_weak = equivalent_linear_iterate(
            layers=layers, rock_Vs=760, rock_rho=2300,
            curves=curves, input_pga=0.005,
        )
        res_strong = equivalent_linear_iterate(
            layers=layers, rock_Vs=760, rock_rho=2300,
            curves=curves, input_pga=0.3,
        )
        # Strong-motion damping in top layer should significantly exceed
        # weak-motion damping (capped by xi_max=25%)
        assert res_strong.xi_eff[0] > 2.5 * res_weak.xi_eff[0]

    def test_top_layer_softens_more_than_bottom(self):
        """Strain is highest near the top -> G/G_max is lowest there."""
        layers = self._build_layers()
        curves = [vucetic_dobry_curves(30) for _ in layers]
        res = equivalent_linear_iterate(
            layers=layers, rock_Vs=760, rock_rho=2300,
            curves=curves, input_pga=0.3,
        )
        assert res.G_over_Gmax[0] < res.G_over_Gmax[-1]

    def test_validates_inputs(self):
        layers = self._build_layers()
        curves = [vucetic_dobry_curves(30) for _ in layers]
        with pytest.raises(ValueError, match="input_pga"):
            equivalent_linear_iterate(
                layers=layers, rock_Vs=760, rock_rho=2300,
                curves=curves, input_pga=-0.1,
            )
        with pytest.raises(ValueError, match="must have the same length"):
            equivalent_linear_iterate(
                layers=layers, rock_Vs=760, rock_rho=2300,
                curves=curves[:2], input_pga=0.1,
            )

    def test_clay_stays_linear_longer_than_sand(self):
        """At the same input PGA, a high-PI clay profile should soften
        less than a low-PI sand profile."""
        layers = self._build_layers()
        sand_curves = [vucetic_dobry_curves(0) for _ in layers]
        clay_curves = [vucetic_dobry_curves(100) for _ in layers]
        res_sand = equivalent_linear_iterate(
            layers=layers, rock_Vs=760, rock_rho=2300,
            curves=sand_curves, input_pga=0.2,
        )
        res_clay = equivalent_linear_iterate(
            layers=layers, rock_Vs=760, rock_rho=2300,
            curves=clay_curves, input_pga=0.2,
        )
        # At same input, sand softens more (lower G/G_max)
        assert res_sand.G_over_Gmax.mean() < res_clay.G_over_Gmax.mean()
