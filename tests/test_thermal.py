"""Phase 40 tests -- heat conduction, thermo-mechanical coupling,
and fire-engineering curves.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ConvectionEdge2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    Quad4,
    SteadyHeatAnalysis,
    ThermalMaterial,
    ThermalQuad4,
    TransientHeatAnalysis,
    apply_thermal_load,
    astm_e119_temperature,
    beam_thermal_axial_force,
    concrete_strength_reduction_ec2,
    hydrocarbon_temperature,
    iso_834_temperature,
    steel_critical_temperature,
    steel_modulus_reduction_ec3,
    steel_strength_reduction_ec3,
)


# ============================================================ thermal material

class TestThermalMaterial:
    def test_basic_properties(self):
        m = ThermalMaterial(tag=1, k=50.0, rho=7850.0, c=460.0,
                              alpha=1.2e-5)
        assert m.rho_c == pytest.approx(7850.0 * 460.0, rel=1.0e-12)
        assert m.diffusivity == pytest.approx(50.0 / (7850.0 * 460.0),
                                                 rel=1.0e-12)

    def test_validates_positive(self):
        with pytest.raises(ValueError):
            ThermalMaterial(tag=1, k=-1.0, rho=7850.0, c=460.0)
        with pytest.raises(ValueError):
            ThermalMaterial(tag=1, k=50.0, rho=-1.0, c=460.0)


# ============================================================ steady-state

def _build_1d_slab_model(*, nx: int = 10, L: float = 1.0,
                          k: float = 50.0):
    mat = ThermalMaterial(tag=1, k=k, rho=7850.0, c=460.0)
    m = Model(ndm=2, ndf=1)
    m.add_material(mat)
    for j in range(2):
        for i in range(nx + 1):
            m.add_node(j * (nx + 1) + i + 1, i * L / nx, j * 0.1)
    et = 1
    for i in range(nx):
        n1 = i + 1
        n2 = n1 + 1
        n3 = n2 + (nx + 1)
        n4 = n1 + (nx + 1)
        m.add_element(ThermalQuad4(et, (n1, n2, n3, n4), mat))
        et += 1
    return m, mat, et


class TestSteadyHeatConduction:
    def test_1d_fourier_linear_profile(self):
        """Slab with Dirichlet BCs: left T=100, right T=0 should
        give a linear profile T(x) = 100*(1 - x/L)."""
        L = 1.0
        m, mat, et = _build_1d_slab_model(nx=10, L=L)
        # Right edge fixed at 0
        for j in range(2):
            m.fix(j * 11 + 11, [1])
        # Left edge: convection penalty -> ~ Dirichlet T=100
        m.add_element(ConvectionEdge2D(et, (1, 12), mat,
                                          h=1.0e8, T_inf=100.0))
        SteadyHeatAnalysis(m).run()
        # Sample at five interior points
        for i in [2, 4, 5, 7, 9]:
            tag = i + 1
            x = m.node(tag).coords[0]
            T_exact = 100.0 * (1.0 - x / L)
            assert m.node(tag).disp[0] == pytest.approx(T_exact,
                                                          abs=1.0e-3)

    def test_rejects_non_thermal_model(self):
        mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
        m = Model(ndm=2, ndf=2)            # mechanical, ndf=2
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        with pytest.raises(ValueError, match="ndf=1"):
            SteadyHeatAnalysis(m)


# ============================================================ transient

class TestTransientHeatConduction:
    def test_1d_uniform_IC_cooling(self):
        """Slab with uniform T=100 and both ends fixed at 0 decays
        like (4/pi)*100*exp(-t/tau) (1-term Fourier) at midspan after
        a few characteristic times."""
        L = 1.0
        m, mat, _ = _build_1d_slab_model(nx=20, L=L)
        alpha = mat.diffusivity
        tau = L * L / (math.pi ** 2 * alpha)
        # Both ends fixed at T=0
        for j in range(2):
            m.fix(j * 21 + 1, [1])
            m.fix(j * 21 + 21, [1])
        m.number_dofs()
        T0 = np.full(m.neq, 100.0)
        dt = tau / 20
        n_steps = 60
        res = TransientHeatAnalysis(
            m, num_steps=n_steps, dt=dt, theta=0.5, T0=T0,
        ).run()
        # At t = 3*tau the max T should match (4/pi)*100*exp(-3) ~ 6.3
        T_max_final = res.T[-1].max()
        T_an = (4.0 / math.pi) * 100.0 * math.exp(-3.0)
        assert T_max_final == pytest.approx(T_an, rel=0.05)

    def test_validates_inputs(self):
        L = 1.0
        m, _, _ = _build_1d_slab_model(nx=4, L=L)
        with pytest.raises(ValueError, match="num_steps"):
            TransientHeatAnalysis(m, num_steps=0, dt=1.0)
        with pytest.raises(ValueError, match="dt"):
            TransientHeatAnalysis(m, num_steps=10, dt=-1.0)
        with pytest.raises(ValueError, match="theta"):
            TransientHeatAnalysis(m, num_steps=10, dt=1.0, theta=2.0)


# ============================================================ thermo-mechanical

class TestThermoMechanical:
    def test_free_thermal_expansion_unit_square(self):
        """A unit-square Quad4 heated by dT free of constraint should
        expand by alpha * dT in both directions; with one corner fixed
        and another rolled, the diagonal corner moves by
        approximately L * alpha * dT.
        """
        alpha = 1.2e-5
        dT = 100.0
        L = 1.0
        steel = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=0.0)
        m = Model(ndm=2, ndf=2)
        m.add_material(steel)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, L, 0.0)
        m.add_node(3, L, L)
        m.add_node(4, 0.0, L)
        m.add_element(Quad4(1, (1, 2, 3, 4), steel, thickness=1.0))
        # Free corner: no rigid-body modes left after 3 fixities
        m.fix(1, [1, 1])
        m.fix(2, [0, 1])
        # Apply uniform temperature change
        temps = {1: 20.0 + dT, 2: 20.0 + dT, 3: 20.0 + dT, 4: 20.0 + dT}
        n_done = apply_thermal_load(m, temperatures=temps,
                                      T_ref=20.0, alpha=alpha)
        assert n_done == 1
        LinearStaticAnalysis(m).run()
        # Node 3 (diagonal corner) should expand ~ L*alpha*dT in both x and y
        expected = L * alpha * dT
        assert m.node(3).disp[0] == pytest.approx(expected, rel=0.10)
        assert m.node(3).disp[1] == pytest.approx(expected, rel=0.10)

    def test_beam_thermal_axial_force(self):
        """Restrained beam under dT generates F = -E A alpha dT."""
        F = beam_thermal_axial_force(
            alpha=1.2e-5, dT=200.0, E=2.0e11, A=0.01,
        )
        expected = -2.0e11 * 0.01 * 1.2e-5 * 200.0
        assert F == pytest.approx(expected, rel=1.0e-12)


# ============================================================ fire curves

class TestFireCurves:
    def test_iso_834_at_known_times(self):
        # T_0 = 20 reference: ISO 834 gives ~ 576 at 5 min, 842 at 30 min
        assert iso_834_temperature(5.0) == pytest.approx(576.0, abs=2.0)
        assert iso_834_temperature(30.0) == pytest.approx(842.0, abs=2.0)
        assert iso_834_temperature(60.0) == pytest.approx(945.0, abs=2.0)

    def test_iso_834_monotonic(self):
        Ts = [iso_834_temperature(t) for t in [1, 5, 10, 30, 60, 120]]
        assert all(Ts[i + 1] > Ts[i] for i in range(len(Ts) - 1))

    def test_astm_matches_iso_834(self):
        # We aliased them; check equivalent values
        for t in [5.0, 30.0, 60.0]:
            assert (astm_e119_temperature(t)
                    == iso_834_temperature(t))

    def test_hydrocarbon_fast_ramp(self):
        # Hydrocarbon should be hot fast
        assert hydrocarbon_temperature(1.0) > 700.0
        assert hydrocarbon_temperature(30.0) > 1050.0

    def test_iso_834_rejects_negative_t(self):
        with pytest.raises(ValueError):
            iso_834_temperature(-1.0)


# ============================================================ EC3 / EC2 reduction

class TestEcReductions:
    def test_steel_k_y_at_20C_is_one(self):
        assert steel_strength_reduction_ec3(20.0) == 1.0

    def test_steel_k_y_at_400C_is_one(self):
        # k_y stays at 1.0 up to 400 C
        assert steel_strength_reduction_ec3(400.0) == pytest.approx(1.0)

    def test_steel_k_y_table_values(self):
        # 500C: 0.78, 600C: 0.47, 700C: 0.23 (EC3 Table 3.1)
        assert steel_strength_reduction_ec3(500.0) == pytest.approx(0.78, abs=1e-9)
        assert steel_strength_reduction_ec3(600.0) == pytest.approx(0.47, abs=1e-9)
        assert steel_strength_reduction_ec3(700.0) == pytest.approx(0.23, abs=1e-9)

    def test_steel_k_E_drops_faster_than_k_y(self):
        # At 500 C: k_E = 0.60 < k_y = 0.78
        assert steel_modulus_reduction_ec3(500.0) < \
            steel_strength_reduction_ec3(500.0)

    def test_concrete_calcareous_better_than_siliceous(self):
        # Both at 700 C
        k_sil = concrete_strength_reduction_ec2(700.0, aggregate="siliceous")
        k_cal = concrete_strength_reduction_ec2(700.0, aggregate="calcareous")
        assert k_cal > k_sil

    def test_concrete_unknown_aggregate_raises(self):
        with pytest.raises(ValueError, match="aggregate"):
            concrete_strength_reduction_ec2(500.0, aggregate="basalt")


class TestCriticalTemperature:
    def test_mu_0p5_gives_about_585C(self):
        # EC3 closed-form for mu_0 = 0.5 -> ~ 585 C
        T_cr = steel_critical_temperature(mu_0=0.5)
        assert T_cr == pytest.approx(585.0, abs=20.0)

    def test_higher_utilisation_lowers_T_cr(self):
        T_low = steel_critical_temperature(mu_0=0.4)
        T_high = steel_critical_temperature(mu_0=0.8)
        assert T_low > T_high

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError):
            steel_critical_temperature(mu_0=-0.1)
        with pytest.raises(ValueError):
            steel_critical_temperature(mu_0=1.5)
