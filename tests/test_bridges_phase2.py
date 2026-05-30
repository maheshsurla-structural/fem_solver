"""Phase 45 tests -- Bridges Phase 2: cables + staged construction + EMM.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)
from femsolver.bridges import (
    CableElement2D,
    ConstructionStage,
    StagedConstructionAnalysis,
    catenary_max_tension,
    catenary_sag,
    effective_modulus_EMM,
    ernst_equivalent_modulus,
)


# ============================================================ Ernst modulus

class TestErnst:
    def test_high_tension_limit_recovers_E(self):
        """At very high T, the sag term vanishes -> E_eq ~ E."""
        E = 2.0e11
        E_eq = ernst_equivalent_modulus(
            E=E, A=0.005, L_h=100.0, gamma_eff=400.0, T=1.0e8,
        )
        assert E_eq == pytest.approx(E, rel=0.001)

    def test_low_tension_drops_E_eq(self):
        """At lower T the equivalent modulus drops significantly."""
        E = 2.0e11
        E_eq_low = ernst_equivalent_modulus(
            E=E, A=0.005, L_h=100.0, gamma_eff=400.0, T=5.0e5,
        )
        # Should drop to under half of E
        assert E_eq_low < 0.6 * E

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            ernst_equivalent_modulus(
                E=-1, A=0.005, L_h=100, gamma_eff=400, T=1e6,
            )
        with pytest.raises(ValueError):
            ernst_equivalent_modulus(
                E=2e11, A=0.005, L_h=100, gamma_eff=-1, T=1e6,
            )


class TestCableElement:
    def test_initial_K_matches_Truss2D_when_no_sag(self):
        """With gamma_eff = 0, the cable reduces to a regular Truss2D."""
        mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
        m = Model(ndm=2, ndf=2)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, 100.0, 0.0)
        m.add_element(CableElement2D(1, (1, 2), mat, 0.005,
                                       gamma_eff=0.0))
        elem = list(m.elements.values())[0]
        K = elem.K_global()
        # E·A/L for a horizontal element of length 100 m
        EAoL = 2.0e11 * 0.005 / 100.0
        assert K[0, 0] == pytest.approx(EAoL, rel=1e-12)
        assert K[2, 2] == pytest.approx(EAoL, rel=1e-12)
        assert K[0, 2] == pytest.approx(-EAoL, rel=1e-12)

    def test_K_with_sag_is_lower(self):
        mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
        m = Model(ndm=2, ndf=2)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, 100.0, 0.0)
        # Operating tension 0.5 MN -> Ernst reduction ~0.5x
        m.add_element(CableElement2D(
            1, (1, 2), mat, 0.005,
            gamma_eff=400.0, T_operating=0.5e6,
        ))
        elem = list(m.elements.values())[0]
        K_sag = elem.K_global()
        # Compare against no-sag
        m2 = Model(ndm=2, ndf=2)
        m2.add_material(mat)
        m2.add_node(1, 0.0, 0.0); m2.add_node(2, 100.0, 0.0)
        m2.add_element(CableElement2D(1, (1, 2), mat, 0.005))
        K_nosag = list(m2.elements.values())[0].K_global()
        assert K_sag[0, 0] < 0.6 * K_nosag[0, 0]


# ============================================================ catenary

class TestCatenary:
    def test_midspan_sag_parabolic_approximation(self):
        """For small w·L/(2H), sag ~ w·L²/(8H) (parabolic limit)."""
        L_h, w, H = 100.0, 100.0, 50.0e3
        sag = catenary_sag(L_h=L_h, w=w, H=H)
        parabolic = w * L_h * L_h / (8.0 * H)
        # Within ~1% for w·L/(2H) ~ 0.1
        assert sag == pytest.approx(parabolic, rel=0.01)

    def test_max_tension_is_at_least_H(self):
        res = catenary_max_tension(L_h=100.0, w=100.0, H=50.0e3)
        assert res.T_max >= res.H

    def test_arc_length_exceeds_chord(self):
        res = catenary_max_tension(L_h=100.0, w=100.0, H=50.0e3)
        assert res.L_arc > 100.0

    def test_zero_weight_recovers_straight(self):
        """w = 0 -> arc = chord, sag = 0."""
        res = catenary_max_tension(L_h=100.0, w=0.0, H=50.0e3)
        assert res.L_arc == pytest.approx(100.0, rel=1e-12)
        assert res.sag_max == 0.0
        assert res.T_max == pytest.approx(50.0e3, rel=1e-9)


# ============================================================ EMM

class TestEffectiveModulusMethod:
    def test_zero_creep_gives_E_c(self):
        assert effective_modulus_EMM(E_c=30e9, phi=0.0) == 30e9

    def test_formula(self):
        # E_eff = E_c / (1 + chi · phi)
        E_eff = effective_modulus_EMM(E_c=30e9, phi=2.0, chi=0.8)
        assert E_eff == pytest.approx(30e9 / 2.6, rel=1e-12)

    def test_unity_chi_is_creep_compliance_form(self):
        # chi=1 reduces to the pure "creep compliance" form: E_eff = E_c / (1 + phi)
        E_eff = effective_modulus_EMM(E_c=30e9, phi=2.0, chi=1.0)
        assert E_eff == pytest.approx(30e9 / 3.0, rel=1e-12)

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            effective_modulus_EMM(E_c=-1, phi=1.0)
        with pytest.raises(ValueError):
            effective_modulus_EMM(E_c=30e9, phi=-0.1)
        with pytest.raises(ValueError):
            effective_modulus_EMM(E_c=30e9, phi=1.0, chi=0.0)


# ============================================================ staged construction

class TestStagedConstruction:
    def _build_simple_beam(self):
        mat = ElasticIsotropic(1, E=30.0e9, nu=0.2, rho=2400.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(5):
            m.add_node(i + 1, i * 1.0, 0.0)
        for i in range(4):
            m.add_element(BeamColumn2D(
                i + 1, (i + 1, i + 2), mat, 0.1, 1e-4,
            ))
        m.fix(1, [1, 1, 1])
        m.fix(5, [0, 1, 0])
        return m, mat

    def test_two_stage_sums_increments_to_cumulative(self):
        m, _ = self._build_simple_beam()
        stages = [
            ConstructionStage(
                name="stage1", duration_days=30.0,
                load_pattern={3: [0.0, -1000.0, 0.0]},
            ),
            ConstructionStage(
                name="stage2", duration_days=18000.0,
                load_pattern={3: [0.0, -1000.0, 0.0]},
            ),
        ]
        ana = StagedConstructionAnalysis(
            m, stages=stages, f_cm=38e6, chi=0.8,
        )
        res = ana.run()
        assert res.n_stages == 2
        cum = sum(res.u_incremental)
        np.testing.assert_allclose(cum, res.u_cumulative, atol=1e-9)

    def test_creep_factor_below_one(self):
        m, _ = self._build_simple_beam()
        stages = [
            ConstructionStage(
                name="stage1", duration_days=18000.0,
                load_pattern={3: [0.0, -1000.0, 0.0]},
                age_at_loading_days=28.0,
            ),
        ]
        ana = StagedConstructionAnalysis(
            m, stages=stages, f_cm=38e6, chi=0.8,
            final_age_days=18250.0,
        )
        res = ana.run()
        # phi(50yr, 28d) ~ 1.9 -> factor = 1/(1+0.8*1.9) = 0.397
        assert 0.2 < res.creep_factors[0] < 0.5

    def test_rejects_empty_stages(self):
        m, _ = self._build_simple_beam()
        with pytest.raises(ValueError, match="at least one"):
            StagedConstructionAnalysis(m, stages=[], f_cm=38e6)
