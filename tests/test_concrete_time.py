"""Phase B.11 tests -- time-dependent concrete properties + structural
creep/shrinkage effects."""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.materials.concrete.concrete_time import (
    aci209_strength_gain,
    en1992_E_cm,
    en1992_beta_cc,
    en1992_strength_gain,
    strength_gain_curve,
)
from femsolver.analysis.time_dependent import (
    StepByStepCreep,
    StepByStepCreepFE,
    age_adjusted_modulus,
    apply_shrinkage_load,
    creep_deflection,
    differential_shrinkage_curvature,
    restraint_force_relaxation,
    shrinkage_axial_force,
)
from femsolver.bridges.creep_shrinkage import cebfip_creep_coefficient


# ============================================================ EN 1992 gain

class TestEN1992StrengthGain:
    def test_beta_cc_unity_at_28(self):
        assert en1992_beta_cc(28, cement_class="N") == pytest.approx(1.0)

    def test_beta_cc_monotonic_and_class_order(self):
        # before 28 d: slow < normal < rapid
        s = en1992_beta_cc(7, cement_class="S")
        n = en1992_beta_cc(7, cement_class="N")
        r = en1992_beta_cc(7, cement_class="R")
        assert s < n < r
        # grows past 28 d
        assert en1992_beta_cc(365, cement_class="N") > 1.0

    def test_E_cm_table_value(self):
        # C40/50: f_cm=48 -> E_cm = 22*(4.8)^0.3 GPa ~ 35.2
        assert en1992_E_cm(48e6) == pytest.approx(22.0 * 4.8 ** 0.3 * 1e9, rel=1e-9)

    def test_strength_gain_at_28(self):
        p = en1992_strength_gain(28, f_ck_28=40e6, cement_class="N")
        assert p.beta_cc == pytest.approx(1.0)
        assert p.f_cm == pytest.approx(48e6, rel=1e-9)
        assert p.f_ck == pytest.approx(40e6, rel=1e-9)
        assert p.f_ctm == pytest.approx(0.30 * 40 ** (2 / 3) * 1e6, rel=1e-6)
        assert p.E_cm == pytest.approx(en1992_E_cm(48e6), rel=1e-9)

    def test_tensile_exponent_change_at_28(self):
        beta7 = en1992_beta_cc(7)
        beta90 = en1992_beta_cc(90)
        p7 = en1992_strength_gain(7, f_ck_28=40e6)
        p90 = en1992_strength_gain(90, f_ck_28=40e6)
        f_ctm28 = 0.30 * 40 ** (2 / 3) * 1e6
        # before 28: alpha=1
        assert p7.f_ctm == pytest.approx(beta7 ** 1.0 * f_ctm28, rel=1e-6)
        # after 28: alpha=2/3
        assert p90.f_ctm == pytest.approx(beta90 ** (2 / 3) * f_ctm28, rel=1e-6)

    def test_early_age_lower_strength_and_E(self):
        p3 = en1992_strength_gain(3, f_ck_28=40e6)
        p28 = en1992_strength_gain(28, f_ck_28=40e6)
        assert p3.f_cm < p28.f_cm
        assert p3.E_cm < p28.E_cm


# ============================================================ ACI 209

class TestACI209:
    def test_ratio_near_unity_at_28(self):
        p = aci209_strength_gain(28, f_c_28=40e6, cement_type="I", curing="moist")
        assert p.beta_cc == pytest.approx(28 / (4 + 0.85 * 28), rel=1e-9)
        assert p.f_cm == pytest.approx(p.beta_cc * 40e6, rel=1e-9)

    def test_type_III_faster_early(self):
        p1 = aci209_strength_gain(3, f_c_28=40e6, cement_type="I")
        p3 = aci209_strength_gain(3, f_c_28=40e6, cement_type="III")
        assert p3.f_cm > p1.f_cm

    def test_modulus_from_strength(self):
        p = aci209_strength_gain(28, f_c_28=40e6)
        assert p.E_cm == pytest.approx(
            4700.0 * math.sqrt(p.f_cm / 1e6) * 1e6, rel=1e-9)

    def test_bad_key_raises(self):
        with pytest.raises(ValueError):
            aci209_strength_gain(28, f_c_28=40e6, cement_type="II")


# ============================================================ curve helper

class TestCurve:
    def test_arrays_and_monotonic(self):
        t = [3, 7, 28, 90, 365]
        c = strength_gain_curve(t, f_ck_28=40e6, code="EN1992")
        assert c["f_cm"].shape == (5,)
        assert np.all(np.diff(c["f_cm"]) > 0)     # strength grows with age
        assert c["beta"][2] == pytest.approx(1.0, abs=1e-6)   # at 28 d

    def test_bad_code_raises(self):
        with pytest.raises(ValueError):
            strength_gain_curve([28], f_ck_28=40e6, code="XYZ")


# ============================================================ structural effects

class TestStructuralEffects:
    def test_aaem(self):
        assert age_adjusted_modulus(E_c=34e9, phi=2.0, chi=0.8) == pytest.approx(
            34e9 / (1 + 0.8 * 2.0), rel=1e-9)
        # chi=1 recovers simple EMM
        assert age_adjusted_modulus(E_c=30e9, phi=1.0, chi=1.0) == pytest.approx(
            15e9, rel=1e-9)

    def test_creep_deflection(self):
        assert creep_deflection(instantaneous=10.0, phi=2.5) == pytest.approx(35.0)

    def test_shrinkage_axial_force_tension(self):
        # shrinkage (negative strain) restrained -> tension (positive)
        N = shrinkage_axial_force(E_eff=20e9, A=0.3, eps_sh=-300e-6)
        assert N == pytest.approx(20e9 * 0.3 * 300e-6, rel=1e-9)
        assert N > 0

    def test_differential_curvature(self):
        k = differential_shrinkage_curvature(eps_top=-400e-6, eps_bot=-100e-6, h=1.0)
        assert k == pytest.approx(300e-6, rel=1e-9)


class TestStepByStepCreep:
    def _phi(self):
        def phi(t, t0):
            dt = t - t0
            return 2.5 * dt / (200.0 + dt) if dt > 0 else 0.0
        return phi

    def test_constant_stress_is_exact_creep(self):
        E = 34e9
        phi = self._phi()
        scc = StepByStepCreep(E_c=E, phi=phi)
        times = np.array([28, 30, 40, 60, 100, 200, 400, 1000, 3650], float)
        s0 = 10e6
        h = scc.strain_history(times, np.full_like(times, s0))
        exact = s0 * (1.0 + np.array([phi(t, 28) for t in times])) / E
        # superposition of a single sustained increment == sigma(1+phi)/E
        assert np.allclose(h.strain, exact, atol=1e-15)

    def test_round_trip_identity(self):
        E = 34e9
        scc = StepByStepCreep(E_c=E, phi=self._phi())
        times = np.array([28, 30, 40, 60, 100, 200, 400, 1000, 3650], float)
        sig = np.array([5, 7, 9, 9, 8, 6, 6, 5, 4], float) * 1e6
        eps = scc.strain_history(times, sig).strain
        sig_rt = scc.relaxation_history(times, eps).stress
        assert np.allclose(sig_rt, sig, rtol=1e-6, atol=1.0)

    def test_relaxation_starts_elastic_and_decreases(self):
        E = 34e9
        scc = StepByStepCreep(E_c=E, phi=self._phi())
        times = np.array([28, 30, 40, 60, 100, 200, 400, 1000, 3650], float)
        eps0 = 3e-4
        r = scc.relaxation_history(times, np.full_like(times, eps0))
        assert r.stress[0] == pytest.approx(E * eps0, rel=1e-9)   # elastic jump
        assert np.all(np.diff(r.stress) <= 1e-3)                  # monotone down
        assert 0.0 < r.stress[-1] < E * eps0                      # relaxes, stays +

    def test_ageing_coefficient_matches_textbook(self):
        """The pure relaxation function implies an ageing coefficient
        chi = 1/(1 - R/E) - 1/phi ~ 0.8 (Bazant/Trost, t0 = 28 d)."""
        E = 34e9
        f_cm = 48e6
        t0, eps0 = 28.0, 1e-3

        def phi(t, tp):
            if t <= tp:
                return 0.0
            return cebfip_creep_coefficient(
                t_days=t, t0_days=tp, f_cm=f_cm, RH=70.0, h_0=0.30).phi

        times = np.unique(np.concatenate(
            [[t0], t0 + np.logspace(-1, np.log10(36500), 150)]))
        scc = StepByStepCreep(E_c=E, phi=phi)
        r = scc.relaxation_history(times, np.full_like(times, eps0))
        R_over_E = r.stress[-1] / (E * eps0)
        phiv = phi(times[-1], t0)
        chi = 1.0 / (1.0 - R_over_E) - 1.0 / phiv
        assert 0.70 <= chi <= 0.90       # textbook chi(inf, 28) ~ 0.80

    def test_creep_recovery_on_unload(self):
        E = 34e9
        scc = StepByStepCreep(E_c=E, phi=self._phi())
        times = np.array([28, 30, 40, 60, 100, 200, 400, 1000, 3650], float)
        # load 10 MPa to t=60, then unload to 0
        sig = np.where(times <= 60, 10e6, 0.0)
        h = scc.strain_history(times, sig)
        before = h.strain[times.tolist().index(60)]
        after = h.strain[-1]
        assert after < before          # recovers
        assert after >= 0.0            # but only partially (residual creep)

    def test_restraint_force_relaxes_below_elastic(self):
        """Shrinkage fully held -> tensile force, far below the
        un-relaxed elastic value because creep relaxes it."""
        E = 34e9
        A = 0.25

        def phi(t, t0):
            dt = t - t0
            return 2.0 * dt / (200.0 + dt) if dt > 0 else 0.0

        def sh(t):
            dt = t - 7
            return -300e-6 * dt / (150.0 + dt) if dt > 0 else 0.0

        times = np.array([28, 40, 100, 400, 1000, 3650, 10000], float)
        rf = restraint_force_relaxation(
            times=times, eps_restrained=0.0, A=A, E_c=E, phi=phi, shrinkage=sh)
        # tensile restraint stress
        assert rf.stress[-1] > 0
        # well below the un-relaxed elastic restraint E*|eps_sh|
        elastic = E * abs(sh(times[-1]))
        assert rf.stress[-1] < 0.6 * elastic


class TestStepByStepCreepFE:
    E = 30e9

    def _phi(self, phi_inf=2.0, b=150.0):
        def phi(t, t0):
            dt = t - t0
            return phi_inf * dt / (b + dt) if dt > 0 else 0.0
        return phi

    def _bar(self, restrain_far=False):
        from femsolver.core.model import Model
        from femsolver.elements.plane import Quad4
        from femsolver.materials.elastic import ElasticIsotropic
        mat = ElasticIsotropic(1, E=self.E, nu=0.0, rho=0.0)
        m = Model(ndm=2, ndf=2)
        m.add_material(mat)
        m.add_node(1, 0, 0); m.add_node(2, 1, 0)
        m.add_node(3, 1, 1); m.add_node(4, 0, 1)
        m.add_element(Quad4(1, (1, 2, 3, 4), mat, thickness=1.0))
        m.fix(1, [1, 1]); m.fix(4, [1, 0])      # x=0 edge held in x
        if restrain_far:
            m.fix(2, [1, 0]); m.fix(3, [1, 0])  # x=1 edge held in x too
        return m

    def test_determinate_creep_grows_by_one_plus_phi(self):
        m = self._bar()
        phi = self._phi()
        P = 1e6

        def loads(model):
            model.add_nodal_load(2, [P / 2, 0])
            model.add_nodal_load(3, [P / 2, 0])

        times = np.array([28, 30, 40, 70, 150, 400, 1000, 3650], float)
        res = StepByStepCreepFE(m, E_c=self.E, phi=phi).run(
            times, sustained_loads=loads, track=[(2, 0)])
        d = res.disp[(2, 0)]
        phiv = np.array([phi(t, 28) for t in times])
        # deflection grows exactly as (1+phi)
        assert np.allclose(d, d[0] * (1.0 + phiv), rtol=1e-6)

    def test_determinate_no_stress_redistribution(self):
        m = self._bar()
        P = 1e6

        def loads(model):
            model.add_nodal_load(2, [P / 2, 0])
            model.add_nodal_load(3, [P / 2, 0])

        times = np.array([28, 40, 150, 1000, 3650], float)
        res = StepByStepCreepFE(m, E_c=self.E, phi=self._phi()).run(
            times, sustained_loads=loads)
        sxx = res.element_stress[1][:, 0]
        # statically determinate -> axial stress unchanged by creep
        assert np.allclose(sxx, sxx[0], rtol=1e-6)

    def test_indeterminate_restrained_shrinkage_relaxes(self):
        m = self._bar(restrain_far=True)
        phi = self._phi()

        def sh(t):
            dt = t - 7
            return -300e-6 * dt / (120.0 + dt) if dt > 0 else 0.0

        times = np.array([28, 40, 150, 1000, 3650, 10000], float)
        res = StepByStepCreepFE(m, E_c=self.E, phi=phi, shrinkage=sh).run(
            times, sustained_loads=lambda model: None)
        sxx = res.element_stress[1][:, 0]
        # tensile restraint stress, far below the un-relaxed elastic value
        assert sxx[-1] > 0
        elastic = self.E * abs(sh(times[-1]))
        assert sxx[-1] < 0.55 * elastic       # creep relieved most of it

    def test_result_shapes_and_unsupported(self):
        m = self._bar()
        times = np.array([28, 100, 1000], float)
        res = StepByStepCreepFE(m, E_c=self.E, phi=self._phi()).run(
            times, sustained_loads=lambda mod: mod.add_nodal_load(2, [1e5, 0]),
            track=[(2, 0)], track_reactions=[(1, 0)])
        assert res.times.shape == (3,)
        assert res.disp[(2, 0)].shape == (3,)
        assert res.reactions[(1, 0)].shape == (3,)
        assert res.element_stress[1].shape[0] == 3
        # a beam element is unsupported by the continuum creep march
        from femsolver.core.model import Model
        from femsolver.elements.beam import BeamColumn2D
        from femsolver.materials.elastic import ElasticIsotropic
        mat = ElasticIsotropic(1, E=self.E, nu=0.2, rho=0.0)
        mb = Model(ndm=2, ndf=3)
        mb.add_material(mat)
        mb.add_node(1, 0, 0); mb.add_node(2, 1, 0)
        mb.add_element(BeamColumn2D(1, (1, 2), mat, 0.1, 1e-4))
        mb.fix(1, [1, 1, 1])
        # no continuum elements -> empty march runs but tracks nothing of note;
        # the kernel raises only if a beam is passed as a creep element, which
        # it isn't here (filtered out), so the run is a no-op creep-wise.
        res2 = StepByStepCreepFE(mb, E_c=self.E, phi=self._phi()).run(
            times, sustained_loads=lambda mod: mod.add_nodal_load(2, [0, -1e3, 0]),
            track=[(2, 1)])
        assert res2.element_stress == {}     # no continuum elements tracked


class TestApplyShrinkageLoad:
    def _unit_cube(self):
        from femsolver.core.model import Model
        from femsolver.elements.solid import Hex8
        from femsolver.materials.elastic import ElasticIsotropic
        mat = ElasticIsotropic(1, E=30e9, nu=0.2, rho=0.0)
        m = Model(ndm=3, ndf=3)
        m.add_material(mat)
        pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        for i, (x, y, z) in enumerate(pts):
            m.add_node(i + 1, x, y, z)
        m.add_element(Hex8(1, (1, 2, 3, 4, 5, 6, 7, 8), mat))
        return m

    def test_free_shrinkage_shortens(self):
        from femsolver.analysis.linear_static import LinearStaticAnalysis
        m = self._unit_cube()
        # determinate restraint: free to shrink in each axis
        m.fix(1, [1, 1, 1]); m.fix(4, [1, 0, 1])
        m.fix(5, [1, 1, 0]); m.fix(2, [0, 1, 1])
        eps_sh = -300e-6
        n = apply_shrinkage_load(m, eps_sh=eps_sh)
        assert n == 1
        LinearStaticAnalysis(m).run()
        # free isotropic shrinkage -> each free face strains by eps_sh
        assert m.node(2).disp[0] == pytest.approx(eps_sh, rel=1e-6)   # x=1
        assert m.node(4).disp[1] == pytest.approx(eps_sh, rel=1e-6)   # y=1

    def test_restrained_shrinkage_develops_reactions(self):
        from femsolver.analysis.linear_static import LinearStaticAnalysis
        m = self._unit_cube()
        # restrain the x=0 face fully and the x=1 face against x only,
        # so the member is axially restrained in x but the solve still
        # has free DOFs (transverse Poisson movement on the x=1 face).
        for nd in (1, 4, 5, 8):          # x=0 face
            m.fix(nd, [1, 1, 1])
        for nd in (2, 3, 6, 7):          # x=1 face: restrain x only
            m.fix(nd, [1, 0, 0])
        apply_shrinkage_load(m, eps_sh=-300e-6)
        LinearStaticAnalysis(m).run()
        # axial restraint of a shortening shrinkage -> tensile reactions:
        # the x=0 face pulls the block toward -x (negative Rx there)
        Rx0 = sum(m.node(nd).reaction[0] for nd in (1, 4, 5, 8))
        assert abs(Rx0) > 1.0                       # genuine restraint force
        # global equilibrium of the self-strained body
        Rx_all = sum(m.node(nd).reaction[0] for nd in range(1, 9))
        assert Rx_all == pytest.approx(0.0, abs=1e-3)
