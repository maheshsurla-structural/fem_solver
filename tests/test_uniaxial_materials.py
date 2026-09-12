"""Tests for the uniaxial-material library.

These exercise the constitutive interface at the *fiber* level — every
property the fiber section then aggregates over a discretised cross
section. Two laws are validated:

* :class:`UniaxialElastic` — stateless, ``sigma = E * eps``.
* :class:`UniaxialBilinear` — bilinear with kinematic hardening; the
  ``b = 0`` limit is elastic-perfectly-plastic.

The tests mirror the algebra of the moment-rotation spring
(:class:`BilinearMomentRotationSpring`) one layer deeper, on (sigma, eps)
instead of (M, theta_h). If the return mapping ever drifts these tests
will catch it before any fiber-section result does.
"""
import numpy as np
import pytest

from femsolver import UniaxialBilinear, UniaxialElastic


# ====================================================== UniaxialElastic ===

def test_elastic_response_is_linear():
    mat = UniaxialElastic(E=2.0e11)
    for eps in (-1.0e-3, 0.0, 1.0e-3, 5.0e-3):
        sigma, Et = mat.get_response(eps)
        assert sigma == pytest.approx(mat.E * eps, rel=1e-14)
        assert Et == pytest.approx(mat.E, rel=1e-14)


def test_elastic_rejects_nonpositive_E():
    with pytest.raises(ValueError):
        UniaxialElastic(E=0.0)
    with pytest.raises(ValueError):
        UniaxialElastic(E=-1.0)


def test_elastic_commit_revert_are_no_ops():
    """An elastic material has no state; the lifecycle calls must be safe
    no-ops so it composes with the FiberSection bookkeeping."""
    mat = UniaxialElastic(E=1.0e11)
    mat.get_response(1.0e-3)
    mat.commit_state()
    mat.revert_state()
    # next call must produce the same answer as before
    sigma, _ = mat.get_response(1.0e-3)
    assert sigma == pytest.approx(1.0e11 * 1.0e-3, rel=1e-14)


def test_elastic_clone_is_independent_object():
    mat = UniaxialElastic(E=1.0e11)
    mat2 = mat.clone()
    assert mat2 is not mat
    # numerical behaviour identical
    s1, _ = mat.get_response(1e-3)
    s2, _ = mat2.get_response(1e-3)
    assert s1 == s2


# ===================================================== UniaxialBilinear ===

def test_bilinear_rejects_invalid_args():
    with pytest.raises(ValueError):
        UniaxialBilinear(E=0.0, sigma_y=1.0)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=0.0)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=1.0, b=-0.01)
    with pytest.raises(ValueError):
        UniaxialBilinear(E=1.0, sigma_y=1.0, b=1.0)


@pytest.mark.parametrize("b", [0.0, 0.05, 0.1])
def test_bilinear_below_yield_is_elastic(b):
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=b)
    eps = 0.5 * mat.sigma_y / mat.E
    sigma, Et = mat.get_response(eps)
    assert sigma == pytest.approx(mat.E * eps, rel=1e-14)
    assert Et == pytest.approx(mat.E, rel=1e-14)
    assert mat.eps_p_trial == 0.0


def test_bilinear_at_yield_is_elastic_then_yields_on_next_step():
    """At the yield surface ``f = 0`` we still consider the step elastic;
    the *next* incremental step is what triggers plastic flow."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    sigma, Et = mat.get_response(mat.sigma_y / mat.E)
    assert sigma == pytest.approx(mat.sigma_y, rel=1e-14)
    assert mat.eps_p_trial == 0.0
    # next step past yield
    mat.commit_state()
    sigma2, Et2 = mat.get_response(2.0 * mat.sigma_y / mat.E)
    assert sigma2 == pytest.approx(mat.sigma_y, rel=1e-14)
    assert mat.eps_p_trial > 0.0
    assert Et2 == 0.0  # EPP post-yield


def test_epp_plateau_clips_stress_at_sigma_y():
    """EPP (b=0): once yielded, sigma stays at sigma_y under monotonic
    loading, plastic strain accumulates linearly with total strain."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    eps_values = [2.0, 3.0, 5.0, 10.0]
    sigmas = []
    for k in eps_values:
        mat.get_response(k * mat.sigma_y / mat.E)
        mat.commit_state()
        sigmas.append(mat.sigma_trial)
    np.testing.assert_allclose(sigmas, mat.sigma_y, rtol=1e-14)
    # plastic strain at the final state ~ (eps - eps_y)
    eps_final = 10.0 * mat.sigma_y / mat.E
    eps_y = mat.sigma_y / mat.E
    assert mat.eps_p_committed == pytest.approx(eps_final - eps_y, rel=1e-14)


def test_epp_unloading_is_elastic_with_original_E():
    """After plastic flow, unloading travels down a line of slope E."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    eps_p = mat.eps_p_committed
    # unload partially
    eps_unload = 2.0 * mat.sigma_y / mat.E
    sigma, Et = mat.get_response(eps_unload)
    assert sigma == pytest.approx(mat.E * (eps_unload - eps_p), rel=1e-14)
    assert Et == pytest.approx(mat.E, rel=1e-14)


def test_epp_reverse_yielding_clips_at_minus_sigma_y():
    """Load + into yield, then strongly negative: stress clips at -sigma_y."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    mat.get_response(-3.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat.sigma_trial == pytest.approx(-mat.sigma_y, rel=1e-14)


@pytest.mark.parametrize("b", [0.05, 0.1, 0.2])
def test_post_yield_tangent_is_b_times_E(b):
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=b)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat.Et == pytest.approx(b * mat.E, rel=1e-12)


def test_post_yield_slope_finite_difference():
    """Finite-difference dM/d_eps on the post-yield branch matches b*E."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.1)
    eps_a = 2.0 * mat.sigma_y / mat.E
    eps_b = 3.0 * mat.sigma_y / mat.E
    mat.get_response(eps_a); mat.commit_state()
    sigma_a = mat.sigma_trial
    mat.get_response(eps_b); mat.commit_state()
    sigma_b = mat.sigma_trial
    assert (sigma_b - sigma_a) / (eps_b - eps_a) == pytest.approx(
        0.1 * mat.E, rel=1e-10
    )


def test_kinematic_hardening_translates_yield_surface():
    """After plastic loading in +, the back-stress q moves positive.
    Reverse yielding then occurs near (q - sigma_y), not at -sigma_y."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.1)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    q = mat.q_committed
    assert q > 0.0
    # Take a small step that just crosses reverse yield
    eps_reverse = (q - 1.2 * mat.sigma_y) / mat.E + mat.eps_p_committed
    mat.get_response(eps_reverse)
    # Should now be on reverse yield surface, sigma close to q - sigma_y
    assert mat.sigma_trial < q - 0.9 * mat.sigma_y


def test_revert_undoes_uncommitted_plastic_flow():
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E)
    assert mat.eps_p_trial > 0.0
    mat.revert_state()
    assert mat.eps_p_trial == 0.0
    assert mat.q_trial == 0.0


def test_clone_is_independent():
    """Each fiber must own its plastic state; cloning is the mechanism."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.0)
    mat.get_response(3.0 * mat.sigma_y / mat.E); mat.commit_state()
    mat2 = mat.clone()
    # mat2 has the same committed state at the moment of cloning
    assert mat2.eps_p_committed == mat.eps_p_committed
    # mutating the original after cloning must not change the clone
    mat.get_response(10.0 * mat.sigma_y / mat.E); mat.commit_state()
    assert mat2.eps_p_committed != mat.eps_p_committed


# ============================================================ tension stiffening

from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    ConcreteParabolaRectangle,
    ConcreteTensionStiffening,
    ConcreteTrilinear,
    ReinforcingSteelKinematic,
    UniaxialReinforcingSteel,
)


def _kp():
    return ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=12e6, eps_cu=0.0035)


class TestConcreteTensionStiffening:
    """Compression delegates to the wrapped model; tension is linear
    elastic to f_ct at eps_cr, then exponentially-decaying (continuous)."""

    def test_compression_matches_base(self):
        mat = ConcreteTensionStiffening(_kp(), f_ct=3.4e6, E_ct=_kp().E0)
        base = _kp()
        for eps in (-0.0005, -0.001, -0.002):
            assert mat.get_response(eps) == pytest.approx(base.get_response(eps))

    def test_tension_elastic_below_cracking(self):
        base = _kp()
        mat = ConcreteTensionStiffening(base, f_ct=3.4e6, E_ct=base.E0)
        s, Et = mat.get_response(0.5 * mat.eps_cr)
        assert s == pytest.approx(base.E0 * 0.5 * mat.eps_cr)
        assert Et == pytest.approx(base.E0)

    def test_peak_at_f_ct_and_continuous(self):
        mat = ConcreteTensionStiffening(_kp(), f_ct=3.4e6, E_ct=_kp().E0)
        assert mat.get_response(mat.eps_cr)[0] == pytest.approx(3.4e6, rel=1e-9)
        # continuous across cracking
        s_after = mat.get_response(mat.eps_cr * 1.0001)[0]
        assert s_after == pytest.approx(3.4e6, rel=1e-2)

    def test_tension_decays_and_stays_positive(self):
        mat = ConcreteTensionStiffening(_kp(), f_ct=3.4e6, E_ct=_kp().E0,
                                        eps_decay=1e-3)
        s1 = mat.get_response(0.001)[0]
        s2 = mat.get_response(0.003)[0]
        assert 0.0 < s2 < s1 < 3.4e6

    def test_origin_returns_compression_modulus(self):
        base = _kp()
        mat = ConcreteTensionStiffening(base, f_ct=3.4e6, E_ct=base.E0)
        s, Et = mat.get_response(0.0)
        assert s == 0.0
        assert Et == pytest.approx(base.E0)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ConcreteTensionStiffening(_kp(), f_ct=3.4e6, E_ct=-1.0)
        with pytest.raises(ValueError):
            ConcreteTensionStiffening(_kp(), f_ct=3.4e6, E_ct=_kp().E0,
                                      eps_decay=0.0)


class TestConcreteParabolaRectangle:
    """EC2 parabola-rectangle: parabola to the peak at eps_c2, constant
    plateau to eps_cu2, zero beyond; tension zero."""

    def _pr(self):
        return ConcreteParabolaRectangle(fpc=30e6, eps_c2=0.002,
                                         eps_cu2=0.0035, n=2.0)

    def test_initial_modulus(self):
        m = self._pr()
        assert m.E0 == pytest.approx(2.0 * 30e6 / 0.002)
        s, Et = m.get_response(0.0)
        assert s == 0.0
        assert Et == pytest.approx(m.E0)

    def test_peak_at_eps_c2(self):
        m = self._pr()
        s, Et = m.get_response(-0.002)
        assert s == pytest.approx(-30e6, rel=1e-9)   # full fpc (compression -)
        assert Et == pytest.approx(0.0, abs=1.0)

    def test_parabola_midpoint(self):
        # r = 0.5 -> sigma = fpc * (1 - (1-0.5)^2) = 0.75 fpc
        s, _ = self._pr().get_response(-0.001)
        assert s == pytest.approx(-0.75 * 30e6, rel=1e-9)

    def test_plateau_is_constant(self):
        m = self._pr()
        for eps in (-0.0025, -0.003, -0.0035):
            assert m.get_response(eps)[0] == pytest.approx(-30e6, rel=1e-9)

    def test_zero_past_ultimate(self):
        assert self._pr().get_response(-0.004)[0] == 0.0

    def test_tension_is_zero(self):
        s, Et = self._pr().get_response(0.001)
        assert s == 0.0 and Et == 0.0

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ConcreteParabolaRectangle(fpc=-1.0)
        with pytest.raises(ValueError):
            ConcreteParabolaRectangle(fpc=30e6, eps_c2=0.004, eps_cu2=0.0035)
        with pytest.raises(ValueError):
            ConcreteParabolaRectangle(fpc=30e6, n=0.0)


class TestUniaxialReinforcingSteel:
    """Park strain-hardening: elastic to yield, plateau to eps_sh, then a
    curved rise to (eps_su, f_su); odd-symmetric, monotonic."""

    def _s(self):
        return UniaxialReinforcingSteel(E=200e9, f_y=414e6, f_su=621e6,
                                        eps_sh=0.008, eps_su=0.10)

    def test_elastic_and_E0(self):
        m = self._s()
        assert m.E0 == pytest.approx(200e9)
        s, Et = m.get_response(0.001)          # below yield
        assert s == pytest.approx(200e9 * 0.001)
        assert Et == pytest.approx(200e9)

    def test_yield_plateau(self):
        m = self._s()
        for eps in (414e6 / 200e9, 0.004, 0.008):   # eps_y .. eps_sh
            assert m.get_response(eps)[0] == pytest.approx(414e6, rel=1e-6)

    def test_passes_through_ultimate(self):
        # curve is calibrated to hit (eps_su, f_su) and (eps_sh, f_y)
        assert self._s().get_response(0.10)[0] == pytest.approx(621e6, rel=1e-6)

    def test_hardening_monotonic(self):
        m = self._s()
        xs = np.linspace(0.008, 0.10, 25)
        ss = [m.get_response(x)[0] for x in xs]
        assert all(ss[i + 1] >= ss[i] - 1.0 for i in range(len(ss) - 1))
        assert 414e6 < ss[len(ss) // 2] < 621e6      # strictly hardening

    def test_odd_symmetric(self):
        m = self._s()
        for eps in (0.001, 0.02, 0.09):
            assert m.get_response(-eps)[0] == pytest.approx(-m.get_response(eps)[0])

    def test_holds_past_ultimate(self):
        assert self._s().get_response(0.15)[0] == pytest.approx(621e6, rel=1e-6)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            UniaxialReinforcingSteel(E=200e9, f_y=414e6, f_su=400e6,
                                     eps_sh=0.008, eps_su=0.10)     # f_su<f_y
        with pytest.raises(ValueError):
            UniaxialReinforcingSteel(E=200e9, f_y=414e6, f_su=621e6,
                                     eps_sh=0.008, eps_su=0.005)    # su<=sh
        with pytest.raises(ValueError):
            UniaxialReinforcingSteel(E=200e9, f_y=414e6, f_su=621e6,
                                     eps_sh=0.001, eps_su=0.10)     # sh<eps_y


class TestReinforcingSteelKinematic:
    """Park backbone + kinematic hardening (cyclic): monotonic response equals
    the monotonic backbone; reversals unload elastically (slope E) and
    translate the backbone through the back-stress (plan P3/G3)."""

    def _s(self):
        return ReinforcingSteelKinematic(E=200e9, f_y=414e6, f_su=621e6,
                                         eps_sh=0.008, eps_su=0.10)

    def test_monotonic_reproduces_backbone(self):
        """A committed monotonic push matches the monotonic Park backbone to
        machine precision (same curve, so the fibre sees identical stress)."""
        bb = UniaxialReinforcingSteel(E=200e9, f_y=414e6, f_su=621e6,
                                      eps_sh=0.008, eps_su=0.10)
        kin = self._s()
        for eps in np.linspace(0.0, 0.095, 120):
            s_bb = bb.get_response(eps)[0]
            s_k, _ = kin.get_response(eps)
            kin.commit_state()
            assert s_k == pytest.approx(s_bb, abs=1.0)   # <1 Pa of 4e8

    def test_benchmark_backbone_points(self):
        """Caltrans A615 Gr60 expected (plan §2.2), kip/in^2 units: the
        backbone hits (eps_sh, f_y)=(0.0075, 68) and (eps_su, f_su)=(0.09, 95),
        for both the Midas eps_su=0.09 and CSI eps_su=0.06 variants (O1)."""
        for esu in (0.09, 0.06):
            m = ReinforcingSteelKinematic(E=29000.0, f_y=68.0, f_su=95.0,
                                          eps_sh=0.0075, eps_su=esu)
            assert m.get_response(0.0075)[0] == pytest.approx(68.0, rel=1e-6)
            assert m.get_response(esu)[0] == pytest.approx(95.0, rel=1e-6)

    def test_origin_tangent(self):
        m = self._s()
        assert m.get_response(0.0) == (0.0, 200e9)

    def test_elastic_unload_slope(self):
        """Right after a tension excursion the material unloads with the
        initial elastic modulus, not the (soft) hardening tangent."""
        m = self._s()
        for eps in np.linspace(0.0, 0.02, 60):
            m.get_response(eps)
            m.commit_state()
        s_peak = m.sigma_trial
        de = -1e-6
        s_un, Et = m.get_response(0.02 + de)
        assert Et == pytest.approx(200e9, rel=1e-6)
        assert (s_un - s_peak) / de == pytest.approx(200e9, rel=1e-3)

    def test_kinematic_shift_bauschinger(self):
        """Kinematic hardening: after yielding in tension, reverse yield in
        compression starts early (at sigma = q - f_y, |.| < f_y), and the
        elastic stress span across a reversal is ~2 f_y."""
        m = self._s()
        for eps in np.linspace(0.0, 0.02, 80):
            m.get_response(eps)
            m.commit_state()
        q = m.q_committed
        assert q > 0.0                          # tension raised the back-stress
        # sweep down: elastic unloading first (Et == E), then compression
        # re-yield. Capture the stress at the last elastic step -- it should
        # sit near sigma = q - f_y.
        prev_s = m.sigma_trial
        was_elastic = False
        onset = None
        for eps in np.linspace(0.02, -0.02, 800):
            s, Et = m.get_response(eps)
            m.commit_state()
            if Et > 0.99 * m.E:
                was_elastic = True
            elif was_elastic and onset is None:     # re-entered plasticity
                onset = prev_s
            prev_s = s
        assert onset == pytest.approx(q - 414e6, rel=0.05)
        assert abs(onset) < 414e6                # yields before -f_y (Bauschinger)

    def test_commit_revert_lifecycle(self):
        """A reverted trial leaves no history; a committed one advances it."""
        m = self._s()
        for eps in np.linspace(0.0, 0.02, 60):
            m.get_response(eps)
            m.commit_state()
        p_before = m.p_committed
        m.get_response(0.03)                     # trial only, no commit
        m.revert_state()
        assert m.p_trial == p_before
        # committing the same push does advance accumulated plastic strain
        m.get_response(0.03)
        m.commit_state()
        assert m.p_committed > p_before

    def test_clone_is_independent(self):
        m = self._s()
        for eps in np.linspace(0.0, 0.02, 60):
            m.get_response(eps)
            m.commit_state()
        c = m.clone()
        c.get_response(0.05)
        c.commit_state()
        assert c.p_committed != m.p_committed    # state did not leak back


class TestConcreteTrilinear:
    """Three straight segments: elastic to the first knee, rise to the peak,
    descent to the residual, then a plateau; tension zero; continuous."""

    def _tl(self):
        return ConcreteTrilinear(fpc=30e6, eps_c0=0.002, fpcu=6e6,
                                 eps_cu=0.0035, f1_ratio=0.4)

    def test_initial_modulus_and_origin(self):
        m = self._tl()
        assert m.E0 == pytest.approx(2.0 * 30e6 / 0.002)
        s, Et = m.get_response(0.0)
        assert s == 0.0 and Et == pytest.approx(m.E0)

    def test_first_knee(self):
        # eps1 = f1_ratio * eps_c0 / 2 = 0.0004, sigma1 = 0.4 fpc
        s, Et = self._tl().get_response(-0.0004)
        assert s == pytest.approx(-0.4 * 30e6, rel=1e-9)
        assert Et == pytest.approx(2.0 * 30e6 / 0.002)   # still elastic slope

    def test_peak_and_residual(self):
        m = self._tl()
        assert m.get_response(-0.002)[0] == pytest.approx(-30e6, rel=1e-9)
        assert m.get_response(-0.0035)[0] == pytest.approx(-6e6, rel=1e-9)
        assert m.get_response(-0.005)[0] == pytest.approx(-6e6, rel=1e-9)

    def test_segments_are_linear(self):
        m = self._tl()
        # midpoint of the rise-to-peak segment equals the chord midpoint
        s1 = m.get_response(-0.0004)[0]
        s2 = m.get_response(-0.002)[0]
        smid = m.get_response(-0.0012)[0]        # midway 0.0004..0.002
        assert smid == pytest.approx(0.5 * (s1 + s2), rel=1e-6)

    def test_tension_zero(self):
        assert self._tl().get_response(0.001) == (0.0, 0.0)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ConcreteTrilinear(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035,
                              f1_ratio=1.5)
        with pytest.raises(ValueError):
            ConcreteTrilinear(fpc=30e6, eps_c0=0.002, fpcu=40e6, eps_cu=0.0035)
