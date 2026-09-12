"""Tests for the reinforced-concrete layered shell section
(``ReinforcedConcreteShellSection``) -- the Midas-style multi-layered
grid / concrete shell.

Validation strategy
-------------------
1. **Guard rails** -- invalid thickness / Simpson points / k_shear / nu /
   rebar area / bending direction are rejected.
2. **Simpson integration** -- weights sum to the zone thickness.
3. **Elastic stiffness** -- a plain-concrete section reproduces the
   analytic ``A = E0 t`` and ``D = E0 t^3 / 12`` and has ~zero coupling.
4. **Kinematics** -- ``eps(z) = eps_m + z kappa`` matches the guide's
   worked example (kappa = 0.02/m, z = 0.1 m -> 0.002).
5. **Smeared steel** -- 78 mm^2 @ 100 mm -> 780 mm^2/m; symmetric steel
   adds exactly ``2 As Es`` to ``A`` and ``2 As Es z^2`` to ``D``.
6. **Coupling** -- asymmetric (single-face) steel produces the exact
   ``B = As Es z`` coupling term.
7. **Rigorous equivalence** -- the section's local-x response (forces
   AND full tangent) is bit-identical to an independently-tested
   :class:`FiberSection2D` built from the same through-thickness points,
   across random *nonlinear* (cracked / tension-stiffened) strain states.
8. **Element drop-in** -- the section plugs into ``ShellMITC4`` and yields
   a symmetric, finite 24x24 stiffness.
9. **State lifecycle** -- commit / revert / clone give independent state.
10. **Moment-curvature driver** -- pure bending holds ``N ~ 0``, detects
    reinforcement yield, and the ultimate moment is of the right order.
"""
import numpy as np
import pytest

from femsolver import (
    ConcreteKentPark,
    ConcreteZone,
    Fiber,
    FiberSection2D,
    Model,
    RebarLayer,
    ReinforcedConcreteShellSection,
    ShellMITC4,
    shell_moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteTensionStiffening,
    UniaxialReinforcingSteel,
)
from femsolver.shell_sections.rc_layered import _simpson_points


# ------------------------------------------------------------ helpers

E_S = 200e9
FC = 30e6
T = 0.30


def _conc():
    return ConcreteKentPark(fpc=FC, eps_c0=0.002, fpcu=6e6, eps_cu=0.005)


def _conc_ts():
    """Concrete with tension stiffening (has a cracking transition)."""
    return ConcreteTensionStiffening(
        ConcreteKentPark(fpc=FC, eps_c0=0.002, fpcu=6e6, eps_cu=0.005),
        f_ct=0.62 * np.sqrt(FC / 1e6) * 1e6, E_ct=_conc().E0,
        eps_decay=3e-4,
    )


def _steel():
    return UniaxialReinforcingSteel(E_S, 500e6, 600e6, 0.01, 0.08)


# ====================================================== guard rails

def test_rejects_nonpositive_thickness():
    with pytest.raises(ValueError, match="thickness"):
        ReinforcedConcreteShellSection.from_midas_grid(0.0, _conc())


def test_rejects_empty_zones():
    with pytest.raises(ValueError, match="at least one concrete zone"):
        ReinforcedConcreteShellSection(0.3, [])


def test_rejects_even_simpson_points():
    with pytest.raises(ValueError, match="odd number of points"):
        ReinforcedConcreteShellSection.from_midas_grid(
            T, _conc(), n_simpson=4)


def test_rejects_bad_k_shear():
    with pytest.raises(ValueError, match="k_shear"):
        ReinforcedConcreteShellSection.from_midas_grid(
            T, _conc(), k_shear=1.5)


def test_rejects_bad_nu():
    with pytest.raises(ValueError, match="nu_concrete"):
        ReinforcedConcreteShellSection.from_midas_grid(
            T, _conc(), nu_concrete=0.6)


def test_rejects_nonpositive_rebar_area():
    with pytest.raises(ValueError, match="area_per_width"):
        ReinforcedConcreteShellSection(
            T, [ConcreteZone(-0.15, 0.15, _conc())],
            [RebarLayer(z=0.1, area_per_width=0.0, material=_steel())],
        )


def test_driver_rejects_bad_direction():
    sec = ReinforcedConcreteShellSection.from_midas_grid(T, _conc_ts())
    with pytest.raises(ValueError, match="direction"):
        shell_moment_curvature(sec, direction="z")


# ====================================================== Simpson rule

def test_simpson_weights_sum_to_thickness():
    for n_pts in (3, 5, 7):
        for n_div in (1, 2, 4):
            z, w = _simpson_points(-0.15, 0.15, n_pts, n_div)
            assert w.sum() == pytest.approx(0.30, rel=1e-12)
            assert z.min() == pytest.approx(-0.15)
            assert z.max() == pytest.approx(0.15)


def test_simpson_integrates_quadratic_exactly():
    # Simpson is exact for cubics: int_{-0.15}^{0.15} z^2 dz = 2*0.15^3/3
    z, w = _simpson_points(-0.15, 0.15, 3, 1)
    assert float(w @ (z ** 2)) == pytest.approx(2 * 0.15 ** 3 / 3, rel=1e-12)


# ====================================================== elastic stiffness

def test_elastic_membrane_and_bending_match_analytic():
    sec = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), n_simpson=5, n_div=4)
    E0 = _conc().E0
    assert sec.D_membrane()[0, 0] == pytest.approx(E0 * T, rel=1e-10)
    assert sec.D_bending()[0, 0] == pytest.approx(E0 * T ** 3 / 12, rel=1e-10)
    # symmetric plain concrete -> no membrane-bending coupling
    assert np.max(np.abs(sec.D_coupling())) < 1e-3


def test_shear_stiffness():
    sec = ReinforcedConcreteShellSection.from_midas_grid(T, _conc())
    E0 = _conc().E0
    G = E0 / (2 * (1 + 0.2))
    assert sec.D_shear()[0, 0] == pytest.approx(5 / 6 * G * T, rel=1e-10)
    assert sec.D_shear()[0, 1] == 0.0


# ====================================================== kinematics

def test_strain_through_thickness_matches_guide_example():
    sec = ReinforcedConcreteShellSection.from_midas_grid(T, _conc())
    # guide: kappa = 0.02/m, rebar at z = 0.1 m -> bending strain 0.002
    ex, ey, gxy = sec.strain_at([0, 0, 0], [0.02, 0, 0], z=0.1)
    assert ex == pytest.approx(0.002)
    ex, _, _ = sec.strain_at([0, 0, 0], [0.02, 0, 0], z=-0.1)
    assert ex == pytest.approx(-0.002)   # opposite face, opposite sign


# ====================================================== smeared steel

def test_smeared_steel_area_per_width():
    r = RebarLayer.from_bars(z=0.1, bar_area=78e-6, spacing=0.1,
                             material=_steel())
    assert r.area_per_width == pytest.approx(780e-6)   # 780 mm^2/m


def test_symmetric_steel_transformed_contribution_exact():
    plain = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), n_simpson=5, n_div=4)
    rl = [RebarLayer.from_bars(+0.1, 78e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 78e-6, 0.1, _steel())]
    rc = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), rl, n_simpson=5, n_div=4)
    As = 780e-6
    assert rc.D_membrane()[0, 0] - plain.D_membrane()[0, 0] == \
        pytest.approx(2 * As * E_S, rel=1e-10)
    assert rc.D_bending()[0, 0] - plain.D_bending()[0, 0] == \
        pytest.approx(2 * As * E_S * 0.1 ** 2, rel=1e-10)
    # symmetric steel keeps coupling ~ 0
    assert abs(rc.D_coupling()[0, 0]) < 1e-3


def test_asymmetric_steel_produces_exact_coupling():
    As = 1570e-6
    rc = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), [RebarLayer(z=-0.1, area_per_width=As, material=_steel())],
        n_simpson=5, n_div=4)
    # B[0,0] = sum As Es z ; steel below mid-surface (z<0) -> negative
    assert rc.D_coupling()[0, 0] == pytest.approx(As * E_S * (-0.1), rel=1e-10)


def test_grid_y_steel_activates_yy_not_xx():
    As = 780e-6
    rc = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(),
        [RebarLayer(z=0.1, area_per_width=As, material=_steel(),
                    theta_deg=90.0)],
        n_simpson=5, n_div=4)
    plain = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), n_simpson=5, n_div=4)
    # 90-degree steel adds to A_yy, leaves A_xx unchanged
    assert rc.D_membrane()[1, 1] - plain.D_membrane()[1, 1] == \
        pytest.approx(As * E_S, rel=1e-10)
    assert rc.D_membrane()[0, 0] == pytest.approx(plain.D_membrane()[0, 0],
                                                  rel=1e-10)


# ====================================================== rigorous equivalence

def test_local_x_response_matches_fiber_section_bit_for_bit():
    """The section's local-x response (forces + full tangent) must equal
    an independent FiberSection2D built from the same through-thickness
    points -- across random nonlinear strain states. This validates the
    integration, kinematics, steel projection and coupling against a
    separately-tested code path."""
    rl = [RebarLayer.from_bars(+0.1, 78e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 157e-6, 0.1, _steel())]   # asymmetric
    sec = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc_ts(), rl, n_simpson=5, n_div=6)
    # Build the unit-width fiber twin from the same internal points.
    # Mapping y = -z, kappa_beam = kappa_shell makes N, M and tangent equal.
    fibers = [Fiber(y=-p.z, z=0.0, area=p.w, material=p.mat_x.clone())
              for p in sec._cpoints]
    fibers += [Fiber(y=-r.z, z=0.0, area=r.As, material=r.material.clone())
               for r in sec._rpoints]
    fs = FiberSection2D(fibers)

    rng = np.random.default_rng(1234)
    for _ in range(12):
        eps_a = float(rng.normal(0, 5e-4))
        kap = float(rng.normal(0, 0.02))
        s_sh, ks_sh = sec.get_response(np.array([eps_a, 0, 0, kap, 0, 0]))
        s_fb, ks_fb = fs.get_response(np.array([eps_a, kap]))
        assert s_sh[0] == pytest.approx(s_fb[0], abs=1e-6, rel=1e-12)   # N
        assert s_sh[3] == pytest.approx(s_fb[1], abs=1e-6, rel=1e-12)   # M
        assert ks_sh[0, 0] == pytest.approx(ks_fb[0, 0], rel=1e-12)     # A
        assert ks_sh[0, 3] == pytest.approx(ks_fb[0, 1], rel=1e-12)     # B
        assert ks_sh[3, 3] == pytest.approx(ks_fb[1, 1], rel=1e-12)     # D
        sec.revert_state()
        fs.revert_state()


# ====================================================== element drop-in

def test_plugs_into_shell_mitc4():
    rl = [RebarLayer.from_bars(+0.1, 78e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 78e-6, 0.1, _steel())]
    sec = ReinforcedConcreteShellSection.from_midas_grid(T, _conc(), rl)
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 1.0, 1.0, 0.0)
    m.add_node(4, 0.0, 1.0, 0.0)
    el = ShellMITC4(1, (1, 2, 3, 4), material=None, section=sec)
    m.add_element(el)
    K = el.K_global()
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    assert np.all(np.isfinite(K))
    # thickness / k_shear surfaced for the element
    assert el.thickness == pytest.approx(T)


# ====================================================== state lifecycle

def test_commit_revert_and_clone_independent():
    sec = ReinforcedConcreteShellSection.from_midas_grid(T, _conc())
    # push into compression, commit
    sec.section_response([-1e-3, 0, 0], [0, 0, 0])
    sec.commit_state()
    committed = sec._eps_committed.copy()
    assert committed[0] == pytest.approx(-1e-3)
    # a clone is independent
    clone = sec.clone()
    clone.section_response([-4e-3, 0, 0], [0, 0, 0])
    clone.commit_state()
    assert sec._eps_committed[0] == pytest.approx(-1e-3)   # original intact
    assert clone._eps_committed[0] == pytest.approx(-4e-3)
    # revert discards trial
    sec.section_response([-9e-3, 0, 0], [0, 0, 0])
    sec.revert_state()
    assert sec._eps_trial[0] == pytest.approx(-1e-3)


# ====================================================== moment-curvature

def test_moment_curvature_pure_bending():
    # Plain concrete (no tension stiffening): the clean cracked RC
    # response -- a monotonic rise to yield, then a plateau to ultimate.
    rl = [RebarLayer.from_bars(+0.1, 157e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 157e-6, 0.1, _steel())]
    sec = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), rl, n_simpson=5, n_div=6)
    res = shell_moment_curvature(sec, direction="x", N_target=0.0,
                                 kappa_max=0.05, n_steps=80)
    # membrane force held near zero throughout (pure bending)
    assert max(abs(p.N) for p in res.points) < 10.0        # N/m
    # reinforcement yields at some point
    assert res.M_y is not None and res.kappa_y is not None
    # moment monotonically increases up to (at least) yield
    kap_y = res.kappa_y
    pre = [p.M for p in res.points if 0 < p.kappa <= kap_y]
    assert all(b >= a - 1.0 for a, b in zip(pre, pre[1:]))
    # ultimate at least matches yield (strain hardening / plateau)
    assert res.M_u >= res.M_y - 1.0
    # ultimate moment within a sane band of the ACI singly-reinforced Mn
    As, fy, d = 1570e-6, 500e6, 0.25
    a = As * fy / (0.85 * FC * 1.0)
    Mn = As * fy * (d - a / 2)      # ~ per metre width, tension steel only
    assert 0.7 * Mn < res.M_u < 2.0 * Mn


def test_tension_stiffening_gives_cracking_moment():
    # With a tensile strength the driver should flag a cracking knee at a
    # moment near f_ct * t^2 / 6 (elastic section modulus per width).
    rl = [RebarLayer.from_bars(+0.1, 157e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 157e-6, 0.1, _steel())]
    sec = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc_ts(), rl, n_simpson=5, n_div=6)
    res = shell_moment_curvature(sec, direction="x", N_target=0.0,
                                 kappa_max=0.05, n_steps=120)
    f_ct = 0.62 * np.sqrt(FC / 1e6) * 1e6
    M_cr_hand = f_ct * T ** 2 / 6.0
    assert res.M_cr is not None
    assert 0.6 * M_cr_hand < res.M_cr < 2.5 * M_cr_hand


def test_moment_curvature_axial_load_raises_capacity():
    rl = [RebarLayer.from_bars(+0.1, 157e-6, 0.1, _steel()),
          RebarLayer.from_bars(-0.1, 157e-6, 0.1, _steel())]
    sec = ReinforcedConcreteShellSection.from_midas_grid(
        T, _conc(), rl, n_simpson=5, n_div=6)
    m0 = shell_moment_curvature(sec, N_target=0.0, kappa_max=0.03,
                                n_steps=40).M_u
    # a modest membrane compression (N_target < 0 = compression) raises M_u
    m_c = shell_moment_curvature(sec, N_target=-500e3, kappa_max=0.03,
                                 n_steps=40).M_u
    assert m_c > m0
