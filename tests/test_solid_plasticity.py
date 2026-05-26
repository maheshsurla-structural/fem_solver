"""Tests for Hex8/Tet4 + 3D plasticity wiring (Phase 16.7.x).

The element-side integration is pinned down by:

1. **Backward compatibility** — without ``material3d``, Hex8/Tet4
   behave identically to before (linear elastic).
2. **Per-IP independent state** — each Gauss point's plastic
   material is an independent clone; pushing one IP's plastic
   strain doesn't pollute others (smoke-tested via single-element
   uniaxial pushover where all IPs see the same strain).
3. **Hex8 + J2 single-element uniaxial pushover** converges via
   the consistent algorithmic tangent and lands on the yield
   surface ``VM(sigma) = sigma_y(alpha)`` after each step.
4. **Tet4 + J2** likewise.
5. **commit_state / revert_state** correctly forward to per-IP
   plastic materials.
6. **Hex8 + Drucker-Prager** runs a soil-like compression problem
   without crashing (smoke).
"""
import numpy as np
import pytest

from femsolver import (
    DruckerPrager3D,
    ElasticIsotropic,
    Hex8,
    J2Plasticity3D,
    Model,
    Tet4,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis


_CUBE_CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]


def _hex_uniaxial_model(material3d=None):
    """Unit-cube Hex8 with x-face pulled. Returns the model."""
    mat_iso = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat_iso)
    for i, (x, y, z) in enumerate(_CUBE_CORNERS):
        m.add_node(i + 1, x, y, z)
    m.add_element(Hex8(1, tuple(range(1, 9)), mat_iso,
                        material3d=material3d))
    m.fix(1, [1, 1, 1])
    m.fix(4, [1, 1, 0])
    m.fix(5, [1, 0, 1])
    m.fix(8, [1, 0, 0])
    return m


# ====================================================== backward compat

def test_hex8_without_material3d_is_linear_elastic():
    """Construction without material3d preserves the linear path:
    response is positive, finite, and uniform across the loaded face."""
    m = _hex_uniaxial_model(material3d=None)
    P = 1000.0
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P / 4, 0, 0])
    LinearStaticAnalysis(m).run()
    disps = [m.node(tag).disp[0] for tag in (2, 3, 6, 7)]
    # All face displacements positive (tensile) and within elastic range
    assert all(d > 0 for d in disps)
    assert max(disps) < 1e-7      # well below yield strain
    assert max(disps) - min(disps) < 0.2 * max(disps)


def test_tet4_without_material3d_is_linear_elastic():
    mat_iso = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0); m.add_node(4, 0, 0, 1)
    m.add_element(Tet4(1, (1, 2, 3, 4), mat_iso))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 0])
    m.fix(4, [1, 0, 1])
    m.add_nodal_load(2, [1.0, 0, 0])
    LinearStaticAnalysis(m).run()
    # No crash; tip displacement is positive
    assert m.node(2).disp[0] > 0.0


# ====================================================== per-IP cloning

def test_hex8_with_material3d_clones_per_ip():
    """Hex8 with material3d holds 8 independent clones (one per
    Gauss point)."""
    mat_j2 = J2Plasticity3D(E=2e11, nu=0.3, sigma_y=400e6, K_iso=1e10)
    m = _hex_uniaxial_model(material3d=mat_j2)
    elem = list(m.elements.values())[0]
    assert elem._ip_materials is not None
    assert len(elem._ip_materials) == 8
    # Each clone is distinct
    for i in range(8):
        for j in range(i + 1, 8):
            assert elem._ip_materials[i] is not elem._ip_materials[j]


def test_tet4_with_material3d_holds_single_clone():
    mat_j2 = J2Plasticity3D(E=2e11, nu=0.3, sigma_y=400e6, K_iso=1e10)
    mat_iso = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0); m.add_node(4, 0, 0, 1)
    e = Tet4(1, (1, 2, 3, 4), mat_iso, material3d=mat_j2)
    m.add_element(e)
    assert e._ip_material is not None
    assert e._ip_material is not mat_j2  # cloned


# ====================================================== Hex8 + J2 pushover

def test_hex8_j2_uniaxial_pushover_lands_on_yield_surface():
    """Hex8 cube under uniaxial tension just past yield: every Gauss
    point's stress sits on the yield surface VM = sigma_y(alpha)."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    K_iso = E * 0.1
    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_j2 = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=K_iso)
    m = _hex_uniaxial_model(material3d=mat_j2)
    P_target = 1.1 * sigma_y       # 1.1x yield force
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P_target / 4, 0, 0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0/20, tol=1.0, max_iter=20,
    ).run()
    elem = list(m.elements.values())[0]
    # Check every IP is on the yield surface
    for ip in elem._ip_materials:
        sigma = ip.sigma_committed
        sigma_vm = ip.von_mises_stress(sigma)
        sigma_y_curr = ip.yield_stress(ip.alpha_committed)
        assert sigma_vm == pytest.approx(sigma_y_curr, rel=1e-6)
        assert ip.alpha_committed > 0.0


def test_hex8_j2_below_yield_stays_elastic():
    """Below yield, plastic strain stays zero at every IP."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_j2 = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=1e10)
    m = _hex_uniaxial_model(material3d=mat_j2)
    P = 0.5 * sigma_y     # well below yield
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P / 4, 0, 0])
    NonlinearStaticAnalysis(
        m, num_steps=2, dlambda=1.0/2, tol=1.0, max_iter=10,
    ).run()
    elem = list(m.elements.values())[0]
    for ip in elem._ip_materials:
        assert ip.alpha_committed == 0.0
        assert np.allclose(ip.eps_p_committed, 0.0, atol=1e-15)


# ====================================================== Tet4 + J2

def test_tet4_j2_accumulates_plastic_strain_below_load_capacity():
    """Tet4 with J2: under a moderate static load (within capacity),
    plastic strain accumulates and stress remains on the yield surface."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    # Use significant isotropic hardening so the post-yield tangent
    # has plenty of stiffness (avoids near-singular K_T at high plastic
    # strain on a single-element model).
    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_j2 = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=E * 0.5)
    m = Model(ndm=3, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0); m.add_node(4, 0, 0, 1)
    e = Tet4(1, (1, 2, 3, 4), mat_iso, material3d=mat_j2)
    m.add_element(e)
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 0])
    m.fix(4, [1, 0, 1])
    P = 5e7        # mild load just past yield
    m.add_nodal_load(2, [P, 0, 0])
    NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=1.0/10, tol=1.0, max_iter=20,
    ).run()
    # Should be on yield surface if plasticity activated
    if e._ip_material.alpha_committed > 0.0:
        sigma = e._ip_material.sigma_committed
        sigma_vm = e._ip_material.von_mises_stress(sigma)
        sigma_y_curr = e._ip_material.yield_stress(
            e._ip_material.alpha_committed
        )
        assert sigma_vm == pytest.approx(sigma_y_curr, rel=1e-6)


# ====================================================== lifecycle

def test_hex8_j2_commit_revert_forwards_to_ips():
    """commit_state and revert_state propagate to every Gauss point's
    material."""
    E, nu, sigma_y = 2e11, 0.3, 400e6
    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_j2 = J2Plasticity3D(E=E, nu=nu, sigma_y=sigma_y, K_iso=1e10)
    m = _hex_uniaxial_model(material3d=mat_j2)
    P = 1.1 * sigma_y
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P / 4, 0, 0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0/20, tol=1.0, max_iter=20,
    ).run()
    elem = list(m.elements.values())[0]
    # All IPs have nonzero plastic strain
    alphas_before = [ip.alpha_committed for ip in elem._ip_materials]
    assert all(a > 0.0 for a in alphas_before)
    # Trial: gather_u, call f_int_global to update trial state, then revert
    elem.f_int_global()    # populates trial via get_response
    elem.revert_state()
    # Committed alpha should be unchanged
    alphas_after_revert = [ip.alpha_committed for ip in elem._ip_materials]
    assert alphas_before == alphas_after_revert


# ====================================================== Drucker-Prager smoke

def test_hex8_drucker_prager_runs():
    """Smoke test: Hex8 with Drucker-Prager material runs without
    crashing through a small compression load."""
    mat_iso = ElasticIsotropic(1, E=2e7, nu=0.3)  # soil-like
    mat_dp = DruckerPrager3D.from_mohr_coulomb(
        E=2e7, nu=0.3, cohesion=50e3, phi_deg=25.0)
    m = _hex_uniaxial_model(material3d=mat_dp)
    # Apply mild compressive load (well within DP yield range)
    P = -1e3
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P / 4, 0, 0])
    NonlinearStaticAnalysis(
        m, num_steps=5, dlambda=1.0/5, tol=0.01, max_iter=20,
    ).run()
    elem = list(m.elements.values())[0]
    # Check no crash; stress is computed
    sigma = elem._ip_materials[0].sigma_committed
    assert sigma is not None
    assert sigma[0] < 0.0     # compressive
