"""Phase 42.6 -- Large-strain + contact capstone.

Three vignettes that exercise the Phase 42 modules:

1. **Rubber bushing** under uniaxial compression -- Neo-Hookean and
   Mooney-Rivlin stress responses across a stretch range, showing
   how the constants ``c_10`` and ``c_01`` shape the curve.
2. **Finite-strain J2 steel coupon** under monotonic uniaxial
   stretch, showing the elastic-plastic transition and post-yield
   hardening at large strain.
3. **Block on rigid surface** with Coulomb friction (stick-then-slip
   under a ramped tangential load).

Each vignette computes results from the constitutive / element
machinery directly (no full-mesh equilibrium iteration required)
so the example runs in a fraction of a second.

Run::

    python examples/52_large_strain_capstone.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ContactNodeToPlane3D,
    ElasticIsotropic,
    FiniteJ2Plasticity3D,
    Model,
    MooneyRivlin3D,
    NeoHookean3D,
)


def line(width: int = 76) -> None:
    print("-" * width)


# ============================================================ rubber bushing

def vignette_rubber_bushing() -> None:
    print()
    line()
    print("(1) Rubber bushing -- uniaxial compression / tension")
    line()
    print("Stretches go from compression (lam < 1) through neutral to")
    print("tension (lam > 1). Both Neo-Hookean and Mooney-Rivlin use")
    print("the same initial shear modulus mu_0 = 0.7 MPa.")
    print()
    # Material constants chosen for equal initial shear modulus
    nh = NeoHookean3D(tag=1, E=2.10e6, nu=0.499)   # mu_0 ~ 0.7 MPa
    mr = MooneyRivlin3D(tag=1, c_10=0.30e6, c_01=0.05e6, K=2.0e9)
    print(f"NH:  E = 2.10 MPa, nu = 0.499  (mu_0 = E/(2(1+nu)) ~ "
          f"{2.1e6/(2*1.499):.2e} Pa)")
    print(f"MR:  c_10 = 0.30, c_01 = 0.05 MPa, K = 2.0 GPa  "
          f"(mu_0 = 2(c_10+c_01) = {mr.mu_0*1e-6:.2f} MPa)")
    print()
    print(f"  {'lambda':>10}{'NH sigma_xx (kPa)':>22}"
          f"{'MR sigma_xx (kPa)':>22}")
    line(54)
    for lam in [0.50, 0.70, 0.90, 1.00, 1.20, 1.50, 2.00, 3.00, 4.00]:
        F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])
        s_nh, _ = nh.response_sigma(F)
        s_mr, _ = mr.response_sigma(F)
        print(f"  {lam:>10.2f}{s_nh[0]*1e-3:>22.1f}"
              f"{s_mr[0]*1e-3:>22.1f}")
    print()
    print("Note: rubber stiffens nonlinearly at large stretch (Mooney-")
    print("Rivlin's I_2 term gives a slightly stiffer high-strain response).")


# ============================================================ finite-strain J2

def vignette_finite_j2() -> None:
    print()
    line()
    print("(2) Finite-strain J2 steel coupon -- monotonic uniaxial stretch")
    line()
    E, nu, sy0, H = 2.0e11, 0.30, 400e6, 1e9
    mat = FiniteJ2Plasticity3D(
        tag=1, E=E, nu=nu, sigma_y0=sy0, H=H,
    )
    print(f"Steel: E = {E*1e-9:.0f} GPa, nu = {nu}, "
          f"sigma_y0 = {sy0*1e-6:.0f} MPa, H = {H*1e-9:.1f} GPa")
    print()
    print(f"  {'strain':>10}{'lambda':>10}"
          f"{'sigma_eq (MPa)':>18}{'alpha':>12}{'note':>14}")
    line(64)
    # Monotonic uniaxial stretch path
    for eps in [0.0005, 0.0010, 0.0020, 0.0050, 0.0100,
                 0.0200, 0.0500, 0.1000]:
        lam = 1.0 + eps
        F = np.diag([lam, 1.0 / math.sqrt(lam), 1.0 / math.sqrt(lam)])
        S_voigt, _ = mat.response_S(F)
        # Recover Cauchy: sigma = J^{-1} F S F^T
        J = float(np.linalg.det(F))
        F_inv = np.linalg.inv(F)
        S = np.zeros((3, 3))
        S[0, 0], S[1, 1], S[2, 2] = S_voigt[0], S_voigt[1], S_voigt[2]
        S[0, 1] = S[1, 0] = S_voigt[3]
        S[1, 2] = S[2, 1] = S_voigt[4]
        S[0, 2] = S[2, 0] = S_voigt[5]
        sigma_tensor = F @ S @ F.T / J
        # von Mises
        dev = sigma_tensor - np.eye(3) * np.trace(sigma_tensor) / 3.0
        sigma_eq = math.sqrt(1.5 * np.sum(dev ** 2))
        mat.commit_state()
        note = "elastic" if mat.alpha_committed < 1e-9 else "plastic"
        print(f"  {eps:>10.4f}{lam:>10.4f}"
              f"{sigma_eq*1e-6:>18.1f}{mat.alpha_committed:>12.4f}"
              f"{note:>14}")
    print()
    print("Yield at eps ~ 0.002 (sigma_y = 400 MPa); post-yield the")
    print("equivalent stress grows by H · alpha each step.")


# ============================================================ contact

def vignette_contact_friction() -> None:
    print()
    line()
    print("(3) Slave node on rigid surface with Coulomb friction")
    line()
    K_N = 1.0e8
    mu = 0.40

    mat_dummy = ElasticIsotropic(1, E=1.0, nu=0.3, rho=0.0)
    m = Model(ndm=3, ndf=3)
    m.add_material(mat_dummy)
    m.add_node(1, 0.0, 0.0, -0.001)        # 1 mm penetration
    contact = ContactNodeToPlane3D(
        1, (1,), plane_point=[0, 0, 0],
        plane_normal=[0, 0, 1], K_N=K_N, mu=mu,
    )
    m.add_element(contact)

    F_N_expected = K_N * 0.001
    cap_friction = mu * F_N_expected
    print(f"Normal penetration  g_N = -1.0 mm, K_N = {K_N*1e-6:.0f} MN/m")
    print(f"Mu = {mu}, expected friction cap = mu * F_N = "
          f"{cap_friction*1e-3:.1f} kN")
    print()
    print(f"  {'u_x (mm)':>12}{'F_x (kN)':>14}{'F_N (kN)':>14}"
          f"{'state':>10}")
    line(50)
    # Reset committed slip
    contact.u_T_committed = np.zeros(2)
    for u in [0.0, 0.005, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000]:
        m.node(1).disp[:] = [u * 1e-3, 0.0, 0.0]
        f = contact.f_int_global()
        F_x = f[0]
        F_N = f[2]
        # Detect state by comparing the trial friction force to the cap
        state = "STICK" if abs(F_x) < cap_friction * 0.99 else "SLIP"
        print(f"  {u:>12.3f}{F_x*1e-3:>14.2f}{F_N*1e-3:>14.2f}"
              f"{state:>10}")
    print()
    print("Lateral force grows linearly with u_x (stick) until the")
    print("Coulomb cap is reached; beyond that the node slips at a")
    print("constant tangential force = mu * F_N.")


# ============================================================ main

def main() -> None:
    print("=" * 78)
    print("Phase 42.6 -- Large-strain + contact capstone")
    print("=" * 78)

    vignette_rubber_bushing()
    vignette_finite_j2()
    vignette_contact_friction()

    print()
    print("=" * 78)
    print("Theme B closed: hyperelastic materials, TL Hex8 large-strain,")
    print("                finite-strain J2 plasticity, contact + friction.")
    print("=" * 78)


if __name__ == "__main__":
    main()
