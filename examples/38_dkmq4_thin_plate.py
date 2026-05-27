"""Phase 22.7 -- ShellDKMQ4 (thin-plate DKQ) convergence comparison.

This example compares the thin-plate Discrete-Kirchhoff Quadrilateral
(DKQ; ``ShellDKMQ4``) against the general-purpose
:class:`ShellMITC4` on two thin-plate problems:

1. **Cantilever plate** under uniformly distributed tip-edge load.
2. **Simply-supported plate** with central point load.

Both elements converge to the same answer with refinement; DKMQ4
gives the Kirchhoff-thin-plate answer immediately at coarse meshes
(its design intent), while MITC4 must refine to wash out
coarse-mesh discretization artifacts.

For thick plates (``L/t < 20``), DKMQ4 underpredicts the deflection
because it omits transverse-shear strain energy by construction.
Use MITC4 for those.

Run::

    python examples/38_dkmq4_thin_plate.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellDKMQ4,
    ShellMITC4,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ============================================================ helpers

def cantilever(element_cls, *, N: int, L: float = 1.0, b: float = 1.0,
                t: float = 0.01, E: float = 2.0e11, nu: float = 0.3):
    """N x N cantilever plate clamped at x=0 with unit total tip load."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * L / N, j * b / N, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(element_cls(etag, (n1, n2, n3, n4), mat,
                                        thickness=t))
            etag += 1
    for j in range(nL):
        m.fix(j * nL + 1, [1, 1, 1, 1, 1, 1])
    F_per = 1.0 / nL
    for j in range(nL):
        m.add_nodal_load(j * nL + N + 1, [0, 0, -F_per, 0, 0, 0])
    return m, nL


def ss_plate(element_cls, *, N: int, L: float = 1.0,
              t: float = 0.01, E: float = 2.0e11, nu: float = 0.3):
    """N x N SS plate with central point load = 1 N."""
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    h = L / N
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * h, j * h, 0.0)
    etag = 1
    for je in range(N):
        for ie in range(N):
            n1 = je * nL + ie + 1
            n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            m.add_element(element_cls(etag, (n1, n2, n3, n4), mat,
                                        thickness=t))
            etag += 1
    for j in range(nL):
        for i in range(nL):
            tag = j * nL + i + 1
            if i == 0 or i == nL - 1 or j == 0 or j == nL - 1:
                m.fix(tag, [1, 1, 1, 0, 0, 0])
    center = (nL // 2) * nL + nL // 2 + 1
    m.add_nodal_load(center, [0, 0, -1.0, 0, 0, 0])
    return m, center


# ============================================================ main

def main() -> None:
    L = 1.0; t = 0.01; E = 2.0e11; nu = 0.3
    D = E * t ** 3 / (12 * (1 - nu ** 2))

    print("DKMQ4 thin-plate convergence comparison vs MITC4")
    print("=" * 60)
    print(f"  E = {E:.1e} Pa, nu = {nu}, t = {t*1000:.1f} mm")
    print()

    # ---- Cantilever plate ----
    print("Example 1: Cantilever plate, unit tip load (cylindrical bending)")
    w_kirch_cantilever = 1.0 / (3 * D)
    print(f"  Kirchhoff cylindrical-bending answer: "
          f"{w_kirch_cantilever * 1e6:.2f} um")
    print()
    print(f"  {'N':>3} | {'DKMQ4 tip':>13} | {'MITC4 tip':>13} | "
          f"{'DKMQ4-MITC4':>11}")
    print("  " + "-" * 60)
    for N in (1, 2, 4, 8, 16):
        m_dk, nL = cantilever(ShellDKMQ4, N=N, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m_dk).run()
        w_dk = -m_dk.node(nL).disp[2]
        m_mt, _ = cantilever(ShellMITC4, N=N, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m_mt).run()
        w_mt = -m_mt.node(nL).disp[2]
        diff_pct = (w_dk - w_mt) / w_mt * 100.0
        print(f"  {N:>3} | {w_dk*1e6:>10.3f} um | "
              f"{w_mt*1e6:>10.3f} um | {diff_pct:>+9.2f}%")
    print()

    # ---- Simply-supported plate ----
    print("Example 2: SS plate, central point load (singular solution)")
    w_kirch_ss = 0.01160 * 1.0 * L ** 2 / D
    print(f"  Kirchhoff thin-plate answer: {w_kirch_ss * 1e6:.2f} um")
    print()
    print(f"  {'N':>3} | {'DKMQ4 mid':>13} | {'MITC4 mid':>13} | "
          f"{'DKMQ4-MITC4':>11}")
    print("  " + "-" * 60)
    for N in (2, 4, 8, 16):
        m_dk, c_dk = ss_plate(ShellDKMQ4, N=N, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m_dk).run()
        w_dk = -m_dk.node(c_dk).disp[2]
        m_mt, c_mt = ss_plate(ShellMITC4, N=N, L=L, t=t, E=E, nu=nu)
        LinearStaticAnalysis(m_mt).run()
        w_mt = -m_mt.node(c_mt).disp[2]
        diff_pct = (w_dk - w_mt) / w_mt * 100.0
        print(f"  {N:>3} | {w_dk*1e6:>10.3f} um | "
              f"{w_mt*1e6:>10.3f} um | {diff_pct:>+9.2f}%")
    print()

    # ---- Thin-limit test ----
    print("Example 3: Thin-limit sweep (SS plate, N=4)")
    print("  DKMQ4 has NO transverse shear by construction -- the")
    print("  scaled deflection w*D/P should be EXACTLY constant as t->0.")
    print()
    print(f"  {'L/t':>8} | {'DKMQ4':>10} | {'MITC4':>10} | "
          f"{'Kirchhoff':>10}")
    print("  " + "-" * 50)
    for t_thin in (0.01, 0.001, 0.0001):
        D_t = E * t_thin ** 3 / (12 * (1 - nu ** 2))
        w_kirch_thin = 0.01160 * 1.0 * L ** 2 / D_t
        m_dk, c_dk = ss_plate(ShellDKMQ4, N=4, L=L, t=t_thin, E=E, nu=nu)
        LinearStaticAnalysis(m_dk).run()
        w_dk = -m_dk.node(c_dk).disp[2]
        m_mt, c_mt = ss_plate(ShellMITC4, N=4, L=L, t=t_thin, E=E, nu=nu)
        LinearStaticAnalysis(m_mt).run()
        w_mt = -m_mt.node(c_mt).disp[2]
        # Scaled: w D / (P L^2)
        scaled_dk = w_dk * D_t / (1.0 * L ** 2)
        scaled_mt = w_mt * D_t / (1.0 * L ** 2)
        scaled_kirch = w_kirch_thin * D_t / (1.0 * L ** 2)
        print(f"  {L/t_thin:>8.0f} | {scaled_dk:.6f} | "
              f"{scaled_mt:.6f} | {scaled_kirch:.6f}")

    print()
    print("Reading the result:")
    print("* DKMQ4 (thin-plate DKQ) and MITC4 (Mindlin-Reissner with MITC")
    print("  shear tying) both converge to the same answer with mesh")
    print("  refinement -- they discretize the same physics differently.")
    print("* DKMQ4 collapses to the Kirchhoff thin-plate answer exactly")
    print("  at any thickness because it has zero shear by construction;")
    print("  MITC4 includes Mindlin shear which dominates for thick")
    print("  plates and washes out near zero for thin plates.")
    print("* For seismic / structural plate problems (L/t > 20), either")
    print("  element works. Choose DKMQ4 when you want guaranteed thin-")
    print("  plate behavior; choose MITC4 when you need thick-plate /")
    print("  Mindlin-shear contributions.")


if __name__ == "__main__":
    main()
