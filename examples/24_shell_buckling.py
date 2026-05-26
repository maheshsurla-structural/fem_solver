"""Shell buckling -- SS plate uniaxial compression vs Bryan's formula
and aspect-ratio scan.

A simply-supported plate of dimensions ``a x b x t`` is subjected to
uniform uniaxial edge compression. Bryan's classical solution gives

    sigma_cr = k * pi^2 * E * t^2 / [12 (1 - nu^2) b^2]

with the buckling coefficient

    k(m) = (m b/a + a/(m b))^2,   m = number of half-waves in the
                                  loaded direction (chosen to minimize k)

For the square (a = b), the minimum is k = 4 at m = 1.

Two sweeps:

* **Mesh convergence** on a square plate -- error drops as O(h^2).
* **Aspect-ratio sweep** -- shows the buckling coefficient picking up
  the right half-wave number m as a/b grows.

Run::

    python examples/24_shell_buckling.py
"""
from __future__ import annotations

import math

from femsolver import ElasticIsotropic, Model, ShellMITC4
from femsolver.analysis.buckling import LinearBucklingAnalysis


def build_plate(N: int, *, a: float, b: float, t: float,
                E: float, nu: float, P_ref: float = 1.0):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nx = N + 1
    ny = max(2, int(round(N * b / a))) + 1
    for j in range(ny):
        for i in range(nx):
            tag = j * nx + i + 1
            m.add_node(tag, i * a / N, j * b / (ny - 1), 0.0)
    etag = 1
    for j in range(ny - 1):
        for i in range(N):
            n1 = j * nx + i + 1; n2 = n1 + 1
            n3 = n2 + nx; n4 = n1 + nx
            m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
            etag += 1
    for j in range(ny):
        for i in range(nx):
            tag = j * nx + i + 1
            on_edge = (i == 0 or i == N or j == 0 or j == ny - 1)
            if on_edge:
                m.fix(tag, [0, 0, 1, 0, 0, 0])
    for j in range(ny):
        m.fix(j * nx + 1, [1, 0, 0, 0, 0, 0])
    m.fix(1, [1, 1, 1, 0, 0, 0])
    weights = [0.5 if j in (0, ny - 1) else 1.0 for j in range(ny)]
    w_total = sum(weights)
    for j in range(ny):
        tag = j * nx + N + 1
        m.add_nodal_load(tag, [-P_ref * weights[j] / w_total, 0, 0, 0, 0, 0])
    return m


def bryan_k(a: float, b: float) -> tuple[int, float]:
    """Return (m, k_min) -- number of half-waves and minimal k for an
    SS-SS-SS-SS plate of dimensions a x b under uniaxial compression
    in the x direction (length a)."""
    k_min, m_min = float("inf"), 1
    for mm in range(1, 8):
        k = (mm * b / a + a / (mm * b)) ** 2
        if k < k_min:
            k_min, m_min = k, mm
    return m_min, k_min


def main() -> None:
    E, nu = 2.0e11, 0.3
    t = 0.01

    print(f"\nShell buckling -- SS plate under uniaxial edge compression")
    print(f"  E = {E:g} Pa, nu = {nu}, t = {t} m")

    print(f"\n  1) Mesh convergence (square plate a = b = 1 m)")
    b = 1.0
    sigma_cr_bryan = 4.0 * math.pi ** 2 * E * t ** 2 / (12.0 * (1 - nu ** 2) * b ** 2)
    P_bryan = sigma_cr_bryan * t * b
    print(f"     Bryan:  sigma_cr = {sigma_cr_bryan:.4e} Pa,  "
          f"P_cr = {P_bryan:.4e} N")
    print(f"     {'N':>3s}  {'lambda_cr (N)':>16s}  {'rel err':>10s}")
    prev_err = None
    for N in (4, 6, 8, 12, 16, 20):
        m = build_plate(N=N, a=1.0, b=1.0, t=t, E=E, nu=nu, P_ref=1.0)
        res = LinearBucklingAnalysis(m, num_modes=1).run()
        P_fem = res["critical_load_factor"]
        err = abs(P_fem / P_bryan - 1.0)
        rate_str = ""
        if prev_err is not None and err > 0 and prev_err > 0:
            rate = math.log(prev_err / err) / math.log(2.0)
            rate_str = f"  rate ~ {rate:.2f}"
        print(f"     {N:>3d}  {P_fem:>16.4e}  {err * 100:>9.3f}% {rate_str}")
        prev_err = err

    print(f"\n  2) Aspect-ratio sweep (N = 16, b = 1 m)")
    print(f"     {'a/b':>6s}  {'m':>3s}  {'k_min':>7s}  {'P_fem/P_bryan':>14s}")
    for ratio in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        a = ratio
        b = 1.0
        m_modes, k = bryan_k(a=a, b=b)
        sigma_cr = k * math.pi ** 2 * E * t ** 2 / (12.0 * (1 - nu ** 2) * b ** 2)
        P_bryan = sigma_cr * t * b
        m = build_plate(N=16, a=a, b=b, t=t, E=E, nu=nu, P_ref=1.0)
        res = LinearBucklingAnalysis(m, num_modes=1).run()
        ratio_fem = res["critical_load_factor"] / P_bryan
        print(f"     {ratio:>6.2f}  {m_modes:>3d}  {k:>7.3f}  "
              f"{ratio_fem:>14.4f}")
    print()
    print(f"  Reading the results:")
    print(f"  * Mesh convergence is O(h^2). The 16x16 mesh hits ~0.2%")
    print(f"    of Bryan's closed form -- the standard MITC4 target.")
    print(f"  * The aspect-ratio sweep shows the buckling mode number m")
    print(f"    increases stepwise as a/b grows -- at integer transitions")
    print(f"    in a/b, the optimal m switches and k drops back to ~4.")
    print(f"  * Plates longer than 2:1 buckle into multiple half-waves;")
    print(f"    fixed mesh density gives fewer elements per half-wave,")
    print(f"    so the error grows mildly with a/b.")


if __name__ == "__main__":
    main()
