"""Modal analysis of a steel cantilever — first 5 natural frequencies and
mode shapes, compared to the Euler-Bernoulli analytical solution.

For a cantilever the natural frequencies are

    omega_n = (beta_n L)^2 * sqrt(E I / (rho A L^4))
    f_n = omega_n / (2 pi)

with (beta_n L) the n-th root of cosh(x) cos(x) + 1 = 0:

    n = 1  ->  1.8751
    n = 2  ->  4.6941
    n = 3  ->  7.8548
    n = 4  -> 10.9955
    n = 5  -> 14.1372

Run::

    python examples/06_cantilever_modal.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
)


BETA_L = (1.875104068711961, 4.694091132974175, 7.854757438237613,
          10.995540734875467, 14.137168391046470)


def main() -> None:
    L = 2.0           # m
    E = 200e9         # Pa
    rho = 7850.0      # kg/m^3
    A = 1.0e-3        # m^2
    Iz = 8.333e-7     # m^4
    n_elem = 30

    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0)
    m.fix(1, [1, 1, 1])
    # Suppress axial DOFs at all other nodes so the first 5 reported modes
    # are all bending — otherwise the first axial mode (around 630 Hz for
    # this geometry) interleaves with the higher bending modes.
    for i in range(2, n_elem + 2):
        m.fix(i, [1, 0, 0])
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))

    a = EigenAnalysis(m, num_modes=5)
    info = a.run()

    base = math.sqrt(E * Iz / (rho * A * L ** 4))
    f_exact = np.array([(b ** 2) * base / (2.0 * math.pi) for b in BETA_L])

    print(f"system size: {info['neq']} equations, {info['num_modes']} modes")
    print(f"{'mode':>4} {'freq (Hz)':>14} {'exact (Hz)':>14} {'rel err':>12}")
    for k in range(a.num_modes):
        f = a.frequencies[k]
        rel = (f - f_exact[k]) / f_exact[k]
        print(f"{k+1:>4} {f:14.4f} {f_exact[k]:14.4f} {rel:>12.2e}")

    print(f"\nperiods (s): {[f'{p:.4f}' for p in a.periods]}")

    # mass-orthonormalization check
    M = a.M.toarray()
    G = a.mode_shapes.T @ M @ a.mode_shapes
    print(f"max |phi^T M phi - I|: {np.max(np.abs(G - np.eye(a.num_modes))):.2e}")

    # tip displacement of each mode (DOF index 1 = uy of last node)
    tip = m.node(n_elem + 1)
    print("tip transverse mode amplitudes (mass-normalized):")
    for k in range(a.num_modes):
        print(f"  mode {k+1}: uy = {tip.mode_disp[1, k]: .4e}, theta_z = {tip.mode_disp[2, k]: .4e}")


if __name__ == "__main__":
    main()
