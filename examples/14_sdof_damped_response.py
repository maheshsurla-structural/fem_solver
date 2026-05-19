"""Damped SDOF response under three damping levels.

A mass-spring-dashpot is given an initial displacement and released
from rest. Three damping ratios are swept (zeta = 0, 5%, 100%) and the
resulting decay envelopes are printed and characterized:

* **zeta = 0**         — undamped free vibration; energy conserved
                          (Newmark with average acceleration).
* **zeta = 0.05**      — underdamped; logarithmic decrement matches
                          the analytical ``2 pi zeta / sqrt(1 - zeta^2)``.
* **zeta = 1.0**       — critical damping; no oscillation, monotonic
                          decay back to zero with no overshoot.

This is the textbook validation of structural-dynamics codes.

Run::

    python examples/14_sdof_damped_response.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ElasticIsotropic,
    Model,
    Newmark,
    RayleighDamping,
    TransientAnalysis,
    Truss2D,
)


def build_sdof(K: float, M: float):
    """SDOF model: horizontal truss between fixed and free nodes, with
    rho tuned to give mass M at the free node."""
    L = 1.0
    A = 1.0
    E = K * L / A                     # EA/L = K
    rho = 3.0 * M / (A * L)            # consistent mass at free DOF = rho A L / 3 = M
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(Truss2D(1, (1, 2), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    return m


def run_case(K: float, M: float, u0: float, zeta: float, *,
             n_periods: int = 5, steps_per_period: int = 200):
    """Run a free-vibration response and return the tip-displacement
    history."""
    omega = math.sqrt(K / M)
    T = 2.0 * math.pi / omega
    m = build_sdof(K=K, M=M)
    m.number_dofs()
    m.node(2).disp[0] = u0
    damping = (None if zeta == 0.0
               else RayleighDamping(alpha_M=2.0 * zeta * omega, alpha_K=0.0))
    dt = T / steps_per_period
    n_steps = int(n_periods * steps_per_period)
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt, damping=damping, track=(2, 0),
    ).run()
    return omega, T, np.array(res["times"]), np.array(res["tracked_disp"])


def main() -> None:
    K = 100.0
    M = 1.0
    u0 = 0.01
    omega = math.sqrt(K / M)
    T = 2.0 * math.pi / omega

    print("\nSDOF damped response — initial displacement, released from rest")
    print(f"  K = {K} N/m,  M = {M} kg")
    print(f"  omega_n = {omega:.4f} rad/s,  T = {T:.4f} s")
    print(f"  u_0 = {u0} m\n")

    for zeta in (0.0, 0.05, 1.0):
        omega, T, t, u = run_case(K=K, M=M, u0=u0, zeta=zeta)
        print(f"  zeta = {zeta:.2f}:")
        # peak values vs time
        peaks = []
        for i in range(1, len(u) - 1):
            if abs(u[i]) > abs(u[i - 1]) and abs(u[i]) > abs(u[i + 1]):
                peaks.append((t[i], u[i]))
        if peaks:
            for k, (tk, uk) in enumerate(peaks[:5], 1):
                print(f"    peak {k}: t = {tk:.4f} s, u = {uk:+.5e}")
            # Logarithmic decrement (first two same-sign peaks)
            same_sign = [(tk, uk) for tk, uk in peaks if uk * peaks[0][1] > 0]
            if len(same_sign) >= 2:
                u1, u2 = same_sign[0][1], same_sign[1][1]
                log_dec = math.log(abs(u1 / u2))
                analytical = (2.0 * math.pi * zeta /
                              math.sqrt(1.0 - min(zeta, 0.99) ** 2)
                              if zeta < 1.0 else float("inf"))
                print(f"    log decrement: FE = {log_dec:.4f}, "
                      f"analytical = {analytical:.4f}")
        else:
            # No peaks — likely overdamped / critical.
            u_max = max(u.min(), u.max(), key=abs)
            t_max = t[np.argmax(np.abs(u))]
            print(f"    monotonic decay; peak |u| = {abs(u_max):.5e} at "
                  f"t = {t_max:.4f} s")
            print(f"    final u(t=T_5) = {u[-1]:.3e}")
        # Energy at the end vs start
        ke_end = 0.0     # we don't track velocity-vs-time, only disp
        pe_end = 0.5 * K * u[-1] ** 2
        pe_start = 0.5 * K * u0 ** 2
        print(f"    PE_end / PE_start = {pe_end / pe_start:.4e}\n")


if __name__ == "__main__":
    main()
