"""Response-spectrum vs direct time-history on a multi-story frame.

A 5-story shear-stick frame is subjected to a ground motion in the
x direction. We compute the peak roof displacement two ways:

* **Response-spectrum analysis** — modal superposition under the
  pseudo-acceleration spectrum of the ground motion, combined with
  both SRSS and CQC.
* **Direct time-history** — Newmark integration of the same frame
  driven by ``F(t) = -M iota * u_g_ddot(t)`` via the
  ``ground_motion_force`` helper, with Rayleigh damping calibrated
  to the same modal damping ratio.

For a regular stick frame whose modal periods are well separated, the
response-spectrum estimate should match the direct-integration peak
to within the modal-superposition accuracy bound — typically a few
percent. This is the canonical validation of every code's
response-spectrum machinery.

Run::

    python examples/22_response_spectrum_vs_time_history.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    RayleighDamping,
    ResponseSpectrum,
    ResponseSpectrumAnalysis,
    TransientAnalysis,
    ground_motion_force,
)


# =========================================================== frame builder

def build_stick(n_story: int = 5, *,
                E: float = 2.0e10, A: float = 1.0e-2, Iz: float = 1.0e-4,
                L: float = 3.0, rho: float = 7850.0):
    """5-story stick: beam-column elements stacked vertically, fixed
    at the base. Lumped consistent mass."""
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_story + 1):
        m.add_node(i + 1, 0.0, i * L)
    for i in range(n_story):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    return m


# =========================================================== ground motion

def ricker_pulse(t: float, *, t0: float, fp: float, amp: float = 1.0) -> float:
    """Centered Ricker (Mexican-hat) pulse, peak frequency ``fp``."""
    tau = math.pi * fp * (t - t0)
    return amp * (1.0 - 2.0 * tau ** 2) * math.exp(-tau ** 2)


def sample_spectrum_of(accel_func, *, periods: np.ndarray,
                       zeta: float, t_end: float, dt: float) -> np.ndarray:
    """Numerically construct the pseudo-acceleration spectrum of
    ``accel_func`` by integrating an SDOF for each period in
    ``periods`` and returning the peak |u(t)| * omega^2.

    This is a Duhamel-integral-style construction; we use Newmark
    with average acceleration on a unit-mass SDOF.
    """
    n_steps = int(round(t_end / dt))
    times = np.arange(n_steps + 1) * dt
    ag = np.array([accel_func(t) for t in times])
    Sa = np.empty_like(periods)
    # Average-acceleration Newmark constants
    beta, gamma = 0.25, 0.5
    for k, T in enumerate(periods):
        omega = 2.0 * math.pi / T
        K = omega ** 2          # unit mass
        C = 2.0 * zeta * omega
        # State
        u = 0.0; v = 0.0; a = -ag[0]   # SDOF: m a + c v + k u = -m a_g
        u_max = 0.0
        a1 = 1.0 / (beta * dt ** 2)
        a2 = 1.0 / (beta * dt)
        a3 = 0.5 / beta - 1.0
        a4 = gamma / (beta * dt)
        a5 = 1.0 - gamma / beta
        a6 = dt * (1.0 - 0.5 * gamma / beta)
        K_eff = K + a4 * C + a1
        for n in range(1, n_steps + 1):
            F_eff = -ag[n] + a1 * u + a2 * v + a3 * a + C * (a4 * u - a5 * v - a6 * a)
            u_new = F_eff / K_eff
            a_new = a1 * (u_new - u) - a2 * v - a3 * a
            v_new = a4 * (u_new - u) + a5 * v + a6 * a
            u, v, a = u_new, v_new, a_new
            if abs(u) > u_max:
                u_max = abs(u)
        Sa[k] = u_max * omega ** 2     # pseudo-acceleration
    return Sa


# =========================================================== main

def main() -> None:
    n_story = 5
    L = 3.0
    print("\nResponse-spectrum vs time-history on a 5-story stick frame")
    print(f"  5 BeamColumn2D bays, L = {L} m each (roof at {n_story * L} m)")
    print(f"  E = 2.0e10 Pa, A = 1.0e-2 m^2, Iz = 1.0e-4 m^4")
    print(f"  rho = 7850 kg/m^3 (consistent mass)\n")

    # Modal characterisation
    from femsolver.analysis.eigen import EigenAnalysis
    m_eig = build_stick(n_story=n_story)
    eig = EigenAnalysis(m_eig, num_modes=n_story).run()
    T_modes = np.array(eig["periods_s"])
    omega_modes = 2.0 * math.pi / T_modes
    print(f"  Modal periods (s): " + ", ".join(f"{T:.3f}" for T in T_modes))
    print(f"  Period ratio T1/T2 = {T_modes[0] / T_modes[1]:.2f} "
          f"(well-separated > 3)\n")

    # ------- Ground motion: Ricker pulse centered at t0 with f_p = 1 Hz
    # (T_p ~ 1 s, broadband enough to excite modes 1-3 of the stick)
    fp = 1.0
    amp = 1.0   # m/s^2 peak ground acceleration ~ amp
    t0 = 3.0
    t_end = 12.0
    dt = 0.01
    n_steps = int(t_end / dt)

    def ag(t):
        return ricker_pulse(t, t0=t0, fp=fp, amp=amp)

    # ------- Build the pseudo-acceleration response spectrum of ag(t)
    zeta = 0.05
    periods_grid = np.logspace(math.log10(0.05), math.log10(5.0), 80)
    print(f"  Building pseudo-acceleration spectrum of the ground motion")
    print(f"  (80 periods between 0.05 s and 5.0 s, zeta = {zeta})...")
    Sa_grid = sample_spectrum_of(
        ag, periods=periods_grid, zeta=zeta, t_end=t_end, dt=dt,
    )
    spectrum = ResponseSpectrum(periods_grid, Sa_grid, damping_ratio=zeta)
    print(f"  Sa at modal periods (m/s^2): " + ", ".join(
        f"{spectrum.Sa(T):.3f}" for T in T_modes))
    print()

    # ------- Response-spectrum analysis: SRSS + CQC
    print(f"  Response-spectrum analysis (modal superposition):")
    for combo in ("srss", "cqc"):
        m_rs = build_stick(n_story=n_story)
        res = ResponseSpectrumAnalysis(
            m_rs, spectrum, num_modes=n_story,
            direction="x", combination=combo,
        ).run()
        u_roof = m_rs.node(n_story + 1).disp[0]
        print(f"    {combo.upper():4s}: roof |u| = {abs(u_roof):.5e} m, "
              f"sum modal mass = {res['total_participating_mass']:.3f} kg")
    print()

    # ------- Direct time-history with same Rayleigh damping at modes 1 & 2
    print(f"  Direct time-history (Newmark + Rayleigh damping at modes 1,2):")
    m_th = build_stick(n_story=n_story)
    damping = RayleighDamping.from_modes(
        omega_1=omega_modes[0], omega_2=omega_modes[1],
        zeta_1=zeta, zeta_2=zeta,
    )
    g_force = ground_motion_force(m_th, direction="x", accel_function=ag)
    res_th = TransientAnalysis(
        m_th, num_steps=n_steps, dt=dt,
        damping=damping, load_function=g_force,
        track=(n_story + 1, 0),
    ).run()
    u_th = np.array(res_th["tracked_disp"])
    u_th_peak = float(np.max(np.abs(u_th)))
    t_th_peak = float(res_th["times"][int(np.argmax(np.abs(u_th)))])
    print(f"    Time-history peak |u_roof|  = {u_th_peak:.5e} m at t = {t_th_peak:.3f} s")
    print()

    # ------- Compare
    m_rs = build_stick(n_story=n_story)
    ResponseSpectrumAnalysis(
        m_rs, spectrum, num_modes=n_story,
        direction="x", combination="cqc",
    ).run()
    u_cqc = abs(m_rs.node(n_story + 1).disp[0])
    err = (u_cqc - u_th_peak) / u_th_peak
    print(f"  CQC vs time-history error: {err * 100:+.2f} %")
    print()
    print(f"  Reading the result:")
    print(f"  * The pseudo-acceleration spectrum is the design-level summary")
    print(f"    of the time-history — peak response of every SDOF period.")
    print(f"  * Modal superposition (CQC) reconstructs the multi-DOF peak")
    print(f"    from per-mode SDOF peaks; an upper bound on the true peak.")
    print(f"  * For a regular stick with well-separated modes, the error is")
    print(f"    typically a few percent. The CQC peak is non-negative-definite")
    print(f"    and modally separable, which is why design codes prescribe it")
    print(f"    instead of full time-history for routine analyses.")


if __name__ == "__main__":
    main()
