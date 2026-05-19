"""Cantilever-beam free vibration excited in its first mode.

A 3-m cantilever steel beam is given an initial displacement equal to
a small multiple of its first mode shape (computed by
:class:`EigenAnalysis`) and released from rest. The tip oscillates
purely in mode 1; we measure the period from zero crossings and verify
it matches the eigenvalue prediction.

This is the canonical pair "modal analysis + transient verification"
that every dynamics textbook covers. The fact that the FE-measured
period matches the eigenvalue prediction to within ~1 % is the
strongest single end-to-end check of the dynamic infrastructure:
mass matrix, eigenvalue solver, Newmark integrator, and initial-
condition handling all have to agree.

Run::

    python examples/15_cantilever_free_vibration.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
    TransientAnalysis,
)


def main() -> None:
    # 3 m steel cantilever, 100 x 100 mm cross-section
    E = 2.0e11
    A = 1.0e-2
    Iz = 8.333e-6
    L = 3.0
    rho = 7850.0
    n_elem = 4
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)

    def build():
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(n_elem + 1):
            m.add_node(i + 1, i * L / n_elem, 0.0)
        for i in range(n_elem):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        return m

    # ---- Modal analysis: first three modes ---------------------------------
    m_eig = build()
    eig = EigenAnalysis(m_eig, num_modes=3).run()
    periods = eig["periods_s"]
    freqs = eig["frequencies_hz"]

    print(f"\nCantilever beam — modal analysis + transient verification")
    print(f"  L = {L} m,  E = {E:g},  rho = {rho},  cross section 100 x 100 mm")
    print(f"  Discretisation: {n_elem} BeamColumn2D elements\n")
    print(f"  Modal analysis:")
    for k, (T, f) in enumerate(zip(periods, freqs), 1):
        print(f"    Mode {k}:  T = {T:.4e} s  ({f:.3f} Hz)")
    print()

    # ---- Transient: kick the structure into pure mode 1 --------------------
    m_tr = build()
    eig_tr = EigenAnalysis(m_tr, num_modes=3).run()
    T_1 = eig_tr["periods_s"][0]
    tip_tag = n_elem + 1

    # Set initial displacement = (small) * mode_1_shape so the
    # transient response is pure mode 1.
    target_tip_v = 0.001         # 1 mm initial tip displacement
    tip_mode1_v = m_tr.node(tip_tag).mode_disp[1, 0]
    scale = target_tip_v / abs(tip_mode1_v)
    for n in m_tr.nodes.values():
        n.disp[:] = scale * n.mode_disp[:, 0]

    dt = T_1 / 200.0
    n_periods = 4
    n_steps = int(n_periods * 200)

    res = TransientAnalysis(
        m_tr, num_steps=n_steps, dt=dt, track=(tip_tag, 1),
    ).run()
    t = np.array(res["times"])
    u = np.array(res["tracked_disp"])
    v = np.array(res["tracked_velocity"])

    # Measure the FE period from successive downward zero crossings.
    zeros = []
    for i in range(1, len(u)):
        if u[i - 1] > 0 and u[i] <= 0:
            frac = u[i - 1] / (u[i - 1] - u[i])
            zeros.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    if len(zeros) >= 2:
        T_fe = zeros[1] - zeros[0]
    else:
        T_fe = float("nan")

    print(f"  Transient: initial disp = {target_tip_v} m at the tip,")
    print(f"             scaled to mode-1 shape so only mode 1 is excited.")
    print(f"  Time step dt = T_1 / 200 = {dt:.4e} s")
    print(f"  Number of steps = {n_steps}  ({n_periods} periods)\n")
    print(f"  Period comparison:")
    print(f"    T_1 (eigenvalue) = {T_1:.6e} s")
    print(f"    T_1 (FE)         = {T_fe:.6e} s")
    print(f"    Error            = {abs(T_fe - T_1) / T_1 * 100:.3f} %\n")

    # ---- Peak amplitude tracking — energy check
    peaks = []
    for i in range(1, len(u) - 1):
        if abs(u[i]) > abs(u[i - 1]) and abs(u[i]) > abs(u[i + 1]):
            peaks.append((t[i], u[i]))
    print(f"  Tip-displacement peaks (first 6):")
    for k, (tk, uk) in enumerate(peaks[:6], 1):
        print(f"    peak {k}: t = {tk:.4f} s, u = {uk:+.4e}")
    if len(peaks) >= 5:
        ratio = abs(peaks[4][1]) / abs(peaks[0][1])
        print(f"  Amplitude ratio peak_5 / peak_1 = {ratio:.6f}")
        print(f"  Newmark with avg acceleration is energy-conserving in")
        print(f"  the absence of damping; the small deviation from 1 comes")
        print(f"  from the residual period error (a 1 % period error gives")
        print(f"  a ~1 % drift in peak position over 5 cycles).")


if __name__ == "__main__":
    main()
