"""Fibre-consistent section moment-curvature driver (plan P4 / §7.2).

:func:`fiber_section_moment_curvature` drives a *stateful* fiber section
(:class:`~femsolver.sections.response.fiber.FiberSection2D`) through a sequence
of prescribed curvatures while holding a constant axial force, solving the
axial strain at each step so the section is in axial equilibrium. Because it
integrates the section's **own** fibers and uniaxial laws (the confined-core
Mander concrete, the Park/kinematic steel, ...), the M-phi it returns is
exactly the response the fiber hinge integrates over its length -- "what you
analyse is what you run" (plan §15). This is the fibre counterpart of the
design-side :func:`femsolver.sections.analysis.exact_mphi`.

At each curvature ``kappa`` the axial strain ``eps_a`` is found by Newton
iteration on the section axial force::

    s, ks = section.get_response([eps_a, kappa])     # s = [N, Mz]
    r     = s[0] - N_target                          # axial residual
    eps_a -= r / ks[0, 0]                            # tangent EA = ks[0,0]

The section commits after each converged curvature, so path-dependent
materials accumulate history correctly for a monotonic (or, with a
non-monotone ``kappas``, cyclic) sweep.
"""
from __future__ import annotations

import numpy as np


def fiber_section_moment_curvature(
    section,
    kappas,
    *,
    N_target: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 50,
    section_recorder=None,
    fiber_recorder=None,
    stop_on_fail: bool = True,
):
    """Monotonic (or prescribed) moment-curvature of a fiber section at
    constant axial force.

    Parameters
    ----------
    section :
        A 2-D fiber section (``n_resultants == 2``) with the stateful
        ``get_response`` / ``commit_state`` / ``revert_state`` lifecycle.
    kappas : sequence of float
        Curvatures to impose, in order (increasing for a monotonic push).
    N_target : float, default 0.0
        Axial force held constant during the sweep (tension positive, matching
        ``N = sum(sigma*A)``). Use a negative value for axial compression.
    tol : float, default 1e-12
        Convergence tolerance on the axial-strain increment (dimensionless).
    max_iter : int, default 50
        Maximum Newton iterations per curvature step.
    section_recorder, fiber_recorder : optional
        Recorders (:mod:`femsolver.results.recorders`) whose ``record(e)`` is
        called after each converged, committed step.
    stop_on_fail : bool, default True
        Stop the sweep at the first non-converging step (typical once the
        section loses axial stiffness at crushing); if ``False`` the step is
        skipped and the sweep continues.

    Returns
    -------
    dict
        ``kappa``, ``M`` (= ``Mz``), ``eps_a`` and ``N`` lists over the
        converged steps, plus ``n_converged`` and ``converged`` (whether the
        whole sweep ran). The section is left committed at the last converged
        state.
    """
    if getattr(section, "n_resultants", 2) != 2:
        raise ValueError(
            "fiber_section_moment_curvature handles 2-D fiber sections "
            "(n_resultants == 2); 3-D P-M2-M3 is a later phase (P7).")

    kappas = [float(k) for k in kappas]
    kappa_out: list[float] = []
    M_out: list[float] = []
    eps_out: list[float] = []
    N_out: list[float] = []

    eps_a = 0.0
    all_ok = True
    for kappa in kappas:
        converged = False
        for _ in range(max_iter):
            e = np.array([eps_a, float(kappa)])
            s, ks = section.get_response(e)
            r = float(s[0]) - N_target
            EA = float(ks[0, 0])
            if EA == 0.0:
                break
            d_eps = -r / EA
            eps_a += d_eps
            if abs(d_eps) <= tol:
                converged = True
                break
        if not converged:
            section.revert_state()
            all_ok = False
            if stop_on_fail:
                break
            continue
        section.commit_state()
        e = np.array([eps_a, float(kappa)])
        s, _ = section.get_response(e)
        kappa_out.append(float(kappa))
        M_out.append(float(s[1]))
        eps_out.append(float(eps_a))
        N_out.append(float(s[0]))
        if section_recorder is not None:
            section_recorder.record(e, step=len(kappa_out) - 1)
        if fiber_recorder is not None:
            fiber_recorder.record(e, step=len(kappa_out) - 1)

    return {
        "kappa": kappa_out, "M": M_out, "eps_a": eps_out, "N": N_out,
        "n_converged": len(kappa_out),
        "converged": all_ok and len(kappa_out) == len(kappas),
    }
