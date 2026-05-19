"""Linear transient (time-history) analysis driver.

Solves the linear equation of motion

    M u_ddot + C u_dot + K u = F(t)

with a time-stepping integrator (currently :class:`Newmark`) and a
time-varying load. Initial conditions are read from
``Node.disp / Node.velocity`` so the user sets them by assigning to
those arrays before calling :meth:`run`. The driver:

* numbers DOFs, assembles ``M, K, C, F_ref`` once
* hands them to the integrator's ``bind``
* marches forward ``num_steps`` time steps, scattering ``u, u_dot,
  u_ddot`` back onto the nodes at every step
* records a tracked DOF's history into Python lists for inspection

Time-varying loads are supplied via the ``load_function`` keyword. It
accepts either:

* a callable ``f(t) -> float`` — the scalar multiplies the reference
  nodal-load pattern at every step (the common case: a single load
  pattern with a time amplitude)
* a callable ``f(t) -> ndarray`` of length ``model.neq`` — used
  directly as the full force vector (for situations where multiple
  load patterns have different time amplitudes)
* ``None`` (default) — the load is constant at the reference pattern,
  scaled by ``1.0`` (a step-applied load).

For free vibration set the model's nodal loads to zero and just supply
initial conditions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from femsolver.analysis.damping import RayleighDamping
from femsolver.analysis.transient_integrator import (
    Newmark,
    TransientIntegrator,
)
from femsolver.numerics.dof_numbering import rcm_renumber


LoadFunction = Callable[[float], "float | np.ndarray"]


def _resolve_transient_integrator(arg) -> TransientIntegrator:
    if isinstance(arg, TransientIntegrator):
        return arg
    if isinstance(arg, str):
        if arg.lower() in ("newmark", "average_acceleration", "avg_acc"):
            return Newmark()
        raise ValueError(
            f"unknown transient integrator {arg!r}; expected 'newmark'"
        )
    raise TypeError(
        "integrator must be a TransientIntegrator instance or a string"
    )


class TransientAnalysis:
    """Direct-integration linear transient analysis.

    Parameters
    ----------
    model : Model
    num_steps : int
        Number of time steps to march forward.
    dt : float
        Time-step size (s).
    integrator : str or TransientIntegrator, default ``"newmark"``
        Time integrator. The default is Newmark with
        ``beta = 1/4, gamma = 1/2`` (average acceleration).
    damping : RayleighDamping or None
        Damping model. ``None`` (default) means undamped.
    load_function : callable or None
        Time amplitude function. See module docstring for forms.
        ``None`` is a step-applied reference load (constant scale = 1.0).
    track : (int, int) tuple, optional
        ``(node_tag, dof_index)`` to record at every step. Recorded
        ``disp, velocity, acceleration`` histories are returned.
    numberer : {"default", "rcm"}, default ``"default"``
    """

    def __init__(
        self,
        model,
        num_steps: int,
        dt: float,
        *,
        integrator: str | TransientIntegrator = "newmark",
        damping: RayleighDamping | None = None,
        load_function: LoadFunction | None = None,
        track: tuple[int, int] | None = None,
        numberer: str = "default",
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.model = model
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.integrator = _resolve_transient_integrator(integrator)
        self.damping = damping
        self.load_function = load_function
        self.track = track
        if numberer not in ("default", "rcm"):
            raise ValueError(f"unknown numberer {numberer!r}")
        self.numberer = numberer

        # Results
        self.times: list[float] = []
        self.tracked_disp: list[float] = []
        self.tracked_velocity: list[float] = []
        self.tracked_acceleration: list[float] = []

    # ----------------------------------------------------------------- run
    def run(self) -> dict:
        m = self.model
        # NB we do *not* reset Node.disp / Node.velocity — they may
        # hold initial conditions.
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError(
                "no free DOFs — model is fully constrained or empty"
            )

        # Force at t = 0, used by integrator to compute u_ddot_0.
        F0 = self._force_at(0.0)
        self.integrator.set_initial_force(F0)
        self.integrator.bind(m, dt=self.dt, damping=self.damping)

        # Record initial state (t = 0) before stepping.
        self.times.append(0.0)
        if self.track is not None:
            tag, dof = self.track
            node = m.node(tag)
            self.tracked_disp.append(float(node.disp[dof]))
            self.tracked_velocity.append(float(node.velocity[dof]))
            self.tracked_acceleration.append(0.0)   # will be overwritten

        # u_ddot at t = 0 has been computed by the integrator — scatter
        # to the node so the initial acceleration is observable.
        self.integrator._scatter("acceleration", self.integrator.u_ddot)
        if self.track is not None:
            tag, dof = self.track
            self.tracked_acceleration[0] = float(m.node(tag).acceleration[dof])

        # Time-march
        for step in range(1, self.num_steps + 1):
            t_new = step * self.dt
            F_new = self._force_at(t_new)
            self.integrator.step(F_new)
            # Scatter the integrator's state to the nodes
            self.integrator._scatter("disp", self.integrator.u)
            self.integrator._scatter("velocity", self.integrator.u_dot)
            self.integrator._scatter("acceleration", self.integrator.u_ddot)

            self.times.append(t_new)
            if self.track is not None:
                tag, dof = self.track
                node = m.node(tag)
                self.tracked_disp.append(float(node.disp[dof]))
                self.tracked_velocity.append(float(node.velocity[dof]))
                self.tracked_acceleration.append(float(node.acceleration[dof]))

        return {
            "neq": int(m.neq),
            "num_steps": self.num_steps,
            "dt": self.dt,
            "total_time": float(self.dt * self.num_steps),
            "times": list(self.times),
            "tracked_disp": list(self.tracked_disp),
            "tracked_velocity": list(self.tracked_velocity),
            "tracked_acceleration": list(self.tracked_acceleration),
        }

    # -------------------------------------------------------------- force
    def _force_at(self, t: float) -> np.ndarray:
        """Build the free-DOF force vector at time ``t`` using the
        time function and the integrator's reference load."""
        F_ref = self.integrator.F_ref if self.integrator.F_ref is not None \
            else self._build_reference_force()
        if self.load_function is None:
            scale = 1.0
            return F_ref * scale
        val = self.load_function(t)
        if np.isscalar(val):
            return F_ref * float(val)
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != F_ref.size:
            raise ValueError(
                f"load_function returned vector of size {arr.size}, "
                f"expected {F_ref.size} (= neq)"
            )
        return arr

    def _build_reference_force(self) -> np.ndarray:
        """Assemble F_ref lazily (before integrator.bind)."""
        from femsolver.analysis.assembler import assemble_force
        return assemble_force(self.model)
