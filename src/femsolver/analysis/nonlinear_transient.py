"""Nonlinear transient (direct-integration) analysis.

Combines Newmark time-stepping (via :class:`NewmarkNonlinear`) with a
static Newton-Raphson solver inside each step. The dynamic equilibrium
equation

    M u_ddot + C u_dot + f_int(u) = F(t)

is enforced at every time ``t_{n+1}`` via Newton iteration: given the
current iterate ``u``, we compute the residual

    R(u) = F(t_{n+1}) - M u_ddot(u) - C u_dot(u) - f_int(u)

and the effective tangent

    K_eff_T(u) = K_tangent(u) + a4 C + a1 M

(with ``a1, a4`` the Newmark coefficients), then solve
``K_eff_T Δu = R`` and update ``u``. ``u_dot, u_ddot`` are expressed
in terms of ``u`` and the step-start state ``u_n, u_dot_n, u_ddot_n``
through the Newmark relations.

This is the canonical "implicit nonlinear dynamic" analysis used in
performance-based seismic engineering — for example, OpenSees'
``Transient`` analysis with ``Newmark`` integrator and a
``BeamColumn`` element backed by a fiber section. Combined with the
corotational beam (Phase 6) and fiber sections (Phase 5), the solver
can now perform nonlinear seismic time-history analyses on frames.

Architecture
------------
Reuses *every* component of the static-nonlinear infrastructure:

* :class:`~femsolver.analysis.algorithm.Newton`,
  :class:`~femsolver.analysis.algorithm.ModifiedNewton`,
  :class:`~femsolver.analysis.algorithm.LineSearchNewton` — the same
  algorithms that drive nonlinear-static Newton.
* :class:`~femsolver.analysis.convergence.NormUnbalance`,
  :class:`~femsolver.analysis.convergence.NormDispIncr`,
  :class:`~femsolver.analysis.convergence.EnergyIncr` — the same
  convergence tests.

The only new piece is :class:`NewmarkNonlinear`, which makes the
dynamic equilibrium equation appear as a static residual + tangent
pair to the algorithm.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from femsolver.analysis.algorithm import (
    LineSearchNewton,
    ModifiedNewton,
    Newton,
    NotConvergedError,
    SolutionAlgorithm,
)
from femsolver.analysis.convergence import (
    ConvergenceTest,
    EnergyIncr,
    NormDispIncr,
    NormUnbalance,
)
from femsolver.analysis.damping import RayleighDamping
from femsolver.analysis.transient_integrator import NewmarkNonlinear
from femsolver.numerics.dof_numbering import rcm_renumber


LoadFunction = Callable[[float], "float | np.ndarray"]


def _resolve_algorithm(arg) -> SolutionAlgorithm:
    if isinstance(arg, SolutionAlgorithm):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key == "newton":
            return Newton()
        if key in ("modified_newton", "modnewton", "modified-newton"):
            return ModifiedNewton()
        if key in ("line_search", "linesearch", "line_search_newton"):
            return LineSearchNewton()
        raise ValueError(
            f"unknown algorithm {arg!r}; expected 'newton', 'modified_newton', "
            f"or 'line_search'"
        )
    raise TypeError(
        f"algorithm must be str or SolutionAlgorithm, got {type(arg).__name__}"
    )


def _resolve_test(arg, tol: float, max_iter: int) -> ConvergenceTest:
    if isinstance(arg, ConvergenceTest):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key in ("disp_incr", "disp", "norm_disp_incr"):
            return NormDispIncr(tol=tol, max_iter=max_iter)
        if key in ("unbalance", "norm_unbalance", "force"):
            return NormUnbalance(tol=tol, max_iter=max_iter)
        if key in ("energy", "energy_incr"):
            return EnergyIncr(tol=tol, max_iter=max_iter)
        raise ValueError(
            f"unknown convergence test {arg!r}; expected one of "
            f"'disp_incr', 'unbalance', 'energy'"
        )
    raise TypeError(
        f"convergence must be str or ConvergenceTest, got {type(arg).__name__}"
    )


class NonlinearTransientAnalysis:
    """Direct-integration nonlinear transient analysis.

    Parameters
    ----------
    model : Model
    num_steps : int
        Number of time steps.
    dt : float
        Time-step size.
    integrator : NewmarkNonlinear or ``"newmark"`` (default)
        Time integrator. Constructs ``NewmarkNonlinear()`` if string.
    algorithm : str or SolutionAlgorithm, default ``"newton"``
        Inner Newton-iteration algorithm.
    convergence : str or ConvergenceTest, default ``"unbalance"``
        Convergence test for the inner Newton iteration.
    tol : float, default 1e-6
    max_iter : int, default 25
    damping : RayleighDamping or None
    load_function : callable or None
        Time function. See :class:`~femsolver.analysis.transient.TransientAnalysis`
        for accepted forms.
    track : (int, int) tuple, optional
        ``(node_tag, dof_index)`` to record at every step.
    numberer : ``"default"`` or ``"rcm"``
    """

    def __init__(
        self,
        model,
        num_steps: int,
        dt: float,
        *,
        integrator: str | NewmarkNonlinear = "newmark",
        algorithm: str | SolutionAlgorithm = "newton",
        convergence: str | ConvergenceTest = "unbalance",
        tol: float = 1.0e-6,
        max_iter: int = 25,
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

        if isinstance(integrator, str):
            if integrator.lower() in ("newmark", "average_acceleration", "avg_acc"):
                self.integrator = NewmarkNonlinear()
            else:
                raise ValueError(
                    f"unknown integrator {integrator!r}; expected 'newmark'"
                )
        elif isinstance(integrator, NewmarkNonlinear):
            self.integrator = integrator
        else:
            raise TypeError(
                "integrator must be a NewmarkNonlinear instance or string"
            )
        self.algorithm = _resolve_algorithm(algorithm)
        self.convergence = _resolve_test(convergence, tol, max_iter)
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
        self.iter_counts: list[int] = []

    # ------------------------------------------------------------ run
    def run(self) -> dict:
        m = self.model
        # We do NOT reset Node.disp / Node.velocity — they hold initial
        # conditions.
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError(
                "no free DOFs — model is fully constrained or empty"
            )

        # Force at t = 0 so the integrator can compute u_ddot_0
        # consistently.
        F0 = self._force_at(0.0, neq=m.neq)
        self.integrator.set_initial_force(F0)
        self.integrator.bind(m, dt=self.dt, damping=self.damping)

        # scatter_du updates Node.disp incrementally — the Newton
        # algorithm uses this callback to apply iteration steps.
        def scatter_du(du: np.ndarray) -> None:
            for node in m.nodes.values():
                for i in range(node.ndf):
                    eq = int(node.eqn[i])
                    if eq >= 0:
                        node.disp[i] += du[eq]

        # Record initial state at t = 0
        self.times.append(0.0)
        if self.track is not None:
            tag, dof = self.track
            node = m.node(tag)
            self.tracked_disp.append(float(node.disp[dof]))
            self.tracked_velocity.append(float(node.velocity[dof]))
            self.tracked_acceleration.append(float(node.acceleration[dof]))
        self.iter_counts.append(0)

        # Time-march
        for step in range(1, self.num_steps + 1):
            t_new = step * self.dt
            F_new = self._force_at(t_new, neq=m.neq)
            self.integrator.new_step(F_new)
            try:
                report = self.algorithm.solve_step(
                    self.integrator, self.convergence,
                    scatter_du=scatter_du,
                )
            except NotConvergedError:
                # Roll back integrator and elements
                self.integrator.revert_step()
                for e in m.elements.values():
                    e.revert_state()
                raise

            # Commit element state, then integrator (updates u_dot,
            # u_ddot from converged u_{n+1}; advances time).
            for e in m.elements.values():
                e.commit_state()
            self.integrator.commit_step()

            # Record
            self.times.append(t_new)
            self.iter_counts.append(report.iterations)
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
            "iter_counts": list(self.iter_counts),
            "total_iterations": int(sum(self.iter_counts)),
        }

    # ------------------------------------------------------------- force
    def _force_at(self, t: float, *, neq: int) -> np.ndarray:
        """Force vector at time ``t``. Uses ``load_function`` to scale
        or override the model's reference nodal-load pattern."""
        from femsolver.analysis.assembler import assemble_force
        F_ref = assemble_force(self.model)
        if self.load_function is None:
            return F_ref
        val = self.load_function(t)
        if np.isscalar(val):
            return F_ref * float(val)
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != neq:
            raise ValueError(
                f"load_function returned vector of size {arr.size}, "
                f"expected {neq} (= neq)"
            )
        return arr
