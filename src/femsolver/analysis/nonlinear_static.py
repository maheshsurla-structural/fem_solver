"""Nonlinear static analysis driver.

Glues together the integrator, solution algorithm, and convergence test
to march along an equilibrium path under load increments.

Each step the analysis:
  1. Asks the integrator to advance (e.g., ``LoadControl`` increments
     :math:`\\lambda`).
  2. Hands the algorithm an iteration callback that scatters ``du`` onto
     :class:`Node` displacements and lets the algorithm assemble residuals.
  3. On convergence, calls :meth:`Element.commit_state` on every element and
     records ``(lambda, u_dof_of_interest)`` for later plotting.

The class deliberately mirrors the interface of
:class:`LinearStaticAnalysis` where possible — both number DOFs, both
support MP constraints via the transformation handler, both populate
``Node.disp`` and ``Node.reaction`` at the end.
"""
from __future__ import annotations

import numpy as np

from femsolver.analysis.algorithm import (
    LineSearchNewton,
    ModifiedNewton,
    Newton,
    NotConvergedError,
    SolutionAlgorithm,
)
from femsolver.analysis.assembler import assemble_reactions
from femsolver.analysis.convergence import (
    ConvergenceTest,
    NormDispIncr,
    NormUnbalance,
)
from femsolver.analysis.static_integrator import (
    ArcLength,
    DisplacementControl,
    LoadControl,
    StaticIntegrator,
    _assemble_internal_force,
)
from femsolver.numerics.dof_numbering import rcm_renumber


def _resolve_algorithm(arg) -> SolutionAlgorithm:
    if isinstance(arg, SolutionAlgorithm):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key == "newton":
            return Newton()
        if key in ("modified_newton", "modnewton", "modified-newton"):
            return ModifiedNewton()
        if key in ("line_search", "linesearch", "line_search_newton",
                   "linesearchnewton"):
            return LineSearchNewton()
        raise ValueError(
            f"unknown algorithm {arg!r}; expected 'newton', "
            "'modified_newton', or 'line_search'"
        )
    raise TypeError(f"algorithm must be str or SolutionAlgorithm, got {type(arg).__name__}")


def _resolve_test(arg, tol: float, max_iter: int) -> ConvergenceTest:
    if isinstance(arg, ConvergenceTest):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key in ("disp_incr", "disp", "norm_disp_incr"):
            return NormDispIncr(tol=tol, max_iter=max_iter)
        if key in ("unbalance", "norm_unbalance", "force"):
            return NormUnbalance(tol=tol, max_iter=max_iter)
        raise ValueError(
            f"unknown convergence test {arg!r}; expected 'disp_incr' or 'unbalance'"
        )
    raise TypeError(f"convergence must be str or ConvergenceTest, got {type(arg).__name__}")


def _resolve_integrator(arg, dlambda: float) -> StaticIntegrator:
    if isinstance(arg, StaticIntegrator):
        return arg
    if isinstance(arg, str):
        key = arg.lower()
        if key in ("load_control", "loadcontrol", "load"):
            return LoadControl(dlambda=dlambda)
        # Path-following integrators cannot be constructed from a string
        # alone — they need additional configuration (control DOF for
        # displacement control, arc-length parameter for ArcLength). The
        # user must pass an instance directly.
        if key in ("displacement_control", "dispcontrol", "displacement"):
            raise ValueError(
                "displacement control requires a control DOF; pass "
                "DisplacementControl(node_tag, dof_index, du_step) "
                "as the integrator= argument"
            )
        if key in ("arc_length", "arclength"):
            raise ValueError(
                "arc length requires a delta_s; pass "
                "ArcLength(delta_s=...) as the integrator= argument"
            )
        raise ValueError(
            f"unknown integrator {arg!r}; expected 'load_control' (or pass "
            f"a DisplacementControl / ArcLength instance directly)"
        )
    raise TypeError(
        f"integrator must be str or StaticIntegrator, got {type(arg).__name__}"
    )


class NonlinearStaticAnalysis:
    """Incremental-iterative static analysis.

    Parameters
    ----------
    model : Model
    num_steps : int
        Number of load increments. With :class:`LoadControl` and
        ``dlambda``, the final load factor is ``num_steps * dlambda``.
    dlambda : float, default 0.1
        Load-factor increment per step (only used if ``integrator='load_control'``).
    integrator : str or StaticIntegrator, default ``"load_control"``
    algorithm : str or SolutionAlgorithm, default ``"newton"``
    convergence : str or ConvergenceTest, default ``"unbalance"``
    tol : float, default 1e-8
        Convergence tolerance (only used if ``convergence`` is a string).
    max_iter : int, default 25
    numberer : {"default", "rcm"}, default "default"
    track : (int, int) tuple, optional
        ``(node_tag, dof_index)`` to record at each converged step. The
        recorded ``(lambda, u)`` pairs are returned in the analysis result.
    """

    def __init__(
        self,
        model,
        num_steps: int,
        *,
        dlambda: float = 0.1,
        integrator: str = "load_control",
        algorithm: str = "newton",
        convergence: str = "unbalance",
        tol: float = 1e-8,
        max_iter: int = 25,
        numberer: str = "default",
        track: tuple[int, int] | None = None,
        keep_state: bool = False,
        const_force: "np.ndarray | None" = None,
        step_callback=None,
        substep: bool = False,
        max_substep_halvings: int = 6,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.model = model
        # keep_state: continue from the model's current committed node/element
        # state (do NOT zero displacements) -- used for staged continuation.
        # const_force: a constant load vector held in the residual on top of
        # the scaled pattern (e.g. a held axial preload). Both are wired by
        # StagedAnalysis; defaults reproduce the standalone behaviour exactly.
        self.keep_state = bool(keep_state)
        self.const_force = const_force
        self.num_steps = int(num_steps)
        self.integrator = _resolve_integrator(integrator, dlambda)
        self.algorithm = _resolve_algorithm(algorithm)
        self.convergence = _resolve_test(convergence, tol, max_iter)
        if numberer not in ("default", "rcm"):
            raise ValueError(f"unknown numberer {numberer!r}")
        self.numberer = numberer
        self.track = track  # (node_tag, dof_index) or None
        # Optional per-step hook: called after each converged+committed step with
        # a dict {step, num_steps, lambda, iterations, tracked}. Returning False
        # stops the run early (e.g. a UI cancel), keeping the results so far.
        self.step_callback = step_callback
        # Adaptive step-cutting (plan §16 C4): when a step fails to converge,
        # halve the increment and retry (subdividing to cover the nominal step),
        # up to ``max_substep_halvings`` halvings, then grow back. Opt-in and
        # only for integrators that advertise ``supports_substep`` (LoadControl,
        # scalar DisplacementControl); off by default so existing behaviour and
        # the "raise on non-convergence" contract are unchanged.
        self.substep = bool(substep)
        self.max_substep_halvings = int(max_substep_halvings)

        # results
        self.lambdas: list[float] = []
        self.tracked: list[float] = []
        self.iter_counts: list[int] = []
        self.u: np.ndarray | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        m = self.model
        if not self.keep_state:
            m.reset_results()
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError("no free DOFs — model is fully constrained or empty")

        self.integrator.bind(m)
        if self.const_force is not None:
            self.integrator.set_constant_force(self.const_force)

        # state vector lives on Node.disp; we manipulate it via scatter_du
        def scatter_du(du: np.ndarray) -> None:
            for node in m.nodes.values():
                for i in range(node.ndf):
                    eq = int(node.eqn[i])
                    if eq >= 0:
                        node.disp[i] += du[eq]

        def _commit_and_record(report) -> bool:
            """Commit a converged (sub)step, record it, and fire the callback.
            Returns False if the callback asked to cancel."""
            for e in m.elements.values():
                e.commit_state()
            if hasattr(self.integrator, "record_step_iterations"):
                self.integrator.record_step_iterations(report.iterations)
            self.integrator.commit_step()
            self.iter_counts.append(report.iterations)
            self.lambdas.append(self.integrator.lambd)
            if self.track is not None:
                tag, dof = self.track
                self.tracked.append(float(m.node(tag).disp[dof]))
            if self.step_callback is not None:
                info = {
                    "step": len(self.lambdas), "num_steps": self.num_steps,
                    "lambda": self.integrator.lambd,
                    "iterations": report.iterations,
                    "tracked": (self.tracked[-1] if self.track is not None
                                else None),
                }
                if self.step_callback(info) is False:
                    return False
            return True

        use_substep = (self.substep
                       and getattr(self.integrator, "supports_substep", False))
        min_scale = 0.5 ** self.max_substep_halvings

        for step in range(1, self.num_steps + 1):
            if not use_substep:
                self.integrator.new_step()
                try:
                    report = self.algorithm.solve_step(
                        self.integrator, self.convergence,
                        scatter_du=scatter_du)
                except NotConvergedError:
                    self.integrator.revert_step()
                    for e in m.elements.values():
                        e.revert_state()
                    raise
                if not _commit_and_record(report):
                    break                              # cooperative cancel
                continue

            # --- adaptive: cover one nominal step, subdividing on failure ---
            remaining = 1.0
            scale = 1.0
            cancelled = False
            while remaining > 1e-9:
                frac = min(scale, remaining)
                # Snapshot the committed displacements: a failed Newton solve
                # leaves the drifted trial ``disp`` on the nodes (revert_step
                # only rolls back the integrator/elements), so we must restore
                # them before retrying a smaller sub-step.
                disp_snap = {nid: nd.disp.copy()
                             for nid, nd in m.nodes.items()}
                self.integrator.set_step_scale(frac)
                self.integrator.new_step()
                try:
                    report = self.algorithm.solve_step(
                        self.integrator, self.convergence,
                        scatter_du=scatter_du)
                except NotConvergedError:
                    self.integrator.revert_step()
                    for e in m.elements.values():
                        e.revert_state()
                    for nid, nd in m.nodes.items():
                        nd.disp[:] = disp_snap[nid]
                    scale *= 0.5
                    if scale < min_scale:
                        self.integrator.set_step_scale(1.0)
                        raise
                    continue
                if not _commit_and_record(report):
                    cancelled = True
                    break
                remaining -= frac
                scale = min(1.0, scale * 2.0)          # grow back toward full
            self.integrator.set_step_scale(1.0)
            if cancelled:
                break

        # element response and reactions at the final state
        for e in m.elements.values():
            e.recover()
        # reactions: assemble using current internal forces and applied loads
        self._compute_reactions()

        # gather the final u vector (free-DOF view)
        u = np.zeros(m.neq)
        for n in m.nodes.values():
            for i in range(n.ndf):
                eq = int(n.eqn[i])
                if eq >= 0:
                    u[eq] = n.disp[i]
        self.u = u

        return {
            "neq": int(m.neq),
            "num_steps": self.num_steps,
            "final_lambda": float(self.integrator.lambd),
            "lambdas": list(self.lambdas),
            "tracked": list(self.tracked),
            "iter_counts": list(self.iter_counts),
            "total_iterations": int(sum(self.iter_counts)),
        }

    # ----------------------------------------------------------- reactions
    def _compute_reactions(self) -> None:
        """Reactions at fixed DOFs are the internal force at those DOFs minus
        the externally applied load. We compute the *full* internal-force
        vector (at all DOFs, including fixed ones) by walking elements
        and accumulating ``f_int_global`` directly to nodes.
        """
        m = self.model
        for n in m.nodes.values():
            n.reaction[:] = 0.0
        for e in m.elements.values():
            fe = e.f_int_global()
            dofs_per_node = e.dofs_per_node
            for k, nt in enumerate(e.node_tags):
                node = m.node(nt)
                node.reaction[:dofs_per_node] += fe[
                    k * dofs_per_node : (k + 1) * dofs_per_node
                ]
        for n in m.nodes.values():
            for j in range(n.ndf):
                if n.fixity[j]:
                    n.reaction[j] -= n._load[j] * self.integrator.lambd
                else:
                    n.reaction[j] = 0.0
