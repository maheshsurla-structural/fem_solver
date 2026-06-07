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
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.model = model
        self.num_steps = int(num_steps)
        self.integrator = _resolve_integrator(integrator, dlambda)
        self.algorithm = _resolve_algorithm(algorithm)
        self.convergence = _resolve_test(convergence, tol, max_iter)
        if numberer not in ("default", "rcm"):
            raise ValueError(f"unknown numberer {numberer!r}")
        self.numberer = numberer
        self.track = track  # (node_tag, dof_index) or None

        # results
        self.lambdas: list[float] = []
        self.tracked: list[float] = []
        self.iter_counts: list[int] = []
        self.u: np.ndarray | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        m = self.model
        m.reset_results()
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError("no free DOFs — model is fully constrained or empty")

        self.integrator.bind(m)

        # state vector lives on Node.disp; we manipulate it via scatter_du
        def scatter_du(du: np.ndarray) -> None:
            for node in m.nodes.values():
                for i in range(node.ndf):
                    eq = int(node.eqn[i])
                    if eq >= 0:
                        node.disp[i] += du[eq]

        for step in range(1, self.num_steps + 1):
            self.integrator.new_step()
            try:
                report = self.algorithm.solve_step(
                    self.integrator, self.convergence, scatter_du=scatter_du
                )
            except NotConvergedError:
                # roll back: undo the step's load increment and last du
                self.integrator.revert_step()
                for e in m.elements.values():
                    e.revert_state()
                raise

            # commit element state (e.g., plasticity will roll history forward)
            for e in m.elements.values():
                e.commit_state()
            # Report iteration count back to the integrator (used by
            # adaptive arc-length to scale delta_s) before commit_step
            if hasattr(self.integrator, "record_step_iterations"):
                self.integrator.record_step_iterations(report.iterations)
            # commit integrator state (path-following integrators use this
            # to update direction tracking, step-start snapshots, etc.)
            self.integrator.commit_step()

            self.iter_counts.append(report.iterations)
            self.lambdas.append(self.integrator.lambd)
            if self.track is not None:
                tag, dof = self.track
                self.tracked.append(float(m.node(tag).disp[dof]))

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
