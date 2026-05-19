"""Solution algorithms — drive the iterative solve at one analysis step.

A :class:`SolutionAlgorithm` orchestrates the inner Newton iteration:

  1. Form the tangent ``K_T`` (full Newton — every iteration; modified
     Newton — only at iter 0).
  2. Form the residual ``R``.
  3. Test convergence; if converged, stop.
  4. Solve ``K_T du = R`` and update the state ``u <- u + du``.

The algorithm does not know whether the analysis is static or transient
or what integrator is in use — it only talks through the integrator
interface (``tangent``, ``residual``, ``update``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, spsolve

from femsolver.analysis.constraint_handler import TransformationHandler


@dataclass
class IterationReport:
    """Per-step outcome reported by an algorithm."""

    converged: bool
    iterations: int
    final_residual: float = 0.0
    final_du: float = 0.0
    history: list[float] = field(default_factory=list)


class NotConvergedError(RuntimeError):
    """Raised when a step exceeds ``max_iter`` without converging."""


class SolutionAlgorithm(ABC):
    """Base class. Subclasses implement :meth:`solve_step`."""

    def __init__(self, *, constraints: str = "transformation"):
        if constraints != "transformation":
            raise ValueError(
                "only the transformation constraint handler is supported "
                "for nonlinear analysis at present"
            )
        self.handler = TransformationHandler()

    @abstractmethod
    def solve_step(self, integrator, conv_test, *, scatter_du) -> IterationReport:
        """Drive one Newton step until convergence or ``max_iter``."""


class Newton(SolutionAlgorithm):
    """Full Newton-Raphson — re-form the tangent every iteration."""

    name = "Newton"

    def solve_step(self, integrator, conv_test, *, scatter_du) -> IterationReport:
        return _newton_loop(integrator, conv_test, self.handler, scatter_du,
                             reform_tangent_each_iter=True)


class ModifiedNewton(SolutionAlgorithm):
    """Modified Newton — keep the iter-0 tangent for the rest of the step.

    Cheaper per iteration but typically needs more iterations near
    high-curvature points on the equilibrium path.
    """

    name = "ModifiedNewton"

    def solve_step(self, integrator, conv_test, *, scatter_du) -> IterationReport:
        return _newton_loop(integrator, conv_test, self.handler, scatter_du,
                             reform_tangent_each_iter=False)


class LineSearchNewton(SolutionAlgorithm):
    """Newton with backtracking line search on the residual norm.

    The proposed Newton update ``du`` is multiplied by a step factor
    ``alpha`` that is halved (starting from 1.0) until the residual norm
    is acceptably reduced. This makes Newton converge on stiff problems
    where the full step would overshoot — particularly elastic-plastic
    sections at the elastic-plastic interface, where the residual is
    piecewise-linear with a slope discontinuity.

    Parameters
    ----------
    max_backtracks : int, default 20
        Maximum number of halvings before accepting the smallest step.
    descent_factor : float, default 1.0 - 1.0e-4
        Required residual-norm reduction factor: the step is accepted if
        ``|R_new| < descent_factor * |R_old|``. Closer to 1.0 = more
        lenient. The default mirrors the Wolfe-Armijo conventional choice.
    """

    name = "LineSearchNewton"

    def __init__(self, *, max_backtracks: int = 20,
                 descent_factor: float = 1.0 - 1.0e-4,
                 constraints: str = "transformation"):
        super().__init__(constraints=constraints)
        if max_backtracks < 1:
            raise ValueError("max_backtracks must be >= 1")
        if not (0.0 < descent_factor < 1.0):
            raise ValueError("descent_factor must be in (0, 1)")
        self.max_backtracks = int(max_backtracks)
        self.descent_factor = float(descent_factor)

    def solve_step(self, integrator, conv_test, *, scatter_du) -> IterationReport:
        return _newton_loop(
            integrator, conv_test, self.handler, scatter_du,
            reform_tangent_each_iter=True,
            line_search_max_backtracks=self.max_backtracks,
            line_search_descent_factor=self.descent_factor,
        )


# ---------------------------------------------------------------------------
# inner loop, shared between Newton and ModifiedNewton


def _newton_loop(
    integrator, conv_test, handler, scatter_du, *,
    reform_tangent_each_iter: bool,
    line_search_max_backtracks: int = 0,
    line_search_descent_factor: float = 1.0 - 1.0e-4,
) -> IterationReport:
    """Iterate up to ``conv_test.max_iter`` times.

    On each iteration:
      - assemble K_T (or reuse iter-0 tangent for modified Newton)
      - assemble R
      - apply MP constraints via the transformation handler
      - test convergence on the *full-DOF* R and the previous du
      - solve and update state
    """
    history: list[float] = []
    K_T = None
    build = None
    du = np.zeros(integrator.F_ref.size)

    for it in range(conv_test.max_iter + 1):
        R = integrator.residual()

        # MP-constraint reduction
        model = integrator.model
        if model.mp_constraints:
            if build is None or reform_tangent_each_iter:
                build = handler.build(model)
            T = build.T
            R_eff = T.T @ R
        else:
            T = None
            R_eff = R

        # Convergence test. Skip at iter 0 so path-following integrators
        # always perform their predictor solve: for DisplacementControl
        # and ArcLength the iter-0 residual is the *previous step's*
        # converged residual (zero), and an early return would skip the
        # constraint-driven displacement step. For LoadControl the
        # iter-0 residual is the load increment ``dlambda * F_ref`` so
        # this skip costs nothing in practice; in trivial corner cases
        # (zero load increment) it adds one harmless extra iteration.
        if it > 0 and conv_test.check(R, du, it):
            history.append(conv_test.last_value)
            return IterationReport(
                converged=True,
                iterations=it,
                final_residual=float(np.linalg.norm(R)),
                final_du=float(np.linalg.norm(du)),
                history=history,
            )
        # Still record the residual for telemetry, even at iter 0.
        if it > 0 or conv_test.max_iter > 0:
            history.append(conv_test.last_value if it > 0 else float(np.linalg.norm(R)))

        if it == conv_test.max_iter:
            raise NotConvergedError(
                f"Newton iteration failed to converge after "
                f"{conv_test.max_iter} iterations "
                f"(last test value {conv_test.last_value:.3e}, tol "
                f"{conv_test.tol:.3e})"
            )

        if reform_tangent_each_iter or K_T is None:
            K_T = integrator.tangent()
            if T is not None:
                K_T_solve = (T.T @ K_T @ T).tocsc()
            else:
                K_T_solve = K_T.tocsc() if sp.issparse(K_T) else K_T
            # Factor once per tangent reform so the integrator can do
            # cheap repeat solves (needed by path-following constraints).
            try:
                K_solve = factorized(K_T_solve)
            except Exception as exc:
                raise NotConvergedError(
                    f"linear solver factorization failed at iteration {it}: {exc}. "
                    "Likely cause: singular tangent (mechanism, snap-through "
                    "without arc-length, or zero-stiffness DOF)."
                ) from exc

        # Prepare F_ref_eff for path-following integrators that need it.
        F_ref = integrator.F_ref
        F_ref_eff = (T.T @ F_ref) if T is not None else F_ref

        try:
            du_eff = integrator.solve_iteration(K_solve, R_eff, F_ref_eff, T)
        except Exception as exc:
            raise NotConvergedError(
                f"linear solve in Newton iteration {it} failed: {exc}. "
                "Likely cause: singular tangent (mechanism, snap-through "
                "without arc-length, or zero-stiffness DOF)."
            ) from exc

        du = (T @ du_eff) if T is not None else np.asarray(du_eff).ravel()

        # Line search at the algorithm level scales ``du`` by alpha < 1
        # when the full step doesn't reduce the residual. This is fine
        # for LoadControl (lambda is fixed within the step) but violates
        # the constraint of path-following integrators
        # (DisplacementControl, ArcLength) which already updated
        # ``self.lambd`` inside ``solve_iteration``. Path-following
        # integrators advertise this via ``supports_line_search = False``.
        if (
            line_search_max_backtracks > 0
            and getattr(integrator, "supports_line_search", True)
        ):
            du = _backtracking_line_search(
                integrator, scatter_du, du, R,
                max_backtracks=line_search_max_backtracks,
                descent_factor=line_search_descent_factor,
            )
        else:
            scatter_du(du)

    # unreachable — the for-loop returns or raises
    raise NotConvergedError("internal: newton loop exited unexpectedly")


def _backtracking_line_search(
    integrator, scatter_du, du_full, R_old, *,
    max_backtracks: int, descent_factor: float,
) -> np.ndarray:
    """Apply a backtracking line search: scatter ``alpha * du_full``,
    halve ``alpha`` until the residual norm decreases below
    ``descent_factor * |R_old|``, then return the *accepted* du.

    If the full Newton step (``alpha = 1``) already reduces the
    residual sufficiently, no backtracking is performed. If
    ``max_backtracks`` is exhausted, the smallest tried step is left in
    place and returned — Newton's outer loop will still get a chance
    to converge (or fail) on the next iteration.
    """
    norm_R_old = float(np.linalg.norm(R_old))
    # First, try the full step.
    scatter_du(du_full)
    R_new = integrator.residual()
    if np.linalg.norm(R_new) < descent_factor * norm_R_old:
        return du_full
    # Backtrack: undo the full step in stages.
    alpha = 0.5
    last_du = du_full.copy()
    for _ in range(max_backtracks):
        # Undo previous attempt; scatter alpha * du_full (net new step).
        scatter_du(-last_du)
        trial = alpha * du_full
        scatter_du(trial)
        R_new = integrator.residual()
        if np.linalg.norm(R_new) < descent_factor * norm_R_old:
            return trial
        last_du = trial
        alpha *= 0.5
    # Exhausted backtracks — leave the last attempt in place.
    return last_du
