"""Static integrators — assemble the residual and tangent for a given step.

An :class:`Integrator` decouples *how* the load is applied (load control
vs displacement control vs arc length) from the rest of the analysis. For
nonlinear static analysis the residual is

    R(u, lambda) = lambda * F_ref - f_int(u)

where ``F_ref`` is the reference load pattern (the model's nodal +
element-equivalent loads at unit factor) and ``f_int`` is the assembled
internal-force vector.

Each integrator owns its own *step-advance policy*:

* :class:`LoadControl` increments ``lambda`` by a fixed amount; the
  Newton corrector then adjusts ``u`` to bring ``R`` to zero.
* :class:`DisplacementControl` increments a chosen DOF by a fixed
  amount; the Newton corrector adjusts both ``u`` and ``lambda``.
* :class:`ArcLength` advances along an arc-length parameter; the
  Newton corrector enforces a cylindrical (or spherical) constraint on
  the displacement-load vector.

The corrector logic for each is encoded in :meth:`solve_iteration`. The
algorithm (``Newton``, ``ModifiedNewton``, ``LineSearchNewton``) just
hands the integrator a factored solver and the current residual; the
integrator returns the iteration's ``du_eff`` and may update
``self.lambd`` as a side effect.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy.sparse as sp

from femsolver.analysis.assembler import (
    assemble_force,
    assemble_stiffness,
)


class StaticIntegrator(ABC):
    """Base class for static integrators."""

    # Whether the integrator's per-iteration ``du`` may be scaled by a
    # line-search factor at the algorithm level. For path-following
    # integrators (DisplacementControl, ArcLength) the ``du`` returned
    # by :meth:`solve_iteration` carries a *constraint-driven*
    # ``dlambda`` that has already been applied; scaling ``du`` would
    # then violate the constraint. Such integrators set this to False;
    # the algorithm skips the line search and uses the full step. Newton
    # convergence in path-following analyses is typically robust enough
    # that line search is not needed at the algorithm level (the
    # element-internal line searches in HingedBeamColumn2D and the
    # adaptive predictor sign of ArcLength already handle the main
    # difficulty cases).
    supports_line_search: bool = True

    def __init__(self):
        self.model = None
        self._F_ref: np.ndarray | None = None
        self.lambd: float = 0.0

    def bind(self, model) -> None:
        """Capture the reference load pattern. Called once at the start of
        the analysis after DOF numbering."""
        self.model = model
        self._F_ref = assemble_force(model)
        self.lambd = 0.0

    @property
    def F_ref(self) -> np.ndarray:
        if self._F_ref is None:
            raise RuntimeError("integrator not bound to a model")
        return self._F_ref

    @abstractmethod
    def new_step(self) -> None:
        """Advance to the next step (e.g., increment the load factor)."""

    @abstractmethod
    def revert_step(self) -> None:
        """Roll back ``new_step`` after a non-converged step."""

    def commit_step(self) -> None:
        """Commit the just-converged step. Default: no-op. Path-following
        integrators override to snapshot the converged state for the
        next step's constraint reference."""
        return None

    def tangent(self) -> sp.csc_matrix:
        """Assemble and return the tangent stiffness at the current state."""
        return _assemble_tangent(self.model)

    def residual(self) -> np.ndarray:
        """Return ``R = lambda * F_ref - f_int(u)`` at the current state."""
        f_int = _assemble_internal_force(self.model)
        return self.lambd * self.F_ref - f_int

    # ----------------------------------------------------- solve hook
    def solve_iteration(
        self,
        K_solve: Callable[[np.ndarray], np.ndarray],
        R_eff: np.ndarray,
        F_ref_eff: np.ndarray,
        T: sp.csr_matrix | None,
    ) -> np.ndarray:
        """Solve one Newton iteration's linear system.

        Default implementation: standard Newton step ``du = K^{-1} R`` —
        correct for load control where ``lambda`` is fixed within the
        step.

        Path-following integrators (displacement-control, arc-length)
        override this to additionally solve ``K^{-1} F_ref`` and combine
        the two solutions through a constraint that produces a
        non-trivial ``Δλ`` per iteration. The integrator is responsible
        for updating ``self.lambd``.

        Parameters
        ----------
        K_solve : callable
            Factored linear solver: ``K_solve(b)`` returns the solution
            of ``K_eff @ x = b`` where ``K_eff`` is the tangent in
            effective (post-MP-constraint) space.
        R_eff : ndarray
            Residual in effective space.
        F_ref_eff : ndarray
            Reference-load vector in effective space.
        T : sparse matrix or None
            Transformation matrix from effective to full DOF space for
            MP constraints (``None`` when there are no MP constraints).

        Returns
        -------
        du_eff : ndarray
            Displacement increment in effective space (the algorithm
            scatters ``T @ du_eff`` onto ``Node.disp``).
        """
        return K_solve(R_eff)


class LoadControl(StaticIntegrator):
    """Increment the load factor by a fixed amount each step.

    Parameters
    ----------
    dlambda : float
        Load increment per step. The total load factor after ``n_steps``
        is ``n_steps * dlambda``.
    """

    def __init__(self, dlambda: float = 0.1):
        super().__init__()
        if dlambda == 0.0:
            raise ValueError("dlambda must be non-zero")
        self.dlambda = float(dlambda)

    def new_step(self) -> None:
        self.lambd += self.dlambda

    def revert_step(self) -> None:
        self.lambd -= self.dlambda


class DisplacementControl(StaticIntegrator):
    """Increment a chosen DOF by a fixed amount each step.

    Useful for tracing softening branches of a force-displacement curve
    that load control cannot follow (mechanism formation, plastic
    plateau, etc.). The load factor ``lambda`` becomes an unknown and
    is solved alongside the displacement update at each Newton
    iteration through the constraint

        u[control_dof] = u[control_dof]_step_start + du_step

    Parameters
    ----------
    node_tag : int
        Node containing the controlling DOF.
    dof_index : int
        Local DOF index within the node (0 for ``u_x``, 1 for ``u_y``,
        2 for ``theta_z`` in 2-D models, etc.).
    du_step : float
        Displacement increment per step on the controlling DOF. May be
        negative to reverse the load direction.

    Notes
    -----
    The control DOF must be a *free* DOF (otherwise the constraint is
    vacuous) and must not be involved in an MP-constraint (otherwise the
    equation-number lookup would need additional plumbing). Both
    restrictions are checked at :meth:`bind`.
    """

    supports_line_search = False

    def __init__(self, node_tag: int, dof_index: int, du_step: float):
        super().__init__()
        if du_step == 0.0:
            raise ValueError("du_step must be non-zero")
        self.node_tag = int(node_tag)
        self.dof_index = int(dof_index)
        self.du_step = float(du_step)
        # Equation number of the controlling DOF (set by bind()).
        self._eq: int = -1
        # u_d at the start of the current step (set by new_step()).
        self._u_d_step_start: float = 0.0
        # Snapshot for revert.
        self._u_d_before_step_start: float = 0.0
        self._lambd_before_step_start: float = 0.0

    def bind(self, model) -> None:
        super().bind(model)
        node = model.node(self.node_tag)
        if not (0 <= self.dof_index < node.ndf):
            raise ValueError(
                f"DisplacementControl: dof_index {self.dof_index} out of "
                f"range for node {self.node_tag} (ndf={node.ndf})"
            )
        eq = int(node.eqn[self.dof_index])
        if eq < 0:
            raise ValueError(
                f"DisplacementControl: node {self.node_tag} dof "
                f"{self.dof_index} is fixed (eqn={eq}); pick a free DOF"
            )
        if model.mp_constraints:
            raise NotImplementedError(
                "DisplacementControl with MP constraints is not yet "
                "supported; remove the MP constraint or pick a different "
                "control DOF"
            )
        self._eq = eq
        # Record the starting value of the control DOF so the first
        # new_step() has a reference.
        self._u_d_step_start = float(node.disp[self.dof_index])

    def _control_disp(self) -> float:
        return float(self.model.node(self.node_tag).disp[self.dof_index])

    def new_step(self) -> None:
        # Snapshot for potential revert.
        self._u_d_before_step_start = self._u_d_step_start
        self._lambd_before_step_start = self.lambd
        # Set the new step's reference state — the *current* control DOF
        # value (committed state from the previous step).
        self._u_d_step_start = self._control_disp()

    def revert_step(self) -> None:
        self._u_d_step_start = self._u_d_before_step_start
        self.lambd = self._lambd_before_step_start

    def solve_iteration(
        self,
        K_solve: Callable[[np.ndarray], np.ndarray],
        R_eff: np.ndarray,
        F_ref_eff: np.ndarray,
        T,
    ) -> np.ndarray:
        # Two solves: tangent-residual and parametric.
        du_t_eff = K_solve(R_eff)
        du_p_eff = K_solve(F_ref_eff)
        # Without MP constraints, du_eff == du (full). With them, we
        # could compute (T @ du_eff)[eq] but we already reject that
        # case in bind().
        if T is not None:
            # In the rare case T is not None even after bind (e.g.,
            # added after construction), gracefully use the full vector.
            du_t_full = np.asarray(T @ du_t_eff).ravel()
            du_p_full = np.asarray(T @ du_p_eff).ravel()
            du_t_d = float(du_t_full[self._eq])
            du_p_d = float(du_p_full[self._eq])
        else:
            du_t_d = float(du_t_eff[self._eq])
            du_p_d = float(du_p_eff[self._eq])
        if abs(du_p_d) < 1e-300:
            raise RuntimeError(
                "DisplacementControl: control DOF has zero parametric "
                "stiffness component — F_ref does not drive this DOF "
                "and the constraint is undefined"
            )
        # Constraint at this iteration: bring (u_d_current + du_d_this_iter)
        # to (u_d_step_start + du_step). The "du_d_this_iter" is the
        # control-DOF component of the iteration's du.
        target_du_d = (self._u_d_step_start + self.du_step) - self._control_disp()
        dlambda = (target_du_d - du_t_d) / du_p_d
        self.lambd += dlambda
        return du_t_eff + dlambda * du_p_eff


class ArcLength(StaticIntegrator):
    """Arc-length integrator (Crisfield) with optional load-factor weighting
    and adaptive step sizing.

    The path-following constraint, in its spherical form, is

        || u - u_step_start ||^2 + psi^2 * (lambda - lambda_0)^2 * |F|^2 = delta_s^2

    The ``psi = 0`` special case is the **cylindrical** variant (the
    most commonly used). For very stiff problems with extreme
    displacement-to-load ratios the spherical variant with a tuned
    ``psi`` is more robust.

    Both flavors are enforced at each Newton iteration by adjusting
    ``lambda`` alongside the displacement update. The integrator
    automatically detects limit points (snap-through, snap-back,
    post-buckling) using Bergan's generalised-stiffness-parameter
    heuristic.

    Parameters
    ----------
    delta_s : float
        Arc-length increment per step (always positive -- direction is
        controlled by ``initial_direction``).
    psi : float, default 0.0
        Load-factor weighting in the constraint norm. ``0`` =
        cylindrical (no load weighting); typical spherical values are
        ``0.5 ... 2.0``.
    initial_direction : {+1, -1}, default ``+1``
        Sign of ``lambda`` advance on the first step. Subsequent steps
        maintain the sign of the previous step's predictor unless a
        limit point is crossed.
    adaptive : bool, default False
        If True, scale ``delta_s`` between steps by
        ``(target_iterations / actual_iterations)^0.5`` after each
        converged step. Caps the step at
        ``[delta_s_min, delta_s_max]`` if those are given. Commercial
        codes universally use this -- a step that converged in 2
        iterations becomes ~2x larger, a step that needed 12 becomes
        ~0.7x.
    target_iterations : int, default 4
        Reference iteration count for adaptive sizing.
    delta_s_min, delta_s_max : float or None
        Optional caps on the adaptive step size.
    """

    supports_line_search = False

    def __init__(
        self,
        delta_s: float,
        *,
        psi: float = 0.0,
        initial_direction: int = 1,
        adaptive: bool = False,
        target_iterations: int = 4,
        delta_s_min: float | None = None,
        delta_s_max: float | None = None,
    ):
        super().__init__()
        if delta_s <= 0.0:
            raise ValueError("delta_s must be positive")
        if initial_direction not in (1, -1):
            raise ValueError("initial_direction must be +1 or -1")
        if psi < 0.0:
            raise ValueError("psi must be >= 0")
        if target_iterations < 1:
            raise ValueError("target_iterations must be >= 1")
        self.delta_s = float(delta_s)
        self.psi = float(psi)
        self._initial_direction = int(initial_direction)
        self.adaptive = bool(adaptive)
        self.target_iterations = int(target_iterations)
        self.delta_s_min = (
            float(delta_s_min) if delta_s_min is not None else None
        )
        self.delta_s_max = (
            float(delta_s_max) if delta_s_max is not None else None
        )
        # State at the start of the current step.
        self._u_step_start: np.ndarray | None = None
        self._lambd_step_start: float = 0.0
        # Previous step's converged displacement increment -- used for
        # direction tracking (sign of du_p . prev_step_du, the
        # "generalized stiffness parameter" of Bergan).
        self._prev_step_du: np.ndarray | None = None
        # Previous step's converged dlambda -- limit points are detected
        # from sign changes of this quantity (snap-through / snap-back).
        self._prev_step_dlambda: float | None = None
        # Limit-point flags per step (set on commit if sign flipped)
        self._limit_points: list[int] = []
        # Snapshot for revert.
        self._u_before: np.ndarray | None = None
        self._lambd_before: float = 0.0
        self._prev_step_du_before: np.ndarray | None = None
        # Step counter for adaptive sizing
        self._step_index: int = 0
        # Last step's iteration count -- updated by the algorithm via
        # ``record_step_iterations`` after a step finishes.
        self._last_iter_count: int = 0

    def bind(self, model) -> None:
        super().bind(model)
        if model.mp_constraints:
            raise NotImplementedError(
                "ArcLength with MP constraints is not yet supported"
            )

    def _gather_u(self) -> np.ndarray:
        """Snapshot ``Node.disp`` at the free DOFs into an array of size
        ``model.neq``."""
        u = np.zeros(self.model.neq)
        for n in self.model.nodes.values():
            for j in range(n.ndf):
                eq = int(n.eqn[j])
                if eq >= 0:
                    u[eq] = n.disp[j]
        return u

    def new_step(self) -> None:
        # Snapshot for revert.
        self._u_before = (
            None if self._u_step_start is None else self._u_step_start.copy()
        )
        self._lambd_before = self._lambd_step_start
        self._prev_step_du_before = (
            None if self._prev_step_du is None else self._prev_step_du.copy()
        )
        # Record start of step.
        self._u_step_start = self._gather_u()
        self._lambd_step_start = self.lambd

    def revert_step(self) -> None:
        self._u_step_start = self._u_before
        self._lambd_step_start = self._lambd_before
        self._prev_step_du = self._prev_step_du_before

    def commit_step(self) -> None:
        """Snapshot the converged displacement increment of this step so
        the next step's predictor can compute a sign-consistent
        direction (Bergan's GSP heuristic). Also detect limit points
        from the sign change of dlambda between consecutive steps and
        optionally adapt ``delta_s`` from the step's iteration count.
        """
        new_du = self._gather_u() - self._u_step_start
        new_dlambda = self.lambd - self._lambd_step_start
        # Limit-point detection: at a limit point the load-factor
        # increment reverses sign while the displacement keeps
        # advancing. Snap-through (peak) and snap-back (trough) both
        # show as dlambda sign flips between consecutive committed
        # steps. (Pure displacement reversal in u-space is also
        # captured by checking du . prev_du for the rare "u-reversal"
        # limit point.)
        if self._prev_step_du is not None:
            prev_dlambda = self._prev_step_dlambda
            if prev_dlambda is not None and prev_dlambda * new_dlambda < 0.0:
                self._limit_points.append(self._step_index)
            elif float(new_du @ self._prev_step_du) < 0.0:
                self._limit_points.append(self._step_index)
        self._prev_step_du = new_du
        self._prev_step_dlambda = new_dlambda
        self._step_index += 1
        # Adaptive step sizing -- scale delta_s for the next step
        if self.adaptive and self._last_iter_count > 0:
            scale = (self.target_iterations / self._last_iter_count) ** 0.5
            new_delta = self.delta_s * scale
            if self.delta_s_min is not None:
                new_delta = max(new_delta, self.delta_s_min)
            if self.delta_s_max is not None:
                new_delta = min(new_delta, self.delta_s_max)
            self.delta_s = new_delta

    def record_step_iterations(self, n_iter: int) -> None:
        """Tell the integrator how many iterations the last solved step
        took. Used by :attr:`adaptive` step sizing. Driver code calls
        this from ``NonlinearStaticAnalysis.run`` after each step."""
        self._last_iter_count = int(n_iter)

    @property
    def limit_points(self) -> list[int]:
        """Step indices where a limit point was crossed (GSP sign
        flip). Indices are 0-based and refer to the step *after* which
        the sign change was detected."""
        return list(self._limit_points)

    def solve_iteration(
        self,
        K_solve: Callable[[np.ndarray], np.ndarray],
        R_eff: np.ndarray,
        F_ref_eff: np.ndarray,
        T,
    ) -> np.ndarray:
        du_t_eff = K_solve(R_eff)
        du_p_eff = K_solve(F_ref_eff)
        # Current displacement-from-step-start (effective space).
        u_now = self._gather_u()
        delta_u = u_now - self._u_step_start
        # Two regimes:
        #   * predictor (delta_u == 0, first iteration of the step):
        #     use ``|du_p| * |dlambda| = delta_s``;
        #   * corrector (delta_u != 0, ``orthogonality`` to delta_u).
        psi2_FF = (
            (self.psi * self.psi) * float(F_ref_eff @ F_ref_eff)
            if self.psi > 0.0 else 0.0
        )
        delta_u_norm = float(np.linalg.norm(delta_u))
        delta_lambda_step = self.lambd - self._lambd_step_start
        if delta_u_norm < 1.0e-14 and abs(delta_lambda_step) < 1.0e-14:
            # Predictor step. Spherical constraint:
            #     |dlambda * du_p|^2 + psi^2 |F|^2 dlambda^2 = ds^2
            # =>  dlambda^2 (|du_p|^2 + psi^2 |F|^2) = ds^2
            denom2 = float(du_p_eff @ du_p_eff) + psi2_FF
            if denom2 < 1.0e-300:
                raise RuntimeError(
                    "ArcLength: parametric solution has zero norm"
                )
            magnitude = self.delta_s / np.sqrt(denom2)
            # Direction selection (Bergan's GSP)
            if self._prev_step_du is None:
                sign = float(self._initial_direction)
            else:
                dot = float(du_p_eff @ self._prev_step_du)
                sign = 1.0 if dot >= 0.0 else -1.0
            dlambda = sign * magnitude
        else:
            # Corrector step: enforce orthogonality
            #   delta_u . (du_t + dlambda * du_p)
            #   + psi^2 |F|^2 * delta_lambda * dlambda = 0
            num = float(delta_u @ du_t_eff)
            den = float(delta_u @ du_p_eff) + psi2_FF * delta_lambda_step
            if abs(den) < 1.0e-300:
                raise RuntimeError(
                    "ArcLength: corrector denominator vanished -- "
                    "near-orthogonal increment, try a smaller delta_s"
                )
            dlambda = -num / den
        self.lambd += dlambda
        return du_t_eff + dlambda * du_p_eff


# ---------------------------------------------------------------------------
# helpers shared by Static and (eventually) Transient integrators


def _assemble_tangent(model) -> sp.csc_matrix:
    """Assemble the global tangent stiffness using each element's
    :meth:`Element.K_tangent_global`. Reuses the vectorized COO path of
    :func:`assemble_stiffness` by temporarily swapping ``K_global``."""
    # Easiest correct implementation: replicate the assembly here using
    # K_tangent_global so we don't perturb the original method or rely on
    # monkey-patching.
    neq = model.neq
    elements = list(model.elements.values())
    if not elements:
        return sp.csc_matrix((neq, neq))

    total = 0
    cache: list[tuple] = []
    for e in elements:
        Ke = e.K_tangent_global()
        dofs = model.element_dof_map(e)
        cache.append((dofs, Ke))
        total += dofs.size * dofs.size
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=float)
    pos = 0
    for (dofs, Ke) in cache:
        n = dofs.size
        nn = n * n
        rows[pos : pos + nn] = np.repeat(dofs, n)
        cols[pos : pos + nn] = np.tile(dofs, n)
        vals[pos : pos + nn] = np.asarray(Ke, dtype=float).ravel()
        pos += nn
    mask = (rows >= 0) & (cols >= 0)
    if not mask.any():
        return sp.csc_matrix((neq, neq))
    return sp.coo_matrix(
        (vals[mask], (rows[mask], cols[mask])), shape=(neq, neq)
    ).tocsc()


def _assemble_internal_force(model) -> np.ndarray:
    """Assemble ``f_int = sum_e gather(f_int_e)`` over free DOFs."""
    neq = model.neq
    f = np.zeros(neq)
    for e in model.elements.values():
        fe = e.f_int_global()
        if fe is None or not np.any(fe):
            continue
        dofs = model.element_dof_map(e)
        free = dofs >= 0
        if free.any():
            f[dofs[free]] += np.asarray(fe, dtype=float)[free]
    return f
