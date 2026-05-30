"""Transient (time-stepping) integrators.

Currently provides two Newmark variants:

* :class:`Newmark` — for *linear* transient analysis. The effective
  tangent ``K_eff = K + a4 C + a1 M`` is constant in time, factored
  once at :meth:`bind` time, and reused for every step. One linear
  solve per step.

* :class:`NewmarkNonlinear` — for *nonlinear* transient analysis. Looks
  like a :class:`~femsolver.analysis.integrator.StaticIntegrator` from
  the algorithm's perspective: it provides ``residual``, ``tangent``,
  and ``solve_iteration`` so the existing static ``_newton_loop`` can
  be reused unchanged. The residual is the dynamic equilibrium
  ``F(t) - M u_ddot - C u_dot - f_int(u)`` and the tangent picks up
  the ``a4 C + a1 M`` contribution.

With the default ``beta = 1/4, gamma = 1/2`` ("average acceleration")
both variants are unconditionally stable, second-order accurate, and
add no algorithmic damping.

HHT-alpha (Hilber-Hughes-Taylor) is the natural next addition — same
overall architecture, slightly different update rules. Deferred to a
follow-up.

Equations
---------
At step ``n``, the integrator stores ``u_n, ud_n, udd_n``. Given the
external force ``F_{n+1}`` at the *new* time, the Newmark relations are

    udd_{n+1} = (1 / (beta dt^2)) (u_{n+1} - u_n) - (1 / (beta dt)) ud_n
                - (1 / (2 beta) - 1) udd_n
    ud_{n+1}  = (gamma / (beta dt)) (u_{n+1} - u_n) - (gamma / beta - 1) ud_n
                - dt (gamma / (2 beta) - 1) udd_n

Substituting into the equation of motion

    M udd + C ud + K u = F

and grouping terms gives the linear system

    K_eff u_{n+1} = F_{n+1} + M (a1 u_n + a2 ud_n + a3 udd_n)
                            + C (a4 u_n + a5 ud_n + a6 udd_n)

with

    K_eff = K + a4 C + a1 M
    a1 = 1 / (beta dt^2),  a2 = 1 / (beta dt),  a3 = 1 / (2 beta) - 1
    a4 = gamma / (beta dt), a5 = gamma / beta - 1
    a6 = dt (gamma / (2 beta) - 1)

For linear analysis ``M, C, K`` are constant, so ``K_eff`` is factored
once at :meth:`bind` and reused for every step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from femsolver.analysis.assembler import (
    assemble_force,
    assemble_mass,
    assemble_stiffness,
)
from femsolver.analysis.damping import RayleighDamping
from femsolver.analysis.integrator import StaticIntegrator
from femsolver.analysis.integrator import (
    _assemble_internal_force,
    _assemble_tangent,
)


class TransientIntegrator(ABC):
    """Base class for time-stepping integrators in linear transient
    analysis."""

    def __init__(self):
        self.model = None
        self.dt: float = 0.0
        self.t: float = 0.0
        # State at the *current* time (size neq).
        self.u: np.ndarray | None = None
        self.u_dot: np.ndarray | None = None
        self.u_ddot: np.ndarray | None = None
        # System matrices (constant for linear analysis).
        self.M: sp.csc_matrix | None = None
        self.C: sp.csc_matrix | None = None
        self.K: sp.csc_matrix | None = None
        # Reference load vector (size neq); the time function scales this.
        self.F_ref: np.ndarray | None = None

    def bind(self, model, *, dt: float,
             damping: RayleighDamping | None = None) -> None:
        """Capture matrices, initial state, and factor ``K_eff``."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.model = model
        self.dt = float(dt)
        self.t = 0.0

        self.M = assemble_mass(model)
        self.K = assemble_stiffness(model)
        if damping is None:
            self.C = sp.csc_matrix(self.M.shape)
        else:
            self.C = damping.build(self.M, self.K)
        self.F_ref = assemble_force(model)

        # Initial conditions from Node state.
        self.u = self._gather("disp")
        self.u_dot = self._gather("velocity")
        # Initial acceleration: solve M udd_0 = F(0) - C ud_0 - K u_0.
        # F at t=0 is taken from the time function (assumed to be 1.0
        # unless the caller overrides — TransientAnalysis passes F(0)
        # explicitly to bind via set_initial_force).
        # For now use F_ref directly; the analysis will recompute udd_0
        # at the right scale if it differs from the reference.
        F0 = self._initial_force
        rhs0 = F0 - self.C @ self.u_dot - self.K @ self.u
        try:
            M_solve = factorized(self.M)
        except Exception as exc:
            raise RuntimeError(
                f"transient integrator: mass matrix could not be factored "
                f"({exc}). Likely cause: massless free DOFs — every free "
                "DOF needs an element with non-zero rho contributing mass "
                "to it."
            ) from exc
        self.u_ddot = np.asarray(M_solve(rhs0)).ravel()
        self._finalize_bind()

    @property
    def _initial_force(self) -> np.ndarray:
        """Force vector at ``t = 0``. The driver overrides this through
        :meth:`set_initial_force` if the time function is non-trivial
        at ``t = 0`` (default zero, which is the most common case)."""
        return getattr(self, "_F0", np.zeros_like(self.F_ref))

    def set_initial_force(self, F0: np.ndarray) -> None:
        """Set the force vector at ``t = 0`` so that the initial
        acceleration is consistent with ``F(0)``. Called by
        :class:`TransientAnalysis` before :meth:`bind`."""
        self._F0 = F0

    @abstractmethod
    def _finalize_bind(self) -> None:
        """Subclass hook: factor any integrator-specific effective
        matrix (e.g. Newmark's ``K_eff``) once at bind time so per-step
        cost is dominated by a back-substitution."""

    @abstractmethod
    def step(self, F_n_plus_1: np.ndarray) -> None:
        """Advance one time step. Updates ``u, u_dot, u_ddot, t``."""

    # ------------------------------------------------------- bookkeeping
    def _gather(self, attr: str) -> np.ndarray:
        """Read free-DOF values of ``Node.<attr>`` into a length-``neq``
        array."""
        n_eq = self.model.neq
        v = np.zeros(n_eq)
        for n in self.model.nodes.values():
            arr = getattr(n, attr)
            for j in range(n.ndf):
                eq = int(n.eqn[j])
                if eq >= 0:
                    v[eq] = arr[j]
        return v

    def _scatter(self, attr: str, vec: np.ndarray) -> None:
        """Write a free-DOF vector back into ``Node.<attr>``."""
        for n in self.model.nodes.values():
            arr = getattr(n, attr)
            for j in range(n.ndf):
                eq = int(n.eqn[j])
                if eq >= 0:
                    arr[j] = vec[eq]


class Newmark(TransientIntegrator):
    """Implicit Newmark-beta integrator.

    Parameters
    ----------
    beta : float, default 1/4
        Beta parameter. ``1/4`` gives average acceleration (the default,
        unconditionally stable, no algorithmic dissipation).
        ``1/6`` gives linear acceleration (conditionally stable).
    gamma : float, default 1/2
        Gamma parameter. ``1/2`` is the standard choice;
        ``gamma > 1/2`` introduces algorithmic damping.
    """

    def __init__(self, *, beta: float = 0.25, gamma: float = 0.5):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must lie in [0, 1]")
        self.beta = float(beta)
        self.gamma = float(gamma)
        # Newmark coefficients — populated at bind() from dt
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0
        self._a4 = 0.0
        self._a5 = 0.0
        self._a6 = 0.0
        # Factored K_eff solver
        self._K_eff_solve: Callable[[np.ndarray], np.ndarray] | None = None

    def _finalize_bind(self) -> None:
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        self._a1 = 1.0 / (beta * dt * dt)
        self._a2 = 1.0 / (beta * dt)
        self._a3 = 1.0 / (2.0 * beta) - 1.0
        self._a4 = gamma / (beta * dt)
        self._a5 = gamma / beta - 1.0
        self._a6 = dt * (gamma / (2.0 * beta) - 1.0)
        # K_eff = a1 M + a4 C + K   (constant for linear analysis)
        K_eff = (self._a1 * self.M + self._a4 * self.C + self.K).tocsc()
        try:
            self._K_eff_solve = factorized(K_eff)
        except Exception as exc:
            raise RuntimeError(
                f"Newmark: effective tangent could not be factored ({exc}). "
                "Likely cause: zero-stiffness free DOF or numerical "
                "singularity in K + a4 C + a1 M."
            ) from exc

    def step(self, F_n_plus_1: np.ndarray) -> None:
        a1, a2, a3 = self._a1, self._a2, self._a3
        a4, a5, a6 = self._a4, self._a5, self._a6
        M, C = self.M, self.C
        u, ud, udd = self.u, self.u_dot, self.u_ddot
        rhs = (
            F_n_plus_1
            + M @ (a1 * u + a2 * ud + a3 * udd)
            + C @ (a4 * u + a5 * ud + a6 * udd)
        )
        u_new = np.asarray(self._K_eff_solve(rhs)).ravel()
        udd_new = a1 * (u_new - u) - a2 * ud - a3 * udd
        ud_new = a4 * (u_new - u) - a5 * ud - a6 * udd
        self.u, self.u_dot, self.u_ddot = u_new, ud_new, udd_new
        self.t += self.dt


# ============================================================ HHT-alpha

class HHTAlpha(TransientIntegrator):
    """Hilber-Hughes-Taylor alpha-method (HHT-alpha) for linear
    transient analysis.

    The equation of motion is enforced at a *weighted* time between
    ``t_n`` and ``t_{n+1}``::

        M u_ddot_{n+1} + (1 + alpha) C u_dot_{n+1} - alpha C u_dot_n
            + (1 + alpha) K u_{n+1} - alpha K u_n
            = (1 + alpha) F_{n+1} - alpha F_n

    The Newmark u-v-a relations are unchanged, with the parameters
    auto-tuned for optimal high-frequency dissipation::

        beta  = (1 - alpha)^2 / 4
        gamma = 1/2 - alpha

    With ``alpha = 0`` HHT collapses exactly to Newmark with
    ``beta = 1/4, gamma = 1/2`` (average acceleration, no algorithmic
    damping). ``alpha < 0`` introduces second-order-accurate numerical
    damping that grows with frequency -- useful to suppress spurious
    high-frequency content in stiff systems while keeping the lower
    modes accurate.

    Parameters
    ----------
    alpha : float, default -0.05
        HHT parameter in ``[-1/3, 0]``. ``0`` = no damping (= Newmark
        average-acceleration). Smaller (more negative) = more
        high-frequency damping. ``-0.05`` is a mild, common default.

    Notes
    -----
    Unconditionally stable and second-order accurate for any
    ``alpha in [-1/3, 0]``. For nonlinear analysis use
    :class:`NewmarkNonlinear` (HHT-alpha for nonlinear systems would
    parallel this with the same alpha-blending; a documented future
    refinement).
    """

    def __init__(self, *, alpha: float = -0.05):
        super().__init__()
        if not (-1.0 / 3.0 <= alpha <= 0.0):
            raise ValueError(f"alpha must lie in [-1/3, 0], got {alpha}")
        self.alpha = float(alpha)
        # Beta / gamma auto-tuned for HHT-alpha to retain second-order
        # accuracy and unconditional stability.
        self.beta = (1.0 - self.alpha) ** 2 / 4.0
        self.gamma = 0.5 - self.alpha
        # Newmark coefficients populated at bind() from dt
        self._a1 = self._a2 = self._a3 = 0.0
        self._a4 = self._a5 = self._a6 = 0.0
        # Factored K_eff solver
        self._K_eff_solve: Callable[[np.ndarray], np.ndarray] | None = None
        # Previous-step external force (for the (1+alpha)F_{n+1} - alpha F_n term)
        self._F_prev: np.ndarray | None = None

    def _finalize_bind(self) -> None:
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        alpha = self.alpha
        self._a1 = 1.0 / (beta * dt * dt)
        self._a2 = 1.0 / (beta * dt)
        self._a3 = 1.0 / (2.0 * beta) - 1.0
        self._a4 = gamma / (beta * dt)
        self._a5 = gamma / beta - 1.0
        self._a6 = dt * (gamma / (2.0 * beta) - 1.0)
        # K_eff for HHT: a1 M + (1+alpha) a4 C + (1+alpha) K
        K_eff = (
            self._a1 * self.M
            + (1.0 + alpha) * self._a4 * self.C
            + (1.0 + alpha) * self.K
        ).tocsc()
        try:
            self._K_eff_solve = factorized(K_eff)
        except Exception as exc:
            raise RuntimeError(
                f"HHTAlpha: effective tangent could not be factored ({exc})."
            ) from exc
        # F_prev initialized to F(0); the first step uses the user-set
        # initial force at t = 0.
        self._F_prev = self._initial_force.copy()

    def step(self, F_n_plus_1: np.ndarray) -> None:
        a1, a2, a3 = self._a1, self._a2, self._a3
        a4, a5, a6 = self._a4, self._a5, self._a6
        alpha = self.alpha
        M, C, K = self.M, self.C, self.K
        u, ud, udd = self.u, self.u_dot, self.u_ddot
        F_prev = self._F_prev
        rhs = (
            (1.0 + alpha) * F_n_plus_1 - alpha * F_prev
            + M @ (a1 * u + a2 * ud + a3 * udd)
            + C @ ((1.0 + alpha) * a4 * u + ((1.0 + alpha) * a5 + alpha) * ud
                    + (1.0 + alpha) * a6 * udd)
            + alpha * (K @ u)
        )
        u_new = np.asarray(self._K_eff_solve(rhs)).ravel()
        udd_new = a1 * (u_new - u) - a2 * ud - a3 * udd
        ud_new = a4 * (u_new - u) - a5 * ud - a6 * udd
        self.u, self.u_dot, self.u_ddot = u_new, ud_new, udd_new
        self._F_prev = F_n_plus_1
        self.t += self.dt


# ============================================================ generalized-alpha

class GeneralizedAlpha(TransientIntegrator):
    """Chung-Hulbert generalized-alpha integrator for linear transient
    analysis.

    The equation of motion is enforced at *two* intermediate time
    points -- one for the inertia term and one for the stiffness /
    damping / external-force terms::

        (1 - alpha_m) M a_{n+1} + alpha_m M a_n
            + (1 - alpha_f) (C v_{n+1} + K u_{n+1})
            + alpha_f (C v_n + K u_n)
            = (1 - alpha_f) F_{n+1} + alpha_f F_n

    Specifying a single user-friendly parameter ``rho_inf`` (the
    spectral radius at the high-frequency limit) auto-tunes
    ``alpha_m, alpha_f, beta, gamma`` to maximize accuracy at low
    frequencies while damping the high frequencies at the rate set by
    ``rho_inf``::

        alpha_m = (2 rho_inf - 1) / (rho_inf + 1)
        alpha_f =        rho_inf  / (rho_inf + 1)
        beta    = (1 - alpha_m + alpha_f)^2 / 4
        gamma   = 1/2 - alpha_m + alpha_f

    ``rho_inf = 1.0`` is the standard average-acceleration Newmark
    (no algorithmic damping). ``rho_inf = 0.0`` annihilates high-
    frequency modes in one step. Typical engineering choices land in
    ``[0.6, 0.95]``.

    Parameters
    ----------
    rho_inf : float, default 0.8
        Spectral radius at infinity, in ``[0, 1]``. ``1`` = no
        algorithmic damping; smaller = more high-frequency dissipation.

    Notes
    -----
    Unconditionally stable and second-order accurate for any
    ``rho_inf in [0, 1]``. Subsumes Newmark (``rho_inf = 1``), HHT
    (``alpha_m = 0``), and WBZ (``alpha_f = 0``) as special cases. For
    nonlinear problems use :class:`NewmarkNonlinear` until a
    generalized-alpha nonlinear variant is added.
    """

    def __init__(self, *, rho_inf: float = 0.8):
        super().__init__()
        if not (0.0 <= rho_inf <= 1.0):
            raise ValueError(f"rho_inf must lie in [0, 1], got {rho_inf}")
        self.rho_inf = float(rho_inf)
        # Auto-tuned coefficients (Chung-Hulbert optimal)
        self.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.beta = (1.0 - self.alpha_m + self.alpha_f) ** 2 / 4.0
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        # Newmark coefficients populated at bind()
        self._a1 = self._a2 = self._a3 = 0.0
        self._a4 = self._a5 = self._a6 = 0.0
        self._K_eff_solve: Callable[[np.ndarray], np.ndarray] | None = None
        self._F_prev: np.ndarray | None = None

    def _finalize_bind(self) -> None:
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        am = self.alpha_m
        af = self.alpha_f
        self._a1 = 1.0 / (beta * dt * dt)
        self._a2 = 1.0 / (beta * dt)
        self._a3 = 1.0 / (2.0 * beta) - 1.0
        self._a4 = gamma / (beta * dt)
        self._a5 = gamma / beta - 1.0
        self._a6 = dt * (gamma / (2.0 * beta) - 1.0)
        K_eff = (
            (1.0 - am) * self._a1 * self.M
            + (1.0 - af) * self._a4 * self.C
            + (1.0 - af) * self.K
        ).tocsc()
        try:
            self._K_eff_solve = factorized(K_eff)
        except Exception as exc:
            raise RuntimeError(
                f"GeneralizedAlpha: K_eff could not be factored ({exc})."
            ) from exc
        self._F_prev = self._initial_force.copy()

    def step(self, F_n_plus_1: np.ndarray) -> None:
        a1, a2, a3 = self._a1, self._a2, self._a3
        a4, a5, a6 = self._a4, self._a5, self._a6
        am = self.alpha_m
        af = self.alpha_f
        M, C, K = self.M, self.C, self.K
        u, ud, udd = self.u, self.u_dot, self.u_ddot
        F_prev = self._F_prev
        rhs = (
            (1.0 - af) * F_n_plus_1 + af * F_prev
            + M @ ((1.0 - am) * (a1 * u + a2 * ud + a3 * udd) - am * udd)
            + C @ ((1.0 - af) * (a4 * u + a5 * ud + a6 * udd) - af * ud)
            - af * (K @ u)
        )
        u_new = np.asarray(self._K_eff_solve(rhs)).ravel()
        udd_new = a1 * (u_new - u) - a2 * ud - a3 * udd
        ud_new = a4 * (u_new - u) - a5 * ud - a6 * udd
        self.u, self.u_dot, self.u_ddot = u_new, ud_new, udd_new
        self._F_prev = F_n_plus_1
        self.t += self.dt


# ============================================================ central difference

class CentralDifference(TransientIntegrator):
    """Explicit central-difference integrator for linear transient
    analysis (impact / blast / very-short-duration loading).

    Update equations:

        u_ddot_n = (u_{n+1} - 2 u_n + u_{n-1}) / dt^2
        u_dot_n  = (u_{n+1} - u_{n-1}) / (2 dt)

    Enforcing the equation of motion at ``t_n``:

        M u_ddot_n + C u_dot_n + K u_n = F_n

    and solving for ``u_{n+1}``:

        [M/dt^2 + C/(2 dt)] u_{n+1} =
            F_n - K u_n + (2 M / dt^2) u_n
            - (M/dt^2 - C/(2 dt)) u_{n-1}

    With a **lumped** mass and Rayleigh-alpha_M-only damping, the
    left-hand side is diagonal -- no matrix solve per step. Each step
    costs only one matrix-vector multiply (``K u_n``) plus a few
    element-wise updates. The cure is conditional stability: the time
    step must satisfy

        dt < 2 / omega_max

    where ``omega_max`` is the largest natural frequency of the mass-
    normalised stiffness. For typical solid-mechanics problems this
    is a small fraction of the smallest element transit time.

    Parameters
    ----------
    lumped_mass : bool, default True
        Use a row-summed lumped mass matrix (the standard choice for
        explicit integration). If ``False``, the consistent mass is
        used and the per-step cost includes a factored solve.

    Notes
    -----
    For nonlinear problems (most impact / blast / contact analyses),
    a nonlinear central-difference variant is the natural next step;
    deferred to a future Phase 17.x.
    """

    def __init__(self, *, lumped_mass: bool = True):
        super().__init__()
        self.lumped_mass = bool(lumped_mass)
        self._u_prev: np.ndarray | None = None          # u_{n-1}
        self._M_eff_solve: Callable[[np.ndarray], np.ndarray] | None = None
        self._M_two_over_dt2: sp.csc_matrix | None = None
        self._M_over_dt2_minus_C_over_2dt: sp.csc_matrix | None = None

    def bind(self, model, *, dt: float,
             damping: RayleighDamping | None = None) -> None:
        # Override base bind to assemble lumped mass when requested.
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.model = model
        self.dt = float(dt)
        self.t = 0.0
        self.M = assemble_mass(model, lumped=self.lumped_mass)
        self.K = assemble_stiffness(model)
        if damping is None:
            self.C = sp.csc_matrix(self.M.shape)
        else:
            self.C = damping.build(self.M, self.K)
        self.F_ref = assemble_force(model)
        self.u = self._gather("disp")
        self.u_dot = self._gather("velocity")
        F0 = self._initial_force
        rhs0 = F0 - self.C @ self.u_dot - self.K @ self.u
        try:
            M_solve = factorized(self.M)
        except Exception as exc:
            raise RuntimeError(
                f"CentralDifference: mass matrix could not be factored "
                f"({exc})."
            ) from exc
        self.u_ddot = np.asarray(M_solve(rhs0)).ravel()
        self._finalize_bind()

    def _finalize_bind(self) -> None:
        dt = self.dt
        # Effective LHS: M/dt^2 + C/(2 dt)
        M_eff = (self.M * (1.0 / (dt * dt))
                 + self.C * (1.0 / (2.0 * dt))).tocsc()
        try:
            self._M_eff_solve = factorized(M_eff)
        except Exception as exc:
            raise RuntimeError(
                f"CentralDifference: effective mass matrix could not be "
                f"factored ({exc})."
            ) from exc
        # Precompute frequently used combinations
        self._M_two_over_dt2 = (self.M * (2.0 / (dt * dt))).tocsc()
        self._M_over_dt2_minus_C_over_2dt = (
            self.M * (1.0 / (dt * dt)) - self.C * (1.0 / (2.0 * dt))
        ).tocsc()
        # Starting value u_{-1} from Taylor expansion:
        #   u_{-1} = u_0 - dt v_0 + (dt^2 / 2) a_0
        self._u_prev = (
            self.u - dt * self.u_dot + 0.5 * dt * dt * self.u_ddot
        )

    def step(self, F_n: np.ndarray) -> None:
        """Advance one explicit step using force at ``t_n``.

        Note: unlike implicit integrators (which use F at ``t_{n+1}``),
        CD uses F at the *current* time. The driver passes whatever it
        thinks of as "F at this step"; both conventions land on the
        same physical answer for slowly-varying loads but differ by
        ``dt`` for sharp transients.
        """
        dt = self.dt
        rhs = (
            F_n
            - self.K @ self.u
            + self._M_two_over_dt2 @ self.u
            - self._M_over_dt2_minus_C_over_2dt @ self._u_prev
        )
        u_new = np.asarray(self._M_eff_solve(rhs)).ravel()
        udd_new = (u_new - 2.0 * self.u + self._u_prev) / (dt * dt)
        ud_new = (u_new - self._u_prev) / (2.0 * dt)
        # Shift history
        self._u_prev = self.u
        self.u = u_new
        self.u_dot = ud_new
        self.u_ddot = udd_new
        self.t += dt


# ============================================================ nonlinear

class NewmarkNonlinear(StaticIntegrator):
    """Newmark for *nonlinear* transient analysis.

    Presents the same interface as a :class:`StaticIntegrator`
    (``residual, tangent, solve_iteration, new_step, revert_step,
    commit_step``) so the existing static ``_newton_loop`` can drive
    the Newton iteration at each time step. The dynamic content lives
    inside ``residual`` and ``tangent``:

    * **residual**: ``R = F(t_{n+1}) - M u_ddot(u) - C u_dot(u) - f_int(u)``
      where ``u_ddot, u_dot`` are expressed via Newmark in terms of the
      current iterate ``u`` and the step-start state ``u_n, u_dot_n,
      u_ddot_n``.
    * **tangent**: ``K_eff_T = K_tangent(u) + a4 C + a1 M``
      (the consistent linearisation of the residual w.r.t. ``u``).

    The step lifecycle::

        new_step(F_target)   # snapshot u_n, u_dot_n, u_ddot_n; remember F_target
        # ... _newton_loop iterates residual / tangent / solve_iteration ...
        commit_step()        # compute u_dot_{n+1}, u_ddot_{n+1} from
                             # converged u_{n+1}; scatter to nodes; advance t
        # (or, on non-convergence:)
        revert_step()        # restore u, u_dot, u_ddot to step-start values
    """

    # Path-following integrators advertise this so the algorithm's
    # line search backs off; for the dynamic case the corrector is a
    # genuine Newton step, so line search IS valid.
    supports_line_search = True

    def __init__(self, *, beta: float = 0.25, gamma: float = 0.5):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must lie in [0, 1]")
        self.beta = float(beta)
        self.gamma = float(gamma)
        # Time-step bookkeeping
        self.dt: float = 0.0
        self.t: float = 0.0
        # State snapshot at the START of the current step
        # (these are u_n, u_dot_n, u_ddot_n).
        self._u_n: np.ndarray | None = None
        self._ud_n: np.ndarray | None = None
        self._udd_n: np.ndarray | None = None
        # Snapshot for revert (set at new_step entry, before any updates).
        self._u_revert: np.ndarray | None = None
        self._ud_revert: np.ndarray | None = None
        self._udd_revert: np.ndarray | None = None
        self._t_revert: float = 0.0
        # System matrices
        self.M: sp.csc_matrix | None = None
        self.C: sp.csc_matrix | None = None
        # Time-varying force target at t_{n+1}
        self._F_target: np.ndarray | None = None
        # Newmark coefficients (set from dt at bind)
        self._a1 = self._a2 = self._a3 = 0.0
        self._a4 = self._a5 = self._a6 = 0.0

    # ------------------------------------------------------------ bind
    def bind(self, model, *, dt: float | None = None,
             damping: RayleighDamping | None = None) -> None:
        super().bind(model)
        if dt is None:
            raise ValueError("NewmarkNonlinear.bind requires dt")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        # M assembled directly; for C we need K (initial elastic) to
        # build a Rayleigh combination consistent with the initial
        # state.
        self.M = assemble_mass(model)
        if damping is None:
            self.C = sp.csc_matrix(self.M.shape)
        else:
            K_initial = assemble_stiffness(model)
            self.C = damping.build(self.M, K_initial)

        # Initial conditions from Node state.
        self._u_n = self._gather("disp")
        self._ud_n = self._gather("velocity")
        # u_ddot_0 from M u_ddot_0 = F(0) - C u_dot_0 - f_int(u_0)
        F0 = self._initial_force_at_zero()
        f_int_0 = _assemble_internal_force(model)
        rhs0 = F0 - self.C @ self._ud_n - f_int_0
        try:
            from scipy.sparse.linalg import factorized as _fac
            M_solve = _fac(self.M)
        except Exception as exc:
            raise RuntimeError(
                "NewmarkNonlinear: mass matrix could not be factored. "
                "Every free DOF needs an element contributing mass."
            ) from exc
        self._udd_n = np.asarray(M_solve(rhs0)).ravel()
        self._scatter("acceleration", self._udd_n)

        # Newmark coefficients
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        self._a1 = 1.0 / (beta * dt * dt)
        self._a2 = 1.0 / (beta * dt)
        self._a3 = 1.0 / (2.0 * beta) - 1.0
        self._a4 = gamma / (beta * dt)
        self._a5 = gamma / beta - 1.0
        self._a6 = dt * (gamma / (2.0 * beta) - 1.0)

    def _initial_force_at_zero(self) -> np.ndarray:
        """Force vector at ``t = 0`` — taken from any user-set
        ``_F0`` attribute, otherwise zero."""
        return getattr(self, "_F0", np.zeros_like(self.F_ref))

    def set_initial_force(self, F0: np.ndarray) -> None:
        self._F0 = F0

    # ------------------------------------------------------ step lifecycle
    def new_step(self, F_target: np.ndarray | None = None) -> None:
        """Snapshot state at the start of the new step. If
        ``F_target`` is given, store it as the right-hand side at
        ``t_{n+1}``; otherwise the driver should set it via
        :meth:`set_target_force` before any residual call.
        """
        # Snapshot the *committed* state for revert. Use the current
        # _u_n etc., which are the values from the previous converged
        # step (or initial conditions on step 1).
        self._u_revert = self._u_n.copy()
        self._ud_revert = self._ud_n.copy()
        self._udd_revert = self._udd_n.copy()
        self._t_revert = self.t
        if F_target is not None:
            self._F_target = np.asarray(F_target, dtype=float).ravel()

    def set_target_force(self, F_target: np.ndarray) -> None:
        """Set the right-hand side at ``t_{n+1}``. Called by the
        driver before each step's Newton loop begins."""
        self._F_target = np.asarray(F_target, dtype=float).ravel()

    def revert_step(self) -> None:
        """Restore u, u_dot, u_ddot to step-start values after a
        failed Newton iteration. Also scatter back to Node arrays so
        element state determination sees the reverted state."""
        if self._u_revert is None:
            return
        self._u_n = self._u_revert.copy()
        self._ud_n = self._ud_revert.copy()
        self._udd_n = self._udd_revert.copy()
        self.t = self._t_revert
        self._scatter("disp", self._u_n)
        self._scatter("velocity", self._ud_n)
        self._scatter("acceleration", self._udd_n)

    def commit_step(self) -> None:
        """After Newton has converged, update u̇ and ü from the
        converged u_{n+1}, scatter to nodes, advance time."""
        u_new = self._gather("disp")
        delta_u = u_new - self._u_n
        udd_new = (
            self._a1 * delta_u
            - self._a2 * self._ud_n
            - self._a3 * self._udd_n
        )
        ud_new = (
            self._a4 * delta_u
            - self._a5 * self._ud_n
            - self._a6 * self._udd_n
        )
        self._scatter("velocity", ud_new)
        self._scatter("acceleration", udd_new)
        # Roll committed state forward
        self._u_n = u_new
        self._ud_n = ud_new
        self._udd_n = udd_new
        self.t += self.dt

    # ----------------------------------------------- residual / tangent
    def residual(self) -> np.ndarray:
        """Dynamic equilibrium residual at the current iterate."""
        if self._F_target is None:
            raise RuntimeError(
                "NewmarkNonlinear.residual called before set_target_force / new_step"
            )
        u_trial = self._gather("disp")
        delta_u = u_trial - self._u_n
        udd_trial = (
            self._a1 * delta_u
            - self._a2 * self._ud_n
            - self._a3 * self._udd_n
        )
        ud_trial = (
            self._a4 * delta_u
            - self._a5 * self._ud_n
            - self._a6 * self._udd_n
        )
        f_int = _assemble_internal_force(self.model)
        return (
            self._F_target
            - self.M @ udd_trial
            - self.C @ ud_trial
            - f_int
        )

    def tangent(self) -> sp.csc_matrix:
        """Effective tangent ``K_T(u) + a4 C + a1 M`` at the current
        iterate. ``K_T`` is the element-assembled tangent, picking up
        material and (in the corotational case) geometric
        contributions."""
        K_T = _assemble_tangent(self.model)
        return (K_T + self._a4 * self.C + self._a1 * self.M).tocsc()

    # solve_iteration: inherits the default (standard Newton step).

    # ----------------------------------------------------- bookkeeping
    def _gather(self, attr: str) -> np.ndarray:
        n_eq = self.model.neq
        v = np.zeros(n_eq)
        for n in self.model.nodes.values():
            arr = getattr(n, attr)
            for j in range(n.ndf):
                eq = int(n.eqn[j])
                if eq >= 0:
                    v[eq] = arr[j]
        return v

    def _scatter(self, attr: str, vec: np.ndarray) -> None:
        for n in self.model.nodes.values():
            arr = getattr(n, attr)
            for j in range(n.ndf):
                eq = int(n.eqn[j])
                if eq >= 0:
                    arr[j] = vec[eq]
