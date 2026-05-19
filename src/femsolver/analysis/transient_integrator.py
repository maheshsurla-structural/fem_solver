"""Transient (time-stepping) integrators.

Currently provides :class:`Newmark` — the canonical structural-dynamics
implicit integrator. With the default ``beta = 1/4, gamma = 1/2``
("average acceleration") it is unconditionally stable, second-order
accurate, and adds no algorithmic damping. With ``gamma > 1/2`` it
becomes dissipative (the classical Newmark-dissipative recipe), useful
when contact / sharp nonlinearities introduce spurious high-frequency
content.

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
