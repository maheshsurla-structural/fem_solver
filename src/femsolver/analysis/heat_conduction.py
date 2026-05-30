"""Heat-conduction analyses (steady-state + transient).

A thermal model is a :class:`~femsolver.Model` built with ``ndf=1`` —
each node carries a single DOF (temperature). The same assembler and
constraint machinery used for mechanics works for heat conduction;
the only changes are:

* element ``K_global`` returns the **conductivity** matrix (size
  ``n_nodes x n_nodes``);
* element ``M_global`` returns the **capacitance** matrix (used by
  the transient driver);
* nodal "loads" are heat fluxes (W) and the resulting "displacements"
  are temperatures.

This module provides:

* :class:`SteadyHeatAnalysis` — solves ``K_T T = f_T`` with optional
  convection-edge contributions and Dirichlet BCs from ``model.fix``.
* :class:`TransientHeatAnalysis` — generalized-trapezoidal (Crank-
  Nicolson by default) time-marching of
  ``C_T dT/dt + K_T T = f_T(t)``.

The driver also writes the solved temperatures back into
``node.disp`` so users can read them via ``model.node(tag).disp[0]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from femsolver.analysis.assembler import (
    assemble_force,
    assemble_mass,
    assemble_stiffness,
)


# ============================================================ steady-state

@dataclass
class SteadyHeatResult:
    """Result of a steady-state heat-conduction analysis."""

    T: np.ndarray              # solved nodal temperatures (size = n_nodes)
    flux_reactions: dict = field(default_factory=dict)
    iterations: int = 1


class SteadyHeatAnalysis:
    """Solve ``K_T T = f_T`` on a thermal model.

    Parameters
    ----------
    model : Model
        Thermal model built with ``ndf=1``.
    """

    def __init__(self, model):
        if model.ndf != 1:
            raise ValueError(
                f"SteadyHeatAnalysis requires a Model with ndf=1, "
                f"got ndf={model.ndf}"
            )
        self.model = model

    def run(self) -> SteadyHeatResult:
        m = self.model
        m.number_dofs()
        # Assemble conductivity (uses K_global of each element)
        K = assemble_stiffness(m).tocsc()
        f = assemble_force(m)
        if K.nnz == 0:
            raise RuntimeError(
                "thermal conductivity matrix has no entries — does "
                "the model contain any thermal elements?"
            )
        T_free = spla.spsolve(K, f)
        # Scatter back to nodes
        for node in m.nodes.values():
            eq = int(node.eqn[0])
            T_value = float(T_free[eq]) if eq >= 0 else float(node.disp[0])
            node.disp[0] = T_value
        # Build a full temperature vector (one per node, by tag order)
        T_full = np.array([m.node(tag).disp[0]
                            for tag in sorted(m.nodes)])
        return SteadyHeatResult(T=T_full)


# ============================================================ transient

@dataclass
class TransientHeatResult:
    """Result of a transient heat-conduction analysis."""

    times: np.ndarray
    T: np.ndarray                  # shape (n_steps, n_nodes)


class TransientHeatAnalysis:
    """Generalized-trapezoidal time-marching of
    ``C_T dT/dt + K_T T = f_T(t)``.

    The recurrence::

        (C + theta*dt*K) T_{n+1}
            = (C - (1-theta)*dt*K) T_n
              + dt * (theta * f_{n+1} + (1-theta) * f_n)

    is solved at each time step. ``theta = 0.5`` is Crank-Nicolson
    (second-order accurate, unconditionally stable);
    ``theta = 1.0`` is fully implicit (first-order, dissipative).

    Parameters
    ----------
    model : Model
    num_steps : int
    dt : float
    theta : float, default 0.5
    T0 : array-like or float, default 0.0
        Initial temperature at each node.
    load_function : callable, optional
        ``f_T(t)`` returning the global force vector at time ``t``.
        If omitted, the model's static nodal loads are used at every
        step.
    """

    def __init__(
        self,
        model,
        *,
        num_steps: int,
        dt: float,
        theta: float = 0.5,
        T0=0.0,
        load_function=None,
    ):
        if model.ndf != 1:
            raise ValueError(
                "TransientHeatAnalysis requires Model(ndf=1)"
            )
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must be in [0, 1]")
        self.model = model
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.theta = float(theta)
        self.T0 = T0
        self.load_function = load_function

    def run(self) -> TransientHeatResult:
        m = self.model
        m.number_dofs()
        K = assemble_stiffness(m).tocsc()
        C = assemble_mass(m).tocsc()
        f_static = assemble_force(m)

        neq = m.neq
        # Initial temperature
        if np.isscalar(self.T0):
            T = np.full(neq, float(self.T0))
        else:
            T = np.asarray(self.T0, dtype=float).ravel()
            if T.size != neq:
                raise ValueError(
                    f"T0 must be a scalar or array of size neq={neq}, "
                    f"got size {T.size}"
                )

        # Pre-factor the iteration operator (C + theta*dt*K)
        A = (C + self.theta * self.dt * K).tocsc()
        solver = spla.splu(A)
        B = C - (1.0 - self.theta) * self.dt * K

        times = np.zeros(self.num_steps + 1)
        T_hist = np.zeros((self.num_steps + 1, neq))
        T_hist[0] = T

        for n in range(self.num_steps):
            t_n = n * self.dt
            t_np1 = (n + 1) * self.dt
            if self.load_function is None:
                f_n = f_static
                f_np1 = f_static
            else:
                f_n = self.load_function(t_n)
                f_np1 = self.load_function(t_np1)
            rhs = B @ T + self.dt * (self.theta * f_np1
                                       + (1.0 - self.theta) * f_n)
            T = solver.solve(rhs)
            times[n + 1] = t_np1
            T_hist[n + 1] = T

        # Scatter final temperature back to nodes
        for node in m.nodes.values():
            eq = int(node.eqn[0])
            if eq >= 0:
                node.disp[0] = float(T[eq])

        return TransientHeatResult(times=times, T=T_hist)
