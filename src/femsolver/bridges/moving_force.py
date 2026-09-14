"""Moving-load time-history / vehicle–bridge interaction (bridge plan T2.1).

Static influence lines (T1.1/T1.2) give the *worst-position* live-load effect
but not the **dynamic** effect of traffic actually crossing at speed — the
dynamic amplification (impact) factor, and, for regularly-spaced axles at a
critical speed, **resonance** (the governing check for high-speed rail,
EN 1991-2 HSLM).

This module marches a train of moving forces across the bridge and integrates
the structural dynamics (Newmark, via :class:`TransientAnalysis`):

* :class:`VehicleAxles` — a train of constant axle forces + spacings.
* :class:`MovingForceAnalysis` — places the axles at their time-varying
  positions (consistent cubic-Hermite nodal loads so a force between nodes is
  applied exactly), runs the transient solve, and reports the dynamic response
  history alongside the quasi-static history and the **dynamic amplification
  factor** ``DAF = max|dynamic| / max|static|``.

This is the *moving-force* model (the workhorse "moving load — dynamic"
analysis in MIDAS / CSiBridge). The sprung-mass coupled interaction (true VBI
with vehicle DOFs) is the T2.1b follow-up.

Scope: 2-D girder-line models (``ndm=2, ndf=3``), a horizontal lane with a
vertical (transverse) moving load — the standard moving-load beam model.
Bridge self-mass comes from the material density, so give the deck material a
non-zero ``rho``.

References
----------
* Frýba, L. *Vibration of Solids and Structures under Moving Loads.*
* Yang, Y.B., Yau, J.D., Wu, Y.S. *Vehicle–Bridge Interaction Dynamics.*
* EN 1991-2 §6.4 (dynamic analysis, HSLM).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from femsolver.analysis.assembler import assemble_stiffness
from femsolver.analysis.damping import RayleighDamping
from femsolver.analysis.transient import TransientAnalysis
from femsolver.bridges.moving_load import Lane


@dataclass
class VehicleAxles:
    """A train of constant moving axle forces.

    Parameters
    ----------
    axle_loads : array-like (k,)
        Axle forces (N), positive downward magnitudes.
    axle_offsets : array-like (k,)
        Longitudinal position of each axle relative to the lead axle (m),
        non-decreasing; ``0`` is the lead (front) axle, positive values are
        axles behind it.
    name : str
    """

    axle_loads: np.ndarray
    axle_offsets: np.ndarray
    name: str = "vehicle"

    def __post_init__(self) -> None:
        self.axle_loads = np.asarray(self.axle_loads, dtype=float).ravel()
        self.axle_offsets = np.asarray(self.axle_offsets, dtype=float).ravel()
        if self.axle_loads.size != self.axle_offsets.size:
            raise ValueError("axle_loads and axle_offsets length mismatch")
        if self.axle_loads.size == 0:
            raise ValueError("need at least one axle")

    @property
    def length(self) -> float:
        return float(self.axle_offsets.max() - self.axle_offsets.min())


def _hermite(xi: float, L: float):
    """Cubic-Hermite shape functions for a transverse unit load at ``xi`` on a
    beam of length ``L``. Order: ``[v_a, θ_a, v_b, θ_b]``."""
    return (1 - 3 * xi ** 2 + 2 * xi ** 3,
            L * (xi - 2 * xi ** 2 + xi ** 3),
            3 * xi ** 2 - 2 * xi ** 3,
            L * (-xi ** 2 + xi ** 3))


class MovingForceAnalysis:
    """Time-history of a moving axle train crossing a 2-D girder.

    Parameters
    ----------
    model : Model
        2-D frame (``ndf=3``) with mass (material ``rho`` > 0) and supports.
    lane : Lane
        The path (ordered nodes + stations) the load travels; consecutive lane
        nodes must be joined by a beam element. ``load_dof`` is the vertical
        translation (1) and ``gravity_sign`` its direction.
    vehicle : VehicleAxles
    speed : float
        Travel speed (m/s), > 0.
    track : (int, int)
        ``(node_tag, dof)`` response to record.
    dt : float, optional
        Time step. Default: cross time / 400 (auto).
    damping : RayleighDamping, optional
    free_vibration_time : float, default 0.0
        Extra time to march after the train leaves (to catch peak free
        vibration).
    """

    def __init__(self, model, lane: Lane, vehicle: VehicleAxles, speed: float,
                 *, track, dt: float | None = None,
                 damping: RayleighDamping | None = None,
                 free_vibration_time: float = 0.0):
        if model.ndm != 2 or model.ndf != 3:
            raise ValueError("MovingForceAnalysis supports 2-D frames "
                             "(ndm=2, ndf=3) only")
        if speed <= 0.0:
            raise ValueError("speed must be > 0")
        self.model = model
        self.lane = lane
        self.vehicle = vehicle
        self.speed = float(speed)
        self.track = track
        self.damping = damping
        self.free_vibration_time = float(free_vibration_time)

        self._stations = lane.resolve_stations(model)
        self._nodes = list(lane.node_tags)
        self._load_dof = lane.load_dof
        self._rot_dof = lane.load_dof + 1
        self._gsign = lane.gravity_sign

        s_min, s_max = float(self._stations[0]), float(self._stations[-1])
        # front axle enters at s_min → rear axle leaves at s_max
        self._s0 = s_min - float(vehicle.axle_offsets.min())
        travel = (s_max - s_min) + vehicle.length
        self._cross_time = travel / self.speed
        self.dt = float(dt) if dt else self._cross_time / 400.0

    # ------------------------------------------------------- load assembly
    def _axle_stations(self, t: float) -> np.ndarray:
        # lead axle (offset 0) at s0 + v t; an axle at offset o sits behind it
        return self._s0 + self.speed * t - self.vehicle.axle_offsets \
            + self.vehicle.axle_offsets.max()

    def _force_vector(self, t: float) -> np.ndarray:
        m = self.model
        F = np.zeros(m.neq)
        stations = self._axle_stations(t)
        for P, s in zip(self.vehicle.axle_loads, stations):
            self._distribute(F, float(s), float(P))
        return F

    def _distribute(self, F: np.ndarray, s: float, P: float) -> None:
        st = self._stations
        if s < st[0] or s > st[-1]:
            return                                  # axle off the bridge
        i = int(np.searchsorted(st, s, side="right") - 1)
        i = min(max(i, 0), len(st) - 2)
        s_a, s_b = st[i], st[i + 1]
        L = s_b - s_a
        if L <= 0:
            return
        xi = (s - s_a) / L
        N1, N2, N3, N4 = _hermite(xi, L)
        force = self._gsign * P
        m = self.model
        na, nb = m.node(self._nodes[i]), m.node(self._nodes[i + 1])
        for node, (Nt, Nr) in ((na, (N1, N2)), (nb, (N3, N4))):
            eqt = int(node.eqn[self._load_dof])
            eqr = int(node.eqn[self._rot_dof])
            if eqt >= 0:
                F[eqt] += force * Nt
            if eqr >= 0:
                F[eqr] += force * Nr

    # --------------------------------------------------------------- run
    def run(self) -> dict:
        m = self.model
        m.number_dofs()
        from femsolver.analysis.assembler import assemble_mass
        if abs(assemble_mass(m)).max() <= 0.0:
            raise ValueError(
                "the model has no mass — a moving-load time-history needs the "
                "deck material to carry a density (rho > 0).")
        total_time = self._cross_time + self.free_vibration_time
        num_steps = max(1, int(round(total_time / self.dt)))

        # --- dynamic pass (Newmark via TransientAnalysis) ---
        m.reset_results()
        dyn = TransientAnalysis(
            m, num_steps=num_steps, dt=self.dt, damping=self.damping,
            load_function=self._force_vector, track=self.track).run()

        # --- quasi-static pass (factorise once, solve each position) ---
        m.reset_results()
        m.number_dofs()
        K = assemble_stiffness(m)
        lu = splu(K.tocsc())
        tag, dof = self.track
        eq = int(m.node(tag).eqn[dof])
        static = []
        for t in dyn["times"]:
            u = lu.solve(self._force_vector(t))
            static.append(float(u[eq]) if eq >= 0 else 0.0)

        d_dyn = np.asarray(dyn["tracked_disp"])
        d_sta = np.asarray(static)
        peak_dyn = float(np.max(np.abs(d_dyn)))
        peak_sta = float(np.max(np.abs(d_sta)))
        daf = peak_dyn / peak_sta if peak_sta > 0 else float("nan")
        return {
            "times": dyn["times"],
            "dynamic_disp": d_dyn.tolist(),
            "static_disp": d_sta.tolist(),
            "peak_dynamic": peak_dyn,
            "peak_static": peak_sta,
            "DAF": daf,
            "dt": self.dt,
            "cross_time": self._cross_time,
            "speed": self.speed,
        }
