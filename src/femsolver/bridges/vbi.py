"""Sprung-mass vehicle–bridge interaction (bridge plan T2.1b).

The moving-*force* model (:mod:`femsolver.bridges.moving_force`) treats each
axle as a constant force. True **vehicle–bridge interaction (VBI)** gives the
vehicle its own dynamics: a sprung mass on a suspension (spring + damper) whose
wheel stays in contact with the deck, so the contact force depends on the
*relative* motion of vehicle and bridge. This captures suspension resonance,
ride response, and the more accurate dynamic amplification used for high-speed
rail / precise impact studies.

Each vehicle is a quarter-car sprung mass (:class:`SprungMassVehicle`).
:class:`VBIAnalysis` solves the **coupled** bridge + vehicle system directly
(monolithic average-acceleration Newmark) — no per-step iteration. With state
``z = [u (bridge, +y up); d_s (sprung mass, +down)]`` the coupled system is,
for one vehicle whose contact shape-function vector is ``N`` (``w_c = Nᵀu``):

    Mc z¨ + Cc z˙ + Kc z = f,     f = [0; m_s g]
    Kc = [[K + k_s N Nᵀ,  k_s N ],   Cc = [[C + c_s N Nᵀ,  c_s N ],
          [ k_s Nᵀ,        k_s  ]]         [ c_s Nᵀ,         c_s  ]]

symmetric, with the coupling blocks rebuilt each step as the contact point
moves. The static solution reproduces the bridge under the moving weight plus
the suspension's static compression, so the run starts jolt-free.

Scope: 2-D girder line (``ndm=2, ndf=3``); bridge mass from material ``rho``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.analysis.assembler import assemble_mass, assemble_stiffness
from femsolver.analysis.damping import RayleighDamping
from femsolver.bridges.moving_force import _hermite
from femsolver.bridges.moving_load import Lane

_G = 9.80665


@dataclass
class SprungMassVehicle:
    """A quarter-car sprung-mass vehicle.

    Parameters
    ----------
    mass : float
        Sprung mass ``m_s`` (kg).
    stiffness : float
        Suspension stiffness ``k_s`` (N/m).
    damping : float, default 0.0
        Suspension damping ``c_s`` (N·s/m).
    offset : float, default 0.0
        Longitudinal offset of this vehicle behind the lead reference (m).
    name : str
    """

    mass: float
    stiffness: float
    damping: float = 0.0
    offset: float = 0.0
    name: str = "vehicle"

    @property
    def natural_frequency(self) -> float:
        """Bounce frequency ``(1/2π)·√(k_s/m_s)`` (Hz)."""
        return float(np.sqrt(self.stiffness / self.mass) / (2 * np.pi))


class VBIAnalysis:
    """Coupled sprung-mass vehicle–bridge interaction on a 2-D girder.

    Parameters
    ----------
    model : Model
        2-D frame (``ndf=3``) with mass (material ``rho`` > 0) and supports.
    lane : Lane
        Path (ordered nodes + stations); ``load_dof`` = vertical translation.
    vehicles : SprungMassVehicle or list
    speed : float
        Travel speed (m/s), > 0.
    track : (int, int)
        ``(node_tag, dof)`` bridge response to record.
    dt : float, optional
        Time step (default: cross time / 400).
    bridge_damping : RayleighDamping, optional
    free_vibration_time : float, default 0.0
    g : float, default 9.80665
    """

    def __init__(self, model, lane: Lane, vehicles, speed: float, *, track,
                 dt: float | None = None,
                 bridge_damping: RayleighDamping | None = None,
                 free_vibration_time: float = 0.0, g: float = _G):
        if model.ndm != 2 or model.ndf != 3:
            raise ValueError("VBIAnalysis supports 2-D frames (ndf=3) only")
        if speed <= 0.0:
            raise ValueError("speed must be > 0")
        self.model = model
        self.lane = lane
        self.vehicles = ([vehicles] if isinstance(vehicles, SprungMassVehicle)
                         else list(vehicles))
        if not self.vehicles:
            raise ValueError("need at least one vehicle")
        self.speed = float(speed)
        self.track = track
        self.bridge_damping = bridge_damping
        self.free_vibration_time = float(free_vibration_time)
        self.g = float(g)

        self._stations = np.asarray(lane.resolve_stations(model), dtype=float)
        self._nodes = list(lane.node_tags)
        self._ld = lane.load_dof
        self._rd = lane.load_dof + 1

        s_min, s_max = float(self._stations[0]), float(self._stations[-1])
        offs = np.array([v.offset for v in self.vehicles])
        self._s0 = s_min - float(offs.min())
        travel = (s_max - s_min) + float(offs.max() - offs.min())
        self._cross_time = travel / self.speed
        self.dt = float(dt) if dt else self._cross_time / 400.0

    # -------------------------------------------------- contact shape vector
    def _contact_N(self, s: float) -> np.ndarray:
        """Shape-function vector ``N`` (size neq) with ``w_c = Nᵀ u`` — the
        bridge vertical deflection at station ``s``. Zero if off the deck."""
        N = np.zeros(self.model.neq)
        st = self._stations
        if s < st[0] or s > st[-1]:
            return N
        i = int(np.searchsorted(st, s, side="right") - 1)
        i = min(max(i, 0), len(st) - 2)
        s_a, s_b = st[i], st[i + 1]
        L = s_b - s_a
        if L <= 0:
            return N
        xi = (s - s_a) / L
        N1, N2, N3, N4 = _hermite(xi, L)
        m = self.model
        for node, val in ((m.node(self._nodes[i]), (N1, N2)),
                          (m.node(self._nodes[i + 1]), (N3, N4))):
            eqt = int(node.eqn[self._ld])
            eqr = int(node.eqn[self._rd])
            if eqt >= 0:
                N[eqt] += val[0]
            if eqr >= 0:
                N[eqr] += val[1]
        return N

    def _vehicle_station(self, veh, t: float) -> float:
        offs = np.array([v.offset for v in self.vehicles])
        return self._s0 + self.speed * t + offs.max() - veh.offset

    # --------------------------------------------------------------- run
    def run(self) -> dict:
        m = self.model
        m.number_dofs()
        nb = m.neq
        if nb == 0:
            raise RuntimeError("bridge has no free DOFs")
        M_b = np.asarray(assemble_mass(m).todense())
        if np.max(np.abs(M_b)) <= 0.0:
            raise ValueError("the bridge has no mass — give the deck material "
                             "a density (rho > 0).")
        K_b = np.asarray(assemble_stiffness(m).todense())
        C_b = (np.asarray(self.bridge_damping.build(assemble_mass(m),
                                                    assemble_stiffness(m))
                          .todense())
               if self.bridge_damping is not None else np.zeros((nb, nb)))

        nv = len(self.vehicles)
        n = nb + nv
        ms = np.array([v.mass for v in self.vehicles])
        ks = np.array([v.stiffness for v in self.vehicles])
        cs = np.array([v.damping for v in self.vehicles])

        M = np.zeros((n, n))
        M[:nb, :nb] = M_b
        for j in range(nv):
            M[nb + j, nb + j] = ms[j]
        f = np.zeros(n)
        for j in range(nv):
            f[nb + j] = ms[j] * self.g

        def _system(t):
            K = np.zeros((n, n))
            C = np.zeros((n, n))
            K[:nb, :nb] = K_b
            C[:nb, :nb] = C_b
            for j, veh in enumerate(self.vehicles):
                N = self._contact_N(self._vehicle_station(veh, t))
                K[:nb, :nb] += ks[j] * np.outer(N, N)
                C[:nb, :nb] += cs[j] * np.outer(N, N)
                K[:nb, nb + j] += ks[j] * N
                K[nb + j, :nb] += ks[j] * N
                C[:nb, nb + j] += cs[j] * N
                C[nb + j, :nb] += cs[j] * N
                K[nb + j, nb + j] += ks[j]
                C[nb + j, nb + j] += cs[j]
            return K, C

        # --- initial state: static solve of the coupled system at t=0 ---
        K0, _ = _system(0.0)
        z = np.linalg.solve(K0, f)
        zd = np.zeros(n)
        zdd = np.zeros(n)

        dt = self.dt
        beta, gamma = 0.25, 0.5
        total_time = self._cross_time + self.free_vibration_time
        nsteps = max(1, int(round(total_time / dt)))
        tag, dof = self.track
        eqb = int(m.node(tag).eqn[dof])

        times = [0.0]
        bridge_hist = [float(z[eqb]) if eqb >= 0 else 0.0]
        contact_hist = [[float(ms[j] * self.g) for j in range(nv)]]

        for step in range(1, nsteps + 1):
            t = step * dt
            K, C = _system(t)
            z_pred = z + dt * zd + dt * dt * (0.5 - beta) * zdd
            zd_pred = zd + dt * (1.0 - gamma) * zdd
            A = M + gamma * dt * C + beta * dt * dt * K
            rhs = f - C @ zd_pred - K @ z_pred
            zdd = np.linalg.solve(A, rhs)
            z = z_pred + beta * dt * dt * zdd
            zd = zd_pred + gamma * dt * zdd

            times.append(t)
            bridge_hist.append(float(z[eqb]) if eqb >= 0 else 0.0)
            # contact force = m_s (g - d¨_s)   [downward +]
            contact_hist.append([float(ms[j] * (self.g - zdd[nb + j]))
                                 for j in range(nv)])

        # --- static reference (moving weights) for the DAF ---
        static = self._static_pass(times, K_b, eqb)
        d_dyn = np.asarray(bridge_hist)
        peak_dyn = float(np.max(np.abs(d_dyn)))
        peak_sta = float(np.max(np.abs(static)))
        return {
            "times": times,
            "bridge_disp": bridge_hist,
            "static_disp": static.tolist(),
            "contact_force": contact_hist,
            "peak_dynamic": peak_dyn,
            "peak_static": peak_sta,
            "DAF": peak_dyn / peak_sta if peak_sta > 0 else float("nan"),
            "dt": dt,
            "cross_time": self._cross_time,
            "speed": self.speed,
        }

    def _static_pass(self, times, K_b, eqb) -> np.ndarray:
        from scipy.sparse.linalg import splu
        import scipy.sparse as sp
        lu = splu(sp.csc_matrix(K_b))
        out = []
        for t in times:
            F = np.zeros(self.model.neq)
            for veh in self.vehicles:
                N = self._contact_N(self._vehicle_station(veh, t))
                F += -(veh.mass * self.g) * N        # downward weight
            u = lu.solve(F)
            out.append(float(u[eqb]) if eqb >= 0 else 0.0)
        return np.asarray(out)
