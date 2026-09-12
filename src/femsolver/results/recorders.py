"""Recorders for nonlinear section / element / node response (plan G8).

A *recorder* logs one quantity's history across the steps of an analysis and
writes it to CSV -- the femsolver counterpart of Midas ``bMONITOR`` / CSI
``SaveFibResp``. Three cover the fiber-hinge workflow:

* :class:`SectionRecorder` -- section resultants + strain state (``N, Mz[, My]``
  and ``eps_a, kappa``) per step, for a stateful fiber section.
* :class:`FiberRecorder` -- per-fiber ``(y, z, eps, sigma)`` per step (long
  format), for drawing the fiber stress/strain contour or a single bar's
  hysteresis.
* :class:`NodeRecorder` -- a monitored node's displacement and reaction DOFs
  per step, for pushover / cyclic force-displacement curves.

All three are plain data loggers: you call ``record(...)`` once per committed
step and ``to_csv(path)`` at the end. Reading a quantity re-evaluates the
(already-committed) section at its committed strain, so recording never advances
or corrupts material history. The stress read is the committed stress; the
tangent is not logged.
"""
from __future__ import annotations

import csv

import numpy as np


class SectionRecorder:
    """Log a stateful section's resultants and strain state per step.

    Parameters
    ----------
    section :
        A section exposing ``get_response(e) -> (s, ks)`` where ``e`` is the
        generalized strain (``[eps_a, kappa_z]`` in 2-D; ``[eps_a, kappa_z,
        kappa_y, phi]`` in 3-D) and ``s`` the work-conjugate resultants
        (``[N, Mz]`` / ``[N, Mz, My, T]``).
    name : str, optional
        Label used as the default CSV stem.
    """

    def __init__(self, section, *, name: str = "section"):
        self.section = section
        self.name = name
        self.rows: list[dict] = []

    def record(self, e, *, step=None) -> dict:
        """Append one row for the generalized strain ``e`` (the section is
        re-evaluated at ``e`` to read the resultants; non-committing)."""
        e = np.asarray(e, dtype=float).ravel()
        s, _ = self.section.get_response(e)
        s = np.asarray(s, dtype=float).ravel()
        row = {"step": len(self.rows) if step is None else step,
               "eps_a": float(e[0]), "kappa_z": float(e[1]),
               "N": float(s[0]), "Mz": float(s[1])}
        if e.size >= 3:                       # 3-D: biaxial curvature
            row["kappa_y"] = float(e[2])
        if s.size >= 3:
            row["My"] = float(s[2])
        if s.size >= 4:
            row["T"] = float(s[3])
        self.rows.append(row)
        return row

    @property
    def columns(self) -> list[str]:
        return list(self.rows[0].keys()) if self.rows else [
            "step", "eps_a", "kappa_z", "N", "Mz"]

    def arrays(self) -> dict:
        """Column name -> numpy array (handy for tests and plotting)."""
        cols = self.columns
        return {c: np.array([r.get(c, np.nan) for r in self.rows])
                for c in cols}

    def to_csv(self, path) -> None:
        _write_csv(path, self.columns, self.rows)


class FiberRecorder:
    """Log per-fiber ``(y, z, eps, sigma)`` per step, in long (tidy) format.

    Parameters
    ----------
    section :
        A fiber section exposing a ``fibers`` list of objects with ``y, z,
        area, material`` (``FiberSection2D`` / ``FiberSection3D``).
    fiber_ids : sequence of int, optional
        Indices of the fibers to record; default all. Pass a subset (e.g. the
        rebar fibers) to keep the file small.
    name : str, optional
        Label used as the default CSV stem.
    """

    def __init__(self, section, *, fiber_ids=None, name: str = "fibers"):
        self.section = section
        self.name = name
        n = len(section.fibers)
        self.fiber_ids = (list(range(n)) if fiber_ids is None
                          else [int(i) for i in fiber_ids])
        self.rows: list[dict] = []

    def record(self, e, *, step=None) -> None:
        """Append one row per recorded fiber for the generalized strain ``e``.

        Fiber strain follows the section kinematics
        ``eps_f = eps_a - y*kappa_z + z*kappa_y``; stress is read from each
        fiber's material at that (committed) strain."""
        e = np.asarray(e, dtype=float).ravel()
        eps_a = float(e[0])
        kz = float(e[1]) if e.size >= 2 else 0.0
        ky = float(e[2]) if e.size >= 3 else 0.0
        st = len(self.rows) if step is None else step
        for idx in self.fiber_ids:
            f = self.section.fibers[idx]
            eps_f = eps_a - f.y * kz + f.z * ky
            sigma, _ = f.material.get_response(eps_f)
            self.rows.append({"step": st, "fiber": idx,
                              "y": float(f.y), "z": float(f.z),
                              "eps": float(eps_f), "sigma": float(sigma)})

    @property
    def columns(self) -> list[str]:
        return ["step", "fiber", "y", "z", "eps", "sigma"]

    def to_csv(self, path) -> None:
        _write_csv(path, self.columns, self.rows)


class NodeRecorder:
    """Log a monitored node's displacement and reaction DOFs per step.

    Parameters
    ----------
    node :
        A node exposing ``disp`` and ``reaction`` arrays (length ``ndf``) and a
        ``tag``.
    dofs : sequence of int, optional
        DOF indices to record (0-based); default all ``ndf``.
    name : str, optional
        Label used as the default CSV stem.
    """

    def __init__(self, node, *, dofs=None, name: str | None = None):
        self.node = node
        self.name = name or f"node{getattr(node, 'tag', '')}"
        self.dofs = (list(range(node.ndf)) if dofs is None
                     else [int(d) for d in dofs])
        self.rows: list[dict] = []

    def record(self, *, step=None) -> dict:
        """Snapshot the node's current ``disp``/``reaction`` at the recorded
        DOFs (call after a step is committed)."""
        row = {"step": len(self.rows) if step is None else step}
        for d in self.dofs:
            row[f"u{d}"] = float(self.node.disp[d])
        for d in self.dofs:
            row[f"R{d}"] = float(self.node.reaction[d])
        self.rows.append(row)
        return row

    @property
    def columns(self) -> list[str]:
        return (["step"] + [f"u{d}" for d in self.dofs]
                + [f"R{d}" for d in self.dofs])

    def arrays(self) -> dict:
        cols = self.columns
        return {c: np.array([r.get(c, np.nan) for r in self.rows])
                for c in cols}

    def to_csv(self, path) -> None:
        _write_csv(path, self.columns, self.rows)


def _write_csv(path, columns, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
