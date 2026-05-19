"""Nodes carry coordinates, DOF state, equation numbers, displacements, and reactions."""
from __future__ import annotations

import numpy as np


class Node:
    __slots__ = (
        "tag", "coords", "ndf", "fixity", "eqn", "disp", "reaction", "_load",
        "mode_disp",
        # Transient-analysis state: populated by TransientAnalysis at each
        # time step. Zero by default (also serves as initial conditions —
        # set them before running TransientAnalysis to give non-zero ICs).
        "velocity", "acceleration",
    )

    def __init__(self, tag: int, coords: np.ndarray, ndf: int):
        self.tag = int(tag)
        self.coords = np.asarray(coords, dtype=float).ravel()
        self.ndf = int(ndf)
        self.fixity = np.zeros(ndf, dtype=bool)
        self.eqn = np.full(ndf, -1, dtype=np.int64)
        self.disp = np.zeros(ndf, dtype=float)
        self.reaction = np.zeros(ndf, dtype=float)
        self._load = np.zeros(ndf, dtype=float)
        # mode_disp[i, k] = i-th DOF amplitude in k-th mode; populated by EigenAnalysis
        self.mode_disp = np.zeros((ndf, 0), dtype=float)
        # Transient state — defaults are zero (at rest).
        self.velocity = np.zeros(ndf, dtype=float)
        self.acceleration = np.zeros(ndf, dtype=float)

    def fix(self, mask) -> None:
        m = np.asarray(mask, dtype=int).ravel()
        if m.size != self.ndf:
            raise ValueError(f"fixity mask must have length {self.ndf}, got {m.size}")
        self.fixity |= m.astype(bool)

    def free(self, mask) -> None:
        m = np.asarray(mask, dtype=int).ravel()
        self.fixity &= ~m.astype(bool)

    def add_load(self, load) -> None:
        v = np.asarray(load, dtype=float).ravel()
        if v.size != self.ndf:
            raise ValueError(f"nodal load must have length {self.ndf}, got {v.size}")
        self._load += v

    @property
    def load(self) -> np.ndarray:
        return self._load.copy()

    def reset_results(self) -> None:
        self.disp[:] = 0.0
        self.reaction[:] = 0.0
        self.velocity[:] = 0.0
        self.acceleration[:] = 0.0

    def __repr__(self) -> str:
        return f"Node(tag={self.tag}, coords={self.coords.tolist()}, ndf={self.ndf})"
