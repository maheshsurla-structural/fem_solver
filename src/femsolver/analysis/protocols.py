"""Displacement / load protocol generators (plan G7).

Each function returns a 1-D :class:`numpy.ndarray` of *target values* (a
displacement or load history) to drive an analysis step by step -- the
femsolver counterpart of Midas time-history functions and CSI cyclic load
cases. Feed the returned array to a stepped driver (e.g. pair consecutive
targets with :class:`~femsolver.analysis.static_integrator.DisplacementControl`
increments, or apply each as a load target).

* :func:`monotonic`      -- a straight ramp ``0 -> target``.
* :func:`stepped_cyclic` -- symmetric push/pull cycles of growing amplitude
  (the ``+-0.25, +-0.5, +-0.75, +-1.0`` protocol of the benchmark).
* :func:`from_time_function` -- resample an arbitrary ``(t, v)`` function onto
  a uniform ``dt`` grid.
"""
from __future__ import annotations

import numpy as np


def monotonic(target: float, n_steps: int) -> np.ndarray:
    """A monotonic ramp from 0 to ``target`` in ``n_steps`` increments.

    Returns ``n_steps + 1`` points (including the 0 start), so successive
    differences are ``target / n_steps``."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return np.linspace(0.0, float(target), n_steps + 1)


def stepped_cyclic(
    amplitudes=(0.25, 0.5, 0.75, 1.0),
    *,
    scale: float = 1.0,
    cycles: int = 1,
    pts_per_cycle: int = 40,
) -> np.ndarray:
    """Symmetric reversed-cyclic target history of growing amplitude.

    For each amplitude ``a`` in ``amplitudes`` (times ``scale``) run ``cycles``
    full cycles ``0 -> +a -> -a -> 0``, each sampled with ``pts_per_cycle``
    points. The segments are concatenated into one continuous history that
    starts and ends at 0, with no duplicated junction points.

    Parameters
    ----------
    amplitudes : sequence of float
        Peak targets (relative units unless ``scale`` is set), typically the
        drift/ductility levels of the protocol.
    scale : float, default 1.0
        Multiplier applied to every amplitude (e.g. a reference displacement,
        so ``amplitudes`` are drift ratios).
    cycles : int, default 1
        Number of full cycles at each amplitude.
    pts_per_cycle : int, default 40
        Sample points per full cycle (a multiple of 4 gives exact peaks).
    """
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    if pts_per_cycle < 4:
        raise ValueError("pts_per_cycle must be >= 4")
    q = pts_per_cycle // 4
    hist = [0.0]
    for a in amplitudes:
        peak = float(a) * scale
        for _ in range(cycles):
            # 0 -> +peak -> 0 -> -peak -> 0, dropping each segment's first
            # point (already the previous segment's last) to avoid duplicates
            seg = np.concatenate([
                np.linspace(0.0, peak, q + 1)[1:],
                np.linspace(peak, 0.0, q + 1)[1:],
                np.linspace(0.0, -peak, q + 1)[1:],
                np.linspace(-peak, 0.0, q + 1)[1:],
            ])
            hist.extend(seg.tolist())
    return np.asarray(hist, dtype=float)


def from_time_function(times, values, dt: float) -> np.ndarray:
    """Resample an arbitrary ``(times, values)`` function onto a uniform grid
    of spacing ``dt`` (from ``times[0]`` to ``times[-1]``), by linear
    interpolation -- e.g. to drive a quasi-static sweep from a recorded
    time-history function.

    Returns the interpolated value history (the time grid is
    ``times[0] + dt*arange(n)``)."""
    times = np.asarray(times, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if times.size != values.size:
        raise ValueError("times and values must have the same length")
    if times.size < 2:
        raise ValueError("need at least two (t, v) points")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    n = int(np.floor((times[-1] - times[0]) / dt)) + 1
    grid = times[0] + dt * np.arange(n)
    return np.interp(grid, times, values)
