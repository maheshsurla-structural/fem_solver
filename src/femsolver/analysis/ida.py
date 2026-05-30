"""Incremental Dynamic Analysis (Vamvatsikos & Cornell 2002).

For a single ground-motion record (or a suite of them), IDA:

1. Scales the record by a series of intensity-measure (IM) values --
   typically the spectral acceleration ``Sa(T_1)`` at the structure's
   fundamental period, or PGA for a first-pass screening.
2. Runs a nonlinear transient analysis at each scaled level on a
   **fresh** model.
3. Extracts one or more engineering demand parameters (EDPs) from
   each run -- typically the maximum interstory drift ratio.
4. Produces an IM-EDP curve per record.

Collapse detection (Phase 25.2) and multi-record orchestration
(Phase 25.2) build on top of this single-record driver.

This module provides:

* :class:`IDAPoint` -- one (IM, EDPs) pair from one NLTHA run.
* :class:`IDARecord` -- full IM-EDP curve for one record.
* :class:`IDADriver` -- the single-record sweep orchestrator.

Design choices
--------------
* **Model factory pattern** -- the user passes a callable that
  returns a *fresh* Model. Each IDA point starts from undeformed
  state (consistent with the Vamvatsikos-Cornell single-record
  definition: each scale level is an independent NLTHA).
* **IM-to-scale-factor pluggable** -- the user provides a
  ``scale_fn(target_IM, record_descriptor)`` callable. For PGA,
  this is just ``target_IM / max(|a_g|)``. For Sa(T_1), the user
  computes the record's spectral acceleration once and reuses the
  resulting scaling. A built-in :func:`pga_scale_factor` covers
  the most common case.
* **EDP extractor pluggable** -- ``edp_extractor(model)`` is called
  on the post-run model and returns a dict of EDP name -> value.
  A built-in :func:`max_drift_edp` covers max-story-drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from femsolver.analysis.nonlinear_transient import NonlinearTransientAnalysis
from femsolver.analysis.response_spectrum import ground_motion_force


# ============================================================ result types

@dataclass
class IDAPoint:
    """One (IM, EDPs) pair from one NLTHA run.

    Attributes
    ----------
    IM : float
        Intensity measure value (e.g., PGA in m/s², or Sa(T_1) in g).
    scale_factor : float
        Multiplier applied to the record to reach this IM.
    EDPs : dict[str, float]
        EDP name -> value. ``"max_drift_ratio"`` is the convention
        for the primary collapse-driving EDP.
    converged : bool
        Did NLTHA reach the end of the record without
        non-convergence? A False here is a strong collapse
        indicator (Phase 25.2).
    n_steps_completed : int
    """

    IM: float
    scale_factor: float
    EDPs: dict
    converged: bool
    n_steps_completed: int = 0


@dataclass
class IDARecord:
    """IM-EDP curve for one ground-motion record under IDA.

    Attributes
    ----------
    record_name : str
    points : list[IDAPoint]
        Sorted by ascending IM.
    """

    record_name: str
    points: list = field(default_factory=list)

    def IMs(self) -> np.ndarray:
        return np.array([p.IM for p in self.points])

    def EDP_array(self, edp_name: str = "max_drift_ratio") -> np.ndarray:
        return np.array([p.EDPs.get(edp_name, np.nan)
                         for p in self.points])


# ============================================================ scaling helpers

def pga_scale_factor(accel_function, t_end: float, dt: float):
    """Return a closure ``scale_fn(target_IM_pga)`` that computes the
    scale factor needed to bring the record's PGA to ``target_IM_pga``.

    Parameters
    ----------
    accel_function : callable
        Ground acceleration time history ``a_g(t)``.
    t_end : float
        Total record duration (s).
    dt : float
        Sample spacing (s).
    """
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    a_g = np.array([accel_function(t) for t in times])
    pga_record = float(np.max(np.abs(a_g)))
    if pga_record <= 0.0:
        raise ValueError("record has zero PGA; cannot scale")

    def scale_fn(target_pga: float) -> float:
        return target_pga / pga_record

    return scale_fn


# ============================================================ EDP extractors

def max_drift_edp(
    story_node_tags,
    *,
    direction: int = 0,
    base_node_tag: int | None = None,
    story_heights=None,
):
    """Return an EDP extractor callable that tracks maximum interstory
    drift ratio across the analysis.

    Parameters
    ----------
    story_node_tags : sequence of int
        Representative node at each story, ordered lowest to roof.
    direction : int, default 0
        DOF index for the drift component.
    base_node_tag : int, optional
        Base reference node.
    story_heights : sequence of float, optional
        Per-story heights (m). If omitted, computed from node
        coordinates assuming the y-axis is vertical.

    Returns
    -------
    callable
        ``f(model) -> dict[str, float]`` returning at least
        ``"max_drift_ratio"`` and ``"max_roof_drift"``.

    Notes
    -----
    The extractor inspects the final state of the model after NLTHA.
    For a true time-history maximum the user should record drift at
    every step, but for IDA purposes the final-state max is
    typically a good proxy when the record's main pulse occurs
    before the end. For sharper extraction, wrap a per-step recorder
    in a custom extractor.
    """
    story_tags = list(story_node_tags)
    if not story_tags:
        raise ValueError("story_node_tags must be non-empty")

    def extract(model) -> dict:
        # Compute per-story heights from coords if not supplied
        if story_heights is None:
            heights: list[float] = []
            prev_y = (model.node(base_node_tag).coords[1]
                      if base_node_tag is not None else 0.0)
            for tag in story_tags:
                y = model.node(tag).coords[1]
                heights.append(y - prev_y)
                prev_y = y
            h_arr = np.array(heights)
        else:
            h_arr = np.array(story_heights)

        base_d = (float(model.node(base_node_tag).disp[direction])
                  if base_node_tag is not None else 0.0)
        story_disps = np.array([
            float(model.node(t).disp[direction]) - base_d
            for t in story_tags
        ])
        # Interstory drift
        interstory = np.empty_like(story_disps)
        interstory[0] = story_disps[0]
        interstory[1:] = np.diff(story_disps)
        drift_ratios = np.where(h_arr > 0,
                                  np.abs(interstory) / h_arr,
                                  0.0)
        return {
            "max_drift_ratio": float(np.max(drift_ratios)),
            "max_roof_drift": float(np.abs(story_disps[-1])),
        }

    return extract


# ============================================================ driver

class IDADriver:
    """Single-record IDA driver.

    Parameters
    ----------
    model_factory : Callable[[], Model]
        Zero-argument callable returning a fresh model (with materials,
        sections, mass, supports already defined).
    accel_function : Callable[[float], float]
        Ground acceleration ``a_g(t)`` (m/s²). Will be scaled by the
        IM-derived factor at each level.
    direction : str, default "x"
        Ground-motion direction (forwarded to
        :func:`ground_motion_force`).
    t_end : float
        Record duration (s).
    dt : float
        NLTHA time step (s).
    IM_levels : sequence of float
        IM values to evaluate (ascending). Typical: 0.1 g to 2 g in
        0.1 g increments for collapse-capacity work.
    scale_fn : Callable[[float], float]
        Maps target IM to a scalar multiplier on the record. Use
        :func:`pga_scale_factor` for PGA scaling, or supply a custom
        Sa-based function.
    edp_extractor : Callable[[Model], dict]
        Called on the post-run model to extract EDPs.
    damping : RayleighDamping, optional
    record_name : str, default "record_0"
    on_progress : callable, optional
        ``on_progress(i, IM_level)`` called before each point's run
        (useful for long sweeps).
    """

    def __init__(
        self,
        *,
        model_factory: Callable,
        accel_function: Callable[[float], float],
        t_end: float,
        dt: float,
        IM_levels,
        scale_fn: Callable[[float], float],
        edp_extractor: Callable,
        direction: str = "x",
        damping=None,
        record_name: str = "record_0",
        on_progress=None,
    ):
        if t_end <= 0.0:
            raise ValueError(f"t_end must be positive, got {t_end}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        ims = np.asarray(IM_levels, dtype=float)
        if ims.size == 0:
            raise ValueError("IM_levels must be non-empty")
        if np.any(ims <= 0.0):
            raise ValueError("IM_levels must all be positive")
        if not callable(model_factory):
            raise TypeError("model_factory must be callable")

        self.model_factory = model_factory
        self.accel_function = accel_function
        self.t_end = float(t_end)
        self.dt = float(dt)
        self.IM_levels = ims
        self.scale_fn = scale_fn
        self.edp_extractor = edp_extractor
        self.direction = direction
        self.damping = damping
        self.record_name = record_name
        self.on_progress = on_progress

    def run(self) -> IDARecord:
        """Sweep IM levels, run NLTHA at each, build the IDA record."""
        n_steps = int(round(self.t_end / self.dt))
        points: list[IDAPoint] = []

        for i, im in enumerate(self.IM_levels):
            if self.on_progress is not None:
                self.on_progress(i, im)
            sf = float(self.scale_fn(im))

            def scaled_ag(t, _sf=sf):
                return _sf * self.accel_function(t)

            model = self.model_factory()
            gforce = ground_motion_force(
                model, direction=self.direction, accel_function=scaled_ag,
            )
            try:
                ana = NonlinearTransientAnalysis(
                    model, num_steps=n_steps, dt=self.dt,
                    damping=self.damping, load_function=gforce,
                )
                ana.run()
                converged = True
                n_completed = n_steps
            except Exception:
                # NLTHA failed -- strong collapse indicator
                converged = False
                n_completed = len(getattr(ana, "times", [])) - 1
                if n_completed < 0:
                    n_completed = 0

            try:
                edps = self.edp_extractor(model)
            except Exception:
                edps = {"max_drift_ratio": float("nan"),
                         "max_roof_drift": float("nan")}

            points.append(IDAPoint(
                IM=float(im), scale_factor=sf,
                EDPs=edps, converged=converged,
                n_steps_completed=n_completed,
            ))

        return IDARecord(record_name=self.record_name, points=points)
