"""Collapse detection + multi-record IDA orchestration.

Builds on the single-record :class:`IDADriver` (Phase 25.1) to:

* **Detect collapse** for each record via three independent criteria:
  - Hard EDP limit exceeded (e.g. drift > 10%).
  - NLTHA non-convergence.
  - "Flatlining" of the IDA curve -- the EDP grows by more than a
    factor ``flatline_factor`` for an IM step ``< flatline_slope_min``
    (slope collapse per Vamvatsikos-Cornell 2002).
* Run a **suite** of records via the same model factory, producing
  one :class:`IDARecord` per record and an aggregated
  :class:`IDASummary` with per-record collapse IM values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from femsolver.analysis.ida import IDADriver, IDAPoint, IDARecord


# ============================================================ collapse detection

@dataclass
class CollapseResult:
    """Result of collapse detection on one :class:`IDARecord`.

    Attributes
    ----------
    collapse_IM : float
        The smallest IM at which collapse was detected. ``inf`` if
        the record did not collapse within the sweep range.
    cause : str
        ``"drift_limit"`` / ``"non_convergence"`` / ``"flatline"`` /
        ``"no_collapse"``.
    collapse_point_index : int or None
        Index into ``record.points`` of the first collapse point.
        ``None`` if no collapse.
    """

    collapse_IM: float
    cause: str
    collapse_point_index: int | None


def detect_collapse(
    record: IDARecord,
    *,
    edp_name: str = "max_drift_ratio",
    drift_limit: float = 0.10,
    flatline_factor: float = 5.0,
    flatline_slope_min: float = 0.05,
) -> CollapseResult:
    """Identify the lowest IM at which the record collapses.

    Criteria (any one triggers collapse):

    1. **EDP limit**: ``EDP >= drift_limit`` (default 10% drift).
    2. **Non-convergence**: NLTHA at this point did not finish.
    3. **Flatline**: the local IDA slope drops below
       ``flatline_slope_min · initial_slope`` AND the EDP has grown
       by ``flatline_factor`` over the previous point.

    Returns the lowest collapse IM across all three criteria, plus
    the cause label and the index in ``record.points``.
    """
    pts = record.points
    if not pts:
        return CollapseResult(collapse_IM=float("inf"),
                               cause="no_collapse",
                               collapse_point_index=None)

    # Sort by IM (defensive)
    pts_sorted = sorted(pts, key=lambda p: p.IM)
    IMs = np.array([p.IM for p in pts_sorted])
    EDPs = np.array([p.EDPs.get(edp_name, np.nan) for p in pts_sorted])
    converged = np.array([p.converged for p in pts_sorted])

    earliest_idx: int | None = None
    earliest_cause = "no_collapse"

    # Criterion 1: EDP limit
    for i, edp in enumerate(EDPs):
        if not np.isnan(edp) and edp >= drift_limit:
            if earliest_idx is None or i < earliest_idx:
                earliest_idx = i
                earliest_cause = "drift_limit"
            break

    # Criterion 2: non-convergence
    for i, c in enumerate(converged):
        if not c:
            if earliest_idx is None or i < earliest_idx:
                earliest_idx = i
                earliest_cause = "non_convergence"
            break

    # Criterion 3: flatline (slope collapse). Compute initial slope
    # (between the lowest two valid points) and look for a slope drop.
    valid = ~np.isnan(EDPs) & (EDPs > 0)
    if int(np.sum(valid)) >= 2:
        v_idx = np.flatnonzero(valid)
        # Initial slope = (EDP[1] - EDP[0]) / (IM[1] - IM[0])
        i0, i1 = int(v_idx[0]), int(v_idx[1])
        if IMs[i1] > IMs[i0] and EDPs[i1] > EDPs[i0]:
            initial_slope = (EDPs[i1] - EDPs[i0]) / (IMs[i1] - IMs[i0])
            for k in range(2, len(v_idx)):
                i_curr = int(v_idx[k])
                i_prev = int(v_idx[k - 1])
                dIM = IMs[i_curr] - IMs[i_prev]
                dEDP = EDPs[i_curr] - EDPs[i_prev]
                if dIM > 0 and dEDP > 0:
                    slope = dEDP / dIM
                    edp_ratio = EDPs[i_curr] / EDPs[i_prev]
                    if (slope < flatline_slope_min * initial_slope
                            and edp_ratio > flatline_factor):
                        if earliest_idx is None or i_curr < earliest_idx:
                            earliest_idx = i_curr
                            earliest_cause = "flatline"
                        break

    if earliest_idx is None:
        return CollapseResult(collapse_IM=float("inf"),
                               cause="no_collapse",
                               collapse_point_index=None)
    return CollapseResult(
        collapse_IM=float(IMs[earliest_idx]),
        cause=earliest_cause,
        collapse_point_index=earliest_idx,
    )


# ============================================================ multi-record

@dataclass
class IDASummary:
    """Aggregated IDA across a set of records.

    Attributes
    ----------
    records : list[IDARecord]
    collapse_results : list[CollapseResult]
        One per record (parallel to ``records``).
    collapse_IMs : np.ndarray
        Per-record collapse IM. ``inf`` for records that did not
        collapse in the sweep range.
    n_collapsed : int
        Number of records that collapsed.
    median_collapse_IM : float
        Median of the finite collapse IMs (``nan`` if none collapsed).
    """

    records: list = field(default_factory=list)
    collapse_results: list = field(default_factory=list)
    collapse_IMs: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_collapsed: int = 0
    median_collapse_IM: float = float("nan")


def multi_record_ida(
    *,
    model_factory: Callable,
    records,
    IM_levels,
    edp_extractor: Callable,
    direction: str = "x",
    damping=None,
    drift_limit: float = 0.10,
    flatline_factor: float = 5.0,
    flatline_slope_min: float = 0.05,
    on_progress=None,
) -> IDASummary:
    """Run IDA on a list of records and aggregate the results.

    Each entry in ``records`` is a dict with keys:

    * ``"name"``: str -- record label
    * ``"accel_function"``: callable a_g(t)
    * ``"t_end"``: float
    * ``"dt"``: float
    * ``"scale_fn"``: callable target_IM -> scalar multiplier

    All records use the same ``IM_levels`` sweep, the same model
    factory, and the same EDP extractor.

    Parameters
    ----------
    records : list[dict]
    IM_levels : sequence of float
    edp_extractor, direction, damping
        Forwarded to :class:`IDADriver`.
    drift_limit, flatline_factor, flatline_slope_min
        Forwarded to :func:`detect_collapse`.
    on_progress : callable, optional
        ``on_progress(record_idx, record_name, n_records)`` invoked
        before each record.
    """
    n_records = len(records)
    records_out: list[IDARecord] = []
    collapse_results: list[CollapseResult] = []
    for rec_idx, rec in enumerate(records):
        if on_progress is not None:
            on_progress(rec_idx, rec["name"], n_records)
        driver = IDADriver(
            model_factory=model_factory,
            accel_function=rec["accel_function"],
            t_end=rec["t_end"], dt=rec["dt"],
            IM_levels=IM_levels,
            scale_fn=rec["scale_fn"],
            edp_extractor=edp_extractor,
            direction=direction,
            damping=damping,
            record_name=rec["name"],
        )
        ida_rec = driver.run()
        records_out.append(ida_rec)
        coll = detect_collapse(
            ida_rec,
            drift_limit=drift_limit,
            flatline_factor=flatline_factor,
            flatline_slope_min=flatline_slope_min,
        )
        collapse_results.append(coll)

    collapse_IMs = np.array([c.collapse_IM for c in collapse_results])
    finite_IMs = collapse_IMs[np.isfinite(collapse_IMs)]
    median_IM = float(np.median(finite_IMs)) if finite_IMs.size > 0 else float("nan")
    return IDASummary(
        records=records_out,
        collapse_results=collapse_results,
        collapse_IMs=collapse_IMs,
        n_collapsed=int(finite_IMs.size),
        median_collapse_IM=median_IM,
    )
