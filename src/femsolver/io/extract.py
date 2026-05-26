"""Extraction utilities for analysis results.

Convert raw analysis output (lists of times, per-step displacements,
mode shapes, eigen results) into structured arrays that are easy to
hand off to matplotlib, pandas, or downstream analysis code.

* :func:`gather_node_history` -- pull a single DOF history from a
  transient-analysis result dict.
* :func:`mode_shape_table`     -- structured table of modal periods,
  participation factors, and effective masses.
* :func:`capacity_curve`       -- (drift, force) capacity curve from a
  pushover analysis.
"""
from __future__ import annotations

import numpy as np


def gather_node_history(model, results: dict, node_tag: int,
                          dof: int) -> dict:
    """Return a single DOF's time history from a transient analysis.

    Parameters
    ----------
    model : Model
        The model the analysis was run on (used to look up the node).
    results : dict
        A ``TransientAnalysis.run()`` result dictionary. Must contain
        either ``"tracked_disp"`` (if the analysis tracked the same
        DOF) or per-step state captured externally.
    node_tag : int
    dof : int
        DOF index within the node (0 = x, 1 = y, etc.).

    Returns
    -------
    dict
        With keys ``"times"``, ``"disp"``, ``"node"``, ``"dof"``.
    """
    if "times" not in results:
        raise ValueError(
            "results dict has no 'times' key -- was this produced by a "
            "transient analysis?"
        )
    times = np.asarray(results["times"], dtype=float)
    if "tracked_disp" in results and results.get("tracked_node") == node_tag \
            and results.get("tracked_dof") == dof:
        disp = np.asarray(results["tracked_disp"], dtype=float)
    elif "tracked_disp" in results:
        # Track was for a different DOF; just return what we have plus
        # a warning hint.
        disp = np.asarray(results["tracked_disp"], dtype=float)
    else:
        raise ValueError(
            "results dict has no 'tracked_disp'; pass track=(node, dof) "
            "to the transient analysis or capture node histories "
            "manually."
        )
    return {
        "times": times,
        "disp": disp,
        "node": node_tag,
        "dof": dof,
    }


def mode_shape_table(eig_result: dict) -> dict:
    """Turn an EigenAnalysis result into a structured table.

    Parameters
    ----------
    eig_result : dict
        Output of ``EigenAnalysis.run()``. Expected keys:
        ``"frequencies_hz"``, ``"periods_s"``, ``"omegas_rad_s"``,
        optionally ``"modal_results"`` (from ResponseSpectrumAnalysis).

    Returns
    -------
    dict with arrays for each tabulated quantity.
    """
    if "periods_s" not in eig_result:
        raise ValueError(
            "eig_result has no 'periods_s' key -- expected an "
            "EigenAnalysis or ResponseSpectrumAnalysis output."
        )
    periods = np.asarray(eig_result["periods_s"], dtype=float)
    n = periods.size
    out: dict = {
        "mode": np.arange(1, n + 1),
        "period_s": periods,
        "frequency_hz": np.asarray(
            eig_result.get("frequencies_hz", 1.0 / periods),
            dtype=float,
        ),
        "omega_rad_s": np.asarray(
            eig_result.get("omegas_rad_s", 2.0 * np.pi / periods),
            dtype=float,
        ),
    }
    # If response-spectrum analysis: pull participation factor and
    # effective modal mass per mode.
    modal_results = eig_result.get("modal_results")
    if modal_results:
        out["participation"] = np.array(
            [r.get("Gamma", float("nan")) for r in modal_results]
        )
        out["modal_mass_eff"] = np.array(
            [r.get("modal_mass_eff", float("nan")) for r in modal_results]
        )
        out["Sa"] = np.array(
            [r.get("Sa", float("nan")) for r in modal_results]
        )
    return out


def capacity_curve(results: dict, *, drift_key: str = "tracked",
                    force_key: str = "lambdas") -> dict:
    """Extract a capacity curve (drift vs base shear or load factor)
    from a nonlinear-static or pushover analysis result.

    Parameters
    ----------
    results : dict
        Output dict from ``NonlinearStaticAnalysis.run()``. Must
        contain a tracked displacement key and either explicit force
        history or a list of load factors / lambda values.
    drift_key, force_key : str
        Keys to read for drift and force, respectively.

    Returns
    -------
    dict with ``"drift"`` and ``"force"`` arrays of equal length.
    """
    if drift_key not in results:
        raise ValueError(f"results has no '{drift_key}' key")
    if force_key not in results:
        raise ValueError(
            f"results has no '{force_key}' key (got "
            f"{sorted(results.keys())})"
        )
    drift = np.asarray(results[drift_key], dtype=float)
    force = np.asarray(results[force_key], dtype=float)
    if drift.size != force.size:
        # Some analyses log drift at the start of each step and force
        # after — pad the shorter one or truncate.
        n = min(drift.size, force.size)
        drift = drift[:n]
        force = force[:n]
    return {"drift": drift, "force": force}
