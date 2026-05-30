"""Drift checks per ASCE 7-22 §12.12 (seismic) and §C-CC.1.2 (wind).

Given a model with a converged displacement state (or per-combo
displacement envelopes) and a list of story-representative node tags,
produce per-story interstory drifts, amplify them per ASCE 7-22, and
check against the allowable drift limit.

Inputs
------
* Story node tags in elevation order (lowest first, roof last).
* Direction index (0 = x, 1 = y) of the lateral motion.
* Optional base node tag for the zero-drift reference.

ASCE 7-22 seismic drift §12.12 / Table 12.12-1
-----------------------------------------------

The **design story drift** ``Δ`` from the elastic analysis is
multiplied by the **deflection amplification factor** ``C_d``
(structure-system-dependent; e.g., 5.5 for SMF) and divided by the
importance factor ``I_e`` (1.0 for Risk Category I and II, 1.25 for
III, 1.5 for IV)::

    Δ_amplified = C_d · Δ_elastic / I_e

The allowable drift ``Δ_a`` per Table 12.12-1 is typically::

    Δ_a = 0.020 · h_sx  (Risk Cat I/II, all other structures)
    Δ_a = 0.015 · h_sx  (Risk Cat III)
    Δ_a = 0.010 · h_sx  (Risk Cat IV)

The check is ``Δ_amplified <= Δ_a`` at every story.

This module wraps the existing :func:`story_drifts` helper from
Phase 19 with the ASCE 7 amplification and limit check, and is
designed to consume either:

* A model with currently-set displacements (one combo loaded), or
* An :class:`EnvelopeResult` (Phase 31.2) for the envelope across
  many combos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from femsolver.analysis.capacity_design import story_drifts


# ============================================================ result types

@dataclass
class DriftCheck:
    """Per-story drift check result.

    Attributes
    ----------
    story_index : np.ndarray
        1..N story numbers.
    story_height : np.ndarray
        Per-story height ``h_sx`` (m).
    delta_elastic : np.ndarray
        Elastic interstory drift from the analysis (m).
    delta_amplified : np.ndarray
        ``C_d · Δ_elastic / I_e`` (m). The drift used in the ASCE
        7-22 limit check.
    drift_ratio : np.ndarray
        ``Δ_amplified / h_sx`` (dimensionless).
    drift_limit : float
        Allowable ratio (e.g., 0.020 for Risk Cat I/II).
    delta_allowable : np.ndarray
        ``drift_limit · h_sx`` (m).
    governing_combo : list[str] or None
        If sourced from envelope: the combination name that produced
        each story's worst drift. ``None`` for single-state input.
    passes_per_story : np.ndarray of bool
        ``Δ_amplified <= Δ_allowable`` per story.
    passes : bool
        ``True`` iff all stories pass.
    """

    story_index: np.ndarray
    story_height: np.ndarray
    delta_elastic: np.ndarray
    delta_amplified: np.ndarray
    drift_ratio: np.ndarray
    drift_limit: float
    delta_allowable: np.ndarray
    governing_combo: list | None
    passes_per_story: np.ndarray
    passes: bool


def _allowable_drift_ratio(risk_category: str) -> float:
    """ASCE 7-22 Table 12.12-1 allowable story-drift ratio."""
    rc = risk_category.upper()
    if rc in ("I", "II"):
        return 0.020
    if rc == "III":
        return 0.015
    if rc == "IV":
        return 0.010
    raise ValueError(
        f"risk_category must be 'I', 'II', 'III', or 'IV', got {risk_category!r}"
    )


# ============================================================ single-state check

def drift_check(
    model,
    story_node_tags,
    *,
    direction: int = 0,
    base_node_tag: int | None = None,
    C_d: float = 5.5,
    I_e: float = 1.0,
    risk_category: str = "II",
    drift_limit_override: float | None = None,
) -> DriftCheck:
    """Seismic drift check per ASCE 7-22 §12.12 on the model's
    current displacement state.

    Parameters
    ----------
    model : Model
        Model with a converged displacement state (i.e., after
        :class:`LinearStaticAnalysis.run` or similar).
    story_node_tags : sequence of int
        Representative node at each story, ordered lowest -> roof.
    direction : int, default 0
        DOF index of the drift component (0 = x, 1 = y).
    base_node_tag : int, optional
        Node defining the zero-drift reference (typically the base).
        If ``None``, the base is treated as having zero displacement.
    C_d : float, default 5.5
        Deflection amplification factor per ASCE 7-22 Table 12.2-1.
        Common values: 5.5 (SMF), 4 (IMF), 3.5 (OMF), 3.25
        (SCBF), 4.5 (BRBF).
    I_e : float, default 1.0
        Importance factor per ASCE 7-22 Table 1.5-2. 1.0 for Risk
        Cat I/II, 1.25 for III, 1.5 for IV.
    risk_category : str, default "II"
        Used to look up the default drift limit. Override with
        ``drift_limit_override`` if needed.
    drift_limit_override : float, optional
        Custom allowable drift ratio (replaces the
        ``risk_category`` lookup).

    Returns
    -------
    DriftCheck
    """
    sd = story_drifts(
        model, story_node_tags, direction=direction,
        base_node_tag=base_node_tag,
    )
    delta_elastic = np.asarray(sd["interstory_drift"], dtype=float)
    h_story = np.asarray(sd["story_height"], dtype=float)
    delta_amplified = C_d * delta_elastic / I_e
    drift_ratio = np.where(h_story > 0,
                              np.abs(delta_amplified) / h_story,
                              0.0)
    limit = (drift_limit_override if drift_limit_override is not None
             else _allowable_drift_ratio(risk_category))
    delta_allowable = limit * h_story
    passes_per_story = drift_ratio <= limit
    passes = bool(np.all(passes_per_story))
    return DriftCheck(
        story_index=np.asarray(sd["story"]),
        story_height=h_story,
        delta_elastic=delta_elastic,
        delta_amplified=delta_amplified,
        drift_ratio=drift_ratio,
        drift_limit=limit,
        delta_allowable=delta_allowable,
        governing_combo=None,
        passes_per_story=passes_per_story,
        passes=passes,
    )


# ============================================================ multi-combo helper

def drift_check_worst_combo(
    model,
    patterns,
    combinations,
    story_node_tags,
    *,
    direction: int = 0,
    base_node_tag: int | None = None,
    C_d: float = 5.5,
    I_e: float = 1.0,
    risk_category: str = "II",
    drift_limit_override: float | None = None,
) -> DriftCheck:
    """Loop over each combination, apply it, run LinearStaticAnalysis,
    compute drift, and keep the worst-case (largest |Δ|) per story.

    Returns a single ``DriftCheck`` whose ``governing_combo`` field
    names the combination responsible for each story's worst drift.
    """
    from femsolver.analysis.linear_static import LinearStaticAnalysis
    from femsolver.analysis.loads import apply_combination

    story_tags = list(story_node_tags)
    n_stories = len(story_tags)
    if n_stories == 0:
        raise ValueError("story_node_tags must be non-empty")
    combos = list(combinations)
    if not combos:
        raise ValueError("combinations must be non-empty")

    delta_elastic_worst = np.zeros(n_stories)
    governing: list[str] = [""] * n_stories
    story_height: np.ndarray | None = None

    for combo in combos:
        apply_combination(model, patterns, combo)
        LinearStaticAnalysis(model).run()
        dc = drift_check(
            model, story_tags,
            direction=direction, base_node_tag=base_node_tag,
            C_d=C_d, I_e=I_e,
            risk_category=risk_category,
            drift_limit_override=drift_limit_override,
        )
        if story_height is None:
            story_height = dc.story_height.copy()
        for i in range(n_stories):
            if abs(dc.delta_elastic[i]) > abs(delta_elastic_worst[i]):
                delta_elastic_worst[i] = dc.delta_elastic[i]
                governing[i] = combo.name

    # Final DriftCheck from the per-story worst Δ_elastic
    delta_amplified = C_d * delta_elastic_worst / I_e
    drift_ratio = np.where(story_height > 0,
                              np.abs(delta_amplified) / story_height,
                              0.0)
    limit = (drift_limit_override if drift_limit_override is not None
             else _allowable_drift_ratio(risk_category))
    delta_allowable = limit * story_height
    passes_per_story = drift_ratio <= limit
    passes = bool(np.all(passes_per_story))
    return DriftCheck(
        story_index=np.arange(1, n_stories + 1),
        story_height=story_height,
        delta_elastic=delta_elastic_worst,
        delta_amplified=delta_amplified,
        drift_ratio=drift_ratio,
        drift_limit=limit,
        delta_allowable=delta_allowable,
        governing_combo=governing,
        passes_per_story=passes_per_story,
        passes=passes,
    )
