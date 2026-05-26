"""Ply failure criteria for composite laminates.

Given per-ply stresses in the lamina (1-2) material axes (typically
recovered via :meth:`LayeredShellSection.ply_stresses`), evaluate one
of the standard failure indices:

* **Max-stress** -- failure if any principal stress exceeds the
  corresponding allowable. Simple but ignores stress interactions.

* **Max-strain** -- the strain-space analogue of max-stress. Useful
  for ductile-matrix or strain-controlled fatigue.

* **Tsai-Hill** -- a quadratic stress-interaction criterion. The
  matrix and fiber strengths are coupled through a single FI
  expression, but tension and compression must be handled by
  choosing X = X_T or X_C depending on the sign of sigma_11
  (similarly Y).

* **Tsai-Wu** -- the most general quadratic criterion, with explicit
  linear and quadratic terms in stress (and the cross-term coefficient
  F_12). Smooth and continuous across stress sign changes.

For every criterion the "failure index" FI is defined so that
``FI >= 1`` indicates failure. The "strength ratio" SR (only mathematically
clean for Tsai-Wu) is the scalar load multiplier that brings the
stress state to FI = 1, i.e. ``sigma_failure = SR * sigma``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class PlyStrength:
    """Allowable strengths of an orthotropic ply in lamina (1-2) axes.

    Attributes
    ----------
    Xt : float
        Tensile strength in the fiber (1) direction (positive).
    Xc : float
        Compressive strength in the fiber direction (positive magnitude).
    Yt : float
        Tensile strength transverse to fibers (positive).
    Yc : float
        Compressive transverse strength (positive magnitude).
    S : float
        In-plane shear strength (positive).
    """

    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S: float

    def __post_init__(self) -> None:
        for name, v in (("Xt", self.Xt), ("Xc", self.Xc),
                          ("Yt", self.Yt), ("Yc", self.Yc),
                          ("S", self.S)):
            if v <= 0.0:
                raise ValueError(f"{name} must be positive, got {v}")


# ============================================================ criteria

def max_stress_index(sigma_local: Sequence[float],
                       strength: PlyStrength) -> float:
    """Max-stress failure index. ``sigma_local`` must be in the
    lamina (1-2) axes as a 3-vector ``(sigma_11, sigma_22, sigma_12)``.
    Returns ``max(ratios)``; failure if return >= 1.
    """
    s = np.asarray(sigma_local, dtype=float)
    s11, s22, s12 = float(s[0]), float(s[1]), float(s[2])
    ratios = []
    if s11 >= 0.0:
        ratios.append(s11 / strength.Xt)
    else:
        ratios.append(-s11 / strength.Xc)
    if s22 >= 0.0:
        ratios.append(s22 / strength.Yt)
    else:
        ratios.append(-s22 / strength.Yc)
    ratios.append(abs(s12) / strength.S)
    return max(ratios)


def max_strain_index(eps_local: Sequence[float],
                       eps_limits: "PlyStrength") -> float:
    """Max-strain failure index. ``eps_local = (eps_11, eps_22, gamma_12)``
    in lamina (1-2) axes; ``eps_limits`` uses the same field names as
    :class:`PlyStrength` but interpreted as strain limits."""
    # The math is identical to max-stress on strains, so reuse:
    return max_stress_index(eps_local, eps_limits)


def tsai_hill_index(sigma_local: Sequence[float],
                      strength: PlyStrength) -> float:
    """Tsai-Hill failure index.

    .. math::

        FI = (sigma_{11}/X)^2 - (sigma_{11} sigma_{22} / X^2)
             + (sigma_{22}/Y)^2 + (sigma_{12}/S)^2

    with ``X = X_T`` if ``sigma_{11} > 0`` else ``X_C``, and similarly
    for ``Y`` based on the sign of ``sigma_{22}``.
    """
    s = np.asarray(sigma_local, dtype=float)
    s11, s22, s12 = float(s[0]), float(s[1]), float(s[2])
    X = strength.Xt if s11 >= 0.0 else strength.Xc
    Y = strength.Yt if s22 >= 0.0 else strength.Yc
    return ((s11 / X) ** 2
            - (s11 * s22) / (X * X)
            + (s22 / Y) ** 2
            + (s12 / strength.S) ** 2)


def tsai_wu_index(sigma_local: Sequence[float],
                    strength: PlyStrength,
                    F12_factor: float = -0.5) -> float:
    """Tsai-Wu quadratic failure index.

    Parameters
    ----------
    sigma_local : (sigma_11, sigma_22, sigma_12)
    strength : ``PlyStrength``
    F12_factor : float, default -0.5
        Cross-term coefficient. Tsai-Wu's original empirical choice is
        ``F12 = F12_factor * sqrt(F11 F22)`` with ``F12_factor = -1/2``,
        which makes the failure envelope an ellipse aligned with the
        tension-tension quadrant. ``-1`` gives a more conservative
        envelope; ``0`` gives a decoupled bound.
    """
    s = np.asarray(sigma_local, dtype=float)
    s11, s22, s12 = float(s[0]), float(s[1]), float(s[2])
    Xt, Xc = strength.Xt, strength.Xc
    Yt, Yc = strength.Yt, strength.Yc
    S = strength.S
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S * S)
    F12 = F12_factor * math.sqrt(F11 * F22)
    return (F1 * s11 + F2 * s22
            + F11 * s11 * s11 + F22 * s22 * s22
            + F66 * s12 * s12
            + 2.0 * F12 * s11 * s22)


def tsai_wu_strength_ratio(sigma_local: Sequence[float],
                              strength: PlyStrength,
                              F12_factor: float = -0.5) -> float:
    """Strength ratio R such that ``sigma * R`` is on the failure
    envelope (FI = 1). The largest positive root of the quadratic

        a R^2 + b R - 1 = 0

    where ``a`` is the quadratic part and ``b`` is the linear part of
    the Tsai-Wu index.
    """
    s = np.asarray(sigma_local, dtype=float)
    s11, s22, s12 = float(s[0]), float(s[1]), float(s[2])
    Xt, Xc = strength.Xt, strength.Xc
    Yt, Yc = strength.Yt, strength.Yc
    S = strength.S
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S * S)
    F12 = F12_factor * math.sqrt(F11 * F22)
    a = (F11 * s11 * s11 + F22 * s22 * s22 + F66 * s12 * s12
          + 2.0 * F12 * s11 * s22)
    b = F1 * s11 + F2 * s22
    if a <= 0.0:
        # Zero / negative quadratic part -- failure impossible for this
        # stress state (e.g. pure compressive when only-tensile)
        return float("inf")
    disc = b * b + 4.0 * a
    sd = math.sqrt(disc)
    R_pos = (-b + sd) / (2.0 * a)
    return R_pos


# ============================================================ section helper

def evaluate_laminate(section, eps_membrane, kappa, strengths,
                       *, criterion: str = "tsai_wu",
                       F12_factor: float = -0.5,
                       z: str = "all") -> list[dict]:
    """Evaluate a failure criterion at every ply of a layered shell
    section.

    Parameters
    ----------
    section : LayeredShellSection
    eps_membrane, kappa : sequences of 3 floats
        Mid-plane membrane strain and curvature in global (laminate)
        axes.
    strengths : PlyStrength or list of PlyStrength
        Allowables. Pass a single ``PlyStrength`` to apply the same to
        every layer, or a list of length ``len(section.layers)`` to
        give per-layer allowables.
    criterion : ``"max_stress"`` | ``"tsai_hill"`` | ``"tsai_wu"``
    F12_factor : float, default -0.5 (Tsai-Wu only)
    z : ``"all"`` | ``"top"`` | ``"mid"`` | ``"bot"``

    Returns
    -------
    list of dict, one per (ply, z-station). Each dict contains the
    fields produced by :meth:`LayeredShellSection.ply_stresses` plus:

    * ``"FI"`` : failure index from the selected criterion.
    * ``"SR"`` : strength ratio (Tsai-Wu only; ``None`` otherwise).
    """
    if criterion not in ("max_stress", "tsai_hill", "tsai_wu"):
        raise ValueError(
            f"unknown criterion {criterion!r}; expected 'max_stress', "
            "'tsai_hill', or 'tsai_wu'"
        )
    layer_stress = section.ply_stresses(eps_membrane, kappa, z=z)
    # Normalize strengths into a per-layer list
    if isinstance(strengths, PlyStrength):
        strengths_list = [strengths] * len(section.layers)
    else:
        strengths_list = list(strengths)
        if len(strengths_list) != len(section.layers):
            raise ValueError(
                f"strengths list has {len(strengths_list)} items but "
                f"section has {len(section.layers)} layers"
            )
    out = []
    for rec in layer_stress:
        strength = strengths_list[rec["layer"]]
        sigma = rec["sigma_local"]
        if criterion == "max_stress":
            FI = max_stress_index(sigma, strength)
            SR = None
        elif criterion == "tsai_hill":
            FI = tsai_hill_index(sigma, strength)
            SR = None
        else:
            FI = tsai_wu_index(sigma, strength, F12_factor=F12_factor)
            SR = tsai_wu_strength_ratio(sigma, strength,
                                          F12_factor=F12_factor)
        rec = dict(rec)
        rec["FI"] = float(FI)
        rec["SR"] = (None if SR is None else float(SR))
        out.append(rec)
    return out
