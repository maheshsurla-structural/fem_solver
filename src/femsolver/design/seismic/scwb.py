"""Strong-Column-Weak-Beam (SCWB) check.

Implements the SCWB / SCWB-ratio joint check from both major codes:

* **ACI 318-19 §18.7.3** (special moment frames -- concrete):

      Σ M_nc ≥ (6 / 5) · Σ M_nb       (Eq 18.7.3-1)

  where ``Σ M_nc`` is the sum of the **nominal** flexural strengths
  of the columns framing into the joint (top + bottom column above and
  below), and ``Σ M_nb`` is the sum of the **nominal** flexural
  strengths of the beams framing into the joint, both evaluated at
  the joint face.

* **AISC 341-22 §E3.4a** (special moment frames -- steel):

      Σ M*_pc / Σ M*_pb ≥ 1.0          (Eq E3-1)

  where ``M*_pc`` is the column moment summed for top and bottom
  columns (accounting for axial-load interaction in the column
  P-M envelope) and ``M*_pb`` includes the steel material
  over-strength factor ``R_y`` and 1.1 multiplier (``1.1 R_y F_y Z_b``).

This module provides a **code-neutral** ratio check; the caller is
responsible for computing the moment-capacity terms appropriate to
their code path (Phase 29 / 30 design routines yield the M_n values
to feed in).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# Required ratios per code
SCWB_RATIO_ACI_SMF = 6.0 / 5.0       # ACI 318-19 18.7.3.2
SCWB_RATIO_AISC_SMF = 1.0            # AISC 341-22 E3.4a


@dataclass
class SCWBCheck:
    """Result of a strong-column-weak-beam joint check.

    Attributes
    ----------
    sum_M_nc : float
        Sum of nominal column flexural strengths at the joint (N·m).
    sum_M_nb : float
        Sum of nominal beam flexural strengths at the joint (N·m).
    ratio : float
        ``sum_M_nc / sum_M_nb``. Larger is stronger column relative
        to beam.
    ratio_required : float
        The code-required minimum (6/5 for ACI SMF, 1.0 for AISC).
    passes : bool
        ``True`` iff ``ratio >= ratio_required``.
    code : str
        Human-readable code citation.
    notes : str
    """

    sum_M_nc: float
    sum_M_nb: float
    ratio: float
    ratio_required: float
    passes: bool
    code: str
    notes: str = ""


def scwb_check(
    *,
    column_M_n: Iterable[float],
    beam_M_n: Iterable[float],
    ratio_required: float | None = None,
    code: str = "ACI",
) -> SCWBCheck:
    """Strong-column-weak-beam joint check.

    Parameters
    ----------
    column_M_n : Iterable[float]
        Nominal flexural strengths of the columns framing into the
        joint (typically two -- column above and column below). Pass
        absolute values; this function takes the magnitudes.
    beam_M_n : Iterable[float]
        Nominal flexural strengths of the beams framing into the
        joint. For a typical SMF interior joint these are the left
        and right beams' negative-moment capacities (i.e., the
        capacities that resist seismic-induced beam moments).
    ratio_required : float, optional
        Custom required ratio. Defaults: 6/5 for ``code="ACI"``,
        1.0 for ``code="AISC"``.
    code : str, default ``"ACI"``
        ``"ACI"`` -> uses 6/5 default + cites ACI 318-19 18.7.3.
        ``"AISC"`` -> uses 1.0 default + cites AISC 341-22 E3.4a.

    Returns
    -------
    SCWBCheck
    """
    sum_Mc = sum(abs(m) for m in column_M_n)
    sum_Mb = sum(abs(m) for m in beam_M_n)
    if sum_Mb <= 0.0:
        # No beams -> degenerate; SCWB is trivially satisfied
        ratio = float("inf")
    else:
        ratio = sum_Mc / sum_Mb

    if ratio_required is None:
        code_norm = code.strip().upper()
        if code_norm == "ACI":
            ratio_required = SCWB_RATIO_ACI_SMF
        elif code_norm == "AISC":
            ratio_required = SCWB_RATIO_AISC_SMF
        else:
            raise ValueError(
                f"code must be 'ACI' or 'AISC', got {code!r}; or pass "
                "ratio_required directly"
            )

    code_norm = code.strip().upper()
    code_str = (
        "ACI 318-19 §18.7.3" if code_norm == "ACI"
        else "AISC 341-22 §E3.4a" if code_norm == "AISC"
        else code
    )

    passes = ratio >= ratio_required - 1.0e-9
    notes_list: list[str] = []
    if not passes:
        notes_list.append(
            f"SCWB ratio {ratio:.3f} < required {ratio_required:.3f}; "
            "joint does not develop the strong-column-weak-beam "
            "mechanism. Enlarge columns or weaken beams."
        )

    return SCWBCheck(
        sum_M_nc=sum_Mc,
        sum_M_nb=sum_Mb,
        ratio=ratio,
        ratio_required=ratio_required,
        passes=passes,
        code=code_str,
        notes="; ".join(notes_list),
    )
