"""Two-way slab Direct Design Method (ACI 318-19 8.10).

The DDM expresses the design moments in each span as fractions of
the *total static moment* per panel::

    M_o = w_u * l_2 * l_n**2 / 8

where ``w_u`` is the factored areal load (N/m^2), ``l_2`` is the
span centre-to-centre perpendicular to the analysis direction (m),
and ``l_n`` is the clear span in the analysis direction (m).

The total moment is then split into negative (support) and positive
(midspan) moments using the longitudinal-distribution factors of
Table 8.10.4.1, and each of those into column- and middle-strip
portions per 8.10.5.

Limitations of DDM (ACI 318-19 8.10.2):

* Three or more continuous spans in each direction.
* Span ratios <= 2.0.
* Successive spans differ by <= 1/3.
* Columns aligned to a regular grid (offset <= 10% of span).
* Live load <= 2 x dead load.
* All loads gravity only.

The functions here just compute the design moments -- *applicability
checking is the user's responsibility*.
"""
from __future__ import annotations

from dataclasses import dataclass


# ============================================================ data


@dataclass
class DDMPanelResult:
    """Design moments for a single DDM panel."""

    M_o: float           # total static moment (N.m)
    # Span-direction moments (interior span)
    M_neg_int: float     # interior negative
    M_pos_int: float     # interior positive
    # Span-direction moments (end span, exterior support)
    M_neg_ext: float
    M_pos_ext: float
    M_neg_int_end: float    # interior negative at first interior support
    # Strip distribution (column / middle for interior support negative)
    M_col_strip_neg_int: float
    M_mid_strip_neg_int: float
    M_col_strip_pos_int: float
    M_mid_strip_pos_int: float


def ddm_panel(
    *,
    w_u: float,
    l_long: float,
    l_short: float,
    direction: str = "long",
    col_size: float = 0.4,
) -> DDMPanelResult:
    """Design moments for a typical DDM panel.

    Parameters
    ----------
    w_u : float
        Factored areal load (N/m^2).
    l_long, l_short : float
        Centre-to-centre spans (m).
    direction : {"long", "short"}
        Direction whose moments are being computed. The clear span
        ``l_n = l_direction - col_size``.
    col_size : float
        Plan dimension of supporting columns (m); subtracted from
        the centre-to-centre span to give the clear span.
    """
    if w_u <= 0:
        raise ValueError(f"w_u must be positive, got {w_u}")
    if l_long <= 0 or l_short <= 0:
        raise ValueError("spans must be positive")
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if col_size <= 0 or col_size >= min(l_long, l_short):
        raise ValueError("col_size must be positive and less than min span")

    if direction == "long":
        l1 = l_long
        l2 = l_short
    else:
        l1 = l_short
        l2 = l_long
    l_n = l1 - col_size
    M_o = w_u * l2 * l_n * l_n / 8.0

    # Interior span (ACI 318-19 Table 8.10.4.1.1)
    M_neg_int = 0.65 * M_o
    M_pos_int = 0.35 * M_o
    # End span -- assuming a flat-slab interior column at exterior face
    # (ACI 318-19 Table 8.10.4.2). Use the most common case: "flat slab
    # with edge beam".
    M_neg_ext = 0.30 * M_o
    M_pos_ext = 0.50 * M_o
    M_neg_int_end = 0.70 * M_o
    # Strip distribution at interior support (column strip = 75% of
    # negative moment, middle strip = 25%; positive: 60% / 40%) --
    # values for l_2 / l_1 = 1.0 from Table 8.10.5.1. For other span
    # ratios, the user should consult the full table.
    M_col_strip_neg_int = 0.75 * M_neg_int
    M_mid_strip_neg_int = 0.25 * M_neg_int
    M_col_strip_pos_int = 0.60 * M_pos_int
    M_mid_strip_pos_int = 0.40 * M_pos_int

    return DDMPanelResult(
        M_o=float(M_o),
        M_neg_int=float(M_neg_int),
        M_pos_int=float(M_pos_int),
        M_neg_ext=float(M_neg_ext),
        M_pos_ext=float(M_pos_ext),
        M_neg_int_end=float(M_neg_int_end),
        M_col_strip_neg_int=float(M_col_strip_neg_int),
        M_mid_strip_neg_int=float(M_mid_strip_neg_int),
        M_col_strip_pos_int=float(M_col_strip_pos_int),
        M_mid_strip_pos_int=float(M_mid_strip_pos_int),
    )


def ddm_minimum_thickness(
    *,
    l_n: float,
    f_y: float = 420e6,
    interior_panel: bool = True,
) -> float:
    """Minimum slab thickness per ACI 318-19 Table 8.3.1.1 for two-way
    slabs without interior beams (flat plates / flat slabs).

    For ``f_y = 420 MPa``:

    * interior panel: ``h_min = l_n / 33``
    * exterior panel without edge beam: ``h_min = l_n / 30``
    * exterior panel with edge beam: ``h_min = l_n / 33``

    For other ``f_y`` the value is scaled by ``(0.4 + f_y / 700 MPa)``.
    """
    if l_n <= 0:
        raise ValueError(f"l_n must be positive, got {l_n}")
    if f_y <= 0:
        raise ValueError(f"f_y must be positive, got {f_y}")
    base = l_n / 33.0 if interior_panel else l_n / 30.0
    f_y_MPa = f_y / 1.0e6
    return float(base * (0.4 + f_y_MPa / 700.0))
