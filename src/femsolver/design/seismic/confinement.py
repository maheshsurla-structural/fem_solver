"""Confined-concrete reinforcement detailing per ACI 318-19 §18.7.5.

For Special Moment Frame (SMF) columns, the ends of each column must
be confined by closely-spaced transverse reinforcement to provide
ductility under cyclic loading. The provisions are:

§18.7.5.1 -- Length of confined zone ``l_o`` at each end:

    l_o = max(
        depth of the member at the joint face   (typically column h),
        clear height / 6,
        450 mm (18 in)
    )

§18.7.5.3 -- Maximum centre-to-centre spacing ``s_o`` within ``l_o``:

    s_o = min(
        b_min / 4,        # 1/4 of the smallest section dimension
        6 d_b_long,        # 6 longitudinal-bar diameters
        s_x                # = 100 + (350 - h_x)/3  mm
    )

where ``h_x`` is the maximum centre-to-centre horizontal spacing
between crosstie or hoop legs across the section. ``s_x`` is bounded:
``s_x`` should be between 100 mm and 150 mm.

§18.7.5.4 -- Transverse reinforcement area ``A_sh`` per direction
(rectangular hoop):

    A_sh / (s · b_c) ≥ max(
        0.3 (A_g / A_ch - 1) f_c' / f_yt,     (Eq 18.7.5.4a)
        0.09 f_c' / f_yt                       (Eq 18.7.5.4b)
    )

where:

* ``b_c`` -- cross-section dim of the confined core (out-to-out of
  hoop centerlines)
* ``A_ch`` -- area of column core out-to-out of hoops
* ``A_g`` -- gross cross-section area
* ``f_yt`` -- yield stress of the transverse hoop reinforcement
"""
from __future__ import annotations

from dataclasses import dataclass

from femsolver.design.concrete.section import (
    ConcreteSection,
    rebar_area,
    rebar_diameter,
)


@dataclass
class ConfinementDetail:
    """Result of an ACI 18.7.5 confinement-detailing evaluation.

    Attributes
    ----------
    l_o : float
        Confined-zone length at each end of the column (m).
    s_o_required : float
        Maximum allowable hoop centre-to-centre spacing within ``l_o``
        (m) per §18.7.5.3.
    s_o_components : dict
        The three individual bounds (``b_min/4``, ``6 d_b_long``,
        ``s_x``) so the user can see which governs.
    Ash_per_s_required : float
        Required ``A_sh / s`` per direction (m²/m) per §18.7.5.4.
    Ash_per_s_provided : float
        Provided ``A_sh / s`` from the rebar layout.
    spacing_ok : bool
        ``True`` iff the provided stirrup spacing ≤ ``s_o_required``.
    reinforcement_ok : bool
        ``True`` iff provided ``A_sh / s`` ≥ required.
    passes : bool
        ``spacing_ok and reinforcement_ok``.
    notes : str
    """

    l_o: float
    s_o_required: float
    s_o_components: dict
    Ash_per_s_required: float
    Ash_per_s_provided: float
    spacing_ok: bool
    reinforcement_ok: bool
    passes: bool
    notes: str = ""


def _confined_core_dims(section: ConcreteSection) -> tuple[float, float, float]:
    """Return (b_c, h_c, A_ch) for a rectangular column.

    Conservatively uses the section's smallest face cover (top or
    bottom) as the depth of the hoop centerline, ignoring side cover
    (which is typically the same). For our standard layout
    ``b_c = b - 2*cover``, ``h_c = h - 2*cover``.
    """
    cover = max(section.rebar.top_cover, section.rebar.bottom_cover)
    b_c = max(section.b - 2.0 * cover, 0.001)
    h_c = max(section.h - 2.0 * cover, 0.001)
    A_ch = b_c * h_c
    return b_c, h_c, A_ch


def confined_concrete_detailing(
    section: ConcreteSection,
    *,
    column_clear_height: float,
    longitudinal_bar: str | None = None,
    h_x: float | None = None,
) -> ConfinementDetail:
    """Compute confinement reinforcement requirements per ACI 18.7.5
    for the **end region** of an SMF column.

    Parameters
    ----------
    section : ConcreteSection
        The column section with rebar layout (including stirrups).
        For an SMF column, the rebar layout provides both the
        longitudinal reinforcement (used for ``d_b_long``) and the
        transverse hoops (used for ``A_sh / s_provided``).
    column_clear_height : float
        Clear column height between joint faces (m). Used to compute
        ``l_o``.
    longitudinal_bar : str, optional
        Designation of the longitudinal bar (e.g., ``"#8"``). If
        omitted, the largest bar in the layout is used.
    h_x : float, optional
        Maximum centre-to-centre horizontal spacing between crosstie
        or hoop legs across the section (m). If omitted, defaults to
        ``b_c`` (the most-conservative single-hoop case).

    Returns
    -------
    ConfinementDetail
    """
    if column_clear_height <= 0.0:
        raise ValueError(
            f"column_clear_height must be positive, got {column_clear_height}"
        )

    # --- l_o per 18.7.5.1 ---
    member_depth = max(section.b, section.h)
    l_o = max(member_depth, column_clear_height / 6.0, 0.450)

    # --- s_o per 18.7.5.3 ---
    b_min = min(section.b, section.h)
    # Largest longitudinal bar diameter
    if longitudinal_bar is None:
        all_long = list(section.rebar.top_bars) + list(section.rebar.bottom_bars)
        if all_long:
            # Pick the largest
            largest = max(all_long, key=rebar_area)
            d_b_long = rebar_diameter(largest)
        else:
            d_b_long = rebar_diameter("#5")     # fallback default
    else:
        d_b_long = rebar_diameter(longitudinal_bar)

    b_c, h_c, A_ch = _confined_core_dims(section)
    if h_x is None:
        h_x = b_c
    # s_x = (350 - h_x[mm]) / 3 + 100, bounded between 100 and 150 mm.
    h_x_mm = h_x * 1000.0
    s_x_mm = 100.0 + (350.0 - h_x_mm) / 3.0
    s_x = max(0.100, min(0.150, s_x_mm / 1000.0))

    s_o_b = b_min / 4.0
    s_o_db = 6.0 * d_b_long
    s_o_required = min(s_o_b, s_o_db, s_x)
    s_o_components = {
        "b_min/4": s_o_b,
        "6 d_b_long": s_o_db,
        "s_x": s_x,
    }

    # --- A_sh / s per 18.7.5.4 ---
    fc = section.material.fc_prime
    fy = section.material.fy
    A_g = section.b * section.h
    ratio_a = 0.3 * (A_g / A_ch - 1.0) * fc / fy
    ratio_b = 0.09 * fc / fy
    ratio_required = max(ratio_a, ratio_b)
    Ash_per_s_required = ratio_required * b_c

    Ash_per_s_provided = (section.rebar.Av / section.rebar.stirrup_spacing
                            if section.rebar.stirrup_spacing > 0 else 0.0)

    spacing_ok = section.rebar.stirrup_spacing <= s_o_required + 1.0e-9
    reinforcement_ok = (
        Ash_per_s_provided >= Ash_per_s_required - 1.0e-12
    )

    notes_list: list[str] = []
    if not spacing_ok:
        notes_list.append(
            f"stirrup spacing s = {section.rebar.stirrup_spacing*1000:.0f} mm "
            f"exceeds s_o = {s_o_required*1000:.0f} mm (governed by "
            f"{min(s_o_components, key=lambda k: s_o_components[k])}). "
            "Reduce hoop spacing within the l_o end zones."
        )
    if not reinforcement_ok:
        notes_list.append(
            f"A_sh/s = {Ash_per_s_provided*1e6:.0f} mm²/m falls short of "
            f"required {Ash_per_s_required*1e6:.0f} mm²/m. Use larger "
            "hoop bars or more crossties."
        )

    return ConfinementDetail(
        l_o=l_o,
        s_o_required=s_o_required,
        s_o_components=s_o_components,
        Ash_per_s_required=Ash_per_s_required,
        Ash_per_s_provided=Ash_per_s_provided,
        spacing_ok=spacing_ok,
        reinforcement_ok=reinforcement_ok,
        passes=(spacing_ok and reinforcement_ok),
        notes="; ".join(notes_list),
    )
