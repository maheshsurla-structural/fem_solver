"""IS 13920:2016 — Ductile Detailing of RC Structures Subjected to
Seismic Forces.

Capacity-design rules and minimum-detailing requirements for ductile
RC frames + walls. This module covers:

* **Strong-column / weak-beam** (Cl. 7.2.1): sum of column moment
  capacities at a joint must be at least 1.4 times the sum of beam
  moment capacities framing into the same joint (sway in either
  direction).
* **Beam capacity-design shear** (Cl. 6.3.3): the shear demand on
  beams is the larger of (a) factored gravity + EQ from analysis and
  (b) the capacity-design shear ``V_p = 1.4 (M_pr_left + M_pr_right) / L_n``
  where ``M_pr`` is the probable moment capacity (1.25 over-strength).
* **Column capacity-design shear** (Cl. 7.5): ``V_p`` from the
  probable moments at top and bottom of the column.
* **Confinement requirements** (Cl. 7.6): minimum confining-reinforcement
  ratio rho_st in plastic-hinge zones.

References
----------
* IS 13920:2016. *Ductile Design and Detailing of Reinforced Concrete
  Structures Subjected to Seismic Forces -- Code of Practice*. BIS.
"""
from __future__ import annotations

from dataclasses import dataclass


# ============================================================ SCWB

@dataclass
class SCWBResult:
    sum_Mc: float                  # sum of column moment capacities (N·m)
    sum_Mb: float                  # sum of beam moment capacities (N·m)
    ratio: float                   # sum_Mc / sum_Mb
    passes: bool                   # ratio >= 1.4
    limit: float = 1.4


def is13920_scwb_check(
    *,
    sum_Mc: float, sum_Mb: float, limit: float = 1.4,
) -> SCWBResult:
    """Strong-column / weak-beam check per IS 13920 Cl. 7.2.1.

    Parameters
    ----------
    sum_Mc : float
        Sum of nominal column flexural capacities at a joint (N·m).
        Compute over both columns framing into the joint (above and
        below), considering both sway directions.
    sum_Mb : float
        Sum of nominal beam flexural capacities at the joint (N·m).
    limit : float, default 1.4
        Required ratio. Some specifications use 1.2; IS 13920 uses 1.4.
    """
    if sum_Mb <= 0.0:
        raise ValueError("sum_Mb must be > 0")
    if sum_Mc < 0.0 or limit <= 0.0:
        raise ValueError("sum_Mc must be >= 0 and limit > 0")
    ratio = sum_Mc / sum_Mb
    return SCWBResult(
        sum_Mc=sum_Mc, sum_Mb=sum_Mb,
        ratio=ratio, passes=bool(ratio >= limit), limit=limit,
    )


# ============================================================ capacity shear -- beam

@dataclass
class CapacityShearBeamResult:
    V_p: float          # capacity-design (probable) shear (N)
    V_gravity: float    # factored-gravity contribution
    V_design: float     # final V_u = max(V_p, V_gravity-based)
    M_pr_left: float
    M_pr_right: float


def is13920_capacity_shear_beam(
    *,
    M_n_pos_left: float, M_n_neg_left: float,
    M_n_pos_right: float, M_n_neg_right: float,
    L_n: float,
    V_gravity: float,
    overstrength: float = 1.25,
) -> CapacityShearBeamResult:
    """Beam capacity-design shear per IS 13920 Cl. 6.3.3.

    Two sway directions are considered; the larger ``V_p`` is taken.
    For sway-right (positive at left, negative at right):

        V_p = overstrength · (M_n_pos_left + M_n_neg_right) / L_n

    For sway-left:

        V_p = overstrength · (M_n_neg_left + M_n_pos_right) / L_n

    Parameters
    ----------
    M_n_pos_left, M_n_neg_left : float
        Beam positive and negative nominal moment capacities at the
        LEFT support (N·m).
    M_n_pos_right, M_n_neg_right : float
        Same at the RIGHT support.
    L_n : float
        Clear span (m).
    V_gravity : float
        Factored shear from gravity (1.2 DL + 1.2 LL or 0.9 DL) (N).
    overstrength : float, default 1.25
        ``M_pr / M_n`` ratio per IS 13920 (sometimes 1.4).
    """
    if L_n <= 0.0:
        raise ValueError("L_n must be > 0")
    M_pr_pl = overstrength * M_n_pos_left
    M_pr_nl = overstrength * M_n_neg_left
    M_pr_pr = overstrength * M_n_pos_right
    M_pr_nr = overstrength * M_n_neg_right
    V_p_sway_right = (M_pr_pl + M_pr_nr) / L_n
    V_p_sway_left = (M_pr_nl + M_pr_pr) / L_n
    V_p = max(V_p_sway_right, V_p_sway_left)
    V_design = max(V_p, V_gravity)
    return CapacityShearBeamResult(
        V_p=V_p, V_gravity=V_gravity, V_design=V_design,
        M_pr_left=max(M_pr_pl, M_pr_nl),
        M_pr_right=max(M_pr_pr, M_pr_nr),
    )


# ============================================================ capacity shear -- column

@dataclass
class CapacityShearColumnResult:
    V_p: float
    V_analysis: float
    V_design: float


def is13920_capacity_shear_column(
    *,
    M_n_top: float, M_n_bot: float,
    h_clear: float,
    V_analysis: float,
    overstrength: float = 1.4,
) -> CapacityShearColumnResult:
    """Column capacity-design shear per IS 13920 Cl. 7.5.

    Parameters
    ----------
    M_n_top, M_n_bot : float
        Nominal moment capacities at the top and bottom of the column
        clear height (N·m).
    h_clear : float
        Column clear height (m).
    V_analysis : float
        Maximum factored shear from analysis (N).
    overstrength : float, default 1.4
        Probable-moment overstrength factor.
    """
    if h_clear <= 0.0:
        raise ValueError("h_clear must be > 0")
    M_pr_top = overstrength * M_n_top
    M_pr_bot = overstrength * M_n_bot
    V_p = (M_pr_top + M_pr_bot) / h_clear
    V_design = max(V_p, V_analysis)
    return CapacityShearColumnResult(
        V_p=V_p, V_analysis=V_analysis, V_design=V_design,
    )


# ============================================================ confinement

@dataclass
class ConfinementResult:
    rho_st_required: float        # transverse reinforcement volumetric ratio
    s_max_required: float         # maximum spacing of hoops (m)
    notes: list


def is13920_confinement(
    *,
    A_g: float, A_k: float,
    f_ck: float, f_yh: float,
    h_clear: float, D: float,
    is_rectangular: bool = True,
) -> ConfinementResult:
    """Confinement reinforcement per IS 13920 Cl. 7.6.

    For rectangular hoops::

        A_sh / (s_v h_c) = 0.18 (s_v h / A_k) (f_ck / f_yh) (A_g/A_k - 1)
                            (simplified to a volumetric ratio bound)

    For spiral hoops::

        rho_s = 0.09 (f_ck / f_yh) (A_g/A_k - 1)

    Maximum spacing s_max:

        - rectangular: min(B/4, 100 mm)  in plastic-hinge region
        - over rest of column: min(B/2, 200 mm)

    Parameters
    ----------
    A_g : float
        Gross cross-section area (m^2).
    A_k : float
        Confined core area (m^2), inside the hoop centerline.
    f_ck, f_yh : float
        Concrete and hoop steel strengths (Pa).
    h_clear : float
        Column clear height (m); used to define plastic-hinge length
        L_o (Cl. 7.6.1).
    D : float
        Larger cross-section dimension (m).
    is_rectangular : bool, default True
    """
    if A_k <= 0.0 or A_g <= A_k:
        raise ValueError("require A_k > 0 and A_g > A_k")
    if f_ck <= 0.0 or f_yh <= 0.0:
        raise ValueError("f_ck and f_yh must be > 0")
    ratio_term = A_g / A_k - 1.0
    if is_rectangular:
        rho_required = 0.18 * (f_ck / f_yh) * ratio_term
        s_max = min(D / 4.0, 0.100)
    else:
        rho_required = 0.09 * (f_ck / f_yh) * ratio_term
        s_max = min(D / 4.0, 0.075)
    L_o = max(D, h_clear / 6.0, 0.450)
    notes = [
        f"Plastic-hinge zone length L_o = {L_o*1000:.0f} mm (max of D, h/6, 450 mm)",
        f"Within L_o: spacing <= {s_max*1000:.0f} mm",
        "Outside L_o: spacing may relax per Cl. 7.6.2",
    ]
    return ConfinementResult(
        rho_st_required=float(rho_required),
        s_max_required=float(s_max),
        notes=notes,
    )
