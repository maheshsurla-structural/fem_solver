"""Punching-shear reinforcement design.

When the demand stress ``v_u`` exceeds ``phi v_c`` (capacity without
reinforcement), the slab needs **stud rails** or **stirrups** at the
column. This module sizes that reinforcement per ACI 318-19, EC2, and
IS 456.

Three failure modes to check when designing punching reinforcement:

1. **Required reinforcement** to bridge the gap between demand and
   the reduced concrete contribution.
2. **Maximum nominal capacity ceiling** -- if ``v_u/phi`` exceeds the
   ceiling, the slab is too thin and must be redesigned (no
   reinforcement can save it).
3. **Outermost perimeter** -- reinforcement is needed until the
   critical section is far enough that unreinforced concrete alone
   suffices (``v_u <= phi v_c_outside``).

The ACI 318-19 §22.6.7 model:

* With reinforcement, the concrete contribution drops to
  ``v_c_with_reinforcement = (1/6) sqrt(f'_c)`` MPa for stud rails,
  ``(1/12) sqrt(f'_c)`` for stirrups (vs ``(1/3) sqrt(f'_c)`` without).
* The shear-reinforcement contribution per peripheral line is
  ``v_s = A_v f_yt / (b_0 s)`` where ``A_v`` is the total area of
  reinforcement on one peripheral line and ``s`` is the radial
  spacing.
* Ceiling: ``v_u/phi <= 0.5 sqrt(f'_c)`` (stud rails) or
  ``0.33 sqrt(f'_c)`` (stirrups).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PunchingReinforcementResult:
    """Outcome of a punching-shear-reinforcement design check."""
    required: bool
    feasible: bool
    v_u: float                  # demand stress (Pa)
    v_c_with_reinf: float       # reduced concrete contribution (Pa)
    v_s_required: float         # required steel contribution per peripheral line (Pa)
    ceiling: float              # nominal-capacity ceiling (Pa)
    A_v_required: float         # m^2 of reinforcement per peripheral line
    s_max: float                # max radial spacing between peripheral lines (m)
    s_first: float              # max distance from column face to first peripheral line (m)
    code: str
    reinforcement_type: str
    note: str = ""


# ============================================================ ACI 318

def aci318_punching_reinforcement(
    *,
    v_u: float,
    f_c: float,
    f_yt: float,
    d: float,
    b_0: float,
    reinforcement_type: str = "stud_rail",
    phi: float = 0.75,
) -> PunchingReinforcementResult:
    """Design punching-shear reinforcement per ACI 318-19 §22.6.7.

    Parameters
    ----------
    v_u : float
        Factored demand stress at the critical section (Pa).
    f_c : float
        Concrete compressive strength (Pa).
    f_yt : float
        Yield strength of shear reinforcement (Pa). Capped at
        420 MPa (ACI 318-19 22.4.3.2).
    d : float
        Effective slab depth (m).
    b_0 : float
        Perimeter of the critical section (m).
    reinforcement_type : {"stud_rail", "stirrup"}, default "stud_rail"
        Stud rails (headed shear studs) allow a higher capacity
        ceiling than conventional stirrups.
    phi : float, default 0.75
        Strength reduction factor for shear.

    Returns
    -------
    :class:`PunchingReinforcementResult`.
    """
    if v_u < 0:
        raise ValueError("v_u must be non-negative")
    if f_c <= 0:
        raise ValueError("f_c must be positive")
    if f_yt <= 0 or d <= 0 or b_0 <= 0:
        raise ValueError("f_yt, d, b_0 must be positive")
    if reinforcement_type not in ("stud_rail", "stirrup"):
        raise ValueError(
            f"reinforcement_type must be 'stud_rail' or 'stirrup', "
            f"got {reinforcement_type!r}"
        )
    if phi <= 0 or phi > 1.0:
        raise ValueError(f"phi must be in (0, 1], got {phi}")

    # Cap f_yt per ACI 318-19 22.4.3.2 (60 ksi = 420 MPa for stirrups,
    # 80 ksi = 550 MPa for stud rails). Use the lower for safety.
    f_yt_cap = 420e6 if reinforcement_type == "stirrup" else 550e6
    f_yt_used = min(f_yt, f_yt_cap)

    f_c_MPa = f_c / 1.0e6
    sqrt_fc = math.sqrt(f_c_MPa) * 1.0e6      # back to Pa-equivalent

    # Concrete contribution WITH reinforcement (reduced):
    if reinforcement_type == "stud_rail":
        v_c_red_MPa = (1.0 / 6.0) * math.sqrt(f_c_MPa)
        ceiling_MPa = 0.5 * math.sqrt(f_c_MPa)
    else:  # stirrup
        v_c_red_MPa = (1.0 / 12.0) * math.sqrt(f_c_MPa)
        ceiling_MPa = 0.33 * math.sqrt(f_c_MPa)
    v_c_red = v_c_red_MPa * 1.0e6
    ceiling = ceiling_MPa * 1.0e6

    # Without-reinforcement capacity (ACI 318-19 22.6.5.2(a))
    v_c_unreinf = (1.0 / 3.0) * sqrt_fc

    # Check if reinforcement is required
    required = v_u > phi * v_c_unreinf
    feasible = v_u <= phi * ceiling

    if not required:
        # Just the unreinforced check passes; no reinforcement needed
        return PunchingReinforcementResult(
            required=False,
            feasible=True,
            v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=0.0,
            ceiling=float(ceiling),
            A_v_required=0.0,
            s_max=0.0,
            s_first=0.0,
            code="ACI 318-19",
            reinforcement_type=reinforcement_type,
            note="v_u <= phi*v_c; no reinforcement needed",
        )

    if not feasible:
        # Even with maximum reinforcement, slab is too thin
        return PunchingReinforcementResult(
            required=True,
            feasible=False,
            v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=float("inf"),
            ceiling=float(ceiling),
            A_v_required=float("inf"),
            s_max=0.0,
            s_first=0.0,
            code="ACI 318-19",
            reinforcement_type=reinforcement_type,
            note=(
                f"v_u/phi = {v_u/phi/1e6:.2f} MPa exceeds ceiling "
                f"{ceiling_MPa:.2f} MPa; SLAB TOO THIN -- redesign"
            ),
        )

    # Required steel contribution
    v_s_required = v_u / phi - v_c_red

    # Spacing limits (ACI 318-19 8.7.6 & 22.6.7.2):
    # - First peripheral line: <= d/2 from column face
    # - Successive lines: <= 0.5d (stirrup) or 0.75d (stud, v_u <= 0.5 sqrt f'_c)
    # - Otherwise stud spacing capped at 0.5d
    if reinforcement_type == "stud_rail":
        # If v_u/phi <= 0.5 sqrt(f'_c) -> spacing up to 0.75d allowed
        if v_u / phi <= 0.5 * math.sqrt(f_c_MPa) * 1.0e6:
            s_max = 0.75 * d
        else:
            s_max = 0.5 * d
    else:
        s_max = 0.5 * d
    s_first = 0.5 * d

    # Required area per peripheral line: v_s = A_v * f_yt / (b_0 * s_max)
    A_v_required = v_s_required * b_0 * s_max / f_yt_used

    return PunchingReinforcementResult(
        required=True,
        feasible=True,
        v_u=float(v_u),
        v_c_with_reinf=float(v_c_red),
        v_s_required=float(v_s_required),
        ceiling=float(ceiling),
        A_v_required=float(A_v_required),
        s_max=float(s_max),
        s_first=float(s_first),
        code="ACI 318-19",
        reinforcement_type=reinforcement_type,
        note=(
            f"DCR = v_u / (phi*ceiling) = "
            f"{v_u / (phi * ceiling):.3f}"
        ),
    )


# ============================================================ EC2

def eurocode_punching_reinforcement(
    *,
    v_u: float,
    f_ck: float,
    f_ywd: float,
    d: float,
    u_1: float,
    sin_alpha: float = 1.0,
    gamma_c: float = 1.5,
) -> PunchingReinforcementResult:
    """EC2 (EN 1992-1-1) 6.4.5 punching-shear reinforcement design.

    EC2 formula::

        v_Rd,cs = 0.75 v_Rd,c + 1.5 (d/s_r) A_sw f_ywd,ef sin(alpha) / u_1

    where the *effective* yield strength is::

        f_ywd,ef = 250 + 0.25 d  <=  f_ywd     (MPa, d in mm)

    Parameters
    ----------
    v_u : float
        Demand stress at u_1 (Pa).
    f_ck : float
        Characteristic concrete strength (Pa).
    f_ywd : float
        Design yield of shear reinforcement (Pa).
    d : float
        Effective depth (m).
    u_1 : float
        Critical perimeter at 2d (m).
    sin_alpha : float, default 1.0
        Sine of the angle between shear reinforcement and slab (1 = perpendicular).
    gamma_c : float, default 1.5
    """
    if v_u < 0:
        raise ValueError("v_u must be non-negative")
    if any(x <= 0 for x in (f_ck, f_ywd, d, u_1)):
        raise ValueError("f_ck, f_ywd, d, u_1 must be positive")

    f_ck_MPa = f_ck / 1.0e6
    d_mm = d * 1000.0

    # Concrete contribution unreinforced (EC2 6.4.4) -- needs rho_l;
    # take a typical 0.5 % flexural ratio to estimate the bound.
    rho_l = 0.005
    k = min(1.0 + math.sqrt(200.0 / d_mm), 2.0)
    C_Rd_c = 0.18 / gamma_c
    v_Rd_c_MPa = C_Rd_c * k * (100.0 * rho_l * f_ck_MPa) ** (1.0 / 3.0)
    v_Rd_c = v_Rd_c_MPa * 1.0e6

    # With reinforcement: concrete reduced to 0.75 * v_Rd,c
    v_c_red = 0.75 * v_Rd_c

    # Effective design yield (EC2 6.4.5)
    f_ywd_ef = min(250e6 + 0.25 * d * 1e9, f_ywd)
    # The above mixes units -- redo: f_ywd_ef_MPa = 250 + 0.25*d_mm
    f_ywd_ef_MPa = min(250.0 + 0.25 * d_mm, f_ywd / 1e6)
    f_ywd_ef = f_ywd_ef_MPa * 1.0e6

    # EC2 ceiling: v_Ed,max <= 0.5 v · f_cd (EC2 6.4.5(3))
    # nu = 0.6(1 - f_ck/250) (in MPa)
    nu = 0.6 * (1.0 - f_ck_MPa / 250.0)
    f_cd = f_ck / gamma_c
    ceiling = 0.5 * nu * f_cd

    required = v_u > v_Rd_c
    feasible = v_u <= ceiling

    if not required:
        return PunchingReinforcementResult(
            required=False, feasible=True, v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=0.0, ceiling=float(ceiling),
            A_v_required=0.0, s_max=0.0, s_first=0.0,
            code="EC2",
            reinforcement_type="stud_rail",
            note="v_u <= v_Rd,c; no reinforcement needed",
        )
    if not feasible:
        return PunchingReinforcementResult(
            required=True, feasible=False, v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=float("inf"),
            ceiling=float(ceiling),
            A_v_required=float("inf"),
            s_max=0.0, s_first=0.0,
            code="EC2",
            reinforcement_type="stud_rail",
            note=(
                f"v_u = {v_u/1e6:.2f} MPa exceeds ceiling "
                f"{ceiling/1e6:.2f} MPa; SLAB TOO THIN"
            ),
        )

    # Required steel contribution per perimeter
    v_s_required = v_u - v_c_red
    # Spacing limits (EC2 9.4.3): s_r <= 0.75 d
    s_max = 0.75 * d
    s_first = 0.5 * d
    # A_sw required: from v_s = 1.5 (d/s_r) A_sw f_ywd,ef sin(alpha) / u_1
    A_sw_required = v_s_required * s_max * u_1 / (1.5 * d * f_ywd_ef * sin_alpha)
    return PunchingReinforcementResult(
        required=True, feasible=True, v_u=float(v_u),
        v_c_with_reinf=float(v_c_red),
        v_s_required=float(v_s_required),
        ceiling=float(ceiling),
        A_v_required=float(A_sw_required),
        s_max=float(s_max), s_first=float(s_first),
        code="EC2",
        reinforcement_type="stud_rail",
        note=f"f_ywd,ef = {f_ywd_ef/1e6:.0f} MPa (capped)",
    )


# ============================================================ IS 456

def is456_punching_reinforcement(
    *,
    v_u: float,
    f_ck: float,
    f_y: float,
    d: float,
    b_0: float,
    beta_c: float = 1.0,
) -> PunchingReinforcementResult:
    """IS 456 (2000) Cl 31.6.3 punching-shear reinforcement design.

    Per IS 456, the *with reinforcement* concrete contribution is the
    same as without (k_s * 0.25 * sqrt(f_ck)), and the reinforcement
    supplies the additional capacity::

        V_us = 0.87 * f_y * A_sv

    distributed around the critical perimeter at spacing s.

    Parameters
    ----------
    v_u : float
        Factored shear stress at the critical section (Pa).
    f_ck : float
        Characteristic concrete strength (Pa).
    f_y : float
        Yield strength of shear reinforcement (Pa).
    d : float
        Effective depth (m).
    b_0 : float
        Critical-section perimeter at d/2 from column face (m).
    beta_c : float, default 1.0
        Column short/long ratio (k_s factor depends on this).
    """
    if v_u < 0:
        raise ValueError("v_u must be non-negative")
    if any(x <= 0 for x in (f_ck, f_y, d, b_0)):
        raise ValueError("f_ck, f_y, d, b_0 must be positive")

    f_ck_MPa = f_ck / 1.0e6
    k_s = min(0.5 + beta_c, 1.0)
    tau_c_unreinf_MPa = k_s * 0.25 * math.sqrt(f_ck_MPa)
    tau_c_unreinf = tau_c_unreinf_MPa * 1.0e6

    # IS 456 ceiling: 1.5 * tau_c_unreinf (Cl 31.6.3.2)
    ceiling = 1.5 * tau_c_unreinf

    # With reinforcement, concrete contribution is reduced to
    # 0.5 * tau_c per IS 456 conservative practice
    v_c_red = 0.5 * tau_c_unreinf

    required = v_u > tau_c_unreinf
    feasible = v_u <= ceiling

    if not required:
        return PunchingReinforcementResult(
            required=False, feasible=True, v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=0.0, ceiling=float(ceiling),
            A_v_required=0.0, s_max=0.0, s_first=0.0,
            code="IS 456",
            reinforcement_type="stirrup",
            note="v_u <= tau_c; no reinforcement needed",
        )
    if not feasible:
        return PunchingReinforcementResult(
            required=True, feasible=False, v_u=float(v_u),
            v_c_with_reinf=float(v_c_red),
            v_s_required=float("inf"),
            ceiling=float(ceiling),
            A_v_required=float("inf"),
            s_max=0.0, s_first=0.0,
            code="IS 456",
            reinforcement_type="stirrup",
            note=f"v_u exceeds 1.5*tau_c ceiling; SLAB TOO THIN",
        )

    v_s_required = v_u - v_c_red
    # Spacing limit: s <= 0.5 d (IS 456 Cl 26.5.1)
    s_max = 0.5 * d
    s_first = 0.5 * d
    # V_us = 0.87 * f_y * A_sv distributed over b_0 at spacing s
    # v_s * b_0 * s = 0.87 * f_y * A_sv
    A_sv_required = v_s_required * b_0 * s_max / (0.87 * f_y)
    return PunchingReinforcementResult(
        required=True, feasible=True, v_u=float(v_u),
        v_c_with_reinf=float(v_c_red),
        v_s_required=float(v_s_required),
        ceiling=float(ceiling),
        A_v_required=float(A_sv_required),
        s_max=float(s_max), s_first=float(s_first),
        code="IS 456",
        reinforcement_type="stirrup",
        note=f"k_s = {k_s:.3f}",
    )
