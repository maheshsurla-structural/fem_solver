"""SteelMemberDesigner -- aggregated checks + auto-sizing for W-shapes.

Composes the per-action checks from Phases 30.2-30.5 into a single
member-level evaluation, and exposes an auto-sizer that picks the
lightest section from a candidate list that satisfies every
governing limit state.

Workflow
--------
::

    demand = SteelMemberDemand(
        P_u=300e3, M_ux=200e3, M_uy=20e3, V_u=80e3,
    )                                     # signed Pu (+compression / -tension)
    res = SteelMemberDesigner.auto_size(
        material=astm_a992(),
        demand=demand,
        L=14 * FT,                         # for axial buckling
        L_b=14 * FT,                       # unbraced for LTB
        C_b=1.0,
    )
    if res.success:
        print(res.best.section.designation, res.best.governing_DCR)

By default, ``auto_size`` searches the full embedded W-shapes catalog.
You can restrict to a specific family (e.g. ``w_series("W14")``) when
auto-sizing within a depth constraint imposed by the architecture.

The aggregated check tracks **all** active limit states; the governing
DCR is ``max(combined.DCR, V_u / φV_n)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from femsolver.design.steel.combined import (
    CombinedForceCheck,
    combined_force_check,
)
from femsolver.design.steel.compression import (
    CompressionCheck,
    compression_strength,
)
from femsolver.design.steel.flexure import FlexureCheck, flexural_strength
from femsolver.design.steel.sections import (
    SteelMaterial,
    SteelSection,
    all_designations,
    get_section,
)
from femsolver.design.steel.tension_shear import (
    ShearCheck,
    TensionCheck,
    shear_strength,
    tension_strength,
)


# ============================================================ demand

@dataclass
class SteelMemberDemand:
    """Required forces on a steel member.

    Attributes
    ----------
    P_u : float
        Required axial force (N). **Positive = compression**,
        negative = tension. Defaults to 0 (pure bending case).
    M_ux : float
        Required strong-axis moment (N·m, magnitude used).
    M_uy : float
        Required weak-axis moment (N·m, magnitude used).
    V_u : float
        Required shear (N, magnitude used).
    """

    P_u: float = 0.0
    M_ux: float = 0.0
    M_uy: float = 0.0
    V_u: float = 0.0

    @property
    def has_axial_or_flexure(self) -> bool:
        return (abs(self.P_u) > 0.0
                or abs(self.M_ux) > 0.0
                or abs(self.M_uy) > 0.0)

    @property
    def has_shear(self) -> bool:
        return abs(self.V_u) > 0.0


# ============================================================ aggregated check

@dataclass
class SteelMemberCheck:
    """Aggregated AISC 360-22 check on one specific section.

    Attributes
    ----------
    section : SteelSection
    combined : CombinedForceCheck or None
        H1 interaction check (computed when there is any P_u, M_ux,
        or M_uy demand). ``None`` for pure-shear cases.
    shear : ShearCheck or None
        Shear check (computed when V_u > 0). ``None`` otherwise.
    governing_DCR : float
        Maximum DCR across all active checks. ``<= 1.0`` means the
        section passes.
    passes : bool
        Convenience flag = ``governing_DCR <= 1.0``.
    weight_per_length : float
        Section weight per unit length (N/m) -- used to compare
        candidates in auto-sizing.
    governing_limit_state : str
        Which check produced ``governing_DCR``: ``"combined"`` (axial +
        flexure interaction), ``"shear"``, or ``"none"`` (zero demand).
    notes : str
    """

    section: SteelSection
    combined: CombinedForceCheck | None
    shear: ShearCheck | None
    governing_DCR: float
    passes: bool
    weight_per_length: float
    governing_limit_state: str
    notes: str = ""


def check_member(
    section: SteelSection,
    material: SteelMaterial,
    demand: SteelMemberDemand,
    *,
    L: float,
    L_b: float | None = None,
    K_x: float = 1.0,
    K_y: float = 1.0,
    L_x: float | None = None,
    L_y: float | None = None,
    C_b: float = 1.0,
    A_e: float | None = None,
) -> SteelMemberCheck:
    """Aggregate per-action checks into a single member-level check.

    Runs the H1 interaction (axial + biaxial flexure) and the §G2
    shear check separately, taking the maximum DCR as governing.
    """
    if L_b is None:
        L_b = L

    combined: CombinedForceCheck | None = None
    if demand.has_axial_or_flexure:
        combined = combined_force_check(
            section, material,
            P_r=demand.P_u, M_rx=demand.M_ux, M_ry=demand.M_uy,
            L=L, L_b=L_b, K_x=K_x, K_y=K_y,
            L_x=L_x, L_y=L_y, C_b=C_b, A_e=A_e,
        )

    shear: ShearCheck | None = None
    shear_dcr = 0.0
    if demand.has_shear:
        shear = shear_strength(section, material)
        if shear.phi_V_n > 0.0:
            shear_dcr = abs(demand.V_u) / shear.phi_V_n
        else:
            shear_dcr = float("inf")

    combined_dcr = combined.DCR if combined is not None else 0.0

    governing_DCR = max(combined_dcr, shear_dcr)
    if governing_DCR == 0.0 and not (
        demand.has_axial_or_flexure or demand.has_shear
    ):
        governing_DCR = 0.0
        governing_limit_state = "none"
    elif shear_dcr > combined_dcr:
        governing_limit_state = "shear"
    elif combined_dcr > 0.0:
        governing_limit_state = "combined"
    else:
        governing_limit_state = "shear"

    notes_list: list[str] = []
    if combined is not None and combined.notes:
        notes_list.append(f"combined: {combined.notes}")
    if shear is not None and shear.notes:
        notes_list.append(f"shear: {shear.notes}")

    return SteelMemberCheck(
        section=section,
        combined=combined,
        shear=shear,
        governing_DCR=governing_DCR,
        passes=(governing_DCR <= 1.0),
        weight_per_length=section.weight_per_length,
        governing_limit_state=governing_limit_state,
        notes="; ".join(notes_list),
    )


# ============================================================ auto-sizing

@dataclass
class SteelDesignResult:
    """Output of an auto-sizing search.

    Attributes
    ----------
    best : SteelMemberCheck or None
        Lightest section in the candidate list that satisfies every
        check (``passes = True``). ``None`` if no candidate passes.
    all_passing : list[SteelMemberCheck]
        Every candidate that passes, sorted from lightest to heaviest.
        Useful for showing 'next-up' alternatives.
    n_candidates_tested : int
    success : bool
        ``True`` if a valid section was found.
    notes : str
    """

    best: SteelMemberCheck | None
    all_passing: list[SteelMemberCheck] = field(default_factory=list)
    n_candidates_tested: int = 0
    success: bool = False
    notes: str = ""


def auto_size(
    material: SteelMaterial,
    demand: SteelMemberDemand,
    *,
    L: float,
    L_b: float | None = None,
    candidates: list[SteelSection] | None = None,
    K_x: float = 1.0,
    K_y: float = 1.0,
    L_x: float | None = None,
    L_y: float | None = None,
    C_b: float = 1.0,
    A_e: float | None = None,
) -> SteelDesignResult:
    """Search a list of candidate W-shapes for the lightest section
    that satisfies every AISC 360-22 limit state under the given
    demand.

    Parameters
    ----------
    material : SteelMaterial
    demand : SteelMemberDemand
    L : float
        Member length for axial buckling (m).
    L_b : float, optional
        Unbraced length for LTB. Defaults to ``L``.
    candidates : list[SteelSection], optional
        Sections to consider. Defaults to the full embedded catalog.
        For depth-constrained design, pass e.g. ``w_series("W14")``.
    K_x, K_y, L_x, L_y, C_b, A_e
        Standard per-check parameters; defaults pass straight through
        to :func:`check_member`.

    Returns
    -------
    SteelDesignResult
        With ``best`` = lightest passing section (or ``None``).
    """
    if candidates is None:
        candidates = [get_section(d) for d in all_designations()]
    if not candidates:
        return SteelDesignResult(
            best=None, all_passing=[], n_candidates_tested=0,
            success=False,
            notes="candidate list is empty",
        )

    passing: list[SteelMemberCheck] = []
    for sec in candidates:
        try:
            chk = check_member(
                sec, material, demand,
                L=L, L_b=L_b, K_x=K_x, K_y=K_y,
                L_x=L_x, L_y=L_y, C_b=C_b, A_e=A_e,
            )
        except Exception:
            # Skip sections that fail to evaluate (e.g. degenerate)
            continue
        if chk.passes:
            passing.append(chk)

    passing.sort(key=lambda c: c.weight_per_length)
    best = passing[0] if passing else None

    notes_list: list[str] = []
    if not passing:
        notes_list.append(
            f"no section in the candidate list ({len(candidates)} "
            "sections) satisfied all checks -- enlarge candidates or "
            "reduce demand"
        )

    return SteelDesignResult(
        best=best,
        all_passing=passing,
        n_candidates_tested=len(candidates),
        success=(best is not None),
        notes="; ".join(notes_list),
    )


# ============================================================ facade

class SteelMemberDesigner:
    """Facade exposing the member-level check and auto-sizer.

    Examples
    --------
    >>> from femsolver.design.steel import (
    ...     SteelMemberDesigner, SteelMemberDemand, astm_a992,
    ... )
    >>> res = SteelMemberDesigner.auto_size(
    ...     material=astm_a992(),
    ...     demand=SteelMemberDemand(P_u=300e3, M_ux=200e3, V_u=80e3),
    ...     L=3.5,
    ... )
    >>> res.best.section.designation
    'W...'
    """

    check_member = staticmethod(check_member)
    auto_size = staticmethod(auto_size)
