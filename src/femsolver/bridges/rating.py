"""AASHTO LRFR load rating — Manual for Bridge Evaluation (MBE) §6A
(bridge plan T2.3).

Load rating answers "how much live load can this *existing* bridge safely
carry?" — the everyday task of a bridge-evaluation engineer, and a headline
MIDAS Civil / CSiBridge capability. The Load-and-Resistance-Factor **Rating**
(LRFR) method expresses the answer as a dimensionless **rating factor** (MBE
Eq. 6A.4.2.1-1):

    RF = ( C - gamma_DC * DC - gamma_DW * DW -/+ gamma_P * P )
         --------------------------------------------------------
                        gamma_LL * ( LL + IM )

* **C** — member capacity.  For a strength limit state
  ``C = phi_c * phi_s * phi * Rn`` (condition factor x system factor x LRFD
  resistance factor x nominal resistance), with the floor
  ``phi_c * phi_s >= 0.85`` (MBE 6A.4.2.1).  For a service limit state ``C``
  is an allowable stress ``f_R`` (pass it as ``Rn`` with the phi-factors 1).
* **DC / DW** — dead-load effects of structural components (DC) and of the
  wearing surface and utilities (DW).
* **P** — permanent loads other than dead load (e.g. secondary prestress);
  pass it signed in the same sense as the effect being rated (positive =
  unfavourable), it is subtracted with ``gamma_P``.
* **LL + IM** — the live-load effect *including* the dynamic load allowance
  (and lane load for HL-93).  Feed it the governing value from a moving-load
  run (:func:`~femsolver.bridges.moving_load.aashto_hl93_envelope`), or use
  :func:`rate_from_influence_line` which does that for you.

``RF >= 1`` means the bridge carries the rating live load.  The rating in
**tons** is ``RF x W`` where ``W`` is the weight of the rating vehicle
(:meth:`RatingResult.rating_tons`).

The load-factor sets are the canonical MBE Strength-I values:

===========  =======  =======  ======================================
Level        gamma_DC gamma_DW gamma_LL
===========  =======  =======  ======================================
Inventory     1.25     1.50    1.75
Operating     1.25     1.50    1.35
Legal         1.25     1.50    1.40 - 1.80  (by ADTT, Table 6A.4.4.2.3a-1)
Permit        1.25     1.50    caller / permit type (Table 6A.4.5.4.2a-1)
===========  =======  =======  ======================================

Everything here is deterministic arithmetic on top of the moving-load engine;
it is validated against closed-form hand calculations in
``tests/test_bridge_rating.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Load-factor sets (MBE Table 6A.4.2.2-1, Strength I)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadFactors:
    """A set of LRFR load factors for one rating level (MBE 6A.4.2.2).

    ``gamma_DW`` defaults to 1.50; use 1.25 when the wearing-surface
    thickness is field-measured.  ``gamma_P`` factors permanent loads other
    than dead load (default 1.0).
    """

    gamma_DC: float = 1.25
    gamma_DW: float = 1.50
    gamma_LL: float = 1.75
    gamma_P: float = 1.0
    name: str = ""
    level: str = ""


def strength_i_inventory(gamma_DW: float = 1.50) -> LoadFactors:
    """Design-load rating at the **Inventory** level (gamma_LL = 1.75) — the
    load the bridge can carry for an indefinite period."""
    return LoadFactors(1.25, gamma_DW, 1.75, 1.0,
                       "Strength I - Inventory", "inventory")


def strength_i_operating(gamma_DW: float = 1.50) -> LoadFactors:
    """Design-load rating at the **Operating** level (gamma_LL = 1.35) — the
    absolute maximum permissible live load."""
    return LoadFactors(1.25, gamma_DW, 1.35, 1.0,
                       "Strength I - Operating", "operating")


# Legal-load live-load factor vs ADTT (MBE Table 6A.4.4.2.3a-1, generalized
# legal loads / routine commercial traffic; linear interpolation permitted).
_LEGAL_GAMMA_LL = ((100.0, 1.40), (1000.0, 1.65), (5000.0, 1.80))


def legal_live_load_factor(adtt: float | None) -> float:
    """Live-load factor for a **legal-load** rating from the one-direction
    ADTT (MBE Table 6A.4.4.2.3a-1, generalized legal loads).

    ``adtt is None`` (unknown ADTT) returns the conservative 1.80.
    Values are linearly interpolated between the tabulated break points
    (100 -> 1.40, 1000 -> 1.65, 5000 -> 1.80) and clamped outside them.
    """
    if adtt is None:
        return 1.80
    lo_adtt, lo_g = _LEGAL_GAMMA_LL[0]
    hi_adtt, hi_g = _LEGAL_GAMMA_LL[-1]
    if adtt <= lo_adtt:
        return lo_g
    if adtt >= hi_adtt:
        return hi_g
    for (a0, g0), (a1, g1) in zip(_LEGAL_GAMMA_LL, _LEGAL_GAMMA_LL[1:]):
        if a0 <= adtt <= a1:
            return g0 + (g1 - g0) * (adtt - a0) / (a1 - a0)
    return hi_g                                            # unreachable


def legal_load(adtt: float | None = None, *, gamma_LL: float | None = None,
               gamma_DW: float = 1.50) -> LoadFactors:
    """Legal-load rating factors (Strength I).  ``gamma_LL`` defaults to the
    ADTT-based value (:func:`legal_live_load_factor`); pass it explicitly to
    override."""
    g = gamma_LL if gamma_LL is not None else legal_live_load_factor(adtt)
    return LoadFactors(1.25, gamma_DW, g, 1.0, "Strength I - Legal", "legal")


def permit_load(gamma_LL: float, *, gamma_DW: float = 1.50) -> LoadFactors:
    """Permit-load rating factors (Strength I).  The permit live-load factor
    (MBE Table 6A.4.5.4.2a-1) depends on the permit type, ADTT, and number of
    loaded lanes and must be supplied by the caller (e.g. 1.15 for a single-
    trip escorted special permit, 1.40 for a routine/annual permit on a busy
    route)."""
    return LoadFactors(1.25, gamma_DW, gamma_LL, 1.0,
                       "Strength I - Permit", "permit")


# --------------------------------------------------------------------------
# Condition (phi_c) and system (phi_s) factors
# --------------------------------------------------------------------------
def condition_factor(*, nbi_rating: int | None = None,
                     condition: str = "good") -> float:
    """Condition factor phi_c (MBE Table 6A.4.2.3-1).

    From the NBI superstructure condition rating when given
    (>=6 good, 5 fair, <=4 poor), otherwise from a ``condition`` keyword
    ("good"/"satisfactory" -> 1.00, "fair" -> 0.95, "poor" -> 0.85).
    """
    if nbi_rating is not None:
        if nbi_rating >= 6:
            return 1.00
        if nbi_rating == 5:
            return 0.95
        return 0.85
    return {"good": 1.00, "satisfactory": 1.00,
            "fair": 0.95, "poor": 0.85}[condition.lower()]


# System factor phi_s for flexural / axial members (MBE Table 6A.4.2.4-1).
# Shear is always 1.00.
_SYSTEM_FACTORS = {
    "welded_two_girder": 0.85,      # welded members, two-girder/truss/arch
    "riveted_two_girder": 0.90,     # riveted members, two-girder/truss/arch
    "multiple_eyebar": 0.90,        # multiple eyebar members in trusses
    "three_girder": 0.85,           # three-girder bridges, spacing 6 ft
    "four_girder": 0.95,            # four-girder bridges, spacing <= 4 ft
    "girder": 1.00,                 # all other girder & slab bridges
    "slab": 1.00,
    "floorbeam": 1.00,              # redundant floorbeams / stringer subsystems
    "nonredundant_floorbeam": 0.85,
}


def system_factor(kind: str = "girder", *, shear: bool = False) -> float:
    """System factor phi_s for flexure/axial (MBE Table 6A.4.2.4-1), keyed by
    superstructure ``kind`` (see ``_SYSTEM_FACTORS``).  ``shear=True`` always
    returns 1.00 (phi_s applies to the flexural/axial capacity only)."""
    if shear:
        return 1.00
    try:
        return _SYSTEM_FACTORS[kind]
    except KeyError as exc:                                 # pragma: no cover
        raise ValueError(
            f"unknown system-factor kind {kind!r}; one of "
            f"{sorted(_SYSTEM_FACTORS)}") from exc


# --------------------------------------------------------------------------
# Core rating factor
# --------------------------------------------------------------------------
@dataclass
class RatingResult:
    """Outcome of a single LRFR rating (MBE Eq. 6A.4.2.1-1)."""

    rf: float                       # rating factor
    level: str                      # "inventory" / "operating" / ...
    capacity: float                 # C = (phi_c*phi_s floored) * phi * Rn
    dead_load_effect: float         # gamma_DC*DC + gamma_DW*DW + gamma_P*P
    live_load_effect: float         # gamma_LL * (LL + IM)
    factors: LoadFactors
    phi: float
    phi_c: float
    phi_s: float
    Rn: float
    DC: float
    DW: float
    P: float
    LL_IM: float
    capacity_floored: bool = False  # phi_c*phi_s hit the 0.85 floor

    @property
    def adequate(self) -> bool:
        """``True`` when the bridge carries the rating live load (RF >= 1)."""
        return self.rf >= 1.0

    def rating_tons(self, vehicle_weight_tons: float) -> float:
        """Rating in tons = RF x rating-vehicle weight (MBE 6A.4.4.4)."""
        return self.rf * vehicle_weight_tons

    def summary(self) -> str:
        tag = self.factors.name or self.level
        return (f"{tag}: RF = {self.rf:.3f} "
                f"({'OK' if self.adequate else 'DEFICIENT'})  "
                f"[C={self.capacity:.3g}, DL={self.dead_load_effect:.3g}, "
                f"LL+IM(factored)={self.live_load_effect:.3g}]")


def rating_factor(*, Rn: float, DC: float, DW: float, LL_IM: float,
                  factors: LoadFactors, phi: float = 1.0,
                  phi_c: float = 1.0, phi_s: float = 1.0,
                  P: float = 0.0) -> RatingResult:
    """Compute the LRFR rating factor (MBE Eq. 6A.4.2.1-1).

    Parameters
    ----------
    Rn : float
        Nominal member resistance (strength limit state) or allowable
        stress / force (service limit state, with the phi-factors 1).
    DC, DW : float
        Unfactored dead-load effects of components (DC) and of the wearing
        surface + utilities (DW), in the same units and sense as ``Rn``.
    LL_IM : float
        Unfactored live-load effect **including** the dynamic load allowance
        (and lane load for HL-93).  Use the governing magnitude for the
        effect being rated (see :func:`rate_from_influence_line`).
    factors : LoadFactors
        Load-factor set (:func:`strength_i_inventory`, ``..._operating``,
        :func:`legal_load`, :func:`permit_load`).
    phi : float
        LRFD resistance factor (strength).  Use 1.0 for a service check.
    phi_c, phi_s : float
        Condition (:func:`condition_factor`) and system
        (:func:`system_factor`) factors.  Their product is floored at 0.85
        per MBE 6A.4.2.1.
    P : float
        Permanent load other than dead load (e.g. secondary prestress),
        signed in the sense of the rated effect (positive = unfavourable);
        subtracted from the capacity with ``gamma_P``.

    Returns
    -------
    RatingResult
    """
    product = phi_c * phi_s
    floored = product < 0.85
    eff_product = 0.85 if floored else product
    C = eff_product * phi * Rn

    dead = (factors.gamma_DC * DC + factors.gamma_DW * DW
            + factors.gamma_P * P)
    live = factors.gamma_LL * abs(LL_IM)

    if live <= 0.0:
        rf = math.inf                                      # no live demand
    else:
        rf = (C - dead) / live
    rf = max(rf, 0.0)                                       # RF is clamped >= 0

    return RatingResult(
        rf=rf, level=factors.level, capacity=C,
        dead_load_effect=dead, live_load_effect=live, factors=factors,
        phi=phi, phi_c=phi_c, phi_s=phi_s, Rn=Rn, DC=DC, DW=DW, P=P,
        LL_IM=abs(LL_IM), capacity_floored=floored)


# --------------------------------------------------------------------------
# Multi-level convenience
# --------------------------------------------------------------------------
@dataclass
class BridgeRating:
    """A member's rating at several LRFR levels (design inventory + operating,
    optionally legal and permit).  ``results`` maps level name -> RatingResult;
    ``controlling`` is the lowest RF among the design-level ratings."""

    results: dict = field(default_factory=dict)

    def __getitem__(self, level: str) -> RatingResult:
        return self.results[level]

    @property
    def controlling(self) -> RatingResult:
        return min(self.results.values(), key=lambda r: r.rf)

    def summary(self) -> str:
        return "\n".join(r.summary() for r in self.results.values())


def rate_member(*, Rn: float, DC: float, DW: float, LL_IM: float,
                phi: float = 1.0, phi_c: float = 1.0, phi_s: float = 1.0,
                P: float = 0.0, gamma_DW: float = 1.50,
                adtt: float | None = None,
                permit_gamma_LL: float | None = None) -> BridgeRating:
    """Rate a member at the standard LRFR levels in one call.

    Always computes design **inventory** and **operating**.  If ``adtt`` is
    given (or you want the conservative legal factor) a **legal** rating is
    added; if ``permit_gamma_LL`` is given a **permit** rating is added.  All
    share the same ``Rn/DC/DW/LL_IM`` and phi-factors.
    """
    common = dict(Rn=Rn, DC=DC, DW=DW, LL_IM=LL_IM,
                  phi=phi, phi_c=phi_c, phi_s=phi_s, P=P)
    out: dict = {
        "inventory": rating_factor(
            factors=strength_i_inventory(gamma_DW), **common),
        "operating": rating_factor(
            factors=strength_i_operating(gamma_DW), **common),
    }
    if adtt is not None:
        out["legal"] = rating_factor(
            factors=legal_load(adtt, gamma_DW=gamma_DW), **common)
    if permit_gamma_LL is not None:
        out["permit"] = rating_factor(
            factors=permit_load(permit_gamma_LL, gamma_DW=gamma_DW),
            **common)
    return BridgeRating(results=out)


# --------------------------------------------------------------------------
# Tie-in to the moving-load engine
# --------------------------------------------------------------------------
def live_load_effect(il, *, im: float = 0.33, lane_load: float = 9.34e3,
                     include_lane: bool = True, sense: str = "governing",
                     n_positions: int = 801) -> float:
    """Governing HL-93 live-load effect (incl. IM) on an influence line.

    Thin wrapper over
    :func:`~femsolver.bridges.moving_load.aashto_hl93_envelope`.  ``sense``
    selects which envelope end to return: ``"max"`` (positive), ``"min"``
    (magnitude of the negative), or ``"governing"`` (the larger magnitude,
    the default for a symmetric-capacity rating).
    """
    from femsolver.bridges.moving_load import aashto_hl93_envelope

    env = aashto_hl93_envelope(il, im=im, lane_load=lane_load,
                               include_lane=include_lane,
                               n_positions=n_positions)
    if sense == "max":
        return abs(env["max"])
    if sense == "min":
        return abs(env["min"])
    return max(abs(env["max"]), abs(env["min"]))


def rate_from_influence_line(il, *, Rn: float, DC: float, DW: float,
                             im: float = 0.33, lane_load: float = 9.34e3,
                             include_lane: bool = True,
                             sense: str = "governing", phi: float = 1.0,
                             phi_c: float = 1.0, phi_s: float = 1.0,
                             P: float = 0.0, gamma_DW: float = 1.50,
                             adtt: float | None = None,
                             permit_gamma_LL: float | None = None,
                             n_positions: int = 801) -> BridgeRating:
    """Rate a member directly from its influence line: sweep the HL-93 live
    load (:func:`live_load_effect`) for the ``LL + IM`` effect, then rate at
    the standard levels (:func:`rate_member`)."""
    ll_im = live_load_effect(il, im=im, lane_load=lane_load,
                             include_lane=include_lane, sense=sense,
                             n_positions=n_positions)
    return rate_member(Rn=Rn, DC=DC, DW=DW, LL_IM=ll_im, phi=phi,
                       phi_c=phi_c, phi_s=phi_s, P=P, gamma_DW=gamma_DW,
                       adtt=adtt, permit_gamma_LL=permit_gamma_LL)
