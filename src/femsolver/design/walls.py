"""ACI 318-19 §18.10 special structural (shear) wall design.

Assembles the axial-flexural (P-M), shear, boundary-element and minimum-
reinforcement provisions for a planar reinforced-concrete wall into a single
demand/capacity check. Inputs are the wall geometry, materials, reinforcement
layout and the pier demand ``(Pu, Mu, Vu)`` — the last of which the desktop app
gets from the shell membrane-stress integration (wall plan W2).

Everything here is closed-form / strip-integration (no FE), pure SI
(N, m, Pa), and code-referenced so it can be checked by hand:

* **P-M capacity** — a rectangular wall section with smeared distributed
  vertical web reinforcement plus concentrated boundary bars at each end, via a
  rectangular (Whitney) stress block and strain-compatibility strip sweep
  (ACI 318-19 §22.2). φ follows the net-tensile-strain classification (§21.2).
* **In-plane shear** — ``Vn = Acv (αc λ √f'c + ρt fy)`` with ``αc`` interpolated
  on ``hw/ℓw`` and capped at ``0.83 Acv √f'c`` (ACI 318-19 §18.10.4).
* **Boundary elements** — the stress-based trigger ``σ > 0.2 f'c`` on the gross
  section (ACI 318-19 §18.10.6.3), with the discontinue-below ``0.15 f'c`` note
  and a boundary-length recommendation.
* **Minimum web reinforcement + two curtains** (ACI 318-19 §18.10.2).

References
----------
* ACI 318-19 §18.10 "Special structural walls".
* ACI 318-19 §22.2 (flexure), §21.2 (strength-reduction factors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

E_S = 200.0e9              # steel elastic modulus (Pa)
EPS_CU = 0.003            # ACI ultimate concrete compressive strain
EPS_TY_REF = 0.002        # reference for compression-controlled (fy/Es ~ Grade 60)


def _sqrt_fc(fc: float) -> float:
    """The ACI empirical ``√f'c`` term as a stress in **Pa**. The code formulas
    are calibrated for ``f'c`` in MPa (``√f'c`` then reads in MPa), so in base
    SI we take ``√(f'c[MPa]) · 1e6``. Mixing ``√(f'c[Pa])`` with the ``ρ·fy``
    term (correctly in Pa) would be dimensionally wrong by ~1000×."""
    return math.sqrt(max(fc, 0.0) / 1.0e6) * 1.0e6


# ============================================================ inputs

@dataclass
class WallGeometry:
    """Planar wall geometry (SI). ``lw`` length, ``t`` web thickness, ``hw``
    total height (for ``hw/ℓw`` and the shear ``αc``)."""
    lw: float
    t: float
    hw: float = 0.0

    @property
    def Ag(self) -> float:
        return self.lw * self.t

    @property
    def Ig(self) -> float:
        return self.t * self.lw ** 3 / 12.0

    @property
    def Acv(self) -> float:
        """Gross area of the wall web resisting shear (ℓw·t)."""
        return self.lw * self.t


@dataclass
class WallMaterial:
    """Concrete + reinforcement strengths (Pa). ``lam`` = λ lightweight factor."""
    fc: float
    fy: float
    fyt: float = 0.0            # transverse (horizontal) bar yield; 0 → fy
    lam: float = 1.0

    def __post_init__(self):
        if self.fyt <= 0.0:
            self.fyt = self.fy


@dataclass
class WallReinforcement:
    """Wall reinforcement. ``rho_l`` distributed vertical (longitudinal) web
    ratio, ``rho_t`` distributed horizontal (transverse) web ratio — both as a
    fraction of the gross web area. ``As_boundary`` is the total vertical steel
    area concentrated in *each* boundary element (m²), its centroid a distance
    ``d_boundary`` from the nearer wall end."""
    rho_l: float = 0.0025
    rho_t: float = 0.0025
    As_boundary: float = 0.0
    d_boundary: float = 0.0


@dataclass
class WallDemand:
    """Factored pier demand (SI). ``Pu`` axial, **compression positive**;
    ``Mu`` in-plane moment; ``Vu`` in-plane shear."""
    Pu: float = 0.0
    Mu: float = 0.0
    Vu: float = 0.0


# ============================================================ shear (§18.10.4)

@dataclass
class WallShearResult:
    alpha_c: float
    Vc: float
    Vs: float
    Vn: float
    Vn_cap: float
    phi: float
    phiVn: float
    dcr: float
    two_curtains_required: bool


def shear_alpha_c(hw_lw: float) -> float:
    """αc for the concrete shear term: 0.25 for squat walls (hw/ℓw ≤ 1.5),
    0.17 for slender (≥ 2.0), linear between (ACI 318-19 §18.10.4.1, SI)."""
    if hw_lw <= 1.5:
        return 0.25
    if hw_lw >= 2.0:
        return 0.17
    return 0.25 + (0.17 - 0.25) * (hw_lw - 1.5) / 0.5


def wall_shear_strength(geom: WallGeometry, mat: WallMaterial,
                        reinf: WallReinforcement, *, Vu: float = 0.0,
                        phi: float = 0.75) -> WallShearResult:
    """In-plane shear strength ``Vn = Acv(αc λ √f'c + ρt fy)`` capped at
    ``0.83 Acv √f'c`` (ACI 318-19 §18.10.4)."""
    Acv = geom.Acv
    hw_lw = (geom.hw / geom.lw) if geom.lw > 0 else 2.0
    ac = shear_alpha_c(hw_lw)
    sqrt_fc = _sqrt_fc(mat.fc)
    Vc = Acv * ac * mat.lam * sqrt_fc
    Vs = Acv * reinf.rho_t * mat.fyt
    Vn = Vc + Vs
    cap = 0.83 * Acv * sqrt_fc
    Vn = min(Vn, cap)
    phiVn = phi * Vn
    dcr = (abs(Vu) / phiVn) if phiVn > 0 else math.inf
    # two curtains of reinforcement (§18.10.2.2)
    two = (abs(Vu) > 0.17 * Acv * mat.lam * sqrt_fc) or (geom.t > 0.25)
    return WallShearResult(alpha_c=ac, Vc=Vc, Vs=Vs, Vn=Vn, Vn_cap=cap,
                           phi=phi, phiVn=phiVn, dcr=dcr,
                           two_curtains_required=two)


# ================================================= minimum web reinf (§18.10.2)

@dataclass
class WallMinReinf:
    rho_l_min: float
    rho_t_min: float
    high_shear: bool
    rho_l_ok: bool
    rho_t_ok: bool


def wall_min_web_reinforcement(geom: WallGeometry, mat: WallMaterial,
                               reinf: WallReinforcement,
                               Vu: float) -> WallMinReinf:
    """Minimum distributed web reinforcement (ACI 318-19 §18.10.2). Above the
    ``0.083 Acv λ√f'c`` shear threshold both ratios must be ≥ 0.0025; below it
    Chapter 11 minimums (≈0.0020 vertical / 0.0025 horizontal-controlled) apply.
    Simplified to 0.0025 / 0.0020 respectively here."""
    thr = 0.083 * geom.Acv * mat.lam * _sqrt_fc(mat.fc)
    high = abs(Vu) > thr
    rho_t_min = 0.0025 if high else 0.0020
    rho_l_min = 0.0025 if high else 0.0020
    return WallMinReinf(rho_l_min=rho_l_min, rho_t_min=rho_t_min,
                        high_shear=high,
                        rho_l_ok=reinf.rho_l >= rho_l_min - 1e-9,
                        rho_t_ok=reinf.rho_t >= rho_t_min - 1e-9)


# ============================================ boundary elements (§18.10.6.3)

@dataclass
class BoundaryElementCheck:
    sigma_max: float          # extreme-fiber compressive stress (Pa)
    ratio: float              # σ_max / f'c
    required: bool            # special boundary element required (either trigger)
    stress_required: bool     # stress trigger: σ_max > 0.2 f'c (§18.10.6.3)
    disp_required: bool       # drift trigger: c ≥ ℓw/(600·δu/hw) (§18.10.6.2)
    can_discontinue: bool     # σ_max < 0.15 f'c
    lbe_min: float            # required boundary length (m)
    c: float                  # neutral-axis depth used (m), 0 if unknown
    drift: float              # δu/hw used for the drift trigger, 0 if unused


def boundary_element_check(geom: WallGeometry, mat: WallMaterial,
                           demand: WallDemand, *, c: float = 0.0,
                           drift: float = 0.0) -> BoundaryElementCheck:
    """Special boundary-element check (ACI 318-19 §18.10.6).

    * **Stress trigger** (§18.10.6.3): required where the extreme-fibre
      compressive stress from factored ``Pu, Mu`` on the gross section exceeds
      ``0.2 f'c`` (may be discontinued below ``0.15 f'c``).
    * **Displacement trigger** (§18.10.6.2, when a neutral-axis depth ``c`` and
      design drift ``drift = δu/hw`` are given): required where
      ``c ≥ ℓw / (600·(δu/hw))`` with ``δu/hw`` not taken less than 0.005.
    * **Boundary length** (§18.10.6.4a): ``ℓbe ≥ max(c − 0.1 ℓw, c/2)`` when
      ``c`` is known, else a ``0.15 ℓw`` placeholder."""
    sigma = 0.0
    if geom.Ag > 0:
        sigma = demand.Pu / geom.Ag + abs(demand.Mu) * (geom.lw / 2.0) / geom.Ig
    ratio = sigma / mat.fc if mat.fc > 0 else math.inf
    stress_req = sigma > 0.2 * mat.fc

    disp_req = False
    if c > 0 and drift > 0:
        d = max(drift, 0.005)                       # §18.10.6.2 lower bound
        c_limit = geom.lw / (600.0 * d)
        disp_req = c >= c_limit

    required = stress_req or disp_req
    if not required:
        lbe = 0.0
    elif c > 0:
        lbe = max(c - 0.1 * geom.lw, c / 2.0)       # §18.10.6.4(a)
    else:
        lbe = 0.15 * geom.lw                         # placeholder without c
    return BoundaryElementCheck(
        sigma_max=sigma, ratio=ratio, required=required,
        stress_required=stress_req, disp_required=disp_req,
        can_discontinue=sigma < 0.15 * mat.fc, lbe_min=lbe, c=c, drift=drift)


# ============================================ axial-flexure P-M (§22.2)

def _phi_flexure(eps_t: float, fy: float) -> float:
    """Strength-reduction factor from the net tensile strain (ACI 318-19
    §21.2.2): 0.90 tension-controlled, 0.65 compression-controlled (tied),
    linear transition."""
    eps_ty = fy / E_S
    if eps_t >= eps_ty + 0.003:            # tension-controlled
        return 0.90
    if eps_t <= eps_ty:                    # compression-controlled
        return 0.65
    return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003


def _beta1(fc: float) -> float:
    """ACI 318-19 §22.2.2.4.3 (SI): β1 = 0.85 for f'c ≤ 28 MPa, −0.05 per 7 MPa
    above, floored at 0.65."""
    if fc <= 28e6:
        return 0.85
    return max(0.65, 0.85 - 0.05 * (fc - 28e6) / 7e6)


def _wall_section_forces(geom, mat, reinf, c: float):
    """(Pn, Mn, eps_t) for a trial neutral-axis depth ``c`` (from the extreme
    compression fiber). Compression positive; moment about the wall centroid.
    Distributed vertical web steel is integrated as a line of area ρl·t per unit
    length; boundary bars are lumped at each end."""
    lw, t = geom.lw, geom.t
    fc, fy = mat.fc, mat.fy
    beta1 = _beta1(fc)
    a = min(beta1 * c, lw)
    # concrete compression (rectangular block), centroid at a/2 from comp face
    Cc = 0.85 * fc * a * t
    xc_C = a / 2.0                                 # from compression face
    P = Cc
    M = Cc * (lw / 2.0 - xc_C)                     # about centroid, comp-side +

    # distributed vertical web steel, integrated over the length
    n = 40
    dz = lw / n
    web_area_per_len = reinf.rho_l * t             # steel area per unit length
    for i in range(n):
        x = (i + 0.5) * dz                          # from compression face
        eps = EPS_CU * (c - x) / c if c > 0 else -1.0
        fs = max(-fy, min(fy, E_S * eps))
        # subtract concrete already counted where steel sits in the block
        conc_disp = 0.85 * fc if x < a else 0.0
        force = (fs - conc_disp) * web_area_per_len * dz
        P += force
        M += force * (lw / 2.0 - x)

    # boundary bars at each end
    for x_be, As in ((reinf.d_boundary, reinf.As_boundary),
                     (lw - reinf.d_boundary, reinf.As_boundary)):
        if As <= 0:
            continue
        eps = EPS_CU * (c - x_be) / c if c > 0 else -1.0
        fs = max(-fy, min(fy, E_S * eps))
        conc_disp = 0.85 * fc if x_be < a else 0.0
        force = (fs - conc_disp) * As
        P += force
        M += force * (lw / 2.0 - x_be)
    # net tensile strain εt of the extreme tension steel (far boundary bar),
    # positive in tension (ACI §21.2); 0 when the whole section is in compression
    d_ext = lw - reinf.d_boundary                   # depth to extreme tension bar
    eps_t = max(0.0, EPS_CU * (d_ext - c) / c) if c > 0 else 0.0
    return P, M, eps_t


@dataclass
class WallPMResult:
    phi: float
    Pn: float
    Mn: float
    phiMn: float
    c: float
    eps_t: float
    dcr: float


def wall_pm_capacity(geom: WallGeometry, mat: WallMaterial,
                     reinf: WallReinforcement, demand: WallDemand) -> WallPMResult:
    """Design flexural capacity ``φMn`` at the demand axial ``Pu`` and the
    moment demand/capacity ratio. Solves for the neutral axis ``c`` giving
    ``Pn = Pu/φ`` by a bisection on the strip-integrated axial force, then
    reports ``φMn`` there (ACI 318-19 §22.2). If ``Pu`` exceeds the section's
    pure-compression capacity the DCR is driven by axial and returned > 1."""
    lw = geom.lw
    # pure-compression cap Pn,max (φ·0.80·[0.85f'c(Ag−Ast)+fy·Ast], §22.4.2)
    Ast = 2 * reinf.As_boundary + reinf.rho_l * geom.Ag
    Pn_max = 0.80 * (0.85 * mat.fc * (geom.Ag - Ast) + mat.fy * Ast)

    Pu = demand.Pu
    # bisection on c for Pn(c) = target; target axial is Pu with a trial φ
    lo, hi = 1e-4, 5.0 * lw
    target = Pu / 0.75                              # provisional φ; refined below
    P_hi, _m, _e = _wall_section_forces(geom, mat, reinf, hi)
    # ensure bracket: if even huge c can't reach target, section is axially short
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        P_mid, _M, _eps = _wall_section_forces(geom, mat, reinf, mid)
        if P_mid < target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    Pn, Mn, eps_t = _wall_section_forces(geom, mat, reinf, c)
    phi = _phi_flexure(eps_t, mat.fy)
    # refine c once with the proper φ target
    target = Pu / phi
    lo, hi = 1e-4, 5.0 * lw
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        P_mid, _M, _eps = _wall_section_forces(geom, mat, reinf, mid)
        if P_mid < target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    Pn, Mn, eps_t = _wall_section_forces(geom, mat, reinf, c)
    phi = _phi_flexure(eps_t, mat.fy)
    phiMn = phi * abs(Mn)
    if 0.65 * Pn_max < Pu:                          # axial governs
        dcr = Pu / (0.65 * Pn_max)
    else:
        dcr = abs(demand.Mu) / phiMn if phiMn > 0 else math.inf
    return WallPMResult(phi=phi, Pn=Pn, Mn=Mn, phiMn=phiMn, c=c,
                        eps_t=eps_t, dcr=dcr)


# ============================================================ assembled check

# ============================================ code-specific detailing (§18.10 / IS 13920 / EC8)

WALL_CODES = ("ACI 318-19", "IS 13920", "EC8")

# Minimum reinforcement ratios by code: web (each direction) and the boundary-
# element vertical reinforcement floor where a boundary element is required.
_CODE_MIN = {
    #            web ρ    boundary ρ (vertical)
    "ACI 318-19": (0.0025, 0.0),     # ACI sets boundary detailing by ties, not ρ
    "IS 13920":   (0.0025, 0.008),   # IS 13920:2016 §10.1.4 / §10.4.4 (0.8%)
    "EC8":        (0.0020, 0.005),   # EN 1998-1 §5.4.3.4.2 confined BE (0.5%)
}


@dataclass
class WallDetailing:
    code: str
    web_rho_min: float
    boundary_rho_min: float
    boundary_rho: float            # provided boundary vertical ρ (As_be/(ℓbe·t))
    boundary_rho_ok: bool
    notes: list = field(default_factory=list)


def wall_detailing(code: str, geom: WallGeometry, mat: WallMaterial,
                   reinf: WallReinforcement,
                   be: BoundaryElementCheck) -> WallDetailing:
    """Code-specific special-wall detailing (ACI 318-19 §18.10, IS 13920:2016
    §10, or EC8 EN 1998-1 §5.4.3.4). Returns the governing web / boundary
    minimum reinforcement ratios, whether the supplied boundary bars meet the
    boundary floor, and code-referenced advisory notes."""
    code = code if code in WALL_CODES else "ACI 318-19"
    web_min, be_rho_min = _CODE_MIN[code]
    notes: list = []

    # boundary vertical reinforcement ratio actually provided
    be_rho = 0.0
    if be.required and be.lbe_min > 0 and geom.t > 0:
        be_rho = reinf.As_boundary / (be.lbe_min * geom.t)
    be_ok = (not be.required) or be_rho_min <= 0.0 or be_rho >= be_rho_min - 1e-9

    if code == "IS 13920":
        if geom.t > 0.20:
            notes.append("IS 13920 §10.1.4: two curtains required (tw > 200 mm).")
        if be.required:
            notes.append("IS 13920 §10.4: boundary element required; vertical "
                         "reinforcement 0.8%–6% of the boundary area.")
    elif code == "EC8":
        notes.append("EC8 §5.4.3.4.2: confine the boundary element over the "
                     "critical height; boundary vertical reinforcement ≥ 0.5%.")
    else:  # ACI 318-19
        if be.required:
            notes.append("ACI 318-19 §18.10.6.4: detail the boundary element "
                         "with transverse ties (hoops/crossties) over ℓbe.")
    if be.required and not be_ok:
        notes.append(f"Boundary vertical ρ {be_rho:.4f} below the {code} "
                     f"minimum {be_rho_min:.4f}.")
    return WallDetailing(code=code, web_rho_min=web_min,
                         boundary_rho_min=be_rho_min, boundary_rho=be_rho,
                         boundary_rho_ok=be_ok, notes=notes)


@dataclass
class WallDesignResult:
    pm: WallPMResult
    shear: WallShearResult
    boundary: BoundaryElementCheck
    min_reinf: WallMinReinf
    detailing: WallDetailing
    dcr: float                     # governing (max of P-M, shear)
    ok: bool
    notes: list = field(default_factory=list)


def design_wall_pier(geom: WallGeometry, mat: WallMaterial,
                     reinf: WallReinforcement, demand: WallDemand, *,
                     phi_shear: float = 0.75, code: str = "ACI 318-19",
                     drift: float = 0.0) -> WallDesignResult:
    """Full special-wall check for one pier demand: P-M, shear, boundary
    elements (stress + drift triggers), minimum reinforcement and code-specific
    detailing, with a governing DCR.

    ``code`` selects the detailing standard (``WALL_CODES``); ``drift`` = δu/hw
    the design roof-drift ratio, enabling the ACI §18.10.6.2 displacement-based
    boundary trigger (0 disables it, leaving only the stress trigger)."""
    pm = wall_pm_capacity(geom, mat, reinf, demand)
    sh = wall_shear_strength(geom, mat, reinf, Vu=demand.Vu, phi=phi_shear)
    be = boundary_element_check(geom, mat, demand, c=pm.c, drift=drift)
    mr = wall_min_web_reinforcement(geom, mat, reinf, demand.Vu)
    det = wall_detailing(code, geom, mat, reinf, be)
    dcr = max(pm.dcr, sh.dcr)
    notes: list = []
    if be.stress_required:
        notes.append("Special boundary elements required (σ > 0.2 f'c).")
    if be.disp_required:
        notes.append(f"Special boundary elements required (drift: c ≥ "
                     f"ℓw/(600·δu/hw), ℓbe ≥ {be.lbe_min:.2f} m).")
    if sh.two_curtains_required:
        notes.append("Two curtains of reinforcement required.")
    if not mr.rho_t_ok:
        notes.append(f"Horizontal web ρt below minimum {mr.rho_t_min:.4f}.")
    if not mr.rho_l_ok:
        notes.append(f"Vertical web ρl below minimum {mr.rho_l_min:.4f}.")
    notes.extend(det.notes)
    ok = (dcr <= 1.0 and mr.rho_t_ok and mr.rho_l_ok and det.boundary_rho_ok)
    return WallDesignResult(pm=pm, shear=sh, boundary=be, min_reinf=mr,
                            detailing=det, dcr=dcr, ok=ok, notes=notes)
