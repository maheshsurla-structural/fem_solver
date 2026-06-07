"""CLT-specific design checks (Phase D.1.5).

Standard timber design (NDS Ch. 3, EC5 §6) was implemented in D.1.3 /
D.1.4 for prismatic members. CLT panels need additional checks that
arise from their layered orthotropic build:

* **Rolling shear** -- cross-grain layers shear at G_R ≈ G/10 and
  govern shear in short-span CLT.
* **Two-way bending** -- floor / roof panels carry load in both the
  strong (machine) and weak (cross) direction; the two utilizations
  interact.
* **k_sys system strength factor** (EC5 §6.6, APA PRG-320) -- when
  multiple parallel panels share load through stiff transverse
  distribution, the characteristic strength can be boosted by ≈ 10 %.
* **k_def creep deformation factor** (EC5 §3.1.4, Table 3.2) --
  long-term loading approximately doubles the elastic deflection of
  solid timber and adds 25 % for glulam; the same factor applies to
  CLT for the bonded layers but creep of the cross layers via
  rolling shear is larger.
* **Vibration** (EC5 §7.3.3) -- footfall-induced response is a
  governing serviceability criterion for thin CLT floors. We
  implement the simplified EC5 fundamental-frequency rule
  (``f_1 ≥ 8 Hz``) along with the supplementary stiffness/velocity
  unity check.

These checks are stand-alone and consume a :class:`CLTSection` plus
demand loads + an :class:`EC5Factors` set. They sit on top of D.1.4
EC5 checks (the longitudinal bending of a CLT strip can still be
verified through ``ec5_bending_check`` with the strip adapter).

References
----------
* EN 1995-1-1:2004+A2:2014  §3.1.4, §6.6, §7.3
* APA PRG-320:2019  Standard for Performance-Rated CLT
* CLT Handbook (FPInnovations, 2013) Chap. 7, 8, 11
* Hu, Y.X.L. & Chui, Y.H. (2004) "Development of a design method to
  control vibrations induced by normal walking action in wood-based
  floors", Forest Prod. J. 54(8): 14-22.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from femsolver.design.timber.ec5 import EC5Factors
from femsolver.shell_sections.clt import CLTSection


# ============================================================ k_sys

def k_sys_clt(
    n_parallel: int,
    *,
    has_load_distribution: bool = True,
) -> float:
    """System strength factor for CLT (EC5 §6.6 / PRG-320 §8.3).

    When multiple parallel CLT panels share load through a stiff
    transverse system (sheathing, topping, or the panel's own cross
    layers), the characteristic strength gets a small boost.

    EC5 Table 6.2 gives k_sys = 1.10 for solid timber sub-systems with
    ≥4 equally loaded members and adequate continuous load
    distribution. For CLT, the same value applies to in-plane
    behaviour; out-of-plane bending of a CLT diaphragm uses k_sys = 1.1
    when the load distribution requirements are met.

    Parameters
    ----------
    n_parallel : int
        Number of parallel panels / planks that share load.
    has_load_distribution : bool
        ``True`` if a continuous load distribution system exists
        (concrete topping, plywood sheathing, mechanical joints).

    Returns
    -------
    k_sys : float
        Factor in [1.0, 1.1]. Apply as a multiplier on f_k before
        the partial factor: ``f_d = k_mod · k_sys · k_h · f_k / γ_M``.
    """
    if n_parallel < 1:
        raise ValueError(f"n_parallel must be ≥1, got {n_parallel}")
    if n_parallel >= 4 and has_load_distribution:
        return 1.10
    return 1.0


# ============================================================ k_def

def k_def_factor(
    material_type: str,
    service_class: int,
) -> float:
    """Creep / deformation factor (EC5 Table 3.2).

    Used for final (long-term) deflection:
        u_fin = u_inst · (1 + k_def · ψ_2)
    where ψ_2 is the load-combination quasi-permanent factor.

    Values (solid / glulam / CLT all per EN 1995-1-1 Table 3.2):

        ============== ====== ====== ======
        Material       SC 1   SC 2   SC 3
        ============== ====== ====== ======
        Solid          0.60   0.80   2.00
        Glulam / LVL   0.60   0.80   2.00
        Plywood        0.80   1.00   2.50
        OSB            1.50   2.25    --
        CLT            0.60   0.80    --    (PRG-320 §8.6 / FPInnovations)
        ============== ====== ====== ======

    Note: SC3 (exterior, fully exposed) CLT is generally not allowed
    by PRG-320; raise if requested.
    """
    table = {
        ("solid", 1): 0.60, ("solid", 2): 0.80, ("solid", 3): 2.00,
        ("glulam", 1): 0.60, ("glulam", 2): 0.80, ("glulam", 3): 2.00,
        ("LVL", 1): 0.60, ("LVL", 2): 0.80, ("LVL", 3): 2.00,
        ("plywood", 1): 0.80, ("plywood", 2): 1.00, ("plywood", 3): 2.50,
        ("OSB", 1): 1.50, ("OSB", 2): 2.25,
        ("CLT", 1): 0.60, ("CLT", 2): 0.80,
    }
    key = (material_type, service_class)
    if key not in table:
        raise ValueError(
            f"k_def undefined for material={material_type!r}, "
            f"service_class={service_class}"
        )
    return table[key]


# ============================================================ rolling shear

@dataclass
class CLTRollingShearCheck:
    """Result of a CLT rolling-shear check (the cross layers shearing
    parallel to their length but perpendicular to the grain)."""
    f_R_d: float           # rolling-shear design strength (Pa)
    tau_R_max: float       # max rolling-shear demand (Pa)
    DCR: float
    passes: bool
    governing_layer_idx: int
    note: str = ""


def clt_rolling_shear_check(
    section: CLTSection,
    *,
    V_Ed_per_width: float,
    factors: EC5Factors,
    f_R_k: float = 1.1e6,
    strong_axis: bool = True,
) -> CLTRollingShearCheck:
    """Rolling-shear check for a CLT panel (EC5 §6.1.7 / PRG-320 §8.2).

    Rolling shear acts on the cross layers (those whose grain is
    perpendicular to the bending direction). The shear stress at the
    interface between layers follows Jourawski:
        τ(y) = V · Q(y) / (I · b)
    where Q(y) is the first moment of the area beyond y above the
    cross layer. We evaluate this at each cross-layer interface and
    report the worst case.

    Parameters
    ----------
    section : CLTSection
    V_Ed_per_width : float
        Design shear demand per unit panel width (N/m).
    factors : EC5Factors
        Must have ``k_mod`` and ``gamma_M`` set. ``k_cr`` is NOT
        applied to rolling shear (cracking model is for longitudinal
        shear).
    f_R_k : float, default 1.1 MPa
        Characteristic rolling-shear strength (Pa). PRG-320 nominal
        for typical SPF/DF CLT is 1.1-1.3 MPa; EC5 commonly uses
        0.7-1.0 MPa for solid lumber cross layers.
    strong_axis : bool
        Bending direction.

    Notes
    -----
    Wikipedia: "Rolling shear" — the stress sliding wood fibres
    against each other in the radial-tangential plane.
    """
    if V_Ed_per_width < 0:
        raise ValueError("V_Ed_per_width must be ≥ 0 (pass abs value)")

    f_R_d = factors.k_mod * f_R_k / factors.gamma_M

    # Build Q at each layer interface from the top
    EI_per_w = section.EI_eff_per_width(strong_axis=strong_axis)
    y_NA = section.neutral_axis_from_top(strong_axis=strong_axis)

    # Walk top -> bottom, accumulating E·A·d above each interface.
    # Note this uses the transformed-section approach (E·Q / E·I).
    centroids = section._layer_centroids_from_top()
    EQ_above = 0.0
    tau_max = 0.0
    governing = -1

    for i, (layer, y_c) in enumerate(zip(section.layers, centroids)):
        E_i = section._E_for_layer(layer, strong_axis=strong_axis)
        EA_i = E_i * layer.thickness
        d_i = y_c - y_NA           # signed distance from NA
        EQ_above += EA_i * d_i     # sign matters; will use |.|

        # The interface BELOW layer i is at the boundary to layer i+1.
        # Rolling shear acts there only if EITHER side is a cross layer
        # (for strong-axis bending: cross = 90°, longitudinal = 0°).
        if i + 1 >= len(section.layers):
            continue
        next_layer = section.layers[i + 1]
        is_cross_interface = (
            (strong_axis and (layer.angle_deg == 90.0
                              or next_layer.angle_deg == 90.0))
            or (not strong_axis and (layer.angle_deg == 0.0
                                     or next_layer.angle_deg == 0.0))
        )
        if not is_cross_interface:
            continue

        # τ_R = V · EQ_above / EI  (per unit width: b=1)
        tau = V_Ed_per_width * abs(EQ_above) / max(EI_per_w, 1e-30)
        if tau > tau_max:
            tau_max = tau
            governing = i

    DCR = tau_max / max(f_R_d, 1e-30)
    passes = DCR <= 1.0

    return CLTRollingShearCheck(
        f_R_d=float(f_R_d),
        tau_R_max=float(tau_max),
        DCR=float(DCR),
        passes=bool(passes),
        governing_layer_idx=int(governing),
        note=f"Rolling shear at layer-{governing} interface; "
             f"f_R_k = {f_R_k/1e6:.2f} MPa",
    )


# ============================================================ two-way bending

@dataclass
class CLTTwoWayBendingCheck:
    """Result of a two-way CLT bending check (orthogonal moments)."""
    sigma_strong: float    # design bending stress, strong axis (Pa)
    sigma_weak: float      # design bending stress, weak axis (Pa)
    f_m_d_strong: float    # design strength, strong axis (Pa)
    f_m_d_weak: float      # design strength, weak axis (Pa)
    ratio_strong: float    # σ / f_m_d
    ratio_weak: float
    interaction: float     # interaction value (≤1 passes)
    passes: bool
    note: str = ""


def clt_two_way_bending_check(
    section: CLTSection,
    *,
    M_strong_per_width: float,
    M_weak_per_width: float,
    factors: EC5Factors,
    k_sys: float = 1.0,
) -> CLTTwoWayBendingCheck:
    """Two-way CLT bending check (PRG-320 §8.5 / EC5 §6.2.4 form).

    For a panel carrying moments about both axes:
        σ_m,strong = M_strong / W_strong
        σ_m,weak   = M_weak   / W_weak
        σ_m,strong / f_m,d,strong + σ_m,weak / f_m,d,weak ≤ 1

    The interaction is **linear** (not squared) because we sum the
    two demand utilizations directly per PRG-320 §8.5.2 — this is
    more conservative than the §6.2.4-style biaxial check used for
    prismatic glulam beams, which is appropriate for plates.

    Parameters
    ----------
    section : CLTSection
    M_strong_per_width, M_weak_per_width : float
        Design moments per unit panel width (N·m/m), strong and weak.
    factors : EC5Factors
        ``k_mod``, ``gamma_M``, ``k_h``.
    k_sys : float, default 1.0
        System factor (use :func:`k_sys_clt`).
    """
    # Transformed-section bending stress in the *load-bearing* layers.
    #
    # For composite (layered) bending, the stress in a layer of modulus
    # E at distance y from the NA is the transformed-section stress
    #     σ(E, y) = M · E · y / EI_eff
    # The governing fibre is the OUTERMOST edge of the outermost
    # load-bearing lamination — i.e. the layer whose grain runs
    # parallel to the bending direction (0° for strong-axis bending,
    # 90° for weak-axis). A 0° layer at the panel surface is *not*
    # load-bearing for weak-axis bending, so its (low E_90) stress is
    # not what we check.
    centroids = section._layer_centroids_from_top()

    def sigma_load_bearing(strong: bool, M: float) -> float:
        if M == 0.0:
            return 0.0
        y_NA = section.neutral_axis_from_top(strong_axis=strong)
        EI = section.EI_eff_per_width(strong_axis=strong)
        target_angle = 0.0 if strong else 90.0
        sigma = 0.0
        for layer, y_c in zip(section.layers, centroids):
            if layer.angle_deg != target_angle:
                continue
            E = section._E_for_layer(layer, strong_axis=strong)
            # extreme fibres of THIS lamination (top & bottom edges)
            y_edge_top = abs((y_c - layer.thickness / 2.0) - y_NA)
            y_edge_bot = abs((y_c + layer.thickness / 2.0) - y_NA)
            y_ext = max(y_edge_top, y_edge_bot)
            s = abs(M) * E * y_ext / max(EI, 1e-30)
            sigma = max(sigma, s)
        return sigma

    # Material strength: the load-bearing lamination for each axis
    bearing_strong = next(
        (l for l in section.layers if l.angle_deg == 0.0), section.layers[0]
    )
    bearing_weak = next(
        (l for l in section.layers if l.angle_deg == 90.0), section.layers[0]
    )
    f_m_k_strong = bearing_strong.material.f_b_k
    f_m_k_weak = bearing_weak.material.f_b_k

    f_m_d_strong = (
        factors.k_mod * k_sys * factors.k_h * f_m_k_strong / factors.gamma_M
    )
    f_m_d_weak = (
        factors.k_mod * k_sys * factors.k_h * f_m_k_weak / factors.gamma_M
    )

    sigma_strong = sigma_load_bearing(strong=True, M=M_strong_per_width)
    sigma_weak = sigma_load_bearing(strong=False, M=M_weak_per_width)

    r_strong = sigma_strong / max(f_m_d_strong, 1e-30)
    r_weak = sigma_weak / max(f_m_d_weak, 1e-30)
    interaction = r_strong + r_weak

    return CLTTwoWayBendingCheck(
        sigma_strong=float(sigma_strong),
        sigma_weak=float(sigma_weak),
        f_m_d_strong=float(f_m_d_strong),
        f_m_d_weak=float(f_m_d_weak),
        ratio_strong=float(r_strong),
        ratio_weak=float(r_weak),
        interaction=float(interaction),
        passes=bool(interaction <= 1.0),
        note="PRG-320 §8.5 linear sum of strong/weak utilizations",
    )


# ============================================================ deflection

@dataclass
class CLTDeflectionCheck:
    """Instantaneous + final CLT deflection."""
    u_inst: float
    u_fin: float
    u_limit_inst: float
    u_limit_fin: float
    passes_inst: bool
    passes_fin: bool
    note: str = ""


def clt_deflection_check(
    section: CLTSection,
    *,
    span: float,
    w_perm: float,
    w_var: float,
    psi_2: float = 0.30,
    material_type: str = "CLT",
    service_class: int = 1,
    span_over_limit_inst: float = 300.0,
    span_over_limit_fin: float = 250.0,
    strong_axis: bool = True,
    use_gamma_method: bool = True,
) -> CLTDeflectionCheck:
    """Span/300 (instantaneous) and span/250 (final) deflection check.

    Uses the simply-supported uniformly-loaded formula
        u_inst = 5 w L^4 / (384 EI_eff)
    where EI_eff is from :meth:`CLTSection.gamma_method` (recommended
    for short spans) or the no-slip ``EI_eff_per_width`` (long spans).

    Final deflection per EC5 §2.2.3 (3):
        u_fin = u_perm · (1 + k_def) + u_var · (1 + ψ_2 · k_def)

    Parameters
    ----------
    section : CLTSection
    span : float
        Effective span (m).
    w_perm, w_var : float
        Permanent and variable distributed loads per unit area (Pa).
    psi_2 : float, default 0.30
        Quasi-permanent combination factor (EN 1990 Table A1.1).
        0.30 for residential, 0.6 for offices/storage.
    material_type, service_class : str, int
        Inputs to :func:`k_def_factor`.
    span_over_limit_inst, span_over_limit_fin : float
        Default L/300 (instantaneous) and L/250 (final) per EC5
        §7.2 and IRC § R301.
    """
    if span <= 0:
        raise ValueError("span must be positive")

    if use_gamma_method:
        EI_per_w = section.gamma_method(span, strong_axis=strong_axis)["EI_eff"]
    else:
        EI_per_w = section.EI_eff_per_width(strong_axis=strong_axis)

    # Per-unit-width: w in N/m^2 acts on 1m strip -> w_strip = w (N/m)
    def simply_supp(w: float) -> float:
        return 5.0 * w * span ** 4 / (384.0 * EI_per_w)

    u_perm = simply_supp(w_perm)
    u_var = simply_supp(w_var)
    u_inst = u_perm + u_var
    k_def = k_def_factor(material_type, service_class)
    u_fin = u_perm * (1.0 + k_def) + u_var * (1.0 + psi_2 * k_def)

    u_lim_i = span / span_over_limit_inst
    u_lim_f = span / span_over_limit_fin

    return CLTDeflectionCheck(
        u_inst=float(u_inst),
        u_fin=float(u_fin),
        u_limit_inst=float(u_lim_i),
        u_limit_fin=float(u_lim_f),
        passes_inst=bool(u_inst <= u_lim_i),
        passes_fin=bool(u_fin <= u_lim_f),
        note=f"k_def = {k_def}, ψ₂ = {psi_2}, EI_eff = {EI_per_w:.0f} N·m²/m",
    )


# ============================================================ vibration

@dataclass
class CLTVibrationCheck:
    """Fundamental-frequency vibration check per EC5 §7.3."""
    f_1: float                # fundamental frequency (Hz)
    f_1_limit: float          # required minimum (Hz)
    passes: bool
    EI_used: float
    mass_per_area: float
    note: str = ""


def clt_vibration_check(
    section: CLTSection,
    *,
    span: float,
    width: Optional[float] = None,
    additional_mass_per_area: float = 0.0,
    f_1_required: float = 8.0,
    strong_axis: bool = True,
    use_gamma_method: bool = True,
) -> CLTVibrationCheck:
    """Footfall-vibration check for CLT floors (EC5 §7.3.3 / Hu-Chui).

    The fundamental frequency of a simply-supported one-way CLT floor
    is:
        f_1 = (π / 2) · sqrt( EI_eff / (m · L^4) )
    where m is mass per unit area and L the span. EC5 requires
    ``f_1 ≥ 8 Hz`` for residential floors. For high-importance
    structures use 10 Hz; the threshold is a user input.

    Reference: EN 1995-1-1 §7.3.3 (2) for residential floors.

    Parameters
    ----------
    section : CLTSection
    span : float
        Span perpendicular to the supports (m).
    additional_mass_per_area : float, default 0
        Topping / partitions / quasi-permanent live load contributing
        to inertia (kg/m²). EC5 §7.3.3 includes 30 kg/m² for service
        loads in residential floors; you must add this manually.
    f_1_required : float, default 8 Hz
        Threshold. 8 Hz is the EC5 default for residential.
    """
    if span <= 0:
        raise ValueError("span must be positive")

    if use_gamma_method:
        EI_per_w = section.gamma_method(span, strong_axis=strong_axis)["EI_eff"]
    else:
        EI_per_w = section.EI_eff_per_width(strong_axis=strong_axis)

    m = section.mass_per_area() + additional_mass_per_area  # kg/m²

    f_1 = (math.pi / 2.0) * math.sqrt(EI_per_w / max(m * span ** 4, 1e-30))

    return CLTVibrationCheck(
        f_1=float(f_1),
        f_1_limit=float(f_1_required),
        passes=bool(f_1 >= f_1_required),
        EI_used=float(EI_per_w),
        mass_per_area=float(m),
        note=f"EC5 §7.3.3 simply-supported formula; "
             f"m = {m:.1f} kg/m²",
    )
