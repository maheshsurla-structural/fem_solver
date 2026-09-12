"""Section-analysis core: moment-curvature + P-M by exact integration.

Strain-banded Gauss-Legendre integration of the *actual* uniaxial
constitutive laws over a section's true geometry:

* :func:`exact_mphi`       -- exact-integration moment-curvature (M3).
* :func:`section_pm_slice`  -- fibre-model P-M interaction slice (P5).

plus the material-from-spec factories (:func:`concrete_uniaxial_from`,
:func:`steel_uniaxial_from`) and the Mander (1988) confinement calculator
(:func:`mander_confinement`) they build on.

Relocated **verbatim** from the Section Designer GUI core
(``section_gui_core.py``) so one section-analysis core is shared by the
engine, tests, the desktop app, and the fiber-hinge stream -- instead of the
GUI owning it. ``section_gui_core`` now re-exports these names. See
``docs/source/fiber_hinge_implementation_plan.md`` §15 (U1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from femsolver.benchmarks.section_designer import (
    E_S,
    SectionCase,
    strong_axis_landmarks,
)
from femsolver.design.concrete import (
    ConcreteMaterial,
    biaxial_pmm_point_aashto,
    biaxial_pmm_point_ec2,
    biaxial_pmm_point_is456,
    biaxial_pmm_surface_aashto,
    biaxial_pmm_surface_ec2,
    biaxial_pmm_surface_is456,
    moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    ConcreteMander,
    ConcreteParabolaRectangle,
    ConcreteTensionStiffening,
    ConcreteTrilinear,
    UniaxialBilinear,
    UniaxialMenegottoPinto,
    UniaxialReinforcingSteel,
)
from femsolver.sections import (
    PrestressTendon,
    RebarBar,
    ReinforcementLayout,
    TendonLayout,
    custom_polygon_section,
)

_GL = np.polynomial.legendre.leggauss(10)


def ec2_fctm(fc: float) -> float:
    """EC2 (EN 1992-1-1 Table 3.1) mean axial tensile strength f_ctm,
    from the concrete cylinder strength ``fc`` (Pa). Returns Pa.

    f_ck <= 50 MPa:  f_ctm = 0.30 * f_ck^(2/3)
    f_ck  > 50 MPa:  f_ctm = 2.12 * ln(1 + f_cm/10), f_cm = f_ck + 8
    """
    fck = fc / 1e6
    if fck <= 50.0:
        fctm = 0.30 * fck ** (2.0 / 3.0)
    else:
        fctm = 2.12 * math.log(1.0 + (fck + 8.0) / 10.0)
    return fctm * 1e6


def _has_confinement_params(m: dict) -> bool:
    """True if a material dict carries enough confinement-steel data to run the
    Mander confinement calculator (else the confined model falls back to the
    unconfined base curve)."""
    return (float(m.get("conf_Asp", 0.0)) > 0.0
            and float(m.get("conf_s", 0.0)) > 0.0)


def mander_confinement(m: dict) -> dict:
    """Compute Mander (1988) confined-concrete properties from the confinement
    reinforcement described in a material dict, for circular or rectangular
    cores. Returns a dict of results **and** every intermediate value so the
    GUI can echo them for checking:

    ``fcc`` (confined peak, Pa), ``eps_cc`` (peak strain), ``eps_cu`` (ultimate
    strain from the Mander energy balance), ``ke`` (confinement effectiveness),
    ``fl`` (effective lateral confining stress, Pa), ``rho_cc`` (longitudinal
    ratio), ``rho_s`` (circular) or ``rho_x``/``rho_y`` (rectangular), etc.

    Parameter keys (SI): ``conf_shape`` ("Circular"/"Rectangular"), ``conf_fyh``
    (hoop yield Pa), ``conf_Asp`` (one tie-leg / spiral-bar area m^2), ``conf_s``
    (hoop centre spacing m), ``conf_sp`` (clear spacing s' m), ``conf_rho_cc``
    (longitudinal steel ratio), ``conf_eps_su_h`` (hoop fracture strain),
    ``conf_Es_h`` (hoop modulus Pa). Circular: ``conf_ds`` (core dia to hoop
    centre m), ``conf_hooptype`` ("Spiral"/"Hoop"). Rectangular: ``conf_bc``,
    ``conf_dc`` (core dims m), ``conf_ny``, ``conf_nz`` (tie legs each dir),
    ``conf_nlong`` (# longitudinal bars, for the sum(w_i^2) term)."""
    fc = float(m["fc"])
    shape = m.get("conf_shape", "Rectangular")
    fyh = float(m.get("conf_fyh", 400e6))
    asp = float(m.get("conf_Asp", 0.0))
    s = float(m.get("conf_s", 0.1))
    sp = float(m.get("conf_sp", max(s - 0.01, 1e-3)))
    rho_cc = min(max(float(m.get("conf_rho_cc", 0.02)), 0.0), 0.2)
    eps_su_h = float(m.get("conf_eps_su_h", 0.10))
    es_h = float(m.get("conf_Es_h", E_S))
    ecu_method = m.get("conf_ecu_method", "energy")

    def OV(name, auto):
        """A per-parameter user override: ``ov_<name>`` in the dict wins over the
        auto-computed value, and downstream terms recompute from it (Midas GSD's
        checkbox-per-field behaviour)."""
        v = m.get("ov_" + name, None)
        try:
            return float(v) if v is not None and v != "" else auto
        except (TypeError, ValueError):
            return auto

    eps_c0 = OV("eps_cy", float(m.get("eps_c0", 0.002)))   # εcy = unconfined peak

    out = dict(shape=shape, rho_cc=rho_cc, Ec=4700.0 * math.sqrt(fc / 1e6) * 1e6)
    if shape == "Circular":
        ds = float(m.get("conf_ds", 0.4))
        spiral = m.get("conf_hooptype", "Spiral") == "Spiral"
        rho_s = OV("rho_s", 4.0 * asp / (ds * s) if ds > 0 and s > 0 else 0.0)
        Ac = math.pi / 4.0 * ds * ds
        Acc = OV("Acc", Ac * max(1.0 - rho_cc, 1e-6))
        ke_geom = (max(1.0 - sp / (2.0 * ds), 0.0)
                   ** (1 if spiral else 2)) / max(1.0 - rho_cc, 1e-6)
        Ae = OV("Ae", ke_geom * Acc)
        ke = OV("ke", Ae / Acc if Acc > 0 else ke_geom)
        fl = OV("fl", 0.5 * ke * rho_s * fyh)       # effective lateral stress
        rho_tot = rho_s
        out.update(rho_s=rho_s, ke=ke, fl=fl, Ac=Ac, Acc=Acc, Ae=Ae)
    else:
        bc = float(m.get("conf_bc", 0.4))
        dc = float(m.get("conf_dc", 0.4))
        ny = float(m.get("conf_ny", 2))
        nz = float(m.get("conf_nz", 2))
        nlong = max(float(m.get("conf_nlong", 8)), 1.0)
        # Midas convention: rho_y = A_sy/(d_c*s), A_sy = ny*A_sp (y-dir legs);
        # rho_z = A_sz/(b_c*s), A_sz = nz*A_sp (z-dir legs).
        rho_y = (ny * asp) / (s * dc) if s > 0 and dc > 0 else 0.0
        rho_z = (nz * asp) / (s * bc) if s > 0 and bc > 0 else 0.0
        perim = 2.0 * (bc + dc)
        # Σw′² for the effectively-confined-area term. EXACT (Midas): the user's
        # clear bar spacings, conf_sum_wi2 = 2·Σw′yi² + 2·Σw′zj². Else fall back
        # to an equal-spacing approximation from the longitudinal-bar count —
        # floored at 4 (a rectangular confined core has ≥4 corner bars) so a
        # degenerate count can't drive the whole perimeter into one gap.
        sum_wi2 = (float(m["conf_sum_wi2"])
                   if float(m.get("conf_sum_wi2", 0.0) or 0.0) > 0.0
                   else perim * perim / max(nlong, 4.0))
        Ac = bc * dc                                # core within hoop centrelines
        Acc = OV("Acc", Ac * max(1.0 - rho_cc, 1e-6))   # net of longitudinal steel
        ke_geom = ((max(1.0 - sum_wi2 / (6.0 * Ac), 0.0))
                   * max(1.0 - sp / (2.0 * bc), 0.0)
                   * max(1.0 - sp / (2.0 * dc), 0.0)) / max(1.0 - rho_cc, 1e-6)
        Ae = OV("Ae", ke_geom * Acc)
        ke = OV("ke", Ae / Acc if Acc > 0 else ke_geom)
        fly = OV("fly", ke * rho_y * fyh)
        flz = OV("flz", ke * rho_z * fyh)
        fl = 0.5 * (fly + flz)          # mean lateral pressure (chart approx.)
        rho_tot = OV("rho_s", rho_y + rho_z)
        out.update(rho_y=rho_y, rho_z=rho_z, ke=ke, fly=fly, flz=flz, fl=fl,
                   sum_wi2=sum_wi2, Ac=Ac, Acc=Acc, Ae=Ae)

    # Confined peak strength (Mander single-pressure equation) and strain.
    if fl > 0.0 and fc > 0.0:
        fcc_auto = fc * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * fl / fc)
                         - 2.0 * fl / fc)
    else:
        fcc_auto = fc
    fcc = OV("fcc", max(fcc_auto, fc))              # confinement never weakens
    eps_cc = OV("eps_cc", eps_c0 * (1.0 + 5.0 * (fcc / fc - 1.0)))

    # Ultimate strain: "energy" (area under the confined curve = unconfined
    # energy U_co ~ 0.017 f'c + the confining-steel energy) or "experiment"
    # (Mander/Priestley ε_cu = 0.004 + 1.4 ρs f_yh ε_su / f'cc), then override.
    eps_yh = fyh / es_h
    u_steel = rho_tot * fyh * max(eps_su_h - 0.5 * eps_yh, 0.0)
    u_co = 0.017 * fc
    if ecu_method == "experiment":
        eps_cu_auto = (0.004 + 1.4 * rho_tot * fyh * eps_su_h / fcc
                       if fcc > 0 else 0.004)
    else:
        target = u_co + u_steel
        law = ConcreteMander(fpc=fcc, eps_c0=eps_cc, min_strength_ratio=0.2)

        def _area(eu):
            n, a, prev = 160, 0.0, 0.0
            for i in range(1, n + 1):
                sig = -law.get_response(-eu * i / n)[0]     # positive magnitude
                a += 0.5 * (prev + sig) * (eu / n)
                prev = sig
            return a
        lo, hi = eps_cc, 0.08
        if _area(hi) < target:
            eps_cu_auto = hi
        else:
            for _ in range(38):
                mid = 0.5 * (lo + hi)
                if _area(mid) < target:
                    lo = mid
                else:
                    hi = mid
            eps_cu_auto = 0.5 * (lo + hi)
    eps_cu = OV("eps_cu", min(max(eps_cu_auto, eps_cc + 1e-3), 0.08))

    out.update(fcc=fcc, eps_cc=eps_cc, eps_cu=eps_cu, eps_cy=eps_c0,
               rho_tot=rho_tot, u_co=u_co, u_steel=u_steel,
               kcc=fcc / fc if fc else 1.0)
    return out


def concrete_uniaxial_from(m: dict):
    """Tension-stiffened concrete constitutive model from a material dict
    (keys: fc, conc_model, eps_c0, eps_cu, fcu_ratio, fr_model, fr_coeff,
    eps_decay, conc_f1_ratio). The result exposes ``.f_ct`` (rupture stress)
    and ``.E_ct`` (E_c). One source of truth for single & composite paths."""
    fc = float(m["fc"])
    eps_c0 = float(m.get("eps_c0", 0.002))
    eps_cu = float(m.get("eps_cu", 0.0035))
    fcu = float(m.get("fcu_ratio", 0.4))
    model = m.get("conc_model", "Kent-Park")
    if model == "Mander":
        # The material is the UNCONFINED base curve (peak f'c at eps_c0).
        # Confinement lives on the SECTION — when it injects its tie steel
        # (conf_* keys, e.g. the confined core built in confined_mphi) the peak
        # rises to fcc' at the larger strain eps_cc = eps_c0*[1 + 5*(fcc/f'c-1)]
        # (Mander/Priestley 1988).
        if _has_confinement_params(m):
            r = mander_confinement(m)
            base = ConcreteMander(fpc=r["fcc"], eps_c0=r["eps_cc"],
                                  min_strength_ratio=fcu)
        else:
            base = ConcreteMander(fpc=fc, eps_c0=eps_c0, min_strength_ratio=fcu)
    elif model == "Parabola-rectangle (EC2)":
        base = ConcreteParabolaRectangle(fpc=fc, eps_c2=eps_c0, eps_cu2=eps_cu,
                                         n=2.0)
    elif model == "Trilinear":
        base = ConcreteTrilinear(
            fpc=fc, eps_c0=eps_c0, fpcu=fcu * fc, eps_cu=eps_cu,
            f1_ratio=min(max(float(m.get("conc_f1_ratio", 0.4)), 0.05), 0.95))
    else:
        base = ConcreteKentPark(fpc=fc, eps_c0=eps_c0, fpcu=fcu * fc,
                                eps_cu=eps_cu)
    f_r = (ec2_fctm(fc) if m.get("fr_model") == "ec2"
           else float(m.get("fr_coeff", 0.62)) * math.sqrt(fc / 1e6) * 1e6)
    return ConcreteTensionStiffening(base, f_ct=f_r, E_ct=base.E0,
                                     eps_decay=float(m.get("eps_decay", 1e-3)))


def steel_uniaxial_from(m: dict):
    """Rebar steel constitutive model from a material dict (keys: fy, Es,
    steel_model, steel_b, steel_fu_ratio, steel_eps_sh, steel_eps_su)."""
    fy = float(m["fy"])
    E_s = float(m.get("Es", E_S))
    model = m.get("steel_model", "Bilinear")
    if model == "Elastic - perfectly plastic":
        return UniaxialBilinear(E=E_s, sigma_y=fy, b=0.0)
    if model == "Menegotto-Pinto":
        return UniaxialMenegottoPinto(E=E_s, sigma_y=fy,
                                      b=float(m.get("steel_b", 0.01)))
    if model == "Park strain-hardening":
        esh = max(float(m.get("steel_eps_sh", 0.008)), fy / E_s + 1e-6)
        return UniaxialReinforcingSteel(
            E=E_s, f_y=fy, f_su=max(float(m.get("steel_fu_ratio", 1.5)),
                                    1.001) * fy,
            eps_sh=esh, eps_su=max(float(m.get("steel_eps_su", 0.10)),
                                   esh + 1e-3))
    return UniaxialBilinear(E=E_s, sigma_y=fy, b=float(m.get("steel_b", 0.01)))


def _mphi_kappas(kcr, kappa_max, n_points=65):
    """Curvature sweep shared by every M-φ path (M4 'No. of points'): resolve
    the steep uncracked branch finely through the cracking region, then step
    coarsely out to ``kappa_max``. ``n_points`` sets the total sample count;
    the default (65) reproduces the historical 10-fine + 55-coarse array."""
    n = max(12, int(n_points))
    n_fine = max(4, n // 6)
    n_coarse = max(6, n - n_fine)
    return np.unique(np.concatenate([
        np.linspace(0.0, 3.0 * kcr, n_fine), [kcr],
        np.linspace(3.0 * kcr, kappa_max, n_coarse)]))


def _mphi_stop(stop, *, crush, eps_steel, eps_su=0.05):
    """Failure-criterion gate for the inline M-φ loops (M4). Returns the
    failure-mode string when the sweep should terminate at this step, else "".
    ``crush`` is True when the governing concrete fibre has reached ε_cu.
    ``stop``: 'concrete' ends on crushing or steel rupture (default), 'steel'
    ends on steel rupture only (concrete allowed to soften past ε_cu), 'peak'
    never ends early (ultimate falls out as the moment peak at κ_max)."""
    if stop == "peak":
        return ""
    if stop != "steel" and crush:
        return "concrete_crushing"
    if eps_steel >= eps_su:
        return "steel_rupture"
    return ""


def _mphi_crossing(kaps, Ms, strains, limit):
    """First (kappa, M) where ``strains`` reaches ``limit``, linearly
    interpolated between the two straddling points; None if never reached
    within the computed sweep."""
    prev = None
    for k, m, e in zip(kaps, Ms, strains):
        if e >= limit:
            if prev is not None and e > prev[2]:
                t = min(1.0, max(0.0, (limit - prev[2]) / (e - prev[2])))
                return (prev[0] + t * (k - prev[0]),
                        prev[1] + t * (m - prev[1]))
            return (k, m)
        prev = (k, m, e)
    return None


def _mphi_ctrl(kaps, Ms, conc_strains, steel_strains, eps_cu, eps_su=0.05):
    """Concrete- vs steel-controlled crossing points (M6): the (κ, M) where the
    governing concrete fibre first reaches ε_cu and where the extreme bar first
    reaches ε_su. ``controls`` names whichever governs (the lower curvature);
    either crossing is None when the sweep never reaches that limit."""
    conc = _mphi_crossing(kaps, Ms, conc_strains, eps_cu)
    steel = _mphi_crossing(kaps, Ms, steel_strains, eps_su)
    if conc and (steel is None or conc[0] <= steel[0]):
        controls = "concrete"
    elif steel:
        controls = "steel"
    else:
        controls = None
    return {"conc_kappa": conc[0] if conc else None,
            "conc_M": conc[1] if conc else None,
            "steel_kappa": steel[0] if steel else None,
            "steel_M": steel[1] if steel else None,
            "controls": controls}


def _rotate_case(case: SectionCase, angle_deg: float) -> SectionCase:
    """Return *case* rigidly rotated by ``-angle_deg`` about the centroid.

    Moment-curvature bends about the section z-axis (``M = -Σ y·σ·A``). To
    analyse bending with the neutral axis inclined at ``angle_deg`` from z,
    we rotate the whole section by the opposite angle so that inclined axis
    lands on the horizontal z-axis. Every built section here is
    centroid-at-origin, so rotating each coordinate about the origin keeps
    the centroid at the origin and leaves the engine's ``M`` about y=0 valid.
    """
    th = math.radians(-angle_deg)
    c, s = math.cos(th), math.sin(th)

    def rot(z, y):
        return (z * c - y * s, z * s + y * c)

    sec = case.section
    poly = sec.geometry.polygon
    outline = [rot(z, y) for (z, y) in poly.exterior.coords[:-1]]
    holes = [[rot(z, y) for (z, y) in ring.coords[:-1]]
             for ring in poly.interiors] or None
    cm = ConcreteMaterial(fc_prime=case.f_c_prime, fy=case.f_y)
    rsec = custom_polygon_section(outline=outline, holes=holes,
                                  material=cm, name=sec.name)
    if sec.reinforcement and sec.reinforcement.bars:
        rsec.reinforcement = ReinforcementLayout(bars=[
            RebarBar(*rot(b.z, b.y), area=b.area, material=b.material,
                     designation=b.designation)
            for b in sec.reinforcement.bars])
    if getattr(sec, "prestress", None) and sec.prestress.tendons:
        rsec.prestress = TendonLayout(tendons=[
            PrestressTendon(*rot(t.z, t.y), area=t.area, material=t.material,
                            f_pe=t.f_pe, bonded=t.bonded,
                            designation=t.designation)
            for t in sec.prestress.tendons])
    return replace(case, section=rsec)


def _width_bands(poly):
    """Section width profile b(w) (the horizontal extent at height ``w``) as a
    list of bands ``(w_lo, w_hi, w_ref, W_ref, slope)`` between consecutive
    vertex w-levels; within a band ``b(w) = W_ref + slope·(w − w_ref)``. The
    boundary is linear inside a vertex-free band, so width is exactly linear
    there — but it can *step* across a hole edge, so each band is sampled at two
    INTERIOR points (never at the vertex levels, which graze hole/apex edges)
    and the line is fitted. Returns ``(w_lo, w_hi, bands)``."""
    from shapely.geometry import LineString
    minz, _miny, maxz, _maxy = poly.bounds
    pad = (maxz - minz) + 1.0

    def width_at(w):
        inter = poly.intersection(
            LineString([(minz - pad, w), (maxz + pad, w)]))
        total = 0.0
        for g in (getattr(inter, "geoms", None) or [inter]):
            cs = list(getattr(g, "coords", []))
            if len(cs) >= 2:
                total += abs(cs[-1][0] - cs[0][0])
        return total

    ws = {y for _, y in poly.exterior.coords}
    for ring in poly.interiors:
        ws.update(y for _, y in ring.coords)
    wl = sorted(ws)
    bands = []
    for a, b in zip(wl[:-1], wl[1:]):
        if b - a < 1e-12:
            continue
        wa, wb = a + 0.25 * (b - a), a + 0.75 * (b - a)
        Wa, Wb = width_at(wa), width_at(wb)
        slope = (Wb - Wa) / (wb - wa) if (wb - wa) > 1e-12 else 0.0
        bands.append((a, b, wa, Wa, slope))
    return wl[0], wl[-1], bands


def exact_mphi(case, P_target_kN, *, na_angle=0.0,
               eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4, fr_coeff=0.62,
               fr_model="sqrt", eps_decay=1.0e-3, E_s=E_S, steel_b=0.01,
               kappa_max=0.06, conc_model="Kent-Park", conc_f1_ratio=0.4,
               steel_model="Bilinear", steel_fu_ratio=1.5,
               steel_eps_sh=0.008, steel_eps_su=0.10,
               n_points=65, stop="concrete"):
    """Exact-integration moment-curvature (M3). Integrates the *same*
    constitutive laws :func:`mphi_data` uses over the true section — strain-
    banded Gauss-Legendre quadrature across a piecewise-linear width profile,
    with rebars as discrete points and no concrete subtracted at the bars
    (matching the fibre model) — so the result differs from the fibre curve
    only by discretisation. Band edges sit at the section's vertex levels and
    at the strains where the concrete law bends (0, ε_c0, ε_cu, cracking), so
    the integrand is smooth within each band and the quadrature is effectively
    exact. Returns the same dict shape as :func:`mphi_data`."""
    if abs(na_angle) > 1e-9:
        case = _rotate_case(case, na_angle)
    concrete = concrete_uniaxial_from(dict(
        fc=case.f_c_prime, conc_model=conc_model, eps_c0=eps_c0, eps_cu=eps_cu,
        fcu_ratio=fcu_ratio, fr_model=fr_model, fr_coeff=fr_coeff,
        eps_decay=eps_decay, conc_f1_ratio=conc_f1_ratio))
    steel = steel_uniaxial_from(dict(
        fy=case.f_y, Es=E_s, steel_model=steel_model, steel_b=steel_b,
        steel_fu_ratio=steel_fu_ratio, steel_eps_sh=steel_eps_sh,
        steel_eps_su=steel_eps_su))
    f_r, E_c = concrete.f_ct, concrete.E_ct

    poly = case.section.geometry.polygon
    w_lo, w_hi, width_bands = _width_bands(poly)
    bars = [(float(b.y), float(b.area)) for b in
            (case.section.reinforcement.bars if case.section.reinforcement
             else [])]
    xg, wg = _GL
    eps_cr = f_r / E_c
    breaks = (eps_cr, 0.0, -eps_c0, -eps_cu)

    def resultants(eps0, kappa):
        """(N, M, EA) at the given reference strain + curvature, tension-
        positive N and M = -∫σ·w (mirrors FiberSection2D). Integrates each
        geometry band, split further at the concrete-law strain breakpoints so
        the integrand is smooth within every Gauss panel."""
        N = M = EA = 0.0
        for wa0, wb0, w_ref, W_ref, slope in width_bands:
            edges = [wa0, wb0]
            if abs(kappa) > 1e-12:                   # add the strain breakpoints
                for e_bp in breaks:
                    wb = (eps0 - e_bp) / kappa
                    if wa0 < wb < wb0:
                        edges.append(wb)
                edges.sort()
            for a, b in zip(edges[:-1], edges[1:]):
                half = 0.5 * (b - a)
                if half <= 1e-12:
                    continue
                mid = 0.5 * (a + b)
                for xi, wi in zip(xg, wg):
                    w = mid + half * xi
                    bwid = W_ref + slope * (w - w_ref)
                    if bwid <= 0.0:
                        continue
                    sig, Et = concrete.get_response(eps0 - w * kappa)
                    gwt = bwid * half * wi
                    N += sig * gwt
                    M -= sig * gwt * w
                    EA += Et * gwt
        for wy, area in bars:                        # rebars (discrete, exact)
            sig, Et = steel.get_response(eps0 - wy * kappa)
            N += sig * area
            M -= sig * area * wy
            EA += Et * area
        return N, M, EA

    N_target = -P_target_kN * 1e3
    kcr = f_r / max(E_c * abs(w_lo), 1e-9)
    kappas = _mphi_kappas(kcr, kappa_max, n_points)
    eps_y = case.f_y / E_s
    eps0 = 0.0
    pts = []
    M_y = kappa_y = None
    failure = ""
    for kappa in kappas:
        for _ in range(60):                          # Newton on eps0 -> N=N_tgt
            N, M, EA = resultants(eps0, kappa)
            resid = N - N_target
            tol = max(1.0, abs(N_target) * 1e-8, 100.0)
            if abs(resid) < tol:
                break
            eps0 -= resid / (EA if abs(EA) >= 1e3 else 1e6)
        N, M, _ = resultants(eps0, kappa)
        eps_top = eps0 - w_hi * kappa                # extreme compression fibre
        eps_steel = max((eps0 - wy * kappa for wy, _a in bars), default=0.0)
        pts.append({"kappa": float(kappa), "M": float(M), "P": float(-N),
                    "axial_strain": float(eps0), "eps_top": float(eps_top),
                    "eps_steel": float(eps_steel)})
        if M_y is None and eps_steel >= eps_y and kappa > 0:
            M_y, kappa_y = float(M), float(kappa)
        failure = _mphi_stop(stop, crush=(-eps_top >= eps_cu),
                             eps_steel=eps_steel)
        if failure:
            break

    kap = [p["kappa"] for p in pts]
    Ms = [p["M"] for p in pts]
    ipk = int(np.argmax(Ms)) if Ms else 0
    M_u, kappa_u = (Ms[ipk], kap[ipk]) if pts else (0.0, 0.0)
    if not failure:
        failure = "kappa_max_reached" if ipk == len(pts) - 1 else "M_peak"
    # on-curve cracking point (extreme tension fibre reaches eps_cr)
    kappa_cr = M_cr = None
    prev = None
    for p in pts:
        eb = p["axial_strain"] - w_lo * p["kappa"]
        if p["kappa"] > 0 and eb >= eps_cr:
            if prev is not None:
                eb0 = prev["axial_strain"] - w_lo * prev["kappa"]
                t = min(1.0, max(0.0, (eps_cr - eb0) / (eb - eb0)
                                 if eb > eb0 else 1.0))
                kappa_cr = prev["kappa"] + t * (p["kappa"] - prev["kappa"])
                M_cr = prev["M"] + t * (p["M"] - prev["M"])
            else:
                kappa_cr, M_cr = p["kappa"], p["M"]
            break
        prev = p

    def near(k):
        return min(pts, key=lambda p: abs(p["kappa"] - k)) if k and pts else None

    milestones = []
    for lab, state, k, m in (("a", "Cracking", kappa_cr, M_cr),
                             ("b", "First yield (tension steel)", kappa_y, M_y),
                             ("d", f"Ultimate ({failure})", kappa_u, M_u)):
        if k is None or m is None:
            continue
        p = near(k)
        milestones.append({
            "label": lab, "state": state, "kappa": float(k), "M": m / 1e3,
            "eps0": float(p["axial_strain"]) if p else 0.0,
            "eps_top": float(p["eps_top"]) if p else 0.0,
            "eps_steel": float(p["eps_steel"]) if p else 0.0})
    mu = (kappa_u / kappa_y) if kappa_y else None
    MkNm = [m / 1e3 for m in Ms]
    ctrl = _mphi_ctrl(kap, MkNm, [-p["eps_top"] for p in pts],
                      [p["eps_steel"] for p in pts], eps_cu)
    return {
        "kappa": kap, "M": MkNm,
        "M_cr": (M_cr or 0) / 1e3, "kappa_cr": kappa_cr,
        "M_y": (M_y / 1e3) if M_y else None, "kappa_y": kappa_y,
        "M_u": M_u / 1e3, "kappa_u": kappa_u, "mu_phi": mu,
        "failure_mode": failure, "milestones": milestones, "ideal": None,
        "y_top": w_hi, "y_bot": w_lo, "rebar_ys": [wy for wy, _a in bars],
        "na_angle": float(na_angle), "ctrl": ctrl,
        "conc_model": conc_model, "steel_model": steel_model}


def section_pm_slice(case, *, theta_deg=0.0, n_points=40,
                     eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4, fr_coeff=0.62,
                     fr_model="sqrt", eps_decay=1.0e-3, E_s=E_S, steel_b=0.01,
                     conc_model="Kent-Park", conc_f1_ratio=0.4,
                     steel_model="Bilinear", steel_fu_ratio=1.5,
                     steel_eps_sh=0.008, steel_eps_su=0.10, **_ignored):
    """Integrated (fibre-model) P-M interaction slice at neutral-axis angle
    ``theta_deg`` (P5): pin the extreme compression fibre at −ε_cu, sweep the
    N-A depth, and integrate the *actual* constitutive laws over the true
    section with the same exact quadrature as :func:`exact_mphi` — the fibre-
    model counterpart to the code's simplified :func:`pmm_slice`. Returns
    ``{"P": [...], "M": [...]}`` in kN / kN·m (nominal strength, φ = 1)."""
    if abs(theta_deg) > 1e-9:
        case = _rotate_case(case, theta_deg)
    concrete = concrete_uniaxial_from(dict(
        fc=case.f_c_prime, conc_model=conc_model, eps_c0=eps_c0, eps_cu=eps_cu,
        fcu_ratio=fcu_ratio, fr_model=fr_model, fr_coeff=fr_coeff,
        eps_decay=eps_decay, conc_f1_ratio=conc_f1_ratio))
    steel = steel_uniaxial_from(dict(
        fy=case.f_y, Es=E_s, steel_model=steel_model, steel_b=steel_b,
        steel_fu_ratio=steel_fu_ratio, steel_eps_sh=steel_eps_sh,
        steel_eps_su=steel_eps_su))
    w_lo, w_hi, width_bands = _width_bands(case.section.geometry.polygon)
    bars = [(float(b.y), float(b.area)) for b in
            (case.section.reinforcement.bars if case.section.reinforcement
             else [])]
    xg, wg = _GL
    eps_cr = concrete.f_ct / concrete.E_ct
    breaks = (eps_cr, 0.0, -eps_c0, -eps_cu)

    def resultants(a, s):
        """(N, M) for a strain field ε(w) = a + s·w (tension-positive N,
        M = -∫σ·w), by strain-banded Gauss quadrature over the width profile."""
        N = M = 0.0
        for wa0, wb0, w_ref, W_ref, slope in width_bands:
            edges = [wa0, wb0]
            if abs(s) > 1e-12:
                for e_bp in breaks:
                    wb = (e_bp - a) / s
                    if wa0 < wb < wb0:
                        edges.append(wb)
                edges.sort()
            for lo, hi in zip(edges[:-1], edges[1:]):
                half = 0.5 * (hi - lo)
                if half <= 1e-12:
                    continue
                mid = 0.5 * (lo + hi)
                for xi, wi in zip(xg, wg):
                    w = mid + half * xi
                    bwid = W_ref + slope * (w - w_ref)
                    if bwid <= 0.0:
                        continue
                    sig, _Et = concrete.get_response(a + s * w)
                    gwt = bwid * half * wi
                    N += sig * gwt
                    M -= sig * gwt * w
        for wy, area in bars:
            sig, _Et = steel.get_response(a + s * wy)
            N += sig * area
            M -= sig * area * wy
        return N, M

    depth = max(w_hi - w_lo, 1e-6)
    cs = np.geomspace(0.03 * depth, 8.0 * depth, n_points)
    P, M = [], []
    for c in cs:                                      # tension end -> compression
        kappa = eps_cu / c                            # extreme comp fibre = −ε_cu
        a = -eps_cu + w_hi * kappa
        N, Mv = resultants(a, -kappa)
        P.append(-N / 1e3)
        M.append(Mv / 1e3)
    # truncate the post-peak softening tail: keep the tension end up to peak
    # axial, then drop any trailing negative-M points so the envelope ends
    # cleanly at the compression apex (M ≈ 0).
    if not P:
        return {"P": [], "M": []}
    i_max = max(range(len(P)), key=lambda i: P[i])
    keep = list(range(i_max + 1))
    while len(keep) > 1 and M[keep[-1]] < 0.0:
        keep.pop()
    return {"P": [P[i] for i in keep], "M": [M[i] for i in keep]}


# ============================================================ P2: Mander wrapper
# A clean, typed, UNIT-AGNOSTIC entry point to the Mander confinement calculator
# above, for the fiber-hinge stream (plan §5.2 / §15 U4 — one confinement calc
# feeds both the Section Designer and the fiber hinge). It just packs a
# ``conf_*`` dict and calls :func:`mander_confinement`; no second calculator.

@dataclass(frozen=True)
class ConfinedCircular:
    """Mander confined-concrete properties for a circular core.

    Units follow the caller's consistent unit system (stresses in the same
    units as ``fco``; strains dimensionless). ``eps_cu`` uses the
    experimental Mander/Priestley formula by default, which is unit-agnostic
    (the energy-balance variant assumes SI, so it is not the default here).
    """
    fcc: float          # confined peak strength
    eps_cc: float       # strain at confined peak
    eps_cu: float       # confined ultimate strain
    ke: float           # confinement effectiveness coefficient
    fl: float           # effective lateral confining stress

    def to_material(self, Ec: float, *, min_strength_ratio: float = 0.2):
        """Build the confined :class:`ConcreteMander` law (``Ec`` in the
        caller's units) ready to drop into a fiber section."""
        return ConcreteMander(fpc=self.fcc, eps_c0=self.eps_cc, Ec=Ec,
                              min_strength_ratio=min_strength_ratio)


def mander_confined_circular(
    *,
    fco: float,
    eps_co: float,
    D_core: float,
    hoop_area: float,
    hoop_spacing: float,
    fyh: float,
    rho_long: float = 0.0,
    hoop_type: str = "hoop",
    clear_spacing: float | None = None,
    eps_su_hoop: float = 0.09,
    ecu_method: str = "experiment",
) -> ConfinedCircular:
    """Mander (1988) confined concrete for a circular column core.

    A thin, typed, unit-agnostic wrapper over :func:`mander_confinement`
    (the same calculator the Section Designer uses). Verified against the
    Midas benchmark card (§2.2): for the 84 in Caltrans column (79 in core,
    #8 hoop @ 6 in, f_yh = 68 ksi, f'c = 5 ksi) it returns
    ``fcc ≈ 6.37 ksi`` (Midas 6.356, 0.24%), ``eps_cc ≈ 0.00526`` (Midas
    0.00523), ``ke = 0.9625`` and ``fl = 0.2182 ksi`` (both exact).

    Parameters
    ----------
    fco, eps_co : float
        Unconfined peak strength and strain.
    D_core : float
        Confined-core diameter (to the hoop centre-line).
    hoop_area, hoop_spacing : float
        One hoop/spiral bar area and its centre-to-centre longitudinal spacing.
    fyh : float
        Hoop yield strength.
    rho_long : float, default 0.0
        Longitudinal steel ratio of the core (for the Acc net-of-steel term).
    hoop_type : {"hoop", "spiral"}
        Circular hoops (exponent 2 in ke) or continuous spiral (exponent 1).
        The benchmark uses **hoops** (matches CSI ``CnfType=Hoop``).
    clear_spacing : float, optional
        Clear vertical spacing s' between hoops. Defaults to ``hoop_spacing``
        (slightly conservative ke); pass the true clear spacing for fidelity.
    eps_su_hoop : float, default 0.09
        Hoop fracture strain (used by the experimental eps_cu).
    ecu_method : {"experiment", "energy"}
        ``"experiment"`` (default) is unit-agnostic; ``"energy"`` assumes SI.
    """
    m = {
        "fc": float(fco), "eps_c0": float(eps_co), "conf_shape": "Circular",
        "conf_fyh": float(fyh), "conf_Asp": float(hoop_area),
        "conf_s": float(hoop_spacing),
        "conf_sp": float(clear_spacing if clear_spacing is not None
                         else hoop_spacing),
        "conf_ds": float(D_core), "conf_rho_cc": float(rho_long),
        "conf_hooptype": "Spiral" if hoop_type.lower() == "spiral" else "Hoop",
        "conf_eps_su_h": float(eps_su_hoop), "conf_ecu_method": ecu_method,
    }
    r = mander_confinement(m)
    return ConfinedCircular(fcc=r["fcc"], eps_cc=r["eps_cc"],
                            eps_cu=r["eps_cu"], ke=r["ke"], fl=r["fl"])


def mphi_data(case: SectionCase, P_target_kN: float, *,
              na_angle: float = 0.0,
              eps_c0: float = 0.002, eps_cu: float = 0.0035,
              fcu_ratio: float = 0.4, fr_coeff: float = 0.62,
              fr_model: str = "sqrt", eps_decay: float = 1.0e-3,
              E_s: float = E_S, steel_b: float = 0.01,
              kappa_max: float = 0.06, conc_model: str = "Kent-Park",
              conc_f1_ratio: float = 0.4,
              steel_model: str = "Bilinear", steel_fu_ratio: float = 1.5,
              steel_eps_sh: float = 0.008, steel_eps_su: float = 0.10,
              n_points: int = 65, stop: str = "concrete",
              n_z: int = 16, n_y: int = 40):
    # Neutral-axis angle: rotate the section so the inclined bending axis
    # aligns with the engine's z-axis, then everything downstream (crack
    # point, milestones, strain profile) is in that rotated frame.
    if abs(na_angle) > 1e-9:
        case = _rotate_case(case, na_angle)
    fc = case.f_c_prime
    concrete = concrete_uniaxial_from(dict(
        fc=fc, conc_model=conc_model, eps_c0=eps_c0, eps_cu=eps_cu,
        fcu_ratio=fcu_ratio, fr_model=fr_model, fr_coeff=fr_coeff,
        eps_decay=eps_decay, conc_f1_ratio=conc_f1_ratio))
    f_r, E_c = concrete.f_ct, concrete.E_ct
    steel = steel_uniaxial_from(dict(
        fy=case.f_y, Es=E_s, steel_model=steel_model, steel_b=steel_b,
        steel_fu_ratio=steel_fu_ratio, steel_eps_sh=steel_eps_sh,
        steel_eps_su=steel_eps_su))
    # Sample curvature finely through the (small) cracking region so the
    # steep uncracked branch is resolved, then coarsely to kappa_max.
    y_top = case.section.geometry.polygon.bounds[3]
    kcr_est = f_r / max(E_c * y_top, 1e-9)
    kappas = _mphi_kappas(kcr_est, kappa_max, n_points)
    # M4 failure criterion -> steer the engine's crush / rupture stops. Concrete
    # crushing at ε_cu (default); 'steel' lets the concrete soften past ε_cu and
    # ends at rebar rupture; 'peak' disables both so the sweep runs to κ_max.
    eff_cu = eps_cu if stop == "concrete" else 1.0
    eff_su = 1.0 if stop == "peak" else 0.05
    res = moment_curvature(
        case.section, P_target=P_target_kN * 1e3,
        concrete_uniaxial=concrete, steel_uniaxial=steel,
        kappas=kappas, f_y=case.f_y, E_s=E_s, f_rupture=f_r,
        eps_cu_crush=eff_cu, eps_steel_rupture=eff_su, n_z=n_z, n_y=n_y)
    pts = res.points

    def _nearest(kap):
        if kap is None or not pts:
            return None
        return min(pts, key=lambda p: abs(p.kappa - kap))

    # On-curve cracking point: first curvature where the extreme tension
    # concrete fibre reaches the cracking strain eps_cr = f_r / E_c. This
    # lies exactly on the (tension-stiffened) fibre curve.
    eps_cr = f_r / E_c
    y_bot = case.section.geometry.polygon.bounds[1]
    kappa_crack, M_crack = res.kappa_cr, res.M_cr
    prev = None
    for p in pts:
        eb = p.axial_strain - y_bot * p.kappa     # extreme tension fibre
        if p.kappa > 0.0 and eb >= eps_cr:
            if prev is not None:
                eb0 = prev.axial_strain - y_bot * prev.kappa
                t = (eps_cr - eb0) / (eb - eb0) if eb > eb0 else 1.0
                t = min(1.0, max(0.0, t))
                kappa_crack = prev.kappa + t * (p.kappa - prev.kappa)
                M_crack = prev.M + t * (p.M - prev.M)
            else:
                kappa_crack, M_crack = float(p.kappa), float(p.M)
            break
        prev = p

    # GSD-style milestone points, each with the section strain state
    # (axial strain eps0, extreme concrete strain, max tension-steel
    # strain) taken from the nearest computed point.
    milestones = []
    for label, state, kap, mom in (
        ("a", "Cracking", kappa_crack, M_crack),
        ("b", "First yield (tension steel)", res.kappa_y, res.M_y),
        ("d", f"Ultimate ({res.failure_mode or 'peak'})",
         res.kappa_u, res.M_u),
    ):
        if kap is None or mom is None:
            continue
        p = _nearest(kap)
        milestones.append({
            "label": label, "state": state,
            "kappa": float(kap), "M": mom / 1e3,
            "eps0": float(p.axial_strain) if p else 0.0,
            "eps_top": float(p.eps_top_concrete) if p else 0.0,
            "eps_steel": float(p.eps_max_steel) if p else 0.0,
        })
    # Full equal-energy bilinear idealization (origin -> yield -> ultimate),
    # in display units (kN.m). The GUI draws it as an overlay polyline.
    ideal = None
    try:
        (ky, My), (ku, Mu) = res.bilinear()
        ideal = {"kappa_y": float(ky), "M_y": My / 1e3,
                 "kappa_u": float(ku), "M_u": Mu / 1e3}
    except Exception:
        pass

    bnd = case.section.geometry.polygon.bounds
    rebar_ys = ([b.y for b in case.section.reinforcement.bars]
                if case.section.reinforcement else [])
    ctrl = _mphi_ctrl([p.kappa for p in pts], [p.M / 1e3 for p in pts],
                      [-p.eps_top_concrete for p in pts],
                      [p.eps_max_steel for p in pts], eps_cu)

    return {
        "kappa": [p.kappa for p in pts],
        "M": [p.M / 1e3 for p in pts],
        "M_cr": (M_crack or 0) / 1e3, "kappa_cr": kappa_crack,
        "M_y": (res.M_y / 1e3) if res.M_y else None, "kappa_y": res.kappa_y,
        "M_u": (res.M_u or 0) / 1e3, "kappa_u": res.kappa_u,
        "mu_phi": res.mu_phi, "failure_mode": res.failure_mode,
        "milestones": milestones, "ideal": ideal,
        "y_top": bnd[3], "y_bot": bnd[1], "rebar_ys": rebar_ys,
        "na_angle": float(na_angle), "ctrl": ctrl,
        "conc_model": conc_model, "steel_model": steel_model,
    }


# ================================================ code-based P-M-M slice
# Relocated **verbatim** from the Section Designer GUI core
# (``section_gui_core.py``) so the code (stress-block) interaction envelope
# lives beside the fibre/exact integrators in one engine core (plan §15 U3).
# ``section_gui_core`` now re-exports this name.

def pmm_slice(case: SectionCase, code: str, n: int = 44,
              theta_deg: float = 0.0):
    """Interaction curve + landmarks for one code, at neutral-axis angle
    ``theta_deg`` (0 = strong-axis P-Mz). The section is rigidly rotated so the
    inclined axis lands on z, then the strong-axis slice is taken -- the same
    convention the moment-curvature view uses for its N-axis angle."""
    if abs(theta_deg) > 1e-9:
        case = _rotate_case(case, theta_deg)
    sec = case.section
    bnd = sec.geometry.polygon.bounds
    depth_y = bnd[3] - bnd[1]
    fc, fy = case.f_c_prime, case.f_y

    if code == "AASHTO LRFD 2024":
        surf = biaxial_pmm_surface_aashto(
            sec, f_c_prime=fc, f_y=fy, spiral=case.spiral,
            prestressed=case.prestressed, n_angles=4, n_depths=n)

        def pf(c):
            return biaxial_pmm_point_aashto(
                sec, 0.0, c, f_c_prime=fc, f_y=fy, spiral=case.spiral,
                prestressed=case.prestressed)
        eps_y = fy / E_S
    elif code == "Eurocode 2":
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=fc, f_yk=fy, n_angles=4, n_depths=n)

        def pf(c):
            return biaxial_pmm_point_ec2(sec, 0.0, c, f_ck=fc, f_yk=fy)
        eps_y = (fy / 1.15) / E_S
    else:  # IS 456:2000
        surf = biaxial_pmm_surface_is456(
            sec, f_ck=fc, f_y=fy, n_angles=4, n_depths=n)

        def pf(c):
            return biaxial_pmm_point_is456(sec, 0.0, c, f_ck=fc, f_y=fy)
        eps_y = (fy / 1.15) / E_S

    pts = sorted(surf.slice_uniaxial_z(), key=lambda p: p.P_n)
    curve = {
        "M_nom": [p.M_nz / 1e3 for p in pts],
        "P_nom": [p.P_n / 1e3 for p in pts],
        "M_des": [p.phi_M_nz / 1e3 for p in pts],
        "P_des": [p.phi_P_n / 1e3 for p in pts],
        "has_design": code == "AASHTO LRFD 2024",
    }
    lm = None
    if not case.prestressed:
        L = strong_axis_landmarks(
            pf, P_o=surf.P_o, P_n_max=surf.P_n_max,
            depth_y=depth_y, eps_y=eps_y)
        # (name, canonical value, kind) -- kind "P"=force (kN), "M"=moment
        lm = [
            ("P_o (squash)", L.P_o / 1e3, "P"),
            ("P_n,max", L.P_n_max / 1e3, "P"),
            ("M_n @ P=0", L.M0_nominal / 1e3, "M"),
            ("φM_n @ P=0", L.M0_phi / 1e3, "M"),
            ("Balanced P_b", L.P_bal / 1e3, "P"),
            ("Balanced M_b", L.M_bal_nominal / 1e3, "M"),
        ]
    return curve, lm


# ==================================================== U3: unified section API
# One result type + backend selector so the Section Designer tool, the tests,
# and the fiber-hinge stream all consume the *same* section-analysis output --
# no session forks a parallel section engine (plan §15 U3). The four backend
# tokens name the four capacity models the codebase already computes:
#
#   C_EXACT   -- exact-integration M-phi over the true geometry (:func:`exact_mphi`)
#   C_FIBRE   -- cell/fibre-model M-phi and P-M (:func:`mphi_data`, :func:`section_pm_slice`)
#   C_NOMINAL -- code stress-block P-M-M envelope, phi = 1        (:func:`pmm_slice`)
#   C_DESIGN  -- code stress-block P-M-M envelope with strength-reduction phi
#
# M-phi accepts {C_EXACT, C_FIBRE}; P-M accepts {C_FIBRE, C_NOMINAL, C_DESIGN}.

C_EXACT = "exact"
C_FIBRE = "fibre"
C_NOMINAL = "nominal"
C_DESIGN = "design"

MPHI_BACKENDS = (C_EXACT, C_FIBRE)
PM_BACKENDS = (C_FIBRE, C_NOMINAL, C_DESIGN)


@dataclass
class MomentCurvatureResult:
    """Unified moment-curvature result (kN, m; moments kN.m). Produced by
    :func:`moment_curvature_analysis` for either the exact or fibre backend;
    both integrate the same uniaxial laws so the curves differ only by
    discretisation. ``raw`` keeps the original backend dict for callers (the
    GUI) that still read the wire format."""
    backend: str
    kappa: list
    M: list
    M_cr: float
    kappa_cr: float
    M_y: object
    kappa_y: object
    M_u: float
    kappa_u: object
    mu_phi: object
    failure_mode: object
    milestones: list
    ideal: object
    y_top: float
    y_bot: float
    rebar_ys: list
    na_angle: float
    ctrl: object
    conc_model: str
    steel_model: str
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def _from_dict(cls, backend: str, d: dict) -> "MomentCurvatureResult":
        return cls(
            backend=backend, raw=d,
            **{k: d.get(k) for k in (
                "kappa", "M", "M_cr", "kappa_cr", "M_y", "kappa_y",
                "M_u", "kappa_u", "mu_phi", "failure_mode", "milestones",
                "ideal", "y_top", "y_bot", "rebar_ys", "na_angle", "ctrl",
                "conc_model", "steel_model")})


@dataclass
class PMInteractionResult:
    """Unified axial-moment interaction slice (axial P in kN, moment M in
    kN.m). Produced by :func:`pm_interaction`. For the fibre backend the curve
    is the nominal fibre-model envelope (phi = 1, no landmarks); for the code
    backends ``P``/``M`` carry the nominal (C_NOMINAL) or design (C_DESIGN)
    envelope and ``landmarks`` the strong-axis capacity points."""
    backend: str
    P: list
    M: list
    code: object = None
    theta_deg: float = 0.0
    has_design: bool = False
    landmarks: object = None
    raw: object = field(default=None, repr=False)


def moment_curvature_analysis(case: SectionCase, P_target_kN: float, *,
                              backend: str = C_EXACT,
                              **kwargs) -> MomentCurvatureResult:
    """One moment-curvature entry point over the two fibre-consistent backends.

    ``backend=C_EXACT`` runs :func:`exact_mphi` (strain-banded Gauss-Legendre
    quadrature over the true geometry); ``backend=C_FIBRE`` runs
    :func:`mphi_data` (cell/fibre mesh). Both take the same material/model
    keyword arguments (``conc_model``, ``steel_model``, ``eps_cu``, ``na_angle``,
    ...); extra kwargs are forwarded unchanged. This is the M-phi half of the
    unified section API the Section Designer tool and the fiber hinge share."""
    if backend == C_EXACT:
        d = exact_mphi(case, P_target_kN, **kwargs)
    elif backend == C_FIBRE:
        d = mphi_data(case, P_target_kN, **kwargs)
    else:
        raise ValueError(
            f"moment_curvature_analysis: backend must be one of "
            f"{MPHI_BACKENDS!r}, got {backend!r}")
    return MomentCurvatureResult._from_dict(backend, d)


def pm_interaction(case: SectionCase, *, backend: str = C_FIBRE,
                   theta_deg: float = 0.0, code: str = "AASHTO LRFD 2024",
                   n_points: int = 40, **kwargs) -> PMInteractionResult:
    """One P-M interaction entry point over the three interaction backends.

    ``backend=C_FIBRE`` integrates the actual uniaxial laws over the true
    section (:func:`section_pm_slice`) -- the same physics the fiber hinge sees.
    ``backend=C_NOMINAL`` / ``C_DESIGN`` return the code stress-block envelope
    (:func:`pmm_slice`) for ``code`` at the nominal (phi = 1) or design
    (strength-reduced) level, plus the strong-axis landmarks. ``theta_deg`` sets
    the neutral-axis angle (0 = strong-axis P-Mz); extra kwargs pass to the
    fibre backend's material/model options."""
    if backend == C_FIBRE:
        d = section_pm_slice(case, theta_deg=theta_deg, n_points=n_points,
                             **kwargs)
        return PMInteractionResult(
            backend=backend, P=d["P"], M=d["M"], code=None,
            theta_deg=float(theta_deg), has_design=False, landmarks=None,
            raw=d)
    if backend in (C_NOMINAL, C_DESIGN):
        curve, lm = pmm_slice(case, code, n=n_points, theta_deg=theta_deg)
        if backend == C_DESIGN:
            P, M = curve["P_des"], curve["M_des"]
        else:
            P, M = curve["P_nom"], curve["M_nom"]
        return PMInteractionResult(
            backend=backend, P=P, M=M, code=code, theta_deg=float(theta_deg),
            has_design=curve["has_design"], landmarks=lm, raw=curve)
    raise ValueError(
        f"pm_interaction: backend must be one of {PM_BACKENDS!r}, "
        f"got {backend!r}")
