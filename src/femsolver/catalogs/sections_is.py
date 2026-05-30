"""IS 808 / SP6-1 Indian hot-rolled section catalogue.

Three families covered:

* **ISMB** -- Indian Standard Medium Weight Beams (most-common I-beam).
* **ISMC** -- Indian Standard Medium Weight Channels.
* **ISA**  -- Indian Standard Equal Angles.

Data taken from IS 808 (2021 amendments). All units SI: mm for
dimensions, mm^2 for area, mm^4 for I, mm^3 for section modulus,
kg/m for unit weight.
"""
from __future__ import annotations

from femsolver.catalogs.sections_ec import SectionProperties


IS_ISMB: dict[str, SectionProperties] = {}
IS_ISMC: dict[str, SectionProperties] = {}
IS_ISA:  dict[str, SectionProperties] = {}


def _ismb(name, mass, h, b, tw, tf, A_mm2, Iy_cm4, Iz_cm4,
          Wely_cm3, Welz_cm3, ry_cm, rz_cm, J_cm4=0.0):
    # Plastic moduli for I-beams approximated as 1.14 * W_el (typical
    # shape factor for narrow-flange I-sections per IS 800).
    Wply = 1.14 * Wely_cm3
    Wplz = 1.34 * Welz_cm3   # higher shape factor in weak axis
    IS_ISMB[name] = SectionProperties(
        name=name, family="ISMB",
        mass=mass, h=h, b=b, t_w=tw, t_f=tf, A=A_mm2,
        I_y=Iy_cm4 * 1e4, I_z=Iz_cm4 * 1e4,
        W_pl_y=Wply * 1e3, W_pl_z=Wplz * 1e3,
        W_el_y=Wely_cm3 * 1e3, W_el_z=Welz_cm3 * 1e3,
        r_y=ry_cm * 10, r_z=rz_cm * 10, J=J_cm4 * 1e4,
    )


# ISMB selected sections (IS 808 Table 1)
# Format: name, mass, h, b, t_w, t_f, A (mm^2), I_y (cm^4), I_z (cm^4),
#         W_el_y (cm^3), W_el_z (cm^3), r_y (cm), r_z (cm)
_ismb("ISMB 100",  8.9, 100,  75, 4.0, 7.2, 1140,  257.4, 40.8,  51.5, 10.9,
      4.75, 1.89)
_ismb("ISMB 125", 13.0, 125,  75, 4.4, 8.1, 1660,  448.7, 43.4,  71.8, 11.6,
      5.20, 1.61)
_ismb("ISMB 150", 14.9, 150,  80, 4.8, 7.6, 1900,  718.0, 52.3,  95.7, 13.1,
      6.15, 1.66)
_ismb("ISMB 175", 19.3, 175,  85, 5.8, 9.1, 2460, 1242.0, 73.7, 142.0, 17.3,
      7.10, 1.73)
_ismb("ISMB 200", 25.4, 200, 100, 5.7, 10.8, 3240, 2235.0, 150.0, 224.0, 30.0,
      8.30, 2.15)
_ismb("ISMB 225", 31.2, 225, 110, 6.5, 11.8, 3960, 3441.0, 218.0, 306.0, 39.7,
      9.32, 2.34)
_ismb("ISMB 250", 37.3, 250, 125, 6.9, 12.5, 4750, 5131.6, 334.5, 410.5, 53.5,
      10.39, 2.65)
_ismb("ISMB 300", 44.2, 300, 140, 7.5, 12.4, 5630, 8603.6, 453.6, 573.6, 64.8,
      12.35, 2.84)
_ismb("ISMB 350", 52.4, 350, 140, 8.1, 14.2, 6670, 13630, 535.9, 778.9, 76.6,
      14.29, 2.84)
_ismb("ISMB 400", 61.5, 400, 140, 8.9, 16.0, 7840, 20458, 622.1, 1022.9, 88.9,
      16.16, 2.82)
_ismb("ISMB 450", 72.4, 450, 150, 9.4, 17.4, 9220, 30391, 834.0, 1350.7, 111.2,
      18.15, 3.01)
_ismb("ISMB 500", 86.9, 500, 180, 10.2, 17.2, 11074, 45219, 1369.8, 1808.8, 152.2,
      20.21, 3.52)
_ismb("ISMB 550", 103.7, 550, 190, 11.2, 19.3, 13211, 64893, 1834.0, 2359.8, 193.0,
      22.16, 3.73)
_ismb("ISMB 600", 122.6, 600, 210, 12.0, 20.8, 15621, 91812, 2647.3, 3060.4, 252.1,
      24.24, 4.12)


def _ismc(name, mass, h, b, tw, tf, A_mm2, Iy_cm4, Iz_cm4,
          Wely_cm3, Welz_cm3, ry_cm, rz_cm):
    IS_ISMC[name] = SectionProperties(
        name=name, family="ISMC",
        mass=mass, h=h, b=b, t_w=tw, t_f=tf, A=A_mm2,
        I_y=Iy_cm4 * 1e4, I_z=Iz_cm4 * 1e4,
        W_pl_y=1.18 * Wely_cm3 * 1e3,
        W_pl_z=1.40 * Welz_cm3 * 1e3,
        W_el_y=Wely_cm3 * 1e3, W_el_z=Welz_cm3 * 1e3,
        r_y=ry_cm * 10, r_z=rz_cm * 10,
    )


# ISMC family (channels). Format identical to ISMB.
_ismc("ISMC 75",   6.8,  75, 40, 4.4, 7.3,  870,  76.1,  12.6, 20.3, 4.66,
      2.96, 1.21)
_ismc("ISMC 100",  9.2, 100, 50, 4.7, 7.5, 1170, 186.3, 25.8, 37.3, 7.41,
      3.99, 1.49)
_ismc("ISMC 125", 12.7, 125, 65, 5.0, 8.1, 1620, 405.4, 53.5, 64.9, 13.3,
      5.00, 1.81)
_ismc("ISMC 150", 16.0, 150, 75, 5.4, 9.0, 2090, 778.8, 86.3, 103.8, 19.4,
      6.10, 2.03)
_ismc("ISMC 175", 19.1, 175, 75, 5.7, 10.2, 2500, 1223, 99.2, 139.8, 22.4,
      6.99, 1.99)
_ismc("ISMC 200", 22.1, 200, 75, 6.1, 11.4, 2890, 1819, 140.4, 181.9, 26.3,
      7.93, 2.21)
_ismc("ISMC 250", 30.4, 250, 80, 7.1, 14.1, 3870, 3816, 219.1, 305.3, 38.8,
      9.94, 2.38)
_ismc("ISMC 300", 36.3, 300, 90, 7.6, 13.6, 4630, 6356, 313.4, 423.7, 47.8,
      11.71, 2.60)
_ismc("ISMC 400", 49.4, 400, 100, 8.8, 15.3, 6293, 15082, 504.8, 754.1, 67.1,
      15.49, 2.83)


# Equal angles (ISA)
def _isa(name, leg, t, mass, A_mm2, Iy_cm4, Iz_cm4, ry_cm, rz_cm):
    """Equal angle of leg x leg x t (mm), unit weight kg/m."""
    IS_ISA[name] = SectionProperties(
        name=name, family="ISA",
        mass=mass, h=leg, b=leg, t_w=t, t_f=t, A=A_mm2,
        I_y=Iy_cm4 * 1e4, I_z=Iz_cm4 * 1e4,
        # Plastic moduli not standardised for angles; user computes
        # from section if needed
        W_pl_y=0.0, W_pl_z=0.0,
        W_el_y=Iy_cm4 / (leg / 2.0) * 1e4 / 1e3,
        W_el_z=Iz_cm4 / (leg / 2.0) * 1e4 / 1e3,
        r_y=ry_cm * 10, r_z=rz_cm * 10,
    )


# Selected ISA equal angles. Data: name, leg (mm), thickness, mass, A, I (cm^4),
# I (same — symmetric), r (cm) -- y and z equal for symmetric.
_isa("ISA 50x50x6",   50,  6, 4.47,  569,  12.10, 12.10, 1.46, 1.46)
_isa("ISA 65x65x6",   65,  6, 5.79,  738,  27.50, 27.50, 1.93, 1.93)
_isa("ISA 75x75x8",   75,  8, 8.86, 1129,  56.40, 56.40, 2.23, 2.23)
_isa("ISA 90x90x8",   90,  8, 10.8, 1376, 101.0, 101.0,  2.71, 2.71)
_isa("ISA 100x100x8", 100, 8, 12.1, 1542, 143.0, 143.0,  3.04, 3.04)
_isa("ISA 110x110x10", 110, 10, 16.6, 2106, 233.0, 233.0, 3.32, 3.32)
_isa("ISA 130x130x10", 130, 10, 19.7, 2510, 388.0, 388.0, 3.93, 3.93)
_isa("ISA 150x150x12", 150, 12, 27.3, 3479, 720.0, 720.0, 4.55, 4.55)
_isa("ISA 200x200x16", 200, 16, 48.5, 6178, 2306, 2306,   6.10, 6.10)


# ============================================================ lookup
_ALL_IS: dict[str, SectionProperties] = {}
_ALL_IS.update(IS_ISMB)
_ALL_IS.update(IS_ISMC)
_ALL_IS.update(IS_ISA)


def indian_section(name: str) -> SectionProperties:
    """Look up an Indian section by name."""
    key = name.strip().upper()
    if key not in _ALL_IS:
        for k in _ALL_IS:
            if k.upper() == key:
                return _ALL_IS[k]
        raise KeyError(
            f"unknown Indian section {name!r}; see "
            f"list_indian_sections() for options"
        )
    return _ALL_IS[key]


def list_indian_sections(family: str | None = None) -> list[str]:
    """List available IS section names, optionally filtered by family."""
    if family is None:
        return list(_ALL_IS.keys())
    f = family.upper()
    if f == "ISMB":
        return list(IS_ISMB.keys())
    if f == "ISMC":
        return list(IS_ISMC.keys())
    if f == "ISA":
        return list(IS_ISA.keys())
    raise ValueError(f"unknown family {family!r}")
