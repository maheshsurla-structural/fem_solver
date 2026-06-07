"""Eurocode hot-rolled section catalogue (IPE, HEA, HEB).

All units SI: mm for dimensions, mm^2 for area, mm^4 for second
moment of area, mm^3 for elastic / plastic section modulus,
kg/m for mass per length.

Source -- ArcelorMittal IPE/HE design data sheets, cross-checked
against EN 10365.

Reference values are deliberately rounded to 4-significant-figure
precision (matches typical design-handbook accuracy).

.. note::

    **Theme II.7 migration:** the unified Section Designer accesses
    these tables through
    :func:`femsolver.sections.eurocode_section` and
    :meth:`femsolver.sections.SectionLibrary.eurocode`. New user
    code should prefer the unified path; the dataclasses in this
    module remain available for backward compatibility and for the
    EC3 design code's internal use.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SectionProperties:
    """Cross-section properties shared by EC, IS, AISC catalogues."""
    name: str
    family: str          # e.g., "IPE", "HEA", "HEB", "ISMB", "W"
    mass: float          # kg/m
    h: float             # depth (mm)
    b: float             # width (mm)
    t_w: float           # web thickness (mm)
    t_f: float           # flange thickness (mm)
    A: float             # area (mm^2)
    I_y: float           # I about strong axis (mm^4)
    I_z: float           # I about weak axis (mm^4)
    W_pl_y: float        # plastic modulus, strong axis (mm^3)
    W_pl_z: float        # plastic modulus, weak axis (mm^3)
    W_el_y: float        # elastic modulus, strong axis (mm^3)
    W_el_z: float        # elastic modulus, weak axis (mm^3)
    r_y: float           # radius of gyration, strong (mm)
    r_z: float           # radius of gyration, weak (mm)
    J: float = 0.0       # torsion constant (mm^4)


# ============================================================ IPE
# IPE family (narrow-flange I-beam, EN 10365). Selected representative
# sections covering the design range 80-600.

EC_IPE: dict[str, SectionProperties] = {}


def _add_ipe(name, mass, h, b, t_w, t_f, A, I_y, I_z,
              W_pl_y, W_pl_z, W_el_y, W_el_z, r_y, r_z, J):
    EC_IPE[name] = SectionProperties(
        name=name, family="IPE",
        mass=mass, h=h, b=b, t_w=t_w, t_f=t_f, A=A,
        I_y=I_y, I_z=I_z,
        W_pl_y=W_pl_y, W_pl_z=W_pl_z,
        W_el_y=W_el_y, W_el_z=W_el_z,
        r_y=r_y, r_z=r_z, J=J,
    )


# Tabulated from ArcelorMittal IPE catalogue (in mm and cm^3 -> mm^3)
# format: name, mass, h, b, t_w, t_f, A, I_y (cm4), I_z (cm4),
#         W_pl_y (cm3), W_pl_z (cm3), W_el_y (cm3), W_el_z (cm3),
#         r_y (cm), r_z (cm), J (cm4)
# Then we convert.
_IPE_RAW = [
    ("IPE 80",  6.0,  80,  46, 3.8, 5.2,  764,  80.1, 8.49,  23.2, 5.82,
     20.0, 3.69, 3.24, 1.05, 0.70),
    ("IPE 100", 8.1, 100,  55, 4.1, 5.7, 1030, 171.0, 15.9,  39.4, 9.15,
     34.2, 5.79, 4.07, 1.24, 1.20),
    ("IPE 120", 10.4, 120, 64, 4.4, 6.3, 1320, 318.0, 27.7,  60.7, 13.6,
     53.0, 8.65, 4.90, 1.45, 1.74),
    ("IPE 140", 12.9, 140, 73, 4.7, 6.9, 1640, 541.0, 44.9,  88.3, 19.3,
     77.3, 12.3, 5.74, 1.65, 2.45),
    ("IPE 160", 15.8, 160, 82, 5.0, 7.4, 2010, 869.0, 68.3, 124.0, 26.1,
     109.0, 16.7, 6.58, 1.84, 3.60),
    ("IPE 180", 18.8, 180, 91, 5.3, 8.0, 2390, 1320, 101.0, 166.0, 34.6,
     146.0, 22.2, 7.42, 2.05, 4.79),
    ("IPE 200", 22.4, 200, 100, 5.6, 8.5, 2850, 1940, 142.0, 221.0, 44.6,
     194.0, 28.5, 8.26, 2.24, 6.98),
    ("IPE 220", 26.2, 220, 110, 5.9, 9.2, 3340, 2770, 205.0, 285.0, 58.1,
     252.0, 37.3, 9.11, 2.48, 9.07),
    ("IPE 240", 30.7, 240, 120, 6.2, 9.8, 3910, 3890, 284.0, 367.0, 73.9,
     324.0, 47.3, 9.97, 2.69, 12.9),
    ("IPE 270", 36.1, 270, 135, 6.6, 10.2, 4590, 5790, 420.0, 484.0, 96.9,
     429.0, 62.2, 11.2, 3.02, 15.9),
    ("IPE 300", 42.2, 300, 150, 7.1, 10.7, 5380, 8360, 604.0, 628.0, 125.0,
     557.0, 80.5, 12.5, 3.35, 20.1),
    ("IPE 330", 49.1, 330, 160, 7.5, 11.5, 6260, 11770, 788.0, 804.0, 154.0,
     713.0, 98.5, 13.7, 3.55, 28.2),
    ("IPE 360", 57.1, 360, 170, 8.0, 12.7, 7270, 16270, 1043.0, 1019.0, 191.0,
     904.0, 122.8, 15.0, 3.79, 37.3),
    ("IPE 400", 66.3, 400, 180, 8.6, 13.5, 8450, 23130, 1318.0, 1307.0, 229.0,
     1156.0, 146.4, 16.5, 3.95, 51.1),
    ("IPE 450", 77.6, 450, 190, 9.4, 14.6, 9880, 33740, 1676.0, 1702.0, 276.0,
     1500.0, 176.4, 18.5, 4.12, 66.9),
    ("IPE 500", 90.7, 500, 200, 10.2, 16.0, 11600, 48200, 2142.0, 2194.0, 335.0,
     1928.0, 214.2, 20.4, 4.31, 89.3),
    ("IPE 550", 106.0, 550, 210, 11.1, 17.2, 13400, 67120, 2668.0, 2787.0, 400.0,
     2441.0, 254.1, 22.3, 4.45, 123.0),
    ("IPE 600", 122.0, 600, 220, 12.0, 19.0, 15600, 92080, 3387.0, 3512.0, 485.0,
     3069.0, 307.9, 24.3, 4.66, 165.0),
]
for row in _IPE_RAW:
    name, mass, h, b, tw, tf, A_mm2, Iy_cm4, Iz_cm4, Wply_cm3, Wplz_cm3, \
        Wely_cm3, Welz_cm3, ry_cm, rz_cm, J_cm4 = row
    _add_ipe(
        name, mass, h, b, tw, tf, A_mm2,
        Iy_cm4 * 1e4, Iz_cm4 * 1e4,
        Wply_cm3 * 1e3, Wplz_cm3 * 1e3,
        Wely_cm3 * 1e3, Welz_cm3 * 1e3,
        ry_cm * 10.0, rz_cm * 10.0, J_cm4 * 1e4,
    )


# ============================================================ HEA, HEB
# HEA (wide-flange medium) and HEB (wide-flange heavy)
EC_HEA: dict[str, SectionProperties] = {}
EC_HEB: dict[str, SectionProperties] = {}


def _add_hea(name, mass, h, b, t_w, t_f, A, I_y, I_z,
              W_pl_y, W_pl_z, W_el_y, W_el_z, r_y, r_z, J):
    EC_HEA[name] = SectionProperties(
        name=name, family="HEA",
        mass=mass, h=h, b=b, t_w=t_w, t_f=t_f, A=A,
        I_y=I_y, I_z=I_z,
        W_pl_y=W_pl_y, W_pl_z=W_pl_z,
        W_el_y=W_el_y, W_el_z=W_el_z,
        r_y=r_y, r_z=r_z, J=J,
    )


def _add_heb(name, mass, h, b, t_w, t_f, A, I_y, I_z,
              W_pl_y, W_pl_z, W_el_y, W_el_z, r_y, r_z, J):
    EC_HEB[name] = SectionProperties(
        name=name, family="HEB",
        mass=mass, h=h, b=b, t_w=t_w, t_f=t_f, A=A,
        I_y=I_y, I_z=I_z,
        W_pl_y=W_pl_y, W_pl_z=W_pl_z,
        W_el_y=W_el_y, W_el_z=W_el_z,
        r_y=r_y, r_z=r_z, J=J,
    )


_HEA_RAW = [
    ("HEA 100", 16.7, 96, 100, 5.0, 8.0, 2120, 349.0, 134.0, 83.0, 41.1,
     72.8, 26.8, 4.06, 2.51, 5.24),
    ("HEA 120", 19.9, 114, 120, 5.0, 8.0, 2530, 606.0, 231.0, 119.0, 58.9,
     106.0, 38.5, 4.89, 3.02, 5.99),
    ("HEA 140", 24.7, 133, 140, 5.5, 8.5, 3140, 1033, 389.0, 173.0, 84.9,
     155.0, 55.6, 5.73, 3.52, 8.13),
    ("HEA 160", 30.4, 152, 160, 6.0, 9.0, 3880, 1673, 616.0, 245.0, 117.6,
     220.0, 76.9, 6.57, 3.98, 11.6),
    ("HEA 180", 35.5, 171, 180, 6.0, 9.5, 4530, 2510, 925.0, 324.0, 156.5,
     294.0, 102.7, 7.45, 4.52, 14.7),
    ("HEA 200", 42.3, 190, 200, 6.5, 10.0, 5380, 3692, 1336, 429.0, 203.8,
     388.6, 133.6, 8.28, 4.98, 21.0),
    ("HEA 220", 50.5, 210, 220, 7.0, 11.0, 6430, 5410, 1955, 568.5, 270.6,
     515.2, 177.7, 9.17, 5.51, 28.5),
    ("HEA 240", 60.3, 230, 240, 7.5, 12.0, 7680, 7763, 2769, 744.6, 351.7,
     675.1, 230.7, 10.05, 6.00, 41.6),
    ("HEA 260", 68.2, 250, 260, 7.5, 12.5, 8680, 10455, 3668, 919.8, 430.2,
     836.4, 282.1, 10.97, 6.50, 52.4),
    ("HEA 280", 76.4, 270, 280, 8.0, 13.0, 9730, 13670, 4763, 1112, 518.1,
     1012, 340.2, 11.86, 7.00, 62.1),
    ("HEA 300", 88.3, 290, 300, 8.5, 14.0, 11250, 18260, 6310, 1383, 641.2,
     1259, 420.6, 12.74, 7.49, 85.2),
    ("HEA 320", 97.6, 310, 300, 9.0, 15.5, 12440, 22930, 6985, 1628, 709.7,
     1479, 465.7, 13.58, 7.49, 108.0),
    ("HEA 340", 105.0, 330, 300, 9.5, 16.5, 13350, 27690, 7436, 1850, 755.9,
     1678, 495.7, 14.40, 7.46, 127.0),
    ("HEA 360", 112.0, 350, 300, 10.0, 17.5, 14280, 33090, 7887, 2088, 802.3,
     1891, 525.8, 15.22, 7.43, 148.8),
    ("HEA 400", 125.0, 390, 300, 11.0, 19.0, 15900, 45070, 8564, 2562, 872.9,
     2311, 570.9, 16.84, 7.34, 189.0),
    ("HEA 450", 140.0, 440, 300, 11.5, 21.0, 17800, 63720, 9465, 3216, 965.5,
     2896, 631.0, 18.92, 7.29, 243.8),
    ("HEA 500", 155.0, 490, 300, 12.0, 23.0, 19750, 86970, 10370, 3949, 1059,
     3550, 691.1, 20.99, 7.24, 309.3),
    ("HEA 550", 166.0, 540, 300, 12.5, 24.0, 21180, 111900, 10820, 4622, 1107,
     4146, 721.4, 22.99, 7.15, 351.5),
    ("HEA 600", 178.0, 590, 300, 13.0, 25.0, 22650, 141200, 11270, 5350, 1156,
     4787, 751.4, 24.97, 7.05, 397.8),
]
for row in _HEA_RAW:
    name, mass, h, b, tw, tf, A_mm2, Iy_cm4, Iz_cm4, Wply_cm3, Wplz_cm3, \
        Wely_cm3, Welz_cm3, ry_cm, rz_cm, J_cm4 = row
    _add_hea(
        name, mass, h, b, tw, tf, A_mm2,
        Iy_cm4 * 1e4, Iz_cm4 * 1e4,
        Wply_cm3 * 1e3, Wplz_cm3 * 1e3,
        Wely_cm3 * 1e3, Welz_cm3 * 1e3,
        ry_cm * 10.0, rz_cm * 10.0, J_cm4 * 1e4,
    )


_HEB_RAW = [
    ("HEB 100", 20.4, 100, 100, 6.0, 10.0, 2600, 449.5, 167.3, 104.2, 51.42,
     89.91, 33.45, 4.16, 2.53, 9.25),
    ("HEB 120", 26.7, 120, 120, 6.5, 11.0, 3400, 864.4, 317.5, 165.2, 80.97,
     144.1, 52.92, 5.04, 3.06, 13.84),
    ("HEB 140", 33.7, 140, 140, 7.0, 12.0, 4300, 1509, 549.7, 245.4, 119.8,
     215.6, 78.52, 5.93, 3.58, 20.06),
    ("HEB 160", 42.6, 160, 160, 8.0, 13.0, 5430, 2492, 889.2, 354.0, 169.9,
     311.5, 111.2, 6.78, 4.05, 31.24),
    ("HEB 180", 51.2, 180, 180, 8.5, 14.0, 6530, 3831, 1363, 481.4, 231.0,
     425.7, 151.4, 7.66, 4.57, 42.16),
    ("HEB 200", 61.3, 200, 200, 9.0, 15.0, 7810, 5696, 2003, 642.5, 305.8,
     569.6, 200.3, 8.54, 5.07, 59.28),
    ("HEB 220", 71.5, 220, 220, 9.5, 16.0, 9100, 8091, 2843, 827.0, 393.9,
     735.5, 258.5, 9.43, 5.59, 76.57),
    ("HEB 240", 83.2, 240, 240, 10.0, 17.0, 10600, 11260, 3923, 1053, 498.4,
     938.3, 326.9, 10.31, 6.08, 102.7),
    ("HEB 260", 93.0, 260, 260, 10.0, 17.5, 11840, 14920, 5135, 1283, 602.2,
     1148, 395.0, 11.22, 6.58, 123.8),
    ("HEB 280", 103.0, 280, 280, 10.5, 18.0, 13140, 19270, 6595, 1534, 717.6,
     1376, 471.0, 12.11, 7.09, 143.7),
    ("HEB 300", 117.0, 300, 300, 11.0, 19.0, 14900, 25170, 8563, 1869, 870.1,
     1678, 570.9, 12.99, 7.58, 185.0),
    ("HEB 320", 127.0, 320, 300, 11.5, 20.5, 16130, 30820, 9239, 2149, 939.1,
     1926, 615.9, 13.82, 7.57, 225.1),
    ("HEB 340", 134.0, 340, 300, 12.0, 21.5, 17090, 36660, 9690, 2408, 985.7,
     2156, 646.0, 14.65, 7.53, 257.2),
    ("HEB 360", 142.0, 360, 300, 12.5, 22.5, 18060, 43190, 10140, 2683, 1032,
     2400, 676.0, 15.46, 7.49, 292.5),
    ("HEB 400", 155.0, 400, 300, 13.5, 24.0, 19780, 57680, 10820, 3232, 1104,
     2884, 721.3, 17.08, 7.40, 355.7),
    ("HEB 450", 171.0, 450, 300, 14.0, 26.0, 21800, 79890, 11720, 3982, 1198,
     3551, 781.4, 19.14, 7.33, 440.5),
    ("HEB 500", 187.0, 500, 300, 14.5, 28.0, 23860, 107200, 12620, 4815, 1292,
     4287, 841.6, 21.19, 7.27, 538.4),
    ("HEB 550", 199.0, 550, 300, 15.0, 29.0, 25410, 136700, 13080, 5591, 1341,
     4971, 871.9, 23.20, 7.17, 600.3),
    ("HEB 600", 212.0, 600, 300, 15.5, 30.0, 27000, 171000, 13530, 6425, 1391,
     5701, 902.1, 25.17, 7.08, 667.2),
]
for row in _HEB_RAW:
    name, mass, h, b, tw, tf, A_mm2, Iy_cm4, Iz_cm4, Wply_cm3, Wplz_cm3, \
        Wely_cm3, Welz_cm3, ry_cm, rz_cm, J_cm4 = row
    _add_heb(
        name, mass, h, b, tw, tf, A_mm2,
        Iy_cm4 * 1e4, Iz_cm4 * 1e4,
        Wply_cm3 * 1e3, Wplz_cm3 * 1e3,
        Wely_cm3 * 1e3, Welz_cm3 * 1e3,
        ry_cm * 10.0, rz_cm * 10.0, J_cm4 * 1e4,
    )


# ============================================================ lookup

_ALL_EC: dict[str, SectionProperties] = {}
_ALL_EC.update(EC_IPE)
_ALL_EC.update(EC_HEA)
_ALL_EC.update(EC_HEB)


def eurocode_section(name: str) -> SectionProperties:
    """Look up a Eurocode section by name (e.g., ``"IPE 300"``,
    ``"HEA 200"``). Case- and whitespace-tolerant."""
    key = name.strip().upper().replace("  ", " ")
    if key not in _ALL_EC:
        # Try without space
        for k in _ALL_EC:
            if k.replace(" ", "") == key.replace(" ", ""):
                return _ALL_EC[k]
        raise KeyError(f"unknown Eurocode section {name!r}; "
                       f"see list_eurocode_sections() for options")
    return _ALL_EC[key]


def list_eurocode_sections(family: str | None = None) -> list[str]:
    """List available section names, optionally filtered by family
    ``"IPE"`` / ``"HEA"`` / ``"HEB"``."""
    if family is None:
        return list(_ALL_EC.keys())
    f = family.upper()
    if f == "IPE":
        return list(EC_IPE.keys())
    if f == "HEA":
        return list(EC_HEA.keys())
    if f == "HEB":
        return list(EC_HEB.keys())
    raise ValueError(f"unknown family {family!r}")


def auto_select_ec_section(
    *,
    W_pl_required: float,
    family: str = "IPE",
    minimise: str = "mass",
) -> SectionProperties:
    """Pick the lightest section of ``family`` with ``W_pl_y >=
    W_pl_required`` (mm^3).

    Parameters
    ----------
    W_pl_required : float
        Plastic modulus demand (mm^3) = M_demand / f_y for EC3 design.
    family : {"IPE", "HEA", "HEB"}
    minimise : {"mass", "depth"}
        Selection criterion among feasible sections.
    """
    pool = list_eurocode_sections(family)
    feasible = [
        eurocode_section(n) for n in pool
        if eurocode_section(n).W_pl_y >= W_pl_required
    ]
    if not feasible:
        raise ValueError(
            f"no {family} section has W_pl_y >= {W_pl_required:.3e} mm^3; "
            f"largest available has "
            f"{max(eurocode_section(n).W_pl_y for n in pool):.3e}"
        )
    if minimise == "mass":
        return min(feasible, key=lambda s: s.mass)
    if minimise == "depth":
        return min(feasible, key=lambda s: s.h)
    raise ValueError(f"unknown minimise={minimise!r}; use 'mass' or 'depth'")
