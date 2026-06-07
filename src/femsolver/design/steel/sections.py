"""AISC v15.0 Steel Shapes Database -- W-shape catalog.

A curated subset (~30 sections spanning W4 through W36) of the AISC
``Shapes Database v15.0`` is embedded as a Python dictionary. Each
W-shape carries the full set of geometric and section properties
needed for AISC 360-22 checks (Phases 30.2-30.6):

.. note::

    **Theme II.7 migration:** the unified Section Designer accesses
    this database through
    :func:`femsolver.sections.aisc_section` and
    :meth:`femsolver.sections.SectionLibrary.aisc`. New user code
    should prefer the unified path. :meth:`SteelSection.to_unified`
    converts a legacy :class:`SteelSection` to a unified
    :class:`femsolver.sections.Section`.

* **Geometry**: ``A``, ``d``, ``bf``, ``tf``, ``tw``, ``k_des``
* **Inertia**: ``Ix``, ``Iy``
* **Section moduli**: ``Sx``, ``Sy`` (elastic), ``Zx``, ``Zy``
  (plastic)
* **Radii of gyration**: ``rx``, ``ry``
* **Torsion**: ``J`` (torsional constant), ``Cw`` (warping constant)
* ``weight_per_length``: in N/m, derived from ``A · ρ_steel · g``
  (ρ_steel = 7850 kg/m³, g = 9.81 m/s²)

All values are stored **internally in SI**: lengths in m, areas in
m², inertia in m⁴, etc. The source AISC table is in imperial; the
conversions are applied once at table-construction time.

The selected sections span the practical range from light beams
(W4x13) to heavy columns/girders (W36x150). The full AISC v15.0
catalog (>300 W-shapes) can be embedded as a future extension; for
the present design phases this subset gives sufficient coverage for
typical mid-rise buildings.
"""
from __future__ import annotations

from dataclasses import dataclass


# Conversion factors (imperial -> SI)
_IN = 0.0254
_IN2 = _IN * _IN
_IN3 = _IN2 * _IN
_IN4 = _IN3 * _IN
_IN6 = _IN4 * _IN2
_RHO_STEEL = 7850.0          # kg/m³
_G = 9.81                    # m/s²


@dataclass
class SteelSection:
    """W-shape (wide-flange) steel section.

    All quantities in SI: lengths in m, area in m², inertia in m⁴,
    section modulus in m³, torsion constant in m⁴, warping constant
    in m⁶, weight per length in N/m.
    """

    designation: str        # e.g., "W14x90"
    # Geometry
    A: float                # cross-section area
    d: float                # overall depth
    bf: float               # flange width
    tf: float               # flange thickness
    tw: float               # web thickness
    k_des: float            # design k-distance (distance from outer
                              # flange face to web-fillet toe)
    # Inertia and section moduli
    Ix: float; Iy: float
    Sx: float; Sy: float
    Zx: float; Zy: float
    # Radii of gyration
    rx: float; ry: float
    # Torsion
    J: float                # St. Venant torsional constant
    Cw: float               # warping constant
    # Derived (set in __post_init__)
    weight_per_length: float = 0.0      # N/m

    def __post_init__(self) -> None:
        # weight per unit length = A · ρ · g
        if self.weight_per_length <= 0.0:
            self.weight_per_length = self.A * _RHO_STEEL * _G

    def __repr__(self) -> str:
        return (
            f"SteelSection({self.designation!r}, "
            f"A={self.A * 1e4:.1f} cm2, "
            f"d={self.d * 1000:.0f} mm, "
            f"Ix={self.Ix * 1e8:.1f}e4 mm4)"
        )

    # ---------------------------------------------------- migration (II.7)
    def to_unified(self, *, material=None):
        """Convert this legacy AISC :class:`SteelSection` to a unified
        :class:`femsolver.sections.Section`.

        Inverse of :meth:`femsolver.sections.Section.as_aisc_section`
        (added in II.6). Together they let user code mix the two
        worlds during the migration window.

        Parameters
        ----------
        material : optional
            Steel material reference attached as a
            :class:`MaterialZone` so the result can drive
            ``.elastic_section_3d()`` etc. out of the box.
        """
        from femsolver.sections import aisc_section
        return aisc_section(self.designation, material=material)


# ============================================================ database


def _from_imperial(designation: str, *,
                    A_in2: float, d_in: float, bf_in: float,
                    tf_in: float, tw_in: float, k_des_in: float,
                    Ix_in4: float, Iy_in4: float,
                    Sx_in3: float, Sy_in3: float,
                    Zx_in3: float, Zy_in3: float,
                    rx_in: float, ry_in: float,
                    J_in4: float, Cw_in6: float) -> SteelSection:
    """Build a SteelSection from imperial AISC table values."""
    return SteelSection(
        designation=designation,
        A=A_in2 * _IN2,
        d=d_in * _IN,
        bf=bf_in * _IN,
        tf=tf_in * _IN,
        tw=tw_in * _IN,
        k_des=k_des_in * _IN,
        Ix=Ix_in4 * _IN4,
        Iy=Iy_in4 * _IN4,
        Sx=Sx_in3 * _IN3,
        Sy=Sy_in3 * _IN3,
        Zx=Zx_in3 * _IN3,
        Zy=Zy_in3 * _IN3,
        rx=rx_in * _IN,
        ry=ry_in * _IN,
        J=J_in4 * _IN4,
        Cw=Cw_in6 * _IN6,
    )


# Curated AISC v15.0 W-shapes (selected representative sizes from
# each W-series). Values from AISC Manual 15th edition Tables 1-1.
# Format: designation, A, d, bf, tf, tw, kdes, Ix, Iy, Sx, Sy, Zx, Zy,
#         rx, ry, J, Cw  (all imperial; conversion to SI in _from_imperial)
_W_SHAPES_IMPERIAL = [
    # designation,    A,    d,    bf,   tf,    tw,    kdes,  Ix,   Iy,    Sx,   Sy,   Zx,   Zy,   rx,   ry,   J,     Cw
    ("W4x13",        3.83, 4.16, 4.060, 0.345, 0.280, 0.55,  11.3, 3.86,  5.46, 1.90, 6.28, 2.92, 1.72, 1.00, 0.151, 15.7),
    ("W6x9",         2.68, 5.90, 3.940, 0.215, 0.170, 0.42,  16.4, 2.20,  5.56, 1.11, 6.23, 1.72, 2.47, 0.905, 0.0405, 17.7),
    ("W6x15",        4.43, 5.99, 5.990, 0.260, 0.230, 0.46,  29.1, 9.32,  9.72, 3.11, 10.8, 4.75, 2.56, 1.46, 0.101, 76.5),
    ("W6x25",        7.34, 6.38, 6.080, 0.455, 0.320, 0.66,  53.4, 17.1,  16.7, 5.61, 18.9, 8.56, 2.70, 1.52, 0.461, 150.0),
    ("W8x10",        2.96, 7.89, 3.940, 0.205, 0.170, 0.42,  30.8, 2.09,  7.81, 1.06, 8.87, 1.66, 3.22, 0.841, 0.0426, 30.9),
    ("W8x18",        5.26, 8.14, 5.250, 0.330, 0.230, 0.55,  61.9, 7.97,  15.2, 3.04, 17.0, 4.66, 3.43, 1.23, 0.172, 122.0),
    ("W8x35",        10.3, 8.12, 8.020, 0.495, 0.310, 0.71,  127., 42.6,  31.2, 10.6, 34.7, 16.1, 3.51, 2.03, 0.769, 626.0),
    ("W8x67",        19.7, 9.00, 8.280, 0.935, 0.570, 1.17,  272., 88.6,  60.4, 21.4, 70.1, 32.7, 3.72, 2.12, 5.05, 1440.0),
    ("W10x12",       3.54, 9.87, 3.960, 0.210, 0.190, 0.49,  53.8, 2.18,  10.9, 1.10, 12.6, 1.74, 3.90, 0.785, 0.0547, 50.9),
    ("W10x33",       9.71, 9.73, 7.960, 0.435, 0.290, 0.70,  171., 36.6,  35.0, 9.20, 38.8, 14.0, 4.19, 1.94, 0.583, 791.0),
    ("W10x49",       14.4, 9.98, 10.00, 0.560, 0.340, 0.83,  272., 93.4,  54.6, 18.7, 60.4, 28.3, 4.35, 2.54, 1.39, 2070.0),
    ("W10x88",       25.9, 10.84, 10.27, 0.990, 0.605, 1.18, 534., 179.,  98.5, 34.8, 113., 53.0, 4.54, 2.63, 5.61, 4330.0),
    ("W12x14",       4.16, 11.91, 3.970, 0.225, 0.200, 0.53, 88.6, 2.36,  14.9, 1.19, 17.4, 1.90, 4.62, 0.753, 0.0704, 80.4),
    ("W12x22",       6.48, 12.31, 4.030, 0.425, 0.260, 0.71, 156., 4.66,  25.4, 2.31, 29.3, 3.66, 4.91, 0.848, 0.293, 164.0),
    ("W12x35",       10.3, 12.50, 6.560, 0.520, 0.300, 0.82, 285., 24.5,  45.6, 7.47, 51.2, 11.5, 5.25, 1.54, 0.741, 879.0),
    ("W12x65",       19.1, 12.12, 12.00, 0.605, 0.390, 0.90, 533., 174.,  87.9, 29.1, 96.8, 44.1, 5.28, 3.02, 2.18, 6180.0),
    ("W12x96",       28.2, 12.71, 12.16, 0.900, 0.550, 1.25, 833., 270., 131., 44.4, 147., 67.5, 5.44, 3.09, 6.85, 9410.0),
    ("W12x152",      44.7, 13.71, 12.48, 1.400, 0.870, 1.75, 1430., 454., 209., 72.8, 243., 111., 5.66, 3.19, 25.8, 17000.0),
    ("W14x22",       6.49, 13.74, 5.000, 0.335, 0.230, 0.74, 199., 7.00,  29.0, 2.80, 33.2, 4.39, 5.54, 1.04, 0.208, 314.0),
    ("W14x30",       8.85, 13.84, 6.730, 0.385, 0.270, 0.79, 291., 19.6,  42.0, 5.82, 47.3, 9.00, 5.73, 1.49, 0.380, 887.0),
    ("W14x53",       15.6, 13.92, 8.060, 0.660, 0.370, 1.06, 541., 57.7,  77.8, 14.3, 87.1, 22.0, 5.89, 1.92, 1.94, 2540.0),
    ("W14x82",       24.0, 14.31, 10.13, 0.855, 0.510, 1.45, 881., 148., 123., 29.3, 139., 44.8, 6.05, 2.48, 5.07, 6710.0),
    ("W14x90",       26.5, 14.02, 14.52, 0.710, 0.440, 1.31, 999., 362., 143., 49.9, 157., 75.6, 6.14, 3.70, 4.06, 16000.0),
    ("W14x132",      38.8, 14.66, 14.73, 1.030, 0.645, 1.63, 1530., 548., 209., 74.5, 234., 113., 6.28, 3.76, 12.3, 25500.0),
    ("W14x176",      51.8, 15.22, 15.65, 1.310, 0.830, 1.91, 2140., 838., 281., 107., 320., 163., 6.43, 4.02, 26.5, 40500.0),
    ("W14x257",      75.6, 16.38, 15.99, 1.890, 1.175, 2.50, 3400., 1290., 415., 161., 487., 246., 6.71, 4.13, 79.1, 67800.0),
    ("W16x26",       7.68, 15.69, 5.500, 0.345, 0.250, 0.75, 301., 9.59,  38.4, 3.49, 44.2, 5.48, 6.26, 1.12, 0.262, 565.0),
    ("W16x36",       10.6, 15.86, 6.985, 0.430, 0.295, 0.83, 448., 24.5,  56.5, 7.00, 64.0, 10.8, 6.51, 1.52, 0.545, 1460.0),
    ("W16x57",       16.8, 16.43, 7.120, 0.715, 0.430, 1.12, 758., 43.1,  92.2, 12.1, 105., 18.9, 6.72, 1.60, 2.22, 2660.0),
    ("W18x35",       10.3, 17.70, 6.000, 0.425, 0.300, 0.83, 510., 15.3,  57.6, 5.12, 66.5, 8.06, 7.04, 1.22, 0.506, 1140.0),
    ("W18x46",       13.5, 18.06, 6.060, 0.605, 0.360, 1.01, 712., 22.5,  78.8, 7.43, 90.7, 11.7, 7.25, 1.29, 1.22, 1720.0),
    ("W18x60",       17.6, 18.24, 7.555, 0.695, 0.415, 1.18, 984., 50.1, 108., 13.3, 123., 20.6, 7.47, 1.68, 2.17, 3850.0),
    ("W21x44",       13.0, 20.66, 6.500, 0.450, 0.350, 0.95, 843., 20.7,  81.6, 6.37, 95.4, 10.2, 8.06, 1.26, 0.770, 2110.0),
    ("W21x55",       16.2, 20.80, 8.215, 0.522, 0.375, 1.02, 1140., 48.4, 110., 11.8, 126., 18.4, 8.40, 1.73, 1.24, 4980.0),
    ("W21x73",       21.5, 21.24, 8.295, 0.740, 0.455, 1.24, 1600., 70.6, 151., 17.0, 172., 26.6, 8.64, 1.81, 3.02, 7410.0),
    ("W24x55",       16.2, 23.57, 7.005, 0.505, 0.395, 1.11, 1350., 29.1, 114., 8.30, 134., 13.3, 9.11, 1.34, 1.18, 3870.0),
    ("W24x68",       20.1, 23.73, 8.965, 0.585, 0.415, 1.09, 1830., 70.4, 154., 15.7, 177., 24.5, 9.55, 1.87, 1.87, 9430.0),
    ("W24x84",       24.7, 24.10, 9.020, 0.770, 0.470, 1.27, 2370., 94.4, 196., 20.9, 224., 32.6, 9.79, 1.95, 3.70, 12800.0),
    ("W27x84",       24.8, 26.71, 9.960, 0.640, 0.460, 1.24, 2850., 106., 213., 21.2, 244., 33.2, 10.7, 2.07, 2.81, 17900.0),
    ("W27x102",      30.0, 27.09, 10.02, 0.830, 0.515, 1.43, 3620., 139., 267., 27.8, 305., 43.4, 10.99, 2.15, 5.28, 23900.0),
    ("W30x90",       26.4, 29.53, 10.40, 0.610, 0.470, 1.21, 3610., 115., 245., 22.1, 283., 34.7, 11.7, 2.09, 2.84, 24500.0),
    ("W30x108",      31.7, 29.83, 10.48, 0.760, 0.545, 1.37, 4470., 146., 299., 27.9, 346., 43.9, 11.9, 2.15, 4.99, 31400.0),
    ("W33x118",      34.7, 32.86, 11.48, 0.740, 0.550, 1.42, 5900., 187., 359., 32.6, 415., 51.3, 13.0, 2.32, 5.31, 48300.0),
    ("W36x135",      39.7, 35.55, 11.95, 0.790, 0.600, 1.54, 7800., 225., 439., 37.7, 509., 59.7, 14.0, 2.38, 6.99, 70600.0),
    ("W36x150",      44.2, 35.85, 11.97, 0.940, 0.625, 1.69, 9040., 270., 504., 45.1, 581., 70.9, 14.3, 2.47, 10.1, 85200.0),
]


# Build the database once at import
_DATABASE: dict[str, SteelSection] = {}
for _row in _W_SHAPES_IMPERIAL:
    _designation = _row[0]
    _DATABASE[_designation] = _from_imperial(
        _designation,
        A_in2=_row[1], d_in=_row[2], bf_in=_row[3],
        tf_in=_row[4], tw_in=_row[5], k_des_in=_row[6],
        Ix_in4=_row[7], Iy_in4=_row[8],
        Sx_in3=_row[9], Sy_in3=_row[10],
        Zx_in3=_row[11], Zy_in3=_row[12],
        rx_in=_row[13], ry_in=_row[14],
        J_in4=_row[15], Cw_in6=_row[16],
    )


# ============================================================ API

def get_section(designation: str) -> SteelSection:
    """Look up a W-shape by designation, e.g. ``"W14x90"``.

    Raises ``KeyError`` if not found, with a helpful message listing
    near-matches.
    """
    if designation in _DATABASE:
        return _DATABASE[designation]
    # Suggest similar designations
    prefix = designation.split("x")[0] if "x" in designation else designation
    matches = [d for d in _DATABASE if d.startswith(prefix)]
    if matches:
        raise KeyError(
            f"{designation!r} not in database. Available {prefix} sections: "
            f"{matches}"
        )
    raise KeyError(
        f"{designation!r} not in database. Available designations: "
        f"{sorted(_DATABASE.keys())[:10]}... (showing first 10)"
    )


def all_designations() -> list[str]:
    """Return all available W-shape designations in the embedded
    database."""
    return sorted(_DATABASE.keys(), key=_designation_sort_key)


def w_series(series_prefix: str) -> list[SteelSection]:
    """All sections in a given W-series, sorted lightest to heaviest.

    Example: ``w_series("W14")`` returns every W14xNN in the database.
    """
    sections = [
        s for d, s in _DATABASE.items()
        if d.startswith(series_prefix + "x")
    ]
    return sorted(sections, key=lambda s: s.A)


def _designation_sort_key(designation: str) -> tuple:
    """Sort by depth then by weight: W4x... < W6x... < ... < W36x... ."""
    try:
        prefix, weight = designation.split("x")
        depth = int(prefix[1:])
        wt = int(weight)
        return (depth, wt)
    except (ValueError, IndexError):
        return (999, 999)


# ============================================================ steel material

@dataclass
class SteelMaterial:
    """Steel material per ASTM A992 (typical W-shape steel) or A36
    (typical bar steel).

    Attributes
    ----------
    Fy : float
        Specified yield stress (Pa). A992 = 50 ksi = 345 MPa.
    Fu : float
        Specified ultimate tensile stress (Pa). A992 = 65 ksi = 448 MPa.
    E : float, default 200 GPa
        Young's modulus per AISC 360-22 §B3.1 (29,000 ksi = 200 GPa).
    G : float, default 77.2 GPa
        Shear modulus = E / (2(1+ν)) with ν = 0.30. AISC 360-22 §B3.1
        gives 11,200 ksi = 77.2 GPa.
    """

    Fy: float
    Fu: float
    E: float = 200.0e9
    G: float = 77.2e9

    def __post_init__(self) -> None:
        if self.Fy <= 0.0:
            raise ValueError(f"Fy must be positive, got {self.Fy}")
        if self.Fu <= 0.0:
            raise ValueError(f"Fu must be positive, got {self.Fu}")
        if self.Fu < self.Fy:
            raise ValueError(
                f"Fu ({self.Fu/1e6:.0f} MPa) must be >= Fy "
                f"({self.Fy/1e6:.0f} MPa)"
            )


def astm_a992() -> SteelMaterial:
    """ASTM A992 -- the standard W-shape steel grade.

    Fy = 50 ksi (345 MPa), Fu = 65 ksi (448 MPa).
    """
    return SteelMaterial(Fy=50.0e3 * 6894.757, Fu=65.0e3 * 6894.757)


def astm_a36() -> SteelMaterial:
    """ASTM A36 -- general structural steel (older spec, less common
    for new W-shapes).

    Fy = 36 ksi (248 MPa), Fu = 58 ksi (400 MPa).
    """
    return SteelMaterial(Fy=36.0e3 * 6894.757, Fu=58.0e3 * 6894.757)
