"""EN 338 strength classes for solid timber and glulam (Phase D.1.1).

The EC5 design framework uses **strength classes** rather than
species-by-species values. Engineers select a class (C16, C24, C30
for solid timber; GL20, GL24, GL28, GL32 for glulam), and the
characteristic values are read from EN 338 / EN 14080.

This module ships:
* :data:`EC5_C_CLASS` -- C-classes for visually graded solid softwood
  (C14, C16, C18, C22, C24, C27, C30, C35, C40, C45, C50). Values
  from EN 338:2016 Table 1.
* :data:`EC5_GL_CLASS` -- GL-classes for structural glulam (GL20h,
  GL22h, GL24h, GL26h, GL28h, GL30h, GL32h homogeneous + the matching
  combined ``c`` classes). Values from EN 14080:2013 Table 5.
* :data:`EC5_T_CLASS` -- T-classes for tension-graded structural timber
  (T8, T9.5, T11.5, T14, T16, T18, T21, T24, T26, T28, T30). EN 338
  Table 2.

All values are characteristic (5th-percentile) in SI units.
"""
from __future__ import annotations

from femsolver.materials.timber.material import TimberMaterial


def _ec5_solid(
    name: str, *,
    fm_k_MPa: float, ft_0_k_MPa: float, ft_90_k_MPa: float,
    fc_0_k_MPa: float, fc_90_k_MPa: float, fv_k_MPa: float,
    E_0_mean_GPa: float, E_0_05_GPa: float,
    E_90_mean_GPa: float, G_mean_GPa: float,
    rho_k_kgm3: float, rho_mean_kgm3: float,
    is_glulam: bool = False,
) -> TimberMaterial:
    """Build an EC5 strength-class TimberMaterial from EN 338 values."""
    return TimberMaterial(
        name=name, species="EC5 Glulam" if is_glulam else "EC5 Solid",
        grade=name, code="EC5",
        E_0_mean=E_0_mean_GPa * 1e9,
        E_0_05=E_0_05_GPa * 1e9,
        E_90_mean=E_90_mean_GPa * 1e9,
        G_mean=G_mean_GPa * 1e9,
        f_b_k=fm_k_MPa * 1e6,
        f_t_0_k=ft_0_k_MPa * 1e6,
        f_t_90_k=ft_90_k_MPa * 1e6,
        f_c_0_k=fc_0_k_MPa * 1e6,
        f_c_90_k=fc_90_k_MPa * 1e6,
        f_v_k=fv_k_MPa * 1e6,
        density_k=rho_k_kgm3, density_mean=rho_mean_kgm3,
    )


# ============================================================ C-classes (EN 338 Table 1)

EC5_C_CLASS: dict[str, TimberMaterial] = {}

# Format: (name, fm_k, ft_0, ft_90, fc_0, fc_90, fv, E_0_mean, E_0_05,
#         E_90_mean, G_mean, rho_k, rho_mean)
_C_CLASS_DATA = [
    ("C14",  14,  7.2, 0.4, 16, 2.0, 3.0,  7.0, 4.7, 0.23, 0.44, 290, 350),
    ("C16",  16,  8.5, 0.4, 17, 2.2, 3.2,  8.0, 5.4, 0.27, 0.50, 310, 370),
    ("C18",  18, 10.0, 0.4, 18, 2.2, 3.4,  9.0, 6.0, 0.30, 0.56, 320, 380),
    ("C20",  20, 11.5, 0.4, 19, 2.3, 3.6,  9.5, 6.4, 0.32, 0.59, 330, 390),
    ("C22",  22, 13.0, 0.4, 20, 2.4, 3.8, 10.0, 6.7, 0.33, 0.63, 340, 410),
    ("C24",  24, 14.0, 0.4, 21, 2.5, 4.0, 11.0, 7.4, 0.37, 0.69, 350, 420),
    ("C27",  27, 16.0, 0.4, 22, 2.6, 4.0, 11.5, 7.7, 0.38, 0.72, 370, 450),
    ("C30",  30, 18.0, 0.4, 23, 2.7, 4.0, 12.0, 8.0, 0.40, 0.75, 380, 460),
    ("C35",  35, 21.0, 0.4, 25, 2.8, 4.0, 13.0, 8.7, 0.43, 0.81, 400, 480),
    ("C40",  40, 24.0, 0.4, 26, 2.9, 4.0, 14.0, 9.4, 0.47, 0.88, 420, 500),
    ("C45",  45, 27.0, 0.4, 27, 3.1, 4.0, 15.0, 10.0, 0.50, 0.94, 440, 520),
    ("C50",  50, 30.0, 0.4, 29, 3.2, 4.0, 16.0, 10.7, 0.53, 1.00, 460, 550),
]
for row in _C_CLASS_DATA:
    name, fm, ft0, ft90, fc0, fc90, fv, E0m, E005, E90m, Gm, rk, rm = row
    EC5_C_CLASS[name] = _ec5_solid(
        name,
        fm_k_MPa=fm, ft_0_k_MPa=ft0, ft_90_k_MPa=ft90,
        fc_0_k_MPa=fc0, fc_90_k_MPa=fc90, fv_k_MPa=fv,
        E_0_mean_GPa=E0m, E_0_05_GPa=E005,
        E_90_mean_GPa=E90m, G_mean_GPa=Gm,
        rho_k_kgm3=rk, rho_mean_kgm3=rm,
    )


# ============================================================ GL-classes (EN 14080 Table 5)
# Homogeneous (h) glulam: all laminations from same grade.

EC5_GL_CLASS: dict[str, TimberMaterial] = {}

_GL_CLASS_DATA = [
    # name, fm, ft0, ft90, fc0, fc90, fv, E0m, E005, E90m, Gm, rk, rm
    ("GL20h", 20, 16, 0.5, 20, 2.5, 3.5, 8.4, 7.0, 0.30, 0.65, 340, 370),
    ("GL22h", 22, 17.6, 0.5, 22, 2.5, 3.5, 10.5, 8.8, 0.30, 0.65, 370, 410),
    ("GL24h", 24, 19.2, 0.5, 24, 2.5, 3.5, 11.5, 9.6, 0.30, 0.65, 385, 420),
    ("GL26h", 26, 20.8, 0.5, 26, 2.5, 3.5, 12.1, 10.1, 0.30, 0.65, 405, 445),
    ("GL28h", 28, 22.3, 0.5, 28, 2.5, 3.5, 12.6, 10.5, 0.30, 0.65, 425, 460),
    ("GL30h", 30, 24.0, 0.5, 30, 2.5, 3.5, 13.6, 11.3, 0.30, 0.65, 430, 480),
    ("GL32h", 32, 25.6, 0.5, 32, 2.5, 3.5, 14.2, 11.8, 0.30, 0.65, 440, 490),
]
for row in _GL_CLASS_DATA:
    name, fm, ft0, ft90, fc0, fc90, fv, E0m, E005, E90m, Gm, rk, rm = row
    EC5_GL_CLASS[name] = _ec5_solid(
        name,
        fm_k_MPa=fm, ft_0_k_MPa=ft0, ft_90_k_MPa=ft90,
        fc_0_k_MPa=fc0, fc_90_k_MPa=fc90, fv_k_MPa=fv,
        E_0_mean_GPa=E0m, E_0_05_GPa=E005,
        E_90_mean_GPa=E90m, G_mean_GPa=Gm,
        rho_k_kgm3=rk, rho_mean_kgm3=rm,
        is_glulam=True,
    )


# ============================================================ T-classes (EN 338 Table 2)
# Tension-graded structural timber.

EC5_T_CLASS: dict[str, TimberMaterial] = {}

_T_CLASS_DATA = [
    ("T8",   12,  8, 0.4, 16, 2.2, 3.2,  8.0, 5.4, 0.27, 0.50, 310, 370),
    ("T11.5", 17, 11.5, 0.4, 19, 2.3, 3.6, 9.5, 6.4, 0.32, 0.59, 330, 390),
    ("T14",  21, 14, 0.4, 21, 2.5, 4.0, 11.0, 7.4, 0.37, 0.69, 350, 420),
    ("T18",  27, 18, 0.4, 23, 2.7, 4.0, 12.0, 8.0, 0.40, 0.75, 380, 460),
    ("T24",  36, 24, 0.4, 26, 2.9, 4.0, 14.0, 9.4, 0.47, 0.88, 420, 500),
    ("T30",  45, 30, 0.4, 29, 3.2, 4.0, 16.0, 10.7, 0.53, 1.00, 460, 550),
]
for row in _T_CLASS_DATA:
    name, fm, ft0, ft90, fc0, fc90, fv, E0m, E005, E90m, Gm, rk, rm = row
    EC5_T_CLASS[name] = _ec5_solid(
        name,
        fm_k_MPa=fm, ft_0_k_MPa=ft0, ft_90_k_MPa=ft90,
        fc_0_k_MPa=fc0, fc_90_k_MPa=fc90, fv_k_MPa=fv,
        E_0_mean_GPa=E0m, E_0_05_GPa=E005,
        E_90_mean_GPa=E90m, G_mean_GPa=Gm,
        rho_k_kgm3=rk, rho_mean_kgm3=rm,
    )


# ============================================================ API

def get_ec5_class(name: str) -> TimberMaterial:
    """Look up an EC5 strength class by designation
    (e.g. ``"C24"``, ``"GL28h"``, ``"T18"``)."""
    for table in (EC5_C_CLASS, EC5_GL_CLASS, EC5_T_CLASS):
        if name in table:
            return table[name]
    available = sorted(
        list(EC5_C_CLASS) + list(EC5_GL_CLASS) + list(EC5_T_CLASS)
    )
    raise KeyError(
        f"EC5 strength class {name!r} not in database. "
        f"Available: {available}"
    )


def list_ec5_classes() -> dict[str, list[str]]:
    """All EC5 classes grouped by family."""
    return {
        "C": sorted(EC5_C_CLASS.keys()),
        "GL": sorted(EC5_GL_CLASS.keys()),
        "T": sorted(EC5_T_CLASS.keys()),
    }
