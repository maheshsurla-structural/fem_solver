"""NDS-2024 reference design values for structural lumber + glulam.

Source: NDS-2024 Supplement Table 4A (sawn lumber), Table 4D (glulam).
All values stored in **SI units** internally; the source table is in
imperial (psi for stresses, pcf for density, msi for modulus).

The NDS reference values F_b, F_t, F_c, F_v are tabulated for the
"reference condition" (dry service, normal duration, no special
adjustments). Application requires the C-factors per NDS Ch. 4 (load
duration C_D, wet service C_M, temperature C_t, size C_F, etc.) which
are applied in the design module, not at the material level.
"""
from __future__ import annotations

from femsolver.materials.timber.material import TimberMaterial


# Imperial -> SI conversions
_PSI = 6_894.757         # Pa per psi
_KSI = 1000 * _PSI       # Pa per ksi
_MSI = 1e6 * _PSI        # Pa per msi (million psi)
_PCF_TO_KGM3 = 16.0185   # kg/m^3 per pcf


def _from_imperial(
    name: str, species: str, grade: str,
    *,
    Fb_psi: float, Ft_psi: float, Fc_psi: float,
    Fc_perp_psi: float, Fv_psi: float,
    E_msi: float, E_min_msi: float,
    density_pcf: float,
) -> TimberMaterial:
    """Build a TimberMaterial from NDS imperial reference values.

    E_min is NDS's "minimum modulus for stability calculations"
    (NDS §3.7) -- it's roughly the 5th-percentile E used for column
    buckling. We map it to ``E_0_05``.
    """
    # NDS doesn't publish E_90 or G explicitly; standard ratios:
    # E_90 ~ E/30 (typical for softwoods)
    # G ~ E/16
    E_0 = E_msi * _MSI
    E_90 = E_0 / 30.0
    G = E_0 / 16.0
    # NDS tension perpendicular is typically not used in design
    # (assumed zero); we set it as a small fraction of f_t_0 to avoid
    # divide-by-zero in design checks that read it.
    Ft_perp = 0.025 * Ft_psi * _PSI
    rho_mean = density_pcf * _PCF_TO_KGM3
    return TimberMaterial(
        name=name, species=species, grade=grade, code="NDS",
        E_0_mean=E_0,
        E_0_05=E_min_msi * _MSI,
        E_90_mean=E_90,
        G_mean=G,
        f_b_k=Fb_psi * _PSI,
        f_t_0_k=Ft_psi * _PSI,
        f_t_90_k=Ft_perp,
        f_c_0_k=Fc_psi * _PSI,
        f_c_90_k=Fc_perp_psi * _PSI,
        f_v_k=Fv_psi * _PSI,
        density_k=0.85 * rho_mean,     # ~5th percentile typically 85% of mean
        density_mean=rho_mean,
    )


# ============================================================ SAWN LUMBER
# NDS-2024 Supplement Table 4A (visually graded dimension lumber 2"-4" thick).
# Values are for normal duration, dry service. Size factor C_F applies
# at design time per NDS §4.3.6.

NDS_SAWN_LUMBER: dict[str, TimberMaterial] = {}


def _add(name, **kwargs):
    NDS_SAWN_LUMBER[name] = _from_imperial(name=name, **kwargs)


# Douglas Fir-Larch (one of the most common North American softwoods)
_add(
    "DFL-SS",  species="Douglas Fir-Larch", grade="Select Structural",
    Fb_psi=1500, Ft_psi=1000, Fc_psi=1700,
    Fc_perp_psi=625, Fv_psi=180,
    E_msi=1.9, E_min_msi=0.690, density_pcf=32.0,
)
_add(
    "DFL-1",  species="Douglas Fir-Larch", grade="No. 1",
    Fb_psi=1000, Ft_psi=675, Fc_psi=1500,
    Fc_perp_psi=625, Fv_psi=180,
    E_msi=1.7, E_min_msi=0.620, density_pcf=32.0,
)
_add(
    "DFL-2",  species="Douglas Fir-Larch", grade="No. 2",
    Fb_psi=900, Ft_psi=575, Fc_psi=1350,
    Fc_perp_psi=625, Fv_psi=180,
    E_msi=1.6, E_min_msi=0.580, density_pcf=32.0,
)

# Southern Pine (common in southeastern US, often higher density)
_add(
    "SP-SS",  species="Southern Pine", grade="Select Structural",
    Fb_psi=2850, Ft_psi=1600, Fc_psi=2100,
    Fc_perp_psi=565, Fv_psi=175,
    E_msi=1.8, E_min_msi=0.660, density_pcf=36.0,
)
_add(
    "SP-1",  species="Southern Pine", grade="No. 1",
    Fb_psi=1850, Ft_psi=1050, Fc_psi=1850,
    Fc_perp_psi=565, Fv_psi=175,
    E_msi=1.7, E_min_msi=0.620, density_pcf=36.0,
)
_add(
    "SP-2",  species="Southern Pine", grade="No. 2",
    Fb_psi=1500, Ft_psi=825, Fc_psi=1650,
    Fc_perp_psi=565, Fv_psi=175,
    E_msi=1.6, E_min_msi=0.580, density_pcf=36.0,
)

# Spruce-Pine-Fir (Canadian softwood, very common framing lumber)
_add(
    "SPF-SS",  species="Spruce-Pine-Fir", grade="Select Structural",
    Fb_psi=1250, Ft_psi=700, Fc_psi=1400,
    Fc_perp_psi=425, Fv_psi=135,
    E_msi=1.5, E_min_msi=0.550, density_pcf=27.0,
)
_add(
    "SPF-1",  species="Spruce-Pine-Fir", grade="No. 1",
    Fb_psi=875, Ft_psi=450, Fc_psi=1150,
    Fc_perp_psi=425, Fv_psi=135,
    E_msi=1.4, E_min_msi=0.510, density_pcf=27.0,
)
_add(
    "SPF-2",  species="Spruce-Pine-Fir", grade="No. 2",
    Fb_psi=875, Ft_psi=450, Fc_psi=1150,
    Fc_perp_psi=425, Fv_psi=135,
    E_msi=1.4, E_min_msi=0.510, density_pcf=27.0,
)


# ============================================================ GLULAM
# NDS-2024 Supplement Table 5A (structural glued-laminated timber).
# Values for stress class combinations -- balanced layups for beams.

NDS_GLULAM: dict[str, TimberMaterial] = {}


def _add_gl(name, **kwargs):
    NDS_GLULAM[name] = _from_imperial(name=name, **kwargs)


# 24F-V4: 2400 psi bending, balanced layup (V4 = visually graded)
_add_gl(
    "24F-V4",  species="Glulam (DF/DF balanced)", grade="24F-1.8E",
    Fb_psi=2400, Ft_psi=1100, Fc_psi=1600,
    Fc_perp_psi=650, Fv_psi=265,
    E_msi=1.8, E_min_msi=0.95, density_pcf=35.0,
)
# 24F-V8: same bending, different layup (V8 = symmetric layup for both sides)
_add_gl(
    "24F-V8",  species="Glulam (DF/DF symmetric)", grade="24F-1.7E",
    Fb_psi=2400, Ft_psi=1100, Fc_psi=1600,
    Fc_perp_psi=650, Fv_psi=265,
    E_msi=1.7, E_min_msi=0.90, density_pcf=35.0,
)
# 24F-V3: Southern Pine variant
_add_gl(
    "24F-V3",  species="Glulam (SP/SP balanced)", grade="24F-1.8E",
    Fb_psi=2400, Ft_psi=1150, Fc_psi=1850,
    Fc_perp_psi=740, Fv_psi=300,
    E_msi=1.8, E_min_msi=0.95, density_pcf=37.0,
)
# 30F-V4: heavier glulam (3000 psi bending)
_add_gl(
    "30F-V4",  species="Glulam (DF/DF, heavy)", grade="30F-2.1E",
    Fb_psi=3000, Ft_psi=1450, Fc_psi=2000,
    Fc_perp_psi=740, Fv_psi=300,
    E_msi=2.1, E_min_msi=1.11, density_pcf=37.0,
)


# ============================================================ API

def get_nds_timber(name: str) -> TimberMaterial:
    """Look up an NDS timber by designation (e.g. ``"DFL-SS"``,
    ``"24F-V4"``).

    Raises ``KeyError`` if not found, with a helpful message listing
    available designations.
    """
    if name in NDS_SAWN_LUMBER:
        return NDS_SAWN_LUMBER[name]
    if name in NDS_GLULAM:
        return NDS_GLULAM[name]
    available = sorted(
        list(NDS_SAWN_LUMBER) + list(NDS_GLULAM)
    )
    raise KeyError(
        f"NDS timber {name!r} not in database. "
        f"Available: {available}"
    )


def list_nds_species() -> list[str]:
    """All sawn-lumber designations in the NDS database."""
    return sorted(NDS_SAWN_LUMBER.keys())
