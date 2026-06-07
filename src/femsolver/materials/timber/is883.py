"""IS 883:2016 timber classification (Phase D.1.1).

IS 883 (Code of Practice for Design of Structural Timber in Building)
groups Indian timber species into three strength classes based on
modulus of elasticity, bending strength, and resistance to wear /
weathering:

* **Class I (Group A)**  -- high-strength hardwoods (Teak, Sal, Padouk,
  Deodar). Typically E >= 12.6 GPa, f_b >= 18 MPa.
* **Class II (Group B)** -- medium-strength hardwoods + premium
  softwoods. E in 9.8-12.6 GPa, f_b in 12-18 MPa.
* **Class III (Group C)** -- lower-strength species, common framing
  softwoods. E in 5.6-9.8 GPa, f_b in 8.5-12 MPa.

The IS 883 design provisions apply allowable-stress factors per
Cl. 5 (load duration, moisture, etc.). Reference values stored here
are the "inside location" permissible stresses (which are then
adjusted by partial factors at design time).
"""
from __future__ import annotations

from femsolver.materials.timber.material import TimberMaterial


# IS 883:2016 Table 1 (basic allowable stresses for inside location,
# dry condition). Values in MPa for stress, GPa for E.

_IS_DATA = [
    # name, species_label, fm, ft0, fc0, fc90, fv, E0mean, rho_k, rho_mean
    ("IS-Class-I",  "Group A (Teak / Sal)",  18.0, 10.5, 11.7, 4.0, 1.10,
        12.6, 700, 800),
    ("IS-Class-II", "Group B (Deodar / Babul / Hollock)", 12.0, 8.5, 7.8, 2.6, 0.84,
        9.8,  570, 660),
    ("IS-Class-III","Group C (Spruce / Fir / Devdar)", 8.5, 6.0, 4.9, 1.5, 0.64,
        5.6, 450, 530),
]


IS883_CLASSES: dict[str, TimberMaterial] = {}


for row in _IS_DATA:
    name, species, fm, ft0, fc0, fc90, fv, E0m, rho_k, rho_mean = row
    IS883_CLASSES[name] = TimberMaterial(
        name=name, species=species, grade=name, code="IS 883",
        E_0_mean=E0m * 1e9,
        E_0_05=0.67 * E0m * 1e9,
        E_90_mean=E0m * 1e9 / 30.0,
        G_mean=E0m * 1e9 / 16.0,
        f_b_k=fm * 1e6,
        f_t_0_k=ft0 * 1e6,
        f_t_90_k=0.025 * ft0 * 1e6,  # negligible
        f_c_0_k=fc0 * 1e6,
        f_c_90_k=fc90 * 1e6,
        f_v_k=fv * 1e6,
        density_k=float(rho_k),
        density_mean=float(rho_mean),
    )


def get_is883_class(name: str) -> TimberMaterial:
    """Look up an IS 883 timber class."""
    if name in IS883_CLASSES:
        return IS883_CLASSES[name]
    raise KeyError(
        f"IS 883 class {name!r} not in database. "
        f"Available: {sorted(IS883_CLASSES)}"
    )
