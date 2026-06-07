"""Indian catalogue loader (ISMB, ISMC, ISA).

Wraps :data:`femsolver.data.sections_is.IS_ISMB`, ``IS_ISMC``,
``IS_ISA`` (IS 808 / SP6-1 section tables) as unified :class:`Section`
instances.

Family handling:
    ISMB  -> :class:`ISectionGeometry`   (I-beam)
    ISMC  -> :class:`ChannelGeometry`    (channel)
    ISA   -> :class:`AngleGeometry`      (equal angle)
"""
from __future__ import annotations

from typing import Any, Optional

from femsolver.sections.catalogue.geometry import CataloguedGeometry
from femsolver.sections.library import SectionLibrary
from femsolver.sections.parametric.geometry import (
    AngleGeometry,
    ChannelGeometry,
    ISectionGeometry,
)
from femsolver.sections.section import MaterialZone, Section


_MM = 1e-3
_MM2 = 1e-6
_MM3 = 1e-9
_MM4 = 1e-12


def _base_geometry_for_family(sp):
    """Build the right parametric base geometry for the family."""
    h_m = sp.h * _MM
    b_m = sp.b * _MM
    tw_m = sp.t_w * _MM
    tf_m = sp.t_f * _MM

    if sp.family == "ISMB":
        return ISectionGeometry(h=h_m, b=b_m, t_f=tf_m, t_w=tw_m)
    if sp.family == "ISMC":
        return ChannelGeometry(h=h_m, b=b_m, t_f=tf_m, t_w=tw_m)
    if sp.family == "ISA":
        # Equal angle: t_w == t_f
        return AngleGeometry(a=h_m, b=b_m, t=tw_m)
    raise ValueError(f"unknown IS family {sp.family!r}")


def _build_section_from_is(sp, material=None) -> Section:
    base = _base_geometry_for_family(sp)
    geom = CataloguedGeometry(
        base,
        area=sp.A * _MM2,
        I_zz=sp.I_y * _MM4,    # IS strong -> unified I_zz
        I_yy=sp.I_z * _MM4,    # IS weak   -> unified I_yy
        J=max(sp.J * _MM4, 1e-12),
        Z_zz=sp.W_pl_y * _MM3,
        Z_yy=sp.W_pl_z * _MM3,
        S_zz_top=sp.W_el_y * _MM3,
        S_zz_bot=sp.W_el_y * _MM3,
        S_yy_left=sp.W_el_z * _MM3,
        S_yy_right=sp.W_el_z * _MM3,
    )
    zones = []
    if material is not None:
        zones.append(MaterialZone(material=material, name="steel"))
    return Section(
        geometry=geom,
        zones=zones,
        name=sp.name,
        family=sp.family,
        catalogue_ref=sp.name,
    )


def indian_section(designation: str, *, material: Any = None) -> Section:
    """Look up an Indian section (ISMB / ISMC / ISA) by designation."""
    from femsolver.data.sections_is import IS_ISA, IS_ISMB, IS_ISMC
    for table in (IS_ISMB, IS_ISMC, IS_ISA):
        if designation in table:
            return _build_section_from_is(table[designation], material=material)
    raise KeyError(
        f"{designation!r} not in Indian catalogue. "
        f"Try one of ISMB/ISMC/ISA families."
    )


_IS_LIBRARY: Optional[SectionLibrary] = None


def load_indian_library(*, material: Any = None,
                         force_reload: bool = False) -> SectionLibrary:
    """Return a :class:`SectionLibrary` populated with ISMB + ISMC + ISA."""
    global _IS_LIBRARY
    if _IS_LIBRARY is not None and not force_reload:
        return _IS_LIBRARY
    from femsolver.data.sections_is import IS_ISA, IS_ISMB, IS_ISMC

    lib = SectionLibrary()
    for table in (IS_ISMB, IS_ISMC, IS_ISA):
        for sp in table.values():
            lib.register(_build_section_from_is(sp, material=material))
    _IS_LIBRARY = lib
    return lib
