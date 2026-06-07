"""AISC catalogue loader.

Wraps :data:`femsolver.design.steel.sections._DATABASE` (the existing
AISC v15.0 W-shape table) as unified :class:`Section` instances.

Mapping AISC -> unified convention:
    AISC.Ix (strong, horizontal axis) -> Section.I_zz
    AISC.Iy (weak, vertical axis)     -> Section.I_yy
    AISC.Sx, Zx                       -> Section.S_zz_top, Z_zz
    AISC.Sy, Zy                       -> Section.S_yy_left, Z_yy
    AISC.d  (depth)                   -> ISectionGeometry h
    AISC.bf (flange width)            -> ISectionGeometry b
    AISC.tf (flange thickness)        -> ISectionGeometry t_f
    AISC.tw (web thickness)           -> ISectionGeometry t_w
"""
from __future__ import annotations

from typing import Any, Optional

from femsolver.sections.catalogue.geometry import CataloguedGeometry
from femsolver.sections.library import SectionLibrary
from femsolver.sections.parametric.geometry import ISectionGeometry
from femsolver.sections.section import MaterialZone, Section


def _build_section_from_aisc(steel_section, material=None) -> Section:
    """Build a unified Section from a femsolver.design.steel.SteelSection."""
    ss = steel_section
    base = ISectionGeometry(h=ss.d, b=ss.bf, t_f=ss.tf, t_w=ss.tw)
    geom = CataloguedGeometry(
        base,
        area=ss.A,
        I_zz=ss.Ix,
        I_yy=ss.Iy,
        J=ss.J,
        Z_zz=ss.Zx,
        Z_yy=ss.Zy,
        S_zz_top=ss.Sx,
        S_zz_bot=ss.Sx,
        S_yy_left=ss.Sy,
        S_yy_right=ss.Sy,
    )
    zones = []
    if material is not None:
        zones.append(MaterialZone(material=material, name="steel"))
    return Section(
        geometry=geom,
        zones=zones,
        name=ss.designation,
        family="W",
        catalogue_ref=ss.designation,
    )


def aisc_section(designation: str, *, material: Any = None) -> Section:
    """Look up an AISC W-shape by designation, return a unified
    :class:`Section`.

    Parameters
    ----------
    designation : str
        e.g. ``"W14x90"``.
    material : optional
        Steel material reference (e.g. ASTM A992 grade 50). If
        provided, attached as a :class:`MaterialZone` so the section
        can drive elastic adapters out of the box.
    """
    from femsolver.design.steel.sections import get_section
    ss = get_section(designation)
    return _build_section_from_aisc(ss, material=material)


_AISC_LIBRARY: Optional[SectionLibrary] = None


def load_aisc_library(*, material: Any = None,
                       force_reload: bool = False) -> SectionLibrary:
    """Return a :class:`SectionLibrary` populated with every AISC
    W-shape in the embedded database.

    The library is cached so repeated calls return the same instance.
    Pass ``force_reload=True`` to rebuild (mainly for tests).

    Note: the cached library is independent of any ``material`` you
    pass -- the first call wins. To attach different materials per
    project, build sections individually with :func:`aisc_section`.
    """
    global _AISC_LIBRARY
    if _AISC_LIBRARY is not None and not force_reload:
        return _AISC_LIBRARY
    from femsolver.design.steel.sections import _DATABASE

    lib = SectionLibrary()
    for designation, ss in _DATABASE.items():
        lib.register(_build_section_from_aisc(ss, material=material))
    _AISC_LIBRARY = lib
    return lib
