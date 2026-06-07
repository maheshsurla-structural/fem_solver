"""Eurocode catalogue loader (IPE, HEA, HEB).

Wraps :data:`femsolver.data.sections_ec.EC_IPE`, ``EC_HEA``,
``EC_HEB`` (the existing ArcelorMittal section tables) as unified
:class:`Section` instances.

Mapping EC convention -> unified convention:
    EC.I_y (strong, about horizontal axis) -> Section.I_zz
    EC.I_z (weak, about vertical axis)     -> Section.I_yy
    EC.W_el_y, W_pl_y                       -> Section S_zz, Z_zz
    EC.W_el_z, W_pl_z                       -> Section S_yy, Z_yy

EC dimensions in mm, areas in mm^2, I in mm^4 -- converted to SI here.
"""
from __future__ import annotations

from typing import Any, Optional

from femsolver.sections.catalogue.geometry import CataloguedGeometry
from femsolver.sections.library import SectionLibrary
from femsolver.sections.parametric.geometry import ISectionGeometry
from femsolver.sections.section import MaterialZone, Section


_MM = 1e-3
_MM2 = 1e-6
_MM3 = 1e-9
_MM4 = 1e-12


def _build_section_from_ec(sp, material=None) -> Section:
    """Build a unified Section from a femsolver.data.sections_ec
    .SectionProperties (which can be IPE, HEA, HEB, or Indian)."""
    h_m = sp.h * _MM
    b_m = sp.b * _MM
    tw_m = sp.t_w * _MM
    tf_m = sp.t_f * _MM
    base = ISectionGeometry(h=h_m, b=b_m, t_f=tf_m, t_w=tw_m)
    geom = CataloguedGeometry(
        base,
        area=sp.A * _MM2,
        I_zz=sp.I_y * _MM4,    # EC strong  -> unified I_zz
        I_yy=sp.I_z * _MM4,    # EC weak    -> unified I_yy
        J=max(sp.J * _MM4, 1e-12),  # some legacy entries had J=0
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


def eurocode_section(designation: str, *, material: Any = None) -> Section:
    """Look up an EC section (IPE, HEA, HEB) by designation."""
    from femsolver.data.sections_ec import EC_HEA, EC_HEB, EC_IPE
    for table in (EC_IPE, EC_HEA, EC_HEB):
        if designation in table:
            return _build_section_from_ec(table[designation], material=material)
    raise KeyError(
        f"{designation!r} not in Eurocode catalogue. "
        f"Try one of IPE/HEA/HEB families."
    )


_EC_LIBRARY: Optional[SectionLibrary] = None


def load_eurocode_library(*, material: Any = None,
                            force_reload: bool = False) -> SectionLibrary:
    """Return a :class:`SectionLibrary` populated with IPE + HEA + HEB."""
    global _EC_LIBRARY
    if _EC_LIBRARY is not None and not force_reload:
        return _EC_LIBRARY
    from femsolver.data.sections_ec import EC_HEA, EC_HEB, EC_IPE

    lib = SectionLibrary()
    for table in (EC_IPE, EC_HEA, EC_HEB):
        for sp in table.values():
            lib.register(_build_section_from_ec(sp, material=material))
    _EC_LIBRARY = lib
    return lib
