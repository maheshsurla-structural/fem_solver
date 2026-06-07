"""Catalogue loaders for AISC / Eurocode / Indian steel sections
(Theme II.4).

This sub-package wraps the existing pure-data catalogue dictionaries
(``femsolver.design.steel.sections._DATABASE`` for AISC,
``femsolver.data.sections_ec`` for Eurocode,
``femsolver.data.sections_is`` for Indian) into unified
:class:`~femsolver.sections.section.Section` instances and registers
them in a :class:`SectionLibrary`.

Each catalogued section gets:

* A parametric polygon outline (from
  :mod:`femsolver.sections.parametric`) sized to the catalogue
  ``h``, ``b``, ``t_f``, ``t_w`` -- so visualization works.
* All gross properties (A, I_zz, I_yy, J, Z) OVERRIDDEN with the
  catalogue-exact values (so numerics match AISC tables to all
  significant figures, including fillet corrections).
* ``catalogue_ref`` set to the catalogue designation (``"W14x90"``,
  ``"IPE 300"``, ``"ISMB 400"``).
* ``family`` set to the catalogue family code.

The original catalogue dicts are NOT modified. This is purely
additive; design modules that import the legacy dataclasses keep
working.

Public API
----------
* :func:`load_aisc_library` -- returns a :class:`SectionLibrary` with
  all AISC W-shapes registered.
* :func:`load_eurocode_library` -- IPE + HEA + HEB.
* :func:`load_indian_library` -- ISMB + ISMC + ISA.
* :class:`CataloguedGeometry` -- the wrapper class.
"""
from femsolver.sections.catalogue.geometry import CataloguedGeometry
from femsolver.sections.catalogue.aisc import (
    aisc_section,
    load_aisc_library,
)
from femsolver.sections.catalogue.eurocode import (
    eurocode_section,
    load_eurocode_library,
)
from femsolver.sections.catalogue.indian import (
    indian_section,
    load_indian_library,
)

__all__ = [
    "CataloguedGeometry",
    "aisc_section",
    "load_aisc_library",
    "eurocode_section",
    "load_eurocode_library",
    "indian_section",
    "load_indian_library",
]
