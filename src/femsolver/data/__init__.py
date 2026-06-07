"""Engineering catalogues -- section property, bolt, and material
databases for commercial-grade design workflows.

Submodules
----------
* :mod:`sections_ec`   -- Eurocode hot-rolled sections (IPE, HEA, HEB).
* :mod:`sections_is`   -- IS 808 / SP6-1 Indian sections (ISMB, ISMC, ISA).
* :mod:`bolts`         -- Bolt catalogues (A325/A490, ISO 8.8/10.9).
* :mod:`materials`     -- Concrete, steel, rebar grade lookups across
  ACI / EC / IS conventions.
"""
from femsolver.data.sections_ec import (
    SectionProperties,
    EC_IPE,
    EC_HEA,
    EC_HEB,
    eurocode_section,
    list_eurocode_sections,
    auto_select_ec_section,
)
from femsolver.data.sections_is import (
    IS_ISMB,
    IS_ISMC,
    IS_ISA,
    indian_section,
    list_indian_sections,
)
from femsolver.data.bolts import (
    BoltProperties,
    bolt_lookup,
    list_bolt_grades,
)
from femsolver.data.materials import (
    ConcreteGrade,
    SteelGrade,
    RebarGrade,
    concrete_grade,
    steel_grade,
    rebar_grade,
)

__all__ = [
    # sections
    "SectionProperties",
    "EC_IPE", "EC_HEA", "EC_HEB",
    "eurocode_section", "list_eurocode_sections",
    "auto_select_ec_section",
    "IS_ISMB", "IS_ISMC", "IS_ISA",
    "indian_section", "list_indian_sections",
    # bolts
    "BoltProperties",
    "bolt_lookup", "list_bolt_grades",
    # materials
    "ConcreteGrade", "SteelGrade", "RebarGrade",
    "concrete_grade", "steel_grade", "rebar_grade",
]
