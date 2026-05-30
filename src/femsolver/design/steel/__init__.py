"""Steel design per AISC 360-22.

Phase 30 modules:

* :mod:`sections` -- AISC v15.0 W-shapes catalog +
  :class:`SteelSection` / :class:`SteelMaterial` dataclasses
  (Phase 30.1).

Future Phase 30 modules:

* ``compression`` -- AISC Ch. E (Phase 30.2)
* ``flexure`` -- AISC Ch. F with LTB (Phase 30.3)
* ``tension_shear`` -- AISC Ch. D + G (Phase 30.4)
* ``combined`` -- AISC Ch. H (Phase 30.5)
* ``designer`` -- SteelMemberDesigner with auto-sizing (Phase 30.6)
"""
from femsolver.design.steel.sections import (
    SteelMaterial,
    SteelSection,
    all_designations,
    astm_a36,
    astm_a992,
    get_section,
    w_series,
)
from femsolver.design.steel.compression import (
    PHI_COMPRESSION,
    CompressionCheck,
    compression_strength,
)
from femsolver.design.steel.flexure import (
    PHI_FLEXURE,
    FlexureCheck,
    c_b_from_moments,
    flexural_strength,
)
from femsolver.design.steel.tension_shear import (
    PHI_TENSION_RUPTURE,
    PHI_TENSION_YIELD,
    ShearCheck,
    TensionCheck,
    shear_strength,
    tension_strength,
)
from femsolver.design.steel.combined import (
    CombinedForceCheck,
    combined_force_check,
)
from femsolver.design.steel.designer import (
    SteelDesignResult,
    SteelMemberCheck,
    SteelMemberDemand,
    SteelMemberDesigner,
    auto_size,
    check_member,
)

__all__ = [
    "SteelSection",
    "SteelMaterial",
    "get_section",
    "all_designations",
    "w_series",
    "astm_a36",
    "astm_a992",
    "PHI_COMPRESSION",
    "CompressionCheck",
    "compression_strength",
    "PHI_FLEXURE",
    "FlexureCheck",
    "c_b_from_moments",
    "flexural_strength",
    "PHI_TENSION_YIELD",
    "PHI_TENSION_RUPTURE",
    "TensionCheck",
    "ShearCheck",
    "tension_strength",
    "shear_strength",
    "CombinedForceCheck",
    "combined_force_check",
    "SteelMemberDemand",
    "SteelMemberCheck",
    "SteelDesignResult",
    "SteelMemberDesigner",
    "check_member",
    "auto_size",
]
