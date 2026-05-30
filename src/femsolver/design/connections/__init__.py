"""Steel + RC connection models and design calculators.

Contains:

* :mod:`panel_zone` -- Krawinkler bilinear panel-zone backbone.
* :mod:`rbs`         -- Reduced Beam Section (AISC 358).
* :mod:`pr_connection` -- Richard-Abbott 4-parameter PR connections.
* :mod:`bolts_welds`  -- Bolt + weld strength calculators
  (AISC 360 / IS 800).
"""
from femsolver.design.connections.bolts_welds import (
    BlockShearResult,
    BoltBearingResult,
    BoltShearResult,
    WeldStrengthResult,
    block_shear_aisc,
    bolt_bearing_aisc,
    bolt_bearing_is800,
    bolt_shear_aisc,
    bolt_shear_is800,
    fillet_weld_aisc,
    fillet_weld_is800,
)
from femsolver.design.connections.panel_zone import (
    PanelZoneProperties,
    build_panel_zone_material,
    krawinkler_panel_zone,
)
from femsolver.design.connections.pr_connection import (
    Pr_preset,
    RichardAbbottParams,
)
from femsolver.design.connections.rbs import (
    RBSGeometry,
    aisc358_recommended_RBS,
    reduced_beam_section,
)


__all__ = [
    # panel zone
    "PanelZoneProperties",
    "krawinkler_panel_zone",
    "build_panel_zone_material",
    # RBS
    "RBSGeometry",
    "reduced_beam_section",
    "aisc358_recommended_RBS",
    # PR
    "RichardAbbottParams",
    "Pr_preset",
    # bolts + welds
    "BoltShearResult",
    "BoltBearingResult",
    "BlockShearResult",
    "WeldStrengthResult",
    "bolt_shear_aisc",
    "bolt_shear_is800",
    "bolt_bearing_aisc",
    "bolt_bearing_is800",
    "block_shear_aisc",
    "fillet_weld_aisc",
    "fillet_weld_is800",
]
