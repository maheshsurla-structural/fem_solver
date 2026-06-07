"""Cross-section response models for beam-column elements.

A *section* maps a vector of generalized strains (axial strain and
curvatures) to a vector of generalized stress resultants (axial force and
bending moments) plus a tangent stiffness matrix. This decouples the beam
element (which integrates section response along its length) from the
constitutive description of the cross-section, so the same element can host
elastic, fiber, or hinge-based sections without changes to its assembly
code.

Two layers
----------

* **Low-level analysis interface** -- :class:`SectionBase`, with concrete
  implementations :class:`ElasticSection2D/3D` and :class:`FiberSection2D/3D`.
  These are what beam elements consume via ``get_response(e) -> (s, ks)``.
* **High-level canonical section** -- :class:`Section` (Theme II.2+) is the
  unified user-facing object. It carries identity, geometry, materials, and
  optional reinforcement, and produces :class:`SectionBase` instances on
  demand via ``elastic_section_2d/3d()`` and ``fiber_section_2d/3d()``.
  The single source of truth.

The two coexist: low-level constructors keep working. ``Section`` is
additive.
"""
from femsolver.sections.response.base import SectionBase
from femsolver.sections.geometry import (
    Geometry,
    PolygonGeometry,
    polygon_centroid,
    polygon_second_moments,
    shoelace_area,
)
from femsolver.sections.catalogue import (
    aisc_section,
    eurocode_section,
    indian_section,
    load_aisc_library,
    load_eurocode_library,
    load_indian_library,
)
from femsolver.shell_sections.clt import CLTLayer, CLTSection
from femsolver.sections.library import SectionLibrary
from femsolver.sections.parametric import (
    angle_section,
    channel_section,
    circular_section,
    custom_polygon_section,
    hollow_circular_section,
    hollow_rect_section,
    i_section,
    rc_rectangular_section,
    rectangular_section,
    subtract_polygons,
    t_section,
    union_polygons,
)
from femsolver.sections.report import (
    RebarRow,
    SectionReport,
    build_section_report,
)
from femsolver.sections.section import (
    MaterialZone,
    PrestressTendon,
    RebarBar,
    ReinforcementLayout,
    Section,
    TendonLayout,
)
from femsolver.sections.serialization import (
    section_from_dict,
    section_from_json,
    section_to_dict,
    section_to_json,
)
from femsolver.sections.visualization import section_to_svg
from femsolver.sections.response.elastic import (
    ElasticSection2D,
    ElasticSection3D,
)
from femsolver.sections.response.fiber import (
    Fiber,
    FiberSection2D,
    FiberSection3D,
)
from femsolver.shell_sections import (
    ElasticShellSection,
    LayeredShellSection,
    ShellLayer,
    ShellSectionBase,
)
from femsolver.shell_sections.ply_failure import (
    PlyStrength,
    evaluate_laminate,
    max_strain_index,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)
from femsolver.sections.response.wall import (
    WallRegion,
    i_wall_section_3d,
    l_wall_section_3d,
    t_wall_section_3d,
    u_wall_section_3d,
    wall_section_2d,
)
from femsolver.sections.response.wall_shear import (
    CrackedSectionFactors,
    aci318_cracked_factors,
    asce41_wall_factors,
    wall_base_shear_spring_stiffness,
    wall_lateral_stiffness,
    wall_shear_area,
)

__all__ = [
    # Theme II.2 -- unified Section layer
    "Section",
    "MaterialZone",
    "ReinforcementLayout",
    "RebarBar",
    "TendonLayout",
    "PrestressTendon",
    "CLTLayer",
    "CLTSection",
    "SectionLibrary",
    "Geometry",
    "PolygonGeometry",
    "shoelace_area",
    "polygon_centroid",
    "polygon_second_moments",
    # Parametric primitives (Theme II.3)
    "rectangular_section",
    "i_section",
    "t_section",
    "channel_section",
    "angle_section",
    "hollow_rect_section",
    "circular_section",
    "hollow_circular_section",
    # RC composition (Theme II.6)
    "rc_rectangular_section",
    # Custom polygons + Boolean ops (Theme II.5)
    "custom_polygon_section",
    "union_polygons",
    "subtract_polygons",
    # Catalogue lookups (Theme II.4)
    "aisc_section",
    "eurocode_section",
    "indian_section",
    "load_aisc_library",
    "load_eurocode_library",
    "load_indian_library",
    # JSON + SVG + report (Theme II.8)
    "section_to_json",
    "section_from_json",
    "section_to_dict",
    "section_from_dict",
    "section_to_svg",
    "SectionReport",
    "RebarRow",
    "build_section_report",
    # Low-level analysis sections
    "SectionBase",
    "ElasticSection2D",
    "ElasticSection3D",
    "Fiber",
    "FiberSection2D",
    "FiberSection3D",
    "ShellSectionBase",
    "ElasticShellSection",
    "LayeredShellSection",
    "ShellLayer",
    "PlyStrength",
    "max_stress_index",
    "max_strain_index",
    "tsai_hill_index",
    "tsai_wu_index",
    "tsai_wu_strength_ratio",
    "evaluate_laminate",
    "WallRegion",
    "wall_section_2d",
    "t_wall_section_3d",
    "l_wall_section_3d",
    "u_wall_section_3d",
    "i_wall_section_3d",
    "CrackedSectionFactors",
    "aci318_cracked_factors",
    "asce41_wall_factors",
    "wall_base_shear_spring_stiffness",
    "wall_lateral_stiffness",
    "wall_shear_area",
]
