"""Mesh-generation, quality, and stress-recovery utilities
(Phase 47 / Theme L).

Submodules
----------
* :mod:`generators`       -- structured mesh generators (Q4, Hex8,
  polar disk / ring, cylindrical shell).
* :mod:`quality`          -- Jacobian / aspect-ratio / skewness
  metrics + mesh-wide aggregators.
* :mod:`stress_recovery`  -- Gauss-point -> nodal averaging,
  principal stresses, von Mises.
"""
from femsolver.mesh.generators import (
    StructuredMesh,
    disk_quad4,
    rectangle_quad4,
    ring_quad4,
    shell_curved_cylinder,
    solid_hex8,
)
from femsolver.mesh.quality import (
    MeshQualityReport,
    Quad4Quality,
    mesh_quality_report,
    quad4_quality,
)
from femsolver.mesh.stress_recovery import (
    NodalStressField,
    average_quad4_stresses_to_nodes,
    principal_stresses_2d,
    principal_stresses_3d,
    von_mises_2d,
    von_mises_3d,
    von_mises_field,
)

__all__ = [
    # generators
    "StructuredMesh",
    "rectangle_quad4",
    "disk_quad4",
    "ring_quad4",
    "solid_hex8",
    "shell_curved_cylinder",
    # quality
    "Quad4Quality",
    "quad4_quality",
    "MeshQualityReport",
    "mesh_quality_report",
    # stress recovery
    "NodalStressField",
    "average_quad4_stresses_to_nodes",
    "principal_stresses_2d",
    "von_mises_2d",
    "principal_stresses_3d",
    "von_mises_3d",
    "von_mises_field",
]
