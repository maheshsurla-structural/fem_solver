"""Post-processing utilities (Phase 47 / Theme L).

Submodules
----------
* :mod:`plot` -- matplotlib quick-plot helpers (mesh, deformed,
  contour, mode shape, time history).
"""
from femsolver.postproc.plot import (
    plot_contour,
    plot_deformed,
    plot_mode_shape,
    plot_time_history,
    plot_undeformed,
)
from femsolver.postproc.plot_3d import (
    plot_deformed_3d,
    plot_mode_shape_3d,
    plot_scalar_field_3d,
    plot_undeformed_3d,
)

__all__ = [
    # 2D / matplotlib (Phase 47)
    "plot_undeformed",
    "plot_deformed",
    "plot_contour",
    "plot_mode_shape",
    "plot_time_history",
    # 3D / PyVista (Phase 48.2)
    "plot_undeformed_3d",
    "plot_deformed_3d",
    "plot_scalar_field_3d",
    "plot_mode_shape_3d",
]
