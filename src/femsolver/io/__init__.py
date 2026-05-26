"""I/O and post-processing utilities for femsolver models and results.

* JSON model deck: :func:`save_model_json`, :func:`load_model_json`.
* VTK export: :func:`write_vtk` (rich), :func:`write_vtk_unstructured`
  (legacy / backward-compat).
* Beam force diagrams: :func:`beam_force_diagram`.
* Extraction utilities: :func:`gather_node_history`,
  :func:`mode_shape_table`, :func:`capacity_curve`.
"""
from femsolver.io.json_io import save_model_json, load_model_json
from femsolver.io.vtk_io import write_vtk, write_vtk_unstructured
from femsolver.io.diagrams import beam_force_diagram, plot_beam_diagrams
from femsolver.io.extract import (
    capacity_curve,
    gather_node_history,
    mode_shape_table,
)

__all__ = [
    "save_model_json",
    "load_model_json",
    "write_vtk",
    "write_vtk_unstructured",
    "beam_force_diagram",
    "plot_beam_diagrams",
    "gather_node_history",
    "mode_shape_table",
    "capacity_curve",
]
