"""Results: export, plotting, and production deliverables.

One home for everything you do *with* a solved model -- merged from the
former ``io`` (file export), ``postproc`` (plotting), and ``deliverables``
(reports) packages.

Submodules
----------
File export / exchange
    * :mod:`json_io`  -- JSON model deck save/load.
    * :mod:`vtk_io`   -- VTK export for ParaView.
    * :mod:`ida_hdf5` -- HDF5-backed IDA result store.
    * :mod:`extract`  -- node-history / mode-shape / capacity-curve tables.
Plotting
    * :mod:`plot`     -- matplotlib quick plots (mesh, deformed, contour).
    * :mod:`plot_3d`  -- PyVista 3-D plots.
    * :mod:`diagrams` -- beam force diagrams (M, V, N).
Reports / deliverables
    * :mod:`calc_sheet` -- engineering calc sheet (HTML / PDF).
    * :mod:`dxf`        -- native minimal DXF drawing writer.
    * :mod:`bom`        -- bill-of-materials / quantity takeoff.
    * :mod:`qa`         -- model QA / sanity warnings.
"""
# --- file export / exchange ---
from femsolver.results.json_io import save_model_json, load_model_json
from femsolver.results.vtk_io import write_vtk, write_vtk_unstructured
from femsolver.results.ida_hdf5 import (
    load_ida_record,
    load_ida_summary,
    save_ida_record,
    save_ida_summary,
)
from femsolver.results.extract import (
    capacity_curve,
    gather_node_history,
    mode_shape_table,
)
# --- plotting ---
from femsolver.results.diagrams import beam_force_diagram, plot_beam_diagrams
from femsolver.results.plot import (
    plot_contour,
    plot_deformed,
    plot_mode_shape,
    plot_time_history,
    plot_undeformed,
)
from femsolver.results.plot_3d import (
    plot_deformed_3d,
    plot_mode_shape_3d,
    plot_scalar_field_3d,
    plot_undeformed_3d,
)
# --- reports / deliverables ---
from femsolver.results.calc_sheet import (
    CalcCheck,
    CalcInput,
    CalcOutput,
    CalcSection,
    CalcSheet,
    render_calc_sheet_html,
    render_calc_sheet_pdf,
)
from femsolver.results.dxf import DxfDocument, write_model_plan_dxf
from femsolver.results.bom import (
    BomLine,
    BomReport,
    bom_concrete_frame,
    bom_rebar,
    bom_steel_frame,
)
from femsolver.results.qa import QaReport, QaWarning, run_qa_checks

__all__ = [
    # files
    "save_model_json", "load_model_json",
    "write_vtk", "write_vtk_unstructured",
    "save_ida_record", "load_ida_record", "save_ida_summary", "load_ida_summary",
    "capacity_curve", "gather_node_history", "mode_shape_table",
    # plots
    "beam_force_diagram", "plot_beam_diagrams",
    "plot_undeformed", "plot_deformed", "plot_contour", "plot_mode_shape",
    "plot_time_history",
    "plot_undeformed_3d", "plot_deformed_3d", "plot_scalar_field_3d",
    "plot_mode_shape_3d",
    # reports
    "CalcCheck", "CalcInput", "CalcOutput", "CalcSection", "CalcSheet",
    "render_calc_sheet_html", "render_calc_sheet_pdf",
    "DxfDocument", "write_model_plan_dxf",
    "BomLine", "BomReport", "bom_concrete_frame", "bom_steel_frame", "bom_rebar",
    "QaReport", "QaWarning", "run_qa_checks",
]
