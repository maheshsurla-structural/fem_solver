"""Write a deformed-shape VTK unstructured grid for ParaView visualization.

Uses the legacy VTK ASCII format to avoid a hard dependency on `meshio`.
Each element type maps to a VTK cell type:

    Truss2D, Truss3D, BeamColumn2D, BeamColumn3D -> VTK_LINE (3)
    Quad4                                        -> VTK_QUAD (9)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.plane import Quad4


_VTK_LINE = 3
_VTK_QUAD = 9


def _cell_type(element) -> int:
    if isinstance(element, (Truss2D, Truss3D, BeamColumn2D, BeamColumn3D)):
        return _VTK_LINE
    if isinstance(element, Quad4):
        return _VTK_QUAD
    raise NotImplementedError(f"VTK output not implemented for {type(element).__name__}")


def write_vtk_unstructured(model, path: str | Path, deformation_scale: float = 0.0) -> None:
    """Write the model as a VTK legacy unstructured grid.

    Parameters
    ----------
    model : Model
    path : path to .vtk file
    deformation_scale : if > 0, displace nodes by `scale * disp` so the
        visualization shows the deformed shape.
    """
    nodes = list(model.nodes.values())
    tag_to_idx = {n.tag: i for i, n in enumerate(nodes)}

    points = np.zeros((len(nodes), 3))
    for i, n in enumerate(nodes):
        c = n.coords
        if c.size == 2:
            points[i, 0:2] = c
        else:
            points[i, : c.size] = c
        # add scaled displacement (only translational DOFs, first ndm)
        if deformation_scale != 0.0:
            ndm = model.ndm
            d = n.disp[:ndm]
            points[i, :ndm] += deformation_scale * d

    elements = list(model.elements.values())
    cell_data = []
    cell_types = []
    cell_size_total = 0
    for e in elements:
        ct = _cell_type(e)
        cell_types.append(ct)
        idxs = [tag_to_idx[t] for t in e.node_tags]
        cell_data.append(idxs)
        cell_size_total += 1 + len(idxs)

    lines: list[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append(f"femsolver model {getattr(model, 'name', '')}".strip())
    lines.append("ASCII")
    lines.append("DATASET UNSTRUCTURED_GRID")
    lines.append(f"POINTS {len(points)} float")
    for p in points:
        lines.append(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
    lines.append(f"CELLS {len(elements)} {cell_size_total}")
    for idxs in cell_data:
        lines.append(f"{len(idxs)} " + " ".join(str(i) for i in idxs))
    lines.append(f"CELL_TYPES {len(elements)}")
    for ct in cell_types:
        lines.append(str(ct))

    # POINT_DATA: displacement vector (3 components)
    lines.append(f"POINT_DATA {len(points)}")
    lines.append("VECTORS displacement float")
    for n in nodes:
        d = np.zeros(3)
        ndm = model.ndm
        d[:ndm] = n.disp[:ndm]
        lines.append(f"{d[0]:.9g} {d[1]:.9g} {d[2]:.9g}")

    Path(path).write_text("\n".join(lines) + "\n")
