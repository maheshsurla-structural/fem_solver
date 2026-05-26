"""Write a deformed-shape VTK unstructured grid for Paraview / VisIt /
MayaVi visualization.

Uses the legacy VTK ASCII format -- no dependency on ``meshio``. Each
element family is mapped to the appropriate VTK cell type:

* Truss (2D / 3D, corotational variants)                      -> VTK_LINE
* BeamColumn (2D / 3D, all corotational + force-based + hinged) -> VTK_LINE
* Quad4 plane element                                         -> VTK_QUAD
* ShellMITC4 (4-node Mindlin-Reissner shell)                   -> VTK_QUAD
* ShellTri3 (3-node Reissner-Mindlin shell)                    -> VTK_TRIANGLE
* Hex8 (8-node trilinear brick)                                -> VTK_HEXAHEDRON
* Tet4 (4-node linear tetrahedron)                             -> VTK_TETRA

Two output entry points:

* :func:`write_vtk_unstructured` -- legacy / backwards-compatible writer
  that emits a deformed-shape file with the nodal displacement vector.
* :func:`write_vtk` -- richer interface that exposes optional element
  data (e.g. averaged stress, equivalent plastic strain, axial force
  along a beam).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.elements.beam_corot_3d import BeamColumn3DCorotational
from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.elements.beam_hinged import HingedBeamColumn2D
from femsolver.elements.plane import Quad4
from femsolver.elements.shell import ShellMITC4
from femsolver.elements.shell_tri import ShellTri3
from femsolver.elements.solid import Hex8, Tet4
from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.elements.truss_corot import Truss2DCorotational


# ---------------------------------------------------------------- VTK types
_VTK_LINE = 3
_VTK_TRIANGLE = 5
_VTK_QUAD = 9
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12


_LINE_ELEMENTS = (
    Truss2D,
    Truss3D,
    Truss2DCorotational,
    BeamColumn2D,
    BeamColumn3D,
    BeamColumn2DCorotational,
    BeamColumn3DCorotational,
    ForceBeamColumn2DCorotational,
    HingedBeamColumn2D,
)


def _cell_type(element) -> int:
    if isinstance(element, _LINE_ELEMENTS):
        return _VTK_LINE
    if isinstance(element, ShellTri3):
        return _VTK_TRIANGLE
    if isinstance(element, (Quad4, ShellMITC4)):
        return _VTK_QUAD
    if isinstance(element, Tet4):
        return _VTK_TETRA
    if isinstance(element, Hex8):
        return _VTK_HEXAHEDRON
    raise NotImplementedError(
        f"VTK output not implemented for {type(element).__name__}"
    )


# ---------------------------------------------------------------- legacy entry
def write_vtk_unstructured(model, path: str | Path,
                            deformation_scale: float = 0.0) -> None:
    """Write the model as a VTK legacy unstructured grid (deformed
    shape + nodal displacement vector).

    Parameters
    ----------
    model : Model
    path : str or Path
        Output file path. Conventionally ``.vtk``.
    deformation_scale : float
        If non-zero, displace the points by ``scale * Node.disp[:ndm]``
        so that the visualization shows the deformed shape. Use 1.0
        for true scale or a larger value for amplified visualization.

    The displacement vector is written as ``POINT_DATA`` so that
    Paraview's "Warp by Vector" filter (or similar) can apply the
    deformation interactively from a single un-warped file.
    """
    write_vtk(model, path, deformation_scale=deformation_scale)


# ---------------------------------------------------------------- rich entry
def write_vtk(model, path: str | Path, *,
               deformation_scale: float = 0.0,
               point_data: dict[str, np.ndarray] | None = None,
               cell_data: dict[str, np.ndarray] | None = None,
               include_reactions: bool = False,
               include_mode_shapes: bool = False) -> None:
    """Write the model as a VTK legacy unstructured grid with optional
    extra point / cell data fields.

    Parameters
    ----------
    model : Model
    path : str or Path
    deformation_scale : float, default 0
        Scale applied to ``Node.disp`` when writing point coordinates.
    point_data : dict of str -> ndarray
        Additional per-node fields. Scalar fields have shape
        ``(n_nodes,)``; vector fields have shape ``(n_nodes, 3)``.
        Vector fields with fewer than 3 components are zero-padded.
    cell_data : dict of str -> ndarray
        Additional per-element fields. Shapes mirror ``point_data``
        but indexed by element rather than node.
    include_reactions : bool, default False
        If ``True``, write ``Node.reaction`` as a 3-component vector
        ``reaction``.
    include_mode_shapes : bool, default False
        If ``True``, write each column of ``Node.mode_disp`` as
        ``mode_1``, ``mode_2``, ... vectors. Useful for visualizing
        eigenmodes or buckling shapes.
    """
    nodes = list(model.nodes.values())
    tag_to_idx = {n.tag: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    ndm = model.ndm

    # ---------- points
    points = np.zeros((n_nodes, 3))
    for i, n in enumerate(nodes):
        c = n.coords
        points[i, : c.size] = c
        if deformation_scale != 0.0:
            d = n.disp[:ndm]
            points[i, :ndm] += deformation_scale * d

    # ---------- cells
    elements = list(model.elements.values())
    cell_data_idxs: list[list[int]] = []
    cell_types: list[int] = []
    cell_size_total = 0
    for e in elements:
        cell_types.append(_cell_type(e))
        idxs = [tag_to_idx[t] for t in e.node_tags]
        cell_data_idxs.append(idxs)
        cell_size_total += 1 + len(idxs)

    lines: list[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append(f"femsolver model {getattr(model, 'name', '')}".strip())
    lines.append("ASCII")
    lines.append("DATASET UNSTRUCTURED_GRID")
    lines.append(f"POINTS {n_nodes} float")
    for p in points:
        lines.append(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
    lines.append(f"CELLS {len(elements)} {cell_size_total}")
    for idxs in cell_data_idxs:
        lines.append(f"{len(idxs)} " + " ".join(str(i) for i in idxs))
    lines.append(f"CELL_TYPES {len(elements)}")
    for ct in cell_types:
        lines.append(str(ct))

    # ---------- POINT_DATA
    # Always emit displacement; optionally reactions, mode shapes, and
    # user-supplied point fields.
    if n_nodes:
        lines.append(f"POINT_DATA {n_nodes}")
        _write_vector(lines, "displacement", _gather_vector(nodes, "disp", ndm))
        if include_reactions:
            _write_vector(lines, "reaction",
                            _gather_vector(nodes, "reaction", ndm))
        if include_mode_shapes:
            n_modes = max(
                (n.mode_disp.shape[1] if n.mode_disp.ndim == 2 else 0)
                for n in nodes
            )
            for k in range(n_modes):
                vec = np.zeros((n_nodes, 3))
                for i, n in enumerate(nodes):
                    if n.mode_disp.ndim == 2 and k < n.mode_disp.shape[1]:
                        vec[i, :ndm] = n.mode_disp[:ndm, k]
                _write_vector(lines, f"mode_{k + 1}", vec)
        if point_data:
            for name, arr in point_data.items():
                _write_point_field(lines, name, arr, n_nodes)

    # ---------- CELL_DATA
    if cell_data and elements:
        lines.append(f"CELL_DATA {len(elements)}")
        for name, arr in cell_data.items():
            _write_cell_field(lines, name, arr, len(elements))

    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- helpers

def _gather_vector(nodes, attr: str, ndm: int) -> np.ndarray:
    """Stack a 3-component array for each node from ``Node.<attr>``."""
    out = np.zeros((len(nodes), 3))
    for i, n in enumerate(nodes):
        arr = getattr(n, attr)
        out[i, :ndm] = arr[:ndm]
    return out


def _write_vector(lines: list[str], name: str, vec: np.ndarray) -> None:
    lines.append(f"VECTORS {name} float")
    for v in vec:
        lines.append(f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")


def _write_scalar(lines: list[str], name: str, vec: np.ndarray) -> None:
    lines.append(f"SCALARS {name} float 1")
    lines.append("LOOKUP_TABLE default")
    for v in vec:
        lines.append(f"{float(v):.9g}")


def _write_point_field(lines, name, arr, n_nodes):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size != n_nodes:
            raise ValueError(
                f"point_data['{name}']: expected length {n_nodes}, "
                f"got {arr.size}"
            )
        _write_scalar(lines, name, arr)
    else:
        if arr.shape[0] != n_nodes:
            raise ValueError(
                f"point_data['{name}']: first dim must be {n_nodes}, "
                f"got {arr.shape[0]}"
            )
        pad = np.zeros((n_nodes, 3))
        pad[:, :min(3, arr.shape[1])] = arr[:, :min(3, arr.shape[1])]
        _write_vector(lines, name, pad)


def _write_cell_field(lines, name, arr, n_cells):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size != n_cells:
            raise ValueError(
                f"cell_data['{name}']: expected length {n_cells}, "
                f"got {arr.size}"
            )
        _write_scalar(lines, name, arr)
    else:
        if arr.shape[0] != n_cells:
            raise ValueError(
                f"cell_data['{name}']: first dim must be {n_cells}, "
                f"got {arr.shape[0]}"
            )
        pad = np.zeros((n_cells, 3))
        pad[:, :min(3, arr.shape[1])] = arr[:, :min(3, arr.shape[1])]
        _write_vector(lines, name, pad)
