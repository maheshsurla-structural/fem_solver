"""Coupling-beam helper for coupled shear-wall buildings.

A coupling beam in a coupled shear-wall system connects the two walls
at floor levels. In a macro model:

* Each wall is a vertical BeamColumn2D running through the wall
  centroid.
* The coupling beam must connect at the wall **face** (not the
  centroid), so a rigid offset is inserted from each wall's centroid
  node to a new face node, and the beam spans the face-to-face gap.

The geometry::

        +-----------+        +-----------+
        | wall 1    |        | wall 2    |
        |   o-------|========|-------o   |
        |   |       |        |       |   |
        +-----------+        +-----------+
            ^         ^       ^      ^
            centroid  face    face   centroid
            (model    (new    (new   (model
             node)     node)   node)  node)

This module's :func:`add_coupling_beam_2d` adds the two face nodes,
the two rigid-link constraints, and the coupling-beam element to a
model in one call, returning the new tags so the caller can iterate.

The coupling beam can be elastic (simple BeamColumn2D with
``ElasticSection2D`` or area+Iz) or nonlinear (passing in a
:class:`FiberSection2D`). For seismic-design coupling beams,
diagonally-reinforced or steel-plate-wrapped beams are typical -- use
a fiber section with appropriate hinge zones for those.
"""
from __future__ import annotations

from dataclasses import dataclass

from femsolver.constraints.rigid_link import RigidLink
from femsolver.elements.beam import BeamColumn2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.sections.response.base import SectionBase


@dataclass
class CouplingBeamResult:
    """Tags created by :func:`add_coupling_beam_2d`.

    Attributes
    ----------
    face_node_1, face_node_2 : int
        New nodes at wall 1's right face and wall 2's left face.
    rigid_link_1, rigid_link_2 : RigidLink
        Constraint objects added to the model.
    beam_element : int
        Tag of the coupling-beam element.
    """

    face_node_1: int
    face_node_2: int
    rigid_link_1: RigidLink
    rigid_link_2: RigidLink
    beam_element: int


def add_coupling_beam_2d(
    model,
    *,
    centroid_node_1: int,
    centroid_node_2: int,
    L_w1: float, L_w2: float,
    material: ElasticIsotropic,
    next_node_tag: int,
    next_element_tag: int,
    section: SectionBase | None = None,
    A: float | None = None,
    Iz: float | None = None,
) -> CouplingBeamResult:
    """Add a coupling beam between two wall centroids at the SAME elevation.

    Parameters
    ----------
    model : Model
        The 2D frame model (ndm=2, ndf=3).
    centroid_node_1, centroid_node_2 : int
        Wall 1 and wall 2 centroid nodes at this elevation. Must
        already exist in the model.
    L_w1, L_w2 : float
        Wall 1 and wall 2 plan lengths (m). Used to compute the offset
        from centroid to face: ``L_w / 2`` each. Wall 1 is assumed to
        the LEFT of wall 2 (positive x is from wall 1 toward wall 2).
    material : ElasticIsotropic
        Material for the beam (only used if ``section`` is None).
    next_node_tag : int
        First free node tag (face nodes use this and the next).
    next_element_tag : int
        First free element tag (the coupling beam uses this).
    section : SectionBase, optional
        For a nonlinear coupling beam, pass a fiber section. If
        omitted, an elastic beam is built from ``A`` and ``Iz``.
    A, Iz : float, optional
        For an elastic beam.

    Returns
    -------
    CouplingBeamResult
    """
    if model.ndm != 2 or model.ndf != 3:
        raise ValueError(
            f"add_coupling_beam_2d requires ndm=2 ndf=3, got "
            f"ndm={model.ndm} ndf={model.ndf}"
        )
    n1 = model.node(centroid_node_1)
    n2 = model.node(centroid_node_2)
    x1, y1 = n1.coords[0], n1.coords[1]
    x2, y2 = n2.coords[0], n2.coords[1]
    if abs(y1 - y2) > 1.0e-6:
        raise ValueError(
            f"coupling-beam centroid nodes must be at the same elevation, "
            f"got y1={y1}, y2={y2}"
        )
    if x2 <= x1:
        raise ValueError(
            f"wall 2 centroid (x={x2}) must be to the RIGHT of wall 1 "
            f"(x={x1})"
        )

    # Face node positions (wall 1's right face, wall 2's left face)
    x_face_1 = x1 + 0.5 * L_w1
    x_face_2 = x2 - 0.5 * L_w2
    if x_face_2 <= x_face_1:
        raise ValueError(
            f"walls overlap or touch: face_1_x={x_face_1} >= "
            f"face_2_x={x_face_2}"
        )

    # Add face nodes
    face_node_1 = next_node_tag
    face_node_2 = next_node_tag + 1
    model.add_node(face_node_1, x_face_1, y1)
    model.add_node(face_node_2, x_face_2, y2)

    # Add rigid-link constraints (centroid -> face)
    link_1 = RigidLink(retained=centroid_node_1,
                        constrained=face_node_1, kind="beam")
    link_2 = RigidLink(retained=centroid_node_2,
                        constrained=face_node_2, kind="beam")
    model.add_mp_constraint(link_1)
    model.add_mp_constraint(link_2)

    # Build the beam element
    if section is not None:
        beam = BeamColumn2D(next_element_tag,
                            (face_node_1, face_node_2),
                            material, section=section)
    else:
        if A is None or Iz is None:
            raise ValueError(
                "for an elastic coupling beam, supply A and Iz "
                "(or pass a fiber section)"
            )
        beam = BeamColumn2D(next_element_tag,
                            (face_node_1, face_node_2),
                            material, A, Iz)
    model.add_element(beam)

    return CouplingBeamResult(
        face_node_1=face_node_1,
        face_node_2=face_node_2,
        rigid_link_1=link_1,
        rigid_link_2=link_2,
        beam_element=next_element_tag,
    )
