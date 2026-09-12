"""High-level RC fiber-section builders.

These assemble ready-to-use :class:`FiberSection2D` / :class:`FiberSection3D`
objects for common reinforced-concrete shapes, wiring together the low-level
circular meshing in :mod:`femsolver.sections.response.fiber` with distinct
core / cover concrete and a discrete longitudinal-bar ring.

The workhorse here is :func:`rc_circular_column_section` — the one-liner for
a Caltrans-style circular bridge column (confined core + unconfined cover +
perimeter bar ring), the section at the heart of the fiber-hinge benchmark
(see ``docs/source/fiber_hinge_implementation_plan.md``).

Materials are passed in as :class:`UniaxialMaterial` objects (cloned per
fiber), so the confinement model is decoupled from the geometry: the caller
supplies a confined concrete law for the core and an unconfined one for the
cover (the Mander confinement pre-processor that derives the confined
``f'cc, eps_cc`` is a separate concern — plan phase P2).
"""
from __future__ import annotations

import math

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.sections.response.fiber import (
    Fiber,
    FiberSection2D,
    FiberSection3D,
    circular_sector_fibers,
)


def rc_circular_column_section(
    *,
    diameter: float,
    cover: float,
    core_concrete: UniaxialMaterial,
    cover_concrete: UniaxialMaterial,
    n_bars: int,
    bar_area: float,
    steel: UniaxialMaterial,
    bar_circle_diameter: float | None = None,
    n_core_rings: int = 8,
    n_cover_rings: int = 2,
    n_wedges: int = 24,
    subtract_rebar_from_core: bool = True,
    three_d: bool = False,
    GJ: float | None = None,
    centroid_y: float = 0.0,
    centroid_z: float = 0.0,
) -> FiberSection2D | FiberSection3D:
    """Assemble a circular RC column fiber section.

    Layout (concentric, from the centre out):

    * **Confined core** — a solid disc of radius ``D/2 - cover`` meshed
      ``n_core_rings x n_wedges`` with ``core_concrete``.
    * **Unconfined cover** — the annulus from ``D/2 - cover`` to ``D/2``
      meshed ``n_cover_rings x n_wedges`` with ``cover_concrete``.
    * **Longitudinal bars** — ``n_bars`` point fibers of area ``bar_area``,
      equally spaced on a circle of diameter ``bar_circle_diameter``
      (default: the core boundary), each with the ``steel`` law.

    Parameters
    ----------
    diameter : float
        Overall column diameter ``D``.
    cover : float
        Cover thickness measured from the outer face to the core boundary
        (so the confined core diameter is ``D - 2*cover``). See plan §10 O2
        for the exact reference; for the benchmark this reproduces the CSI
        ``CnfDiam``.
    core_concrete, cover_concrete : UniaxialMaterial
        Confined (core) and unconfined (cover) concrete laws.
    n_bars : int
        Number of longitudinal bars (>= 1; >= 2 for a symmetric ring).
    bar_area : float
        Area of one longitudinal bar.
    steel : UniaxialMaterial
        Reinforcing-steel law.
    bar_circle_diameter : float, optional
        Diameter of the bar-centroid circle. Defaults to the core diameter
        ``D - 2*cover`` (bars on the core edge). Pass the real value for
        fidelity.
    n_core_rings, n_cover_rings, n_wedges : int
        Mesh density. Defaults ``8 / 2 / 24`` mirror CSI's cylindrical fiber
        mesh (24 circumferential x 8 radial in the core).
    subtract_rebar_from_core : bool, default True
        If True, uniformly scale the core-concrete fiber areas down by
        ``(A_core_gross - A_steel) / A_core_gross`` so the concrete does not
        double-count the area occupied by the bars, keeping the total
        (concrete + steel) area equal to the gross section area exactly.
    three_d : bool, default False
        Build a :class:`FiberSection3D` (needs ``GJ``) instead of 2-D.
    GJ : float, optional
        Torsional stiffness, required when ``three_d=True``.
    centroid_y, centroid_z : float
        Section-centre coordinates.

    Returns
    -------
    FiberSection2D or FiberSection3D
    """
    if diameter <= 0.0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (0.0 < cover < 0.5 * diameter):
        raise ValueError(
            f"cover must satisfy 0 < cover < D/2, got cover={cover}, "
            f"D={diameter}"
        )
    if n_bars < 1:
        raise ValueError(f"need at least 1 bar, got {n_bars}")
    if bar_area <= 0.0:
        raise ValueError(f"bar_area must be positive, got {bar_area}")
    if three_d and (GJ is None or GJ <= 0.0):
        raise ValueError("three_d=True requires a positive GJ")

    R = 0.5 * diameter
    core_r = R - cover

    core_fibers = circular_sector_fibers(
        0.0, core_r, n_core_rings, n_wedges, core_concrete,
        centroid_y=centroid_y, centroid_z=centroid_z,
    )
    cover_fibers = circular_sector_fibers(
        core_r, R, n_cover_rings, n_wedges, cover_concrete,
        centroid_y=centroid_y, centroid_z=centroid_z,
    )

    a_steel = n_bars * bar_area
    if subtract_rebar_from_core:
        a_core_gross = math.pi * core_r * core_r
        if a_steel >= a_core_gross:
            raise ValueError(
                f"total steel area ({a_steel:g}) >= gross core area "
                f"({a_core_gross:g}); cannot subtract rebar from core"
            )
        scale = (a_core_gross - a_steel) / a_core_gross
        for f in core_fibers:
            f.area *= scale

    bar_r = 0.5 * bar_circle_diameter if bar_circle_diameter is not None else core_r
    if not (0.0 < bar_r < R):
        raise ValueError(
            f"bar circle radius must satisfy 0 < r < D/2, got {bar_r}"
        )
    rebar_fibers = [
        Fiber(
            y=centroid_y + bar_r * math.sin(2.0 * math.pi * k / n_bars),
            z=centroid_z + bar_r * math.cos(2.0 * math.pi * k / n_bars),
            area=bar_area,
            material=steel.clone(),
        )
        for k in range(n_bars)
    ]

    fibers = core_fibers + cover_fibers + rebar_fibers
    if three_d:
        return FiberSection3D(fibers, GJ=GJ)
    return FiberSection2D(fibers)
