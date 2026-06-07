"""Unified Section ABC -- the canonical cross-section type (Theme II.2).

A :class:`Section` is the single source of truth for a beam-column
cross-section. It carries identity (name, family, catalogue ref),
geometry (a :class:`~femsolver.sections.geometry.Geometry`), one or
more :class:`~femsolver.materials.MaterialReference`, and optional
reinforcement (rebar layout, tendon layout). From this single object,
**lazy adapter methods** produce:

* analysis-side sections (:class:`ElasticSection2D/3D`,
  :class:`FiberSection2D/3D`) for beam elements
* design-side sections (RC, steel-W, EC2, IS456) for code checks
* JSON for save/load, SVG/DXF for visualization, BOM data

The design promise: build the Section ONCE. It flows everywhere.

The existing :class:`~femsolver.sections.response.base.SectionBase` is the
*low-level* analysis interface (``get_response(e) -> (s, ks)``).
:class:`Section` is the *high-level* canonical object that PRODUCES
``SectionBase`` instances on demand. The two coexist; nothing breaks.

This module is the Theme II.2 foundation. Concrete subclasses
(``ParametricSection``, ``CataloguedSection``, ``CustomSection``,
``RcSection``, ``CompositeSection``, ``PsCSection``, etc.) arrive in
sub-phases II.3-II.6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from femsolver.sections.geometry import Geometry


# ============================================================ MaterialZone

@dataclass
class MaterialZone:
    """A region of the cross-section assigned to one material.

    For a simple homogeneous section (a steel I-beam, a plain concrete
    rectangle), a :class:`Section` carries one :class:`MaterialZone`
    spanning the entire :class:`Geometry`. For composite sections
    (girder + deck, hollow tube filled with concrete), the section
    carries a list of :class:`MaterialZone` with each entry restricted
    to a sub-polygon of the section.

    Attributes
    ----------
    material : Any
        Reference to a material object (concrete, steel, etc.) -- left
        intentionally untyped here so this module does not import the
        whole materials package. Downstream code introspects what it
        needs (E, fy, fc, etc.).
    geometry : Geometry, optional
        Sub-polygon this material covers. ``None`` means "the whole
        parent Section's geometry".
    name : str
        Descriptive name ("girder concrete", "deck concrete", "flange",
        "web", ...).
    """
    material: Any
    geometry: Optional[Geometry] = None
    name: str = ""


# ============================================================ Reinforcement

@dataclass
class RebarBar:
    """A single rebar at a point.

    Coordinates ``(z, y)`` are in section-local frame, measured from
    the section centroid (NOT from any cover face). ``area`` in m^2,
    ``material`` is a reference to a steel material (with at least an
    ``fy`` attribute).
    """
    z: float
    y: float
    area: float
    material: Any = None
    designation: str = ""    # e.g. "#5", "T20"


@dataclass
class ReinforcementLayout:
    """A collection of rebars in a cross-section.

    This is the new unified reinforcement carrier. The legacy
    :class:`~femsolver.design.concrete.section.RebarLayout` (which
    describes rebars by layer + cover) becomes one factory that
    emits a :class:`ReinforcementLayout` with explicit bar positions.

    Attributes
    ----------
    bars : list[RebarBar]
        All bars in the section, with explicit (z, y, area).
    stirrup_designation : str
        Stirrup bar designation (e.g. "#3", "T10").
    stirrup_spacing : float
        Center-to-center stirrup spacing along the beam axis (m).
    stirrup_legs : int
        Number of vertical legs in a stirrup.
    """
    bars: list[RebarBar] = field(default_factory=list)
    stirrup_designation: str = "#3"
    stirrup_spacing: float = 0.150
    stirrup_legs: int = 2

    @property
    def total_area(self) -> float:
        return float(sum(b.area for b in self.bars))

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    # ----------------------------------------------------- factories
    @classmethod
    def from_rectangular_layers(
        cls,
        *,
        b: float,
        h: float,
        bottom_bars: list[tuple[float, float]] | None = None,
        top_bars: list[tuple[float, float]] | None = None,
        side_bars: list[tuple[float, float, float]] | None = None,
        bottom_cover: float = 0.040,
        top_cover: float = 0.040,
        steel_material: Any = None,
        stirrup_designation: str = "#3",
        stirrup_spacing: float = 0.150,
        stirrup_legs: int = 2,
    ) -> "ReinforcementLayout":
        """Build a layout from layer descriptions on a rectangular cross-section.

        Coordinate convention: section centered at origin; ``y`` is
        the vertical (height) direction. The rectangle spans
        ``y in [-h/2, +h/2]`` and ``z in [-b/2, +b/2]``.

        Bars are described as ``(area_m2, designation)`` per bar; the
        layer of N bars is placed at y = +h/2 - top_cover (or -h/2 +
        bottom_cover), distributed uniformly in z over the inner clear
        width.

        Parameters
        ----------
        b, h : float
            Section width (z-extent) and depth (y-extent), m.
        bottom_bars : list of (area_m2, designation), optional
            Bars in the bottom layer.
        top_bars : list of (area_m2, designation), optional
            Bars in the top layer.
        side_bars : list of (area_m2, designation, y_position), optional
            Optional side-face bars at explicit y positions.
        bottom_cover, top_cover : float
            Clear cover to bar centroid (m).
        steel_material : optional
            Material assigned to every bar (typically a Grade 60 /
            Grade 500 steel).
        stirrup_* : standard stirrup parameters.
        """
        bars: list[RebarBar] = []

        def _distribute(layer, y):
            if not layer:
                return
            n = len(layer)
            # Distribute across inner clear width [-b/2+cover, +b/2-cover]
            # Use side-cover ~= bottom_cover as default
            edge = b / 2.0 - bottom_cover
            if n == 1:
                z_positions = [0.0]
            else:
                z_positions = [
                    -edge + 2 * edge * i / (n - 1) for i in range(n)
                ]
            for (area, desig), z in zip(layer, z_positions):
                bars.append(RebarBar(
                    z=z, y=y, area=float(area),
                    material=steel_material, designation=str(desig),
                ))

        if bottom_bars:
            _distribute(bottom_bars, -h / 2.0 + bottom_cover)
        if top_bars:
            _distribute(top_bars, +h / 2.0 - top_cover)
        if side_bars:
            for area, desig, y in side_bars:
                bars.append(RebarBar(
                    z=0.0, y=float(y), area=float(area),
                    material=steel_material, designation=str(desig),
                ))

        return cls(
            bars=bars,
            stirrup_designation=stirrup_designation,
            stirrup_spacing=stirrup_spacing,
            stirrup_legs=stirrup_legs,
        )

    @classmethod
    def from_perimeter(
        cls,
        section,
        *,
        n_bars: int,
        bar_area: float,
        cover: float,
        material: Any = None,
        designation: str = "",
        include_holes: bool = False,
        n_bars_per_hole: int = 0,
        stirrup_designation: str = "#3",
        stirrup_spacing: float = 0.150,
        stirrup_legs: int = 2,
    ) -> "ReinforcementLayout":
        """Auto-distribute ``n_bars`` evenly around the perimeter of an
        **arbitrary** polygon section, at a clear ``cover`` to the bar
        centroid.

        The outer boundary is offset inward by ``cover`` (``shapely``
        ``buffer(-cover)``) and ``n_bars`` are placed at equal
        arc-length spacing around that offset ring -- works for any
        outline (I, box, L, custom polygon). With ``include_holes`` the
        same is done around each interior void (offset *into* the
        material by ``cover``), with ``n_bars_per_hole`` bars per hole.

        Parameters
        ----------
        section : Section
            Section whose ``geometry.polygon`` defines the outline (and
            any holes).
        n_bars : int
            Number of bars around the outer perimeter.
        bar_area : float
            Area of each bar (m²).
        cover : float
            Clear cover from the boundary to the bar centroid (m).
        material : optional
            Steel material assigned to every bar.
        designation : str
        include_holes : bool, default False
            Also ring each interior hole with reinforcement.
        n_bars_per_hole : int
            Bars per hole when ``include_holes`` is True.
        stirrup_* : standard stirrup parameters.

        Notes
        -----
        Bars are spaced uniformly by arc length around the offset ring.
        For corner-critical columns where a bar must sit exactly at each
        vertex, place those bars explicitly instead.
        """
        from shapely.geometry import Polygon

        if n_bars < 1:
            raise ValueError("n_bars must be >= 1")
        if bar_area <= 0 or cover <= 0:
            raise ValueError("bar_area and cover must be > 0")
        poly = section.geometry.polygon

        def _largest(geom):
            if geom.is_empty:
                raise ValueError(
                    "cover is too large -- the offset region vanished"
                )
            if geom.geom_type == "MultiPolygon":
                return max(geom.geoms, key=lambda g: g.area)
            return geom

        bars: list[RebarBar] = []

        def _ring_bars(ring, n: int) -> None:
            L = ring.length
            for i in range(n):
                p = ring.interpolate((i / n) * L)
                bars.append(RebarBar(
                    z=float(p.x), y=float(p.y), area=float(bar_area),
                    material=material, designation=designation,
                ))

        inner = _largest(poly.buffer(-cover))
        _ring_bars(inner.exterior, n_bars)

        if include_holes and n_bars_per_hole > 0:
            for interior in poly.interiors:
                hole = Polygon(interior)
                grown = _largest(hole.buffer(cover))   # ring into material
                _ring_bars(grown.exterior, n_bars_per_hole)

        return cls(
            bars=bars,
            stirrup_designation=stirrup_designation,
            stirrup_spacing=stirrup_spacing,
            stirrup_legs=stirrup_legs,
        )


@dataclass
class PrestressTendon:
    """A single prestressing tendon at a point.

    Coordinates ``(z, y)`` are in section-local frame, measured from
    the section centroid (matching :class:`RebarBar`).

    ``f_pe`` is the **effective prestress AFTER all losses** (friction,
    anchorage slip, elastic shortening, creep, shrinkage, relaxation).
    This is converted internally to an initial pre-strain
    ``epsilon_pe = f_pe / E_p`` that the strand material sees as a
    strain offset: when concrete strain at the tendon location is
    zero, the strand already carries stress ``f_pe`` in tension.

    Attributes
    ----------
    z, y : float
        Tendon centroid position (m).
    area : float
        Total cross-sectional area of strand at this point (m^2).
    material : UniaxialMaterial
        Strand material (Grade 1860 / Grade 270 typically). Provides
        the Ramberg-Osgood or bilinear stress-strain curve.
    f_pe : float
        Effective prestress AFTER losses (Pa, tension-positive
        magnitude).
    bonded : bool, default True
        ``True`` for bonded post-tensioned / pre-tensioned tendons --
        the tendon follows concrete strain compatibility. ``False``
        for unbonded post-tensioned tendons -- strain depends on
        member-level deformation; not handled at the section level.
    designation : str
        Strand label (e.g. "12T15", "0.6in Gr270").
    """
    z: float
    y: float
    area: float
    material: Any = None
    f_pe: float = 0.0
    bonded: bool = True
    designation: str = ""


@dataclass
class TendonLayout:
    """Prestressing tendons in a PSC cross-section.

    Each :class:`PrestressTendon` carries position, area, strand
    material, and effective prestress after losses.
    """
    tendons: list[PrestressTendon] = field(default_factory=list)

    @property
    def total_area(self) -> float:
        return float(sum(t.area for t in self.tendons))

    @property
    def total_prestress_force(self) -> float:
        """Sum of ``A_p · f_pe`` over all tendons (N, tension-positive)."""
        return float(sum(t.area * t.f_pe for t in self.tendons))

    @property
    def n_tendons(self) -> int:
        return len(self.tendons)


# ============================================================ Section ABC

class Section:
    """Unified, canonical cross-section object.

    Subclasses populate ``geometry``, ``zones`` (one or more
    :class:`MaterialZone`), and optionally ``reinforcement`` /
    ``prestress``. The base class provides lazy adapter methods that
    produce :class:`SectionBase` instances on demand.

    Identity
    --------
    ``name``      User-visible name ("B1 600x300", "C1 W14x90")
    ``family``    Geometric family ("rect" | "I" | "channel" | "T" |
                  "L" | "hollow_rect" | "circular" | "polygon" | ...)
    ``catalogue_ref``  Catalogue identifier ("W14x90" | "IPE300") if
                  this Section comes from a catalogue, else ``None``.

    Composition slots
    -----------------
    ``geometry``       The outer :class:`Geometry`.
    ``zones``          List of :class:`MaterialZone` -- one entry for
                       a homogeneous section, multiple for composite.
    ``reinforcement``  Optional :class:`ReinforcementLayout`.
    ``prestress``      Optional :class:`TendonLayout`.
    """

    # ----------------------------------------------------------- identity
    name: str = ""
    family: str = ""
    catalogue_ref: Optional[str] = None

    def __init__(
        self,
        geometry: Geometry,
        zones: Optional[list[MaterialZone]] = None,
        *,
        name: str = "",
        family: str = "",
        catalogue_ref: Optional[str] = None,
        reinforcement: Optional[ReinforcementLayout] = None,
        prestress: Optional[TendonLayout] = None,
    ):
        self.geometry = geometry
        self.zones: list[MaterialZone] = list(zones) if zones else []
        self.name = str(name)
        self.family = str(family)
        self.catalogue_ref = catalogue_ref
        self.reinforcement = reinforcement
        self.prestress = prestress

    # ----------------------------------------------------------- gross props
    @property
    def area(self) -> float:
        return self.geometry.area

    @property
    def I_zz(self) -> float:
        return self.geometry.I_zz

    @property
    def I_yy(self) -> float:
        return self.geometry.I_yy

    @property
    def J(self) -> float:
        return self.geometry.J

    @property
    def centroid(self) -> tuple[float, float]:
        return self.geometry.centroid

    # ----------------------------------------------------------- primary material
    @property
    def primary_material(self) -> Any:
        """The "main" material -- first zone's material, or None.

        Catalogued steel sections and parametric concrete sections both
        have a single material. Composite sections override this with
        their own logic.
        """
        if not self.zones:
            return None
        return self.zones[0].material

    # ----------------------------------------------------------- adapters (II.6)
    def elastic_section_2d(self):
        """Produce a :class:`ElasticSection2D` from this Section.

        Implemented in II.6. The default implementation tries to read
        ``E`` from :attr:`primary_material` and uses ``EA, EIz`` from
        the geometry. Composite / fiber sections override.
        """
        from femsolver.sections.response.elastic import ElasticSection2D

        mat = self.primary_material
        if mat is None:
            raise ValueError(
                f"Section {self.name!r} has no material; cannot build "
                f"ElasticSection2D"
            )
        E = _resolve_E(mat)
        return ElasticSection2D(E=E, A=self.area, Iz=self.I_zz)

    def elastic_section_3d(self):
        """Produce a :class:`ElasticSection3D`."""
        from femsolver.sections.response.elastic import ElasticSection3D

        mat = self.primary_material
        if mat is None:
            raise ValueError(
                f"Section {self.name!r} has no material; cannot build "
                f"ElasticSection3D"
            )
        E = _resolve_E(mat)
        G = _resolve_G(mat, E)
        return ElasticSection3D(
            E=E, G=G, A=self.area,
            Iy=self.I_yy, Iz=self.I_zz, J=max(self.J, 1e-30),
        )

    # ----------------------------------------------------------- fiber adapters (II.6)
    def fiber_section_2d(
        self,
        *,
        material=None,
        n_z: int = 20,
        n_y: int = 20,
    ):
        """Auto-discretize the polygon into a :class:`FiberSection2D`.

        The polygon's bounding box is gridded into ``n_z x n_y`` cells;
        each cell that intersects the polygon contributes a Fiber whose
        ``(z, y)`` is the centroid of the intersection and whose
        ``area`` is the intersection area. Rebar from :attr:`reinforcement`
        is added as additional fibers.

        Parameters
        ----------
        material : UniaxialMaterial, optional
            Material assigned to every concrete fiber. If ``None``, uses
            the primary material (which must be a
            :class:`UniaxialMaterial`).
        n_z, n_y : int
            Grid resolution along z (width) and y (depth).
        """
        from femsolver.sections.response.fiber import Fiber, FiberSection2D
        from femsolver.materials.uniaxial.base import UniaxialMaterial

        concrete_mat = material if material is not None else self.primary_material
        if concrete_mat is None:
            raise ValueError(
                f"Section {self.name!r}: need a UniaxialMaterial via "
                f"`material=...` to build fibers"
            )
        if not isinstance(concrete_mat, UniaxialMaterial):
            raise TypeError(
                f"fiber discretization requires a UniaxialMaterial; "
                f"got {type(concrete_mat).__name__}"
            )

        fibers = _discretize_polygon_to_fibers(
            self.geometry.polygon, concrete_mat, n_z=n_z, n_y=n_y,
        )

        # Append rebar fibers (over-counting the small concrete-bar
        # overlap is a standard simplification in fiber-section practice;
        # see Mazzoni et al. OpenSees Manual section "Fiber Section").
        if self.reinforcement and self.reinforcement.bars:
            for bar in self.reinforcement.bars:
                bar_mat = bar.material if bar.material is not None else None
                if bar_mat is None or not isinstance(bar_mat, UniaxialMaterial):
                    raise ValueError(
                        f"rebar at (z={bar.z}, y={bar.y}) lacks a "
                        f"UniaxialMaterial; cannot add fiber"
                    )
                fibers.append(Fiber(
                    y=bar.y, z=bar.z, area=bar.area,
                    material=bar_mat.clone(),
                ))

        return FiberSection2D(fibers)

    def fiber_section_3d(
        self,
        *,
        material=None,
        n_z: int = 20,
        n_y: int = 20,
        GJ: float | None = None,
    ):
        """Auto-discretize into a :class:`FiberSection3D`.

        ``GJ`` defaults to ``G_steel * self.J`` if the primary material
        exposes ``G``, otherwise uses an effective G derived from
        ``E`` and ``nu`` on the primary material. If neither is
        available, the user must pass ``GJ`` explicitly.
        """
        from femsolver.sections.response.fiber import Fiber, FiberSection3D
        from femsolver.materials.uniaxial.base import UniaxialMaterial

        concrete_mat = material if material is not None else self.primary_material
        if concrete_mat is None:
            raise ValueError(
                f"Section {self.name!r}: need a UniaxialMaterial via "
                f"`material=...` to build fibers"
            )
        if not isinstance(concrete_mat, UniaxialMaterial):
            raise TypeError(
                f"fiber discretization requires a UniaxialMaterial; "
                f"got {type(concrete_mat).__name__}"
            )

        # Resolve GJ
        if GJ is None:
            # Try to read G from the primary high-level material (not
            # the uniaxial). If unavailable, fall back to nu=0.3.
            ref_mat = self.primary_material
            if ref_mat is not None and hasattr(ref_mat, "G"):
                G = float(getattr(ref_mat, "G"))
            elif ref_mat is not None and hasattr(ref_mat, "E"):
                E = _resolve_E(ref_mat)
                G = _resolve_G(ref_mat, E)
            else:
                # Bare uniaxial with no nu: assume 0.3 around its E
                # via .get_response(0) tangent
                _, Et = concrete_mat.get_response(0.0)
                G = Et / (2 * 1.3)
            GJ = G * max(self.J, 1e-30)

        fibers = _discretize_polygon_to_fibers(
            self.geometry.polygon, concrete_mat, n_z=n_z, n_y=n_y,
        )
        if self.reinforcement and self.reinforcement.bars:
            for bar in self.reinforcement.bars:
                bar_mat = bar.material
                if bar_mat is None or not isinstance(bar_mat, UniaxialMaterial):
                    raise ValueError(
                        f"rebar at (z={bar.z}, y={bar.y}) lacks a "
                        f"UniaxialMaterial"
                    )
                fibers.append(Fiber(
                    y=bar.y, z=bar.z, area=bar.area,
                    material=bar_mat.clone(),
                ))

        return FiberSection3D(fibers, GJ=float(GJ))

    # ----------------------------------------------------------- design adapters (II.6)
    def as_aisc_section(self):
        """Return the legacy :class:`SteelSection` (AISC W-shape) for
        this section, looked up by :attr:`catalogue_ref`.

        Raises if this section is not an AISC-catalogued section.
        """
        if not self.catalogue_ref:
            raise ValueError(
                f"Section {self.name!r} has no catalogue_ref; not an "
                f"AISC-catalogued section"
            )
        from femsolver.design.steel.sections import get_section
        return get_section(self.catalogue_ref)

    def as_eurocode_section(self):
        """Return the legacy EC :class:`SectionProperties` by
        catalogue_ref (IPE / HEA / HEB)."""
        if not self.catalogue_ref:
            raise ValueError(
                f"Section {self.name!r} has no catalogue_ref; not an "
                f"Eurocode-catalogued section"
            )
        from femsolver.data.sections_ec import EC_HEA, EC_HEB, EC_IPE
        for table in (EC_IPE, EC_HEA, EC_HEB):
            if self.catalogue_ref in table:
                return table[self.catalogue_ref]
        raise KeyError(
            f"catalogue_ref {self.catalogue_ref!r} not in EC tables"
        )

    def as_indian_section(self):
        """Return the legacy IS :class:`SectionProperties` by
        catalogue_ref (ISMB / ISMC / ISA)."""
        if not self.catalogue_ref:
            raise ValueError(
                f"Section {self.name!r} has no catalogue_ref; not an "
                f"IS-catalogued section"
            )
        from femsolver.data.sections_is import IS_ISA, IS_ISMB, IS_ISMC
        for table in (IS_ISMB, IS_ISMC, IS_ISA):
            if self.catalogue_ref in table:
                return table[self.catalogue_ref]
        raise KeyError(
            f"catalogue_ref {self.catalogue_ref!r} not in IS tables"
        )

    def as_aci_concrete_section(self, *, concrete_material=None):
        """Return a legacy :class:`~femsolver.design.concrete.section.ConcreteSection`
        view of this section for use with ACI 318 design code.

        Requires this Section to be a rectangular RC section with a
        :class:`ReinforcementLayout`. Raises otherwise.

        Parameters
        ----------
        concrete_material : optional
            ``ConcreteMaterial`` (with ``fc_prime``, ``fy``). If
            None, the primary material must already be a
            ``ConcreteMaterial``.
        """
        from femsolver.design.concrete.section import (
            ConcreteMaterial,
            ConcreteSection,
            RebarLayout,
        )

        # Validate shape: this is a rectangular section
        if self.family != "rect":
            raise ValueError(
                f"as_aci_concrete_section requires family='rect'; "
                f"got {self.family!r}"
            )

        # Resolve concrete material
        cm = concrete_material if concrete_material is not None \
            else self.primary_material
        if cm is None:
            raise ValueError(
                f"Section {self.name!r}: need a ConcreteMaterial via "
                f"`concrete_material=...` or as primary material"
            )

        b = self.geometry.width
        h = self.geometry.depth

        # Build a RebarLayout from this section's reinforcement bars.
        # Heuristic: bars in the bottom half are "bottom_bars", in
        # the top half are "top_bars". Each gets its designation
        # passed through.
        bottom: list[str] = []
        top: list[str] = []
        bottom_y: list[float] = []
        top_y: list[float] = []
        if self.reinforcement:
            for bar in self.reinforcement.bars:
                if bar.y < 0:
                    bottom.append(bar.designation)
                    bottom_y.append(bar.y)
                else:
                    top.append(bar.designation)
                    top_y.append(bar.y)
        bottom_cover = (h / 2.0 + min(bottom_y)) if bottom_y else 0.040
        top_cover = (h / 2.0 - max(top_y)) if top_y else 0.040
        rl = RebarLayout(
            bottom_bars=tuple(bottom),
            top_bars=tuple(top),
            bottom_cover=bottom_cover,
            top_cover=top_cover,
            stirrup_designation=(
                self.reinforcement.stirrup_designation
                if self.reinforcement else "#3"
            ),
            stirrup_spacing=(
                self.reinforcement.stirrup_spacing
                if self.reinforcement else 0.15
            ),
            stirrup_legs=(
                self.reinforcement.stirrup_legs
                if self.reinforcement else 2
            ),
        )
        return ConcreteSection(b=b, h=h, material=cm, rebar=rl)

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        """Light dict representation suitable for logging / JSON.

        For full round-trip use :meth:`to_json` / :meth:`from_json`
        (Theme II.8)."""
        return {
            "type": type(self).__name__,
            "name": self.name,
            "family": self.family,
            "catalogue_ref": self.catalogue_ref,
            "area": self.area,
            "I_zz": self.I_zz,
            "I_yy": self.I_yy,
            "J": self.J,
            "n_zones": len(self.zones),
            "has_reinforcement": self.reinforcement is not None,
            "has_prestress": self.prestress is not None,
        }

    # ----------------------------------------------------------- JSON (II.8)
    def to_json(self, *, indent: int | None = 2) -> str:
        """Round-trip JSON encoding. See
        :mod:`femsolver.sections.serialization` for the schema."""
        from femsolver.sections.serialization import section_to_json
        return section_to_json(self, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Section":
        from femsolver.sections.serialization import section_from_json
        return section_from_json(text)

    # ----------------------------------------------------------- SVG (II.8)
    def to_svg(self, **kwargs) -> str:
        """Render this section as a standalone SVG string."""
        from femsolver.sections.visualization import section_to_svg
        return section_to_svg(self, **kwargs)

    # ----------------------------------------------------------- report (II.8)
    def section_report(self):
        """Build a :class:`SectionReport` for inclusion in HTML / PDF
        design reports."""
        from femsolver.sections.report import build_section_report
        return build_section_report(self)

    # ----------------------------------------------------------- BOM
    def weight_per_length(self) -> float:
        """Self-weight per unit beam length (N/m).

        For a single-material section: ``A * rho * g``. For composite
        sections: sum of zone areas times densities. Density is read
        from the material if it has a ``density`` (or ``rho``)
        attribute; otherwise assumes steel density (7850 kg/m^3).
        """
        g = 9.81
        if len(self.zones) == 0:
            # Geometry-only section -- treat as steel
            return self.area * 7850.0 * g
        if len(self.zones) == 1:
            rho = _resolve_density(self.zones[0].material, default=7850.0)
            return self.area * rho * g
        # Composite: each zone's geometry might be a sub-polygon
        total = 0.0
        for z in self.zones:
            rho = _resolve_density(z.material, default=7850.0)
            A = z.geometry.area if z.geometry is not None else self.area
            total += A * rho * g
        return total

    def paint_area_per_length(self) -> float:
        return self.geometry.paint_area_per_length()

    # ----------------------------------------------------------- repr
    def __repr__(self) -> str:
        ref = f" [{self.catalogue_ref}]" if self.catalogue_ref else ""
        return (
            f"{type(self).__name__}({self.name!r}{ref}, "
            f"family={self.family}, A={self.area:.4g} m^2, "
            f"I_zz={self.I_zz:.4g} m^4)"
        )


# ============================================================ material resolvers

def _resolve_E(material: Any) -> float:
    """Pull elastic modulus from a material reference.

    Looks for ``E``, then ``Ec`` (concrete), then ``Es`` (steel). Raises
    if none of them present."""
    for attr in ("E", "Ec", "Es", "modulus"):
        if hasattr(material, attr):
            val = getattr(material, attr)
            if val is not None and val > 0:
                return float(val)
    raise ValueError(
        f"material {material!r} has no E/Ec/Es attribute; "
        f"cannot build elastic section"
    )


def _resolve_G(material: Any, E: float) -> float:
    """Shear modulus. If material has ``G``, use it. Otherwise assume
    Poisson ratio 0.3."""
    if hasattr(material, "G"):
        G = getattr(material, "G")
        if G is not None and G > 0:
            return float(G)
    nu = float(getattr(material, "nu", 0.3))
    return E / (2.0 * (1.0 + nu))


def _resolve_density(material: Any, default: float = 7850.0) -> float:
    """Material density in kg/m^3. Defaults to steel if unknown."""
    for attr in ("density", "rho"):
        if hasattr(material, attr):
            v = getattr(material, attr)
            if v is not None and v > 0:
                return float(v)
    return default


# ============================================================ polygon discretization

def _discretize_polygon_to_fibers(polygon, material, *, n_z: int, n_y: int):
    """Grid-sample a shapely polygon into fibers.

    Returns a list of :class:`Fiber`. For each grid cell, the
    intersection with the polygon is computed via shapely; cells with
    non-trivial intersection contribute a fiber at the intersection
    centroid with the intersection area.
    """
    from shapely.geometry import box

    from femsolver.sections.response.fiber import Fiber

    minz, miny, maxz, maxy = polygon.bounds
    dz = (maxz - minz) / n_z
    dy = (maxy - miny) / n_y
    cell_area = dz * dy
    # Tolerance for "significant" intersection -- 0.1% of cell area
    eps = 1e-3 * cell_area

    fibers = []
    for iy in range(n_y):
        y0 = miny + iy * dy
        y1 = miny + (iy + 1) * dy
        for iz in range(n_z):
            z0 = minz + iz * dz
            z1 = minz + (iz + 1) * dz
            cell = box(z0, y0, z1, y1)
            inter = polygon.intersection(cell)
            A = inter.area
            if A < eps:
                continue
            c = inter.centroid
            fibers.append(Fiber(
                y=float(c.y), z=float(c.x),
                area=float(A), material=material.clone(),
            ))
    return fibers
