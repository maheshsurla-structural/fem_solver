"""The project — the serializable information model, the app's single source of truth.

A ``Project`` is a *declarative* description of the structure (nodes, members
and their sections + materials, supports, loads) plus metadata (name, units,
design code). It is what the app opens and saves. ``build_model()`` compiles
it into a transient ``femsolver.Model`` for analysis; results are never stored
in the project — they are recomputed from it.

This is the spine every future view (analysis, design, drawings, and the
desktop Section Designer) reads and writes. Today it compiles 2-D frames
(``BeamColumn2D``); element/section breadth grows from here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

SCHEMA = "femsolver-project/1"

# Element-tag offset for surface (Area) elements so they never collide with
# member ids (which are used directly as element tags). Comfortably above any
# realistic node/member count; each area owns a contiguous ``AREA_TAG_STRIDE``
# block for its mesh sub-elements (slab S3).
AREA_TAG_BASE = 2_000_000
AREA_TAG_STRIDE = 10_000
# Node-tag base for auto-created rigid-diaphragm master nodes (slab S8) — far
# above project + mesh-generated node ids so they never collide.
DIAPHRAGM_MASTER_NODE_BASE = 3_000_000
GRAVITY = 9.80665                    # m/s² — for area self-weight (slab S5)


def area_element_tag(area_id: int, k: int = 0) -> int:
    """Deterministic engine element tag for sub-element ``k`` of area
    ``area_id`` — so the load path (:meth:`Project._area_element_tags`) can name
    an area's mesh elements without carrying build state."""
    return AREA_TAG_BASE + area_id * AREA_TAG_STRIDE + k


def _cell_in_opening(openings, u, v) -> bool:
    """True if the parametric cell centre ``(u, v)`` lies in any opening rect
    ``(u0, v0, u1, v1)`` (wall plan W5)."""
    for (u0, v0, u1, v1) in openings:
        if u0 <= u <= u1 and v0 <= v <= v1:
            return True
    return False


def area_quad_cells(area) -> list:
    """``(k, i, j)`` for each *kept* mesh cell of a 4-node area — cells whose
    centre falls inside an opening (W5) are dropped. ``k = j·n1 + i`` is a stable
    tag index independent of which cells are dropped, so element tags never
    shift. Non-quad areas return ``[]``."""
    if len(area.nodes) != 4:
        return []
    n1 = max(1, int(area.mesh[0]))
    n2 = max(1, int(area.mesh[1]))
    ops = getattr(area, "openings", None) or []
    return [(j * n1 + i, i, j)
            for j in range(n2) for i in range(n1)
            if not _cell_in_opening(ops, (i + 0.5) / n1, (j + 0.5) / n2)]


def area_quad_needed_nodes(area) -> set:
    """The grid corners ``(i, j)`` used by at least one kept cell — the only
    mesh nodes to create, so openings leave no orphaned (singular) nodes."""
    needed = set()
    for (_k, i, j) in area_quad_cells(area):
        needed.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
    return needed


# Element-tag block for beam sub-elements produced when a member lying along a
# meshed slab edge is auto-split at the slab's edge nodes (beam-to-slab-edge
# compatibility, epics BE0-BE2). An *unsplit* member keeps its own id as the
# element tag; a split member's sub-elements take contiguous tags from this
# block so results/design can still be mapped back (BE3, not yet wired).
MEMBER_SPLIT_BASE = 4_000_000
MEMBER_SPLIT_STRIDE = 1_000


def member_element_tag(member_id: int, j: int = 0) -> int:
    """Deterministic engine element tag for sub-element ``j`` of a split member
    (BE2). Mirrors :func:`area_element_tag` so :meth:`Project.member_element_tags`
    can recover a member's mesh tags without build state."""
    return MEMBER_SPLIT_BASE + member_id * MEMBER_SPLIT_STRIDE + j


def decode_member_id(tag: int):
    """The project ``Member`` id an engine element tag belongs to (BE3): the tag
    itself for an unsplit member, or the parent member for a split sub-element
    tag. ``None`` for tags in the area/shell range. (Tag layout: member ids <
    ``AREA_TAG_BASE`` 2M ≤ area/shell < ``MEMBER_SPLIT_BASE`` 4M ≤ split members.)"""
    if tag is None:
        return None
    if tag >= MEMBER_SPLIT_BASE:                  # split sub-element
        return (tag - MEMBER_SPLIT_BASE) // MEMBER_SPLIT_STRIDE
    if tag >= AREA_TAG_BASE:                       # area / shell sub-element
        return None
    return tag                                     # unsplit member id == tag


def _coord_key(p, tol: float = 1.0e-6):
    """Rounded-coordinate key for merging coincident mesh nodes (µm tolerance)."""
    q = round(1.0 / tol)
    return (round(p[0] * q), round(p[1] * q), round(p[2] * q))


def _bilinear(corners, s: float, t: float):
    """Point on a 4-corner quad by bilinear map of the unit square: ``s`` runs
    along edge P0→P1, ``t`` along edge P0→P3. Corners are CCW ``[P0,P1,P2,P3]``
    (each an (x, y, z) tuple)."""
    p0, p1, p2, p3 = corners
    a = (1.0 - s) * (1.0 - t)
    b = s * (1.0 - t)
    c = s * t
    d = (1.0 - s) * t
    return tuple(a * p0[k] + b * p1[k] + c * p2[k] + d * p3[k] for k in range(3))


@dataclass
class Material:
    id: int
    name: str
    E: float
    nu: float = 0.3
    kind: str = "elastic_isotropic"
    rho: float = 0.0              # mass density (kg/m³) — self-mass for modal/RS
    fy: float = 345.0e6           # yield (A992 = 345 MPa) — for design checks
    fu: float = 448.0e6           # ultimate (A992 = 448 MPa)
    # Inelastic (fiber) constitutive parameters, per ``kind`` — e.g. concrete
    # {fc, eps_c0, eps_cu, Ec, fpcu_ratio}, steel {E, fy, fu, eps_sh, eps_su}.
    # Elastic materials leave this empty. Consumed by desktop.materials
    # (`uniaxial_law` / `stress_strain_curve`) to build the engine law.
    params: dict = field(default_factory=dict)


@dataclass
class Section:
    id: int
    name: str
    A: float
    Iz: float
    shape: str = ""               # AISC W-shape (e.g. "W12x65"); drives design
    Iy: float = 0.0               # 3-D weak-axis inertia (from shape if named)
    J: float = 0.0                # 3-D torsion constant (from shape if named)
    # A concrete/PSC/composite section authored in the General Section Designer.
    # When present, ``gsd_spec`` (a serialized ``section_gui_core.Spec``) is the
    # source of truth for A / Iz / Iy / J (see ``_resolve_section``) and carries
    # the reinforcement for future P-M-M member design. ``gsd_code`` is its
    # Section-Designer design code.
    gsd_spec: dict | None = None
    gsd_code: str = ""


# Shell / plate element formulations offered to an ``Area`` via its
# ``ShellSection``. Maps the user-facing "type" to the engine element the
# builder (S1) will emit. Mirrors the SAP2000/ETABS area-section menu.
SHELL_KINDS = ("shell-thin", "shell-thick", "plate-thin", "plate-thick",
               "membrane")
SHELL_KIND_LABELS = {
    "shell-thin": "Shell — thin (MITC, DKMQ)",
    "shell-thick": "Shell — thick (MITC)",
    "plate-thin": "Plate — thin (bending only)",
    "plate-thick": "Plate — thick (bending only)",
    "membrane": "Membrane (in-plane only)",
}

# Structural *role* an area object plays (wall plan W0). Following ETABS, a wall
# is not a separate object type — it is an area whose role is ``"wall"`` and
# which may carry a ``pier``/``spandrel`` label so its integrated forces can be
# reported and designed as one wall across several area objects / stories. A
# plain shell that is neither a floor nor a wall is ``"shell"``. Absent role on
# an older file migrates to ``"slab"`` (see ``Project.from_dict``).
AREA_ROLES = ("slab", "wall", "shell")
AREA_ROLE_LABELS = {"slab": "Slab / floor", "wall": "Wall", "shell": "Shell"}


@dataclass
class ShellSection:
    """A shell / plate *thickness* property — the surface-element analogue of
    :class:`Section` (which serves line members). Where a beam ``Section``
    carries (A, Iz, Iy, J), a shell section carries the through-thickness
    description that maps membrane strains, curvatures and transverse shear to
    stress resultants per unit width.

    ``kind`` selects the element formulation the builder emits (see
    ``SHELL_KINDS``): a general *shell* (membrane + bending + drilling, the
    MITC/DKMQ workhorse), a *plate* (bending only), or a *membrane* (in-plane
    only). Like a line ``Section`` this is pure geometry/formulation — the
    material is carried by the owning ``Area`` (mirroring ``Member.material``),
    so the same section can be reused with different materials.

    ``modifiers`` are SAP-style stiffness/mass/weight scale factors applied to
    the element's constitutive matrices; any missing key defaults to 1.0. Keys:
    ``f11 f22 f12`` (membrane), ``m11 m22 m12`` (bending), ``v13 v23`` (shear),
    ``mass``, ``weight``.

    ``layers`` is reserved for the layered / RC-layered path (S4/Phase C);
    empty means the single isotropic layer of ``thickness`` (the MVP path)."""
    id: int
    name: str
    thickness: float                       # total thickness (m), stored SI
    kind: str = "shell-thin"               # one of SHELL_KINDS
    modifiers: dict = field(default_factory=dict)   # SAP stiffness modifiers
    layers: list = field(default_factory=list)      # future: layered/RC stack

    def modifier(self, key: str) -> float:
        """Stiffness/mass/weight modifier ``key`` (1.0 if unset)."""
        try:
            return float(self.modifiers.get(key, 1.0))
        except (TypeError, ValueError):
            return 1.0


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float = 0.0
    supports: tuple = ()          # fixity mask (len ndf); empty = free


@dataclass
class Hinge:
    """A finite-length fiber plastic-hinge *property* (plan §14 GUI-3).

    Assigned to a member (``Member.hinge``) it turns that member's fiber
    (Section-Designer) section into a lumped ``FiberHingeBeamColumn2D`` with a
    plastic hinge of length ``lp`` at each end (the CSI "Fiber P-M2-M3" / Midas
    lumped-hinge idiom). ``relative`` makes ``lp`` / ``lp_j`` fractions of the
    member length (CSI "relative hinge length"); otherwise they are absolute
    lengths in project units. ``lp_j`` ``None`` means end J equals end I
    (a symmetric hinge)."""
    id: int
    name: str
    lp: float = 0.1               # end-I hinge length (ratio if ``relative``)
    lp_j: float | None = None     # end-J hinge length; None = same as end I
    relative: bool = True


@dataclass
class Member:
    id: int
    n1: int
    n2: int
    section: int
    material: int
    kind: str = "beamcolumn2d"
    hinge: int | None = None      # Hinge property id (None = distributed/elastic)
    # Vertical insertion offset (m, 3-D): the beam's centroid sits ``z_offset``
    # below (negative) / above the drawn line. Non-zero on an edge beam makes it
    # act *compositely* with the slab — the beam is built at the offset elevation
    # and rigidly tied back to the drawn (slab) nodes (BE6). 0 = centroidal.
    z_offset: float = 0.0


@dataclass
class Area:
    """A surface (shell / plate) *area object* — the 2-D analogue of
    :class:`Member`. ``nodes`` is a list of 3 (triangle) or 4 (quad) corner
    node ids, ordered counter-clockwise when viewed from the positive-normal
    (top / local-3) side — the ordering ``ShellMITC4`` / ``ShellTri3`` expect.

    ``shell_section`` references a :class:`ShellSection` (thickness +
    formulation); ``material`` is the node-material id, carried here rather than
    on the section exactly as ``Member.material`` is (so one thickness property
    serves many materials).

    Following SAP2000/ETABS, the drawn *area object* is deliberately distinct
    from its analysis *mesh*: ``mesh`` = ``(n1, n2)`` internal-mesh divisions is
    a property resolved to elements at solve time (S3), keeping the model
    editable and the tree clean. ``(1, 1)`` (the default) means the area is a
    single element. ``local_axis`` rotates the in-plane local axes about the
    element normal (degrees), for oriented results / orthotropic sections."""
    id: int
    nodes: list                   # 3 or 4 corner node ids, CCW from +normal
    shell_section: int
    material: int
    mesh: tuple = (1, 1)          # (n1, n2) auto-mesh divisions (S3)
    local_axis: float = 0.0       # local-axis rotation about the normal (deg)
    # Structural role + wall labels (wall plan W0). ``role`` is one of
    # ``AREA_ROLES``; ``pier``/``spandrel`` are optional labels (ETABS-style)
    # grouping wall areas whose forces are integrated and designed together.
    # Both labels are ``None`` unless ``role == "wall"``.
    role: str = "slab"
    pier: "str | None" = None
    spandrel: "str | None" = None
    # Rectangular openings (wall plan W5), each ``(u0, v0, u1, v1)`` in the quad's
    # parametric coordinates: ``u`` runs along nodes[0]→nodes[1] (the base of a
    # W1a wall), ``v`` along nodes[0]→nodes[3] (its height), both in 0..1. Mesh
    # cells whose centre falls inside any opening are dropped, so a door/window
    # is a hole in the panel. Only honoured for 4-node (quad) areas.
    openings: list = field(default_factory=list)

    def __post_init__(self):
        # JSON round-trips ``nodes``/``mesh`` as lists; normalize types so
        # equality and downstream indexing behave.
        self.nodes = [int(n) for n in self.nodes]
        m = tuple(int(x) for x in self.mesh)
        self.mesh = (m + (1, 1))[:2] if m else (1, 1)
        self.role = self.role if self.role in AREA_ROLES else "slab"
        # A pier/spandrel label only means something on a wall; blank strings
        # normalize to ``None`` so "unset" has one representation.
        self.pier = _norm_label(self.pier) if self.role == "wall" else None
        self.spandrel = (_norm_label(self.spandrel)
                         if self.role == "wall" else None)
        # normalize each opening to a sorted 4-tuple clamped to the unit square
        norm = []
        for o in (self.openings or []):
            try:
                u0, v0, u1, v1 = (float(o[0]), float(o[1]),
                                  float(o[2]), float(o[3]))
            except (TypeError, ValueError, IndexError):
                continue
            u0, u1 = sorted((min(max(u0, 0.0), 1.0), min(max(u1, 0.0), 1.0)))
            v0, v1 = sorted((min(max(v0, 0.0), 1.0), min(max(v1, 0.0), 1.0)))
            if u1 - u0 > 1e-9 and v1 - v0 > 1e-9:
                norm.append((u0, v0, u1, v1))
        self.openings = norm


# Load "nature" -> ASCE 7 pattern key used by the code combinations. ``None``
# means the case carries no code factor of its own (user combos only).
NATURE_ASCE = {"dead": "D", "live": "L", "roof_live": "Lr", "snow": "S",
               "rain": "R", "wind": "W", "seismic": "E", "other": None}
NATURE_LABELS = {"dead": "Dead", "live": "Live", "roof_live": "Roof live",
                 "snow": "Snow", "rain": "Rain", "wind": "Wind",
                 "seismic": "Seismic", "other": "Other"}


@dataclass
class LoadCase:
    """A named physical load case (Dead, Live, Wind, …). Loads belong to a
    case; combinations factor cases together. ``nature`` maps the case to an
    ASCE 7 pattern key so code combinations can be generated automatically.

    ``self_weight_factor`` is the SAP2000/MIDAS-style self-weight multiplier:
    when non-zero the case applies the structure's own gravity weight scaled by
    it (1.0 = full self-weight acting global-down), in addition to any explicit
    loads assigned to the case. Members contribute ρ·A·L·g (lumped to their end
    nodes, so it is correct for any orientation, columns included); areas
    contribute ρ·t·g as a surface load. Usually set to 1.0 on a single 'Self
    weight' / dead case."""
    id: int
    name: str
    nature: str = "dead"          # key of NATURE_ASCE
    self_weight_factor: float = 0.0   # gravity self-weight multiplier (1.0 = on)


@dataclass
class NonlinearCase:
    """A saved nonlinear (fiber) analysis case (plan §14 GUI-4) — the desktop
    counterpart of a Midas/CSI nonlinear static load case.

    Displacement-controlled: ``control_node`` DOF ``control_dof`` (0=Ux, 1=Uy,
    2=Rz) is driven to ``target`` following ``protocol``:

    * ``"monotonic"`` — a ramp 0→target over ``n_steps``.
    * ``"cyclic"`` — reversed cycles of growing amplitude; the peaks are
      ``amplitudes`` (fractions) × ``target``, ``cycles`` full cycles each,
      ``pts_per_cycle`` points per cycle.

    ``axial`` (compression magnitude at ``axial_node``/``axial_dof``) is applied
    first and held constant (staged). ``continue_from`` (another case's id)
    continues this case from that case's committed state (staged construction).
    ``tol`` / ``max_iter`` are the Newton controls."""
    id: int
    name: str
    control_node: int = 0
    control_dof: int = 1
    target: float = 0.05
    n_steps: int = 40
    protocol: str = "monotonic"           # "monotonic" | "cyclic"
    amplitudes: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    cycles: int = 1
    pts_per_cycle: int = 40
    axial: float = 0.0
    axial_node: int | None = None
    axial_dof: int = 0
    continue_from: int | None = None      # id of a prior NonlinearCase
    tol: float = 1.0e-6
    max_iter: int = 60
    notes: str = ""                        # free-text case notes (GUI only)
    # Initial condition (E2): ("zero",) = unstressed state, or
    # ("state", nl_case_id) = start from the committed state + stiffness at the
    # end of that Nonlinear Static case. Distinct from ``continue_from``, which
    # *replays* an ancestor's push to trace a continuous pushover curve; this is
    # the general cross-type state seed (see the initial-conditions sub-plan).
    initial_condition: tuple = ("zero",)

    def __post_init__(self):
        self.initial_condition = _coerce_ic(self.initial_condition)


@dataclass
class AnalysisCase:
    """A saved analysis case for a built-in analysis type (modal, buckling,
    response spectrum, moving load, …) — the desktop counterpart of a
    CSiBridge / MIDAS named load case, and the generic sibling of
    :class:`NonlinearCase` (which keeps its own rich list).

    ``type`` selects the analysis (and its setup dialog / adapter in
    :mod:`case_types`); ``params`` is the JSON-friendly saved configuration —
    the **inputs** the user typed (spectrum source + code parameters, lane node
    ids, vehicle key, response target, …), *not* any derived runtime object.
    The runtime config a ``MainWindow.run_*`` consumes is rebuilt from
    ``params`` at Run time (``CaseType.build_config``), so a saved case runs
    without re-prompting. ``notes`` is free-text (GUI only)."""
    id: int
    name: str
    type: str                              # "modal" | "buckling" | ...
    params: dict = field(default_factory=dict)
    notes: str = ""
    # Initial condition (E2) — see :class:`NonlinearCase`. ("zero",) or
    # ("state", nl_case_id); the source is always a Nonlinear Static case.
    initial_condition: tuple = ("zero",)

    def __post_init__(self):
        self.initial_condition = _coerce_ic(self.initial_condition)


@dataclass
class TimeHistoryFunction:
    """A named ground-motion / time-history function — the desktop counterpart of
    CSiBridge / MIDAS *Define ▸ Functions ▸ Time History*. Equally-spaced
    acceleration ordinates ``values`` at step ``dt`` seconds, given in *g* when
    ``in_g`` else m/s². Defined once here and referenced (by ``id``) from a saved
    Time-History :class:`AnalysisCase`; ``source`` keeps the import filename."""
    id: int
    name: str
    dt: float = 0.01
    values: list = field(default_factory=list)
    in_g: bool = False
    source: str = ""

    @property
    def npts(self) -> int:
        return len(self.values)

    @property
    def duration(self) -> float:
        return max(0, len(self.values) - 1) * float(self.dt)


@dataclass
class Load:
    node: int
    values: tuple                 # nodal load vector (len ndf)
    case: int = 1                 # owning LoadCase id


@dataclass
class MemberLoad:
    """A uniform transverse line load on a member, in the member's local axes
    (N/m). ``wz`` is used in 3-D only."""
    member: int
    wy: float = 0.0               # local-y UDL (N/m)
    wz: float = 0.0               # local-z UDL (N/m), 3-D only
    case: int = 1                 # owning LoadCase id


@dataclass
class AreaLoad:
    """A uniform load on a surface (``Area``) object (slab plan S5), magnitude
    ``w`` in pressure units (Pa = N/m²):

    * ``kind="gravity"`` — a downward area load, applied as the global traction
      ``(0, 0, -w)`` (the common slab dead/live/superimposed case).
    * ``kind="pressure"`` — ``w`` acting normal to the area (its local +3),
      e.g. wind on a wall or hydrostatic face pressure.
    * ``kind="selfweight"`` — a downward load auto-computed from the area's
      material density × shell-section thickness × g (``w`` is ignored / 0).

    ``case`` is the owning LoadCase id. Compiled onto the area's shell
    element(s) by :meth:`Project.apply_case` via the element's
    ``add_surface_load`` / ``add_pressure`` (see slab S5a)."""
    area: int
    w: float = 0.0                # magnitude (Pa = N/m²)
    kind: str = "gravity"         # "gravity" (global -Z) | "pressure" (normal)
    case: int = 1                 # owning LoadCase id


@dataclass
class LoadCombination:
    """A weighted sum of load cases. ``factors`` maps case id -> factor."""
    id: int
    name: str
    factors: dict = field(default_factory=dict)   # {case_id: factor}


@dataclass
class Stage:
    """One construction stage: the members that become active ("born") in it,
    in construction order. Members in no stage are treated as built before the
    sequence (initially active). Drives the incremental staged analysis + camber
    (bridge GUI plan G3)."""
    id: int
    name: str
    add_members: list = field(default_factory=list)   # member ids born this stage


@dataclass
class Diaphragm:
    """A rigid floor diaphragm (slab plan S8): the ``nodes`` (typically the
    beam-column joints at one level) are tied so they share the master node's
    two in-plane translations and its rotation about the diaphragm normal, while
    vertical translation and the two in-plane rotations stay independent — the
    classic rigid-floor idealisation that distributes lateral load by rigidity.

    ``perp_dir`` is the global axis normal to the diaphragm plane
    (2 = Z / XY-plane floor, the default; 0 = X, 1 = Y). ``master`` is an
    existing node id to use as the master, or ``None`` to auto-create one at the
    centroid of ``nodes`` at build time. Compiled to a
    :class:`femsolver.RigidDiaphragm` MP-constraint in ``build_model``; a 3-D
    feature (needs ndf=6)."""
    id: int
    name: str
    nodes: list = field(default_factory=list)   # slave node ids
    perp_dir: int = 2                            # 0=X, 1=Y, 2=Z (XY floor)
    master: int | None = None                   # None = auto master at centroid

    def __post_init__(self):
        self.nodes = [int(n) for n in self.nodes]
        self.perp_dir = int(self.perp_dir)


@dataclass
class Story:
    """A building story / level (wall plan W4). A named horizontal level whose
    top is at elevation ``elev`` (global Z) and which is ``height`` tall (down
    to the level below; ``0`` for the base). Following ETABS, stories are a
    *modeling context* — nodes/areas belong to a story by their elevation, not
    by a stored link — used to slice pier forces, label the tree and (W4c)
    replicate 'similar' stories. ``master`` names a story this one copies."""
    id: int
    name: str
    elev: float                    # elevation of the level (global Z), SI
    height: float = 0.0            # story height to the level below (SI)
    master: "str | None" = None    # 'similar to' story name (W4c)

    def __post_init__(self):
        self.elev = float(self.elev)
        self.height = float(self.height)


@dataclass
class GridLine:
    """A named grid line (wall plan W4). ``axis="x"`` is a line of constant X
    (running in the Y direction); ``axis="y"`` is constant Y (running in X).
    ``coord`` is its position (SI). Snapping/rendering use it (W4b)."""
    id: int
    name: str
    axis: str = "x"                # "x" (const-X line) | "y" (const-Y line)
    coord: float = 0.0

    def __post_init__(self):
        self.axis = self.axis if self.axis in ("x", "y") else "x"
        self.coord = float(self.coord)


@dataclass
class Project:
    name: str = "Untitled"
    ndm: int = 2
    ndf: int = 3
    force_unit: str = "N"          # project stores SI base units (N, m, Pa)
    length_unit: str = "m"
    design_code: str = "AISC 360"
    materials: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    shell_sections: list = field(default_factory=list)  # ShellSection (slab plan S0)
    hinges: list = field(default_factory=list)         # Hinge properties (GUI-3)
    nodes: list = field(default_factory=list)
    members: list = field(default_factory=list)
    areas: list = field(default_factory=list)          # Area (shell/plate; slab S0)
    # Auto-connect beams to a meshed slab edge (BE0-BE2): split a member lying
    # along a slab edge at the slab's edge nodes so the two share nodes (true
    # compatibility). Off = legacy one-element-per-member behaviour.
    auto_connect_beams: bool = True
    load_cases: list = field(default_factory=list)    # LoadCase
    loads: list = field(default_factory=list)          # nodal Load
    member_loads: list = field(default_factory=list)   # MemberLoad (line loads)
    area_loads: list = field(default_factory=list)      # AreaLoad (slab plan S5)
    combinations: list = field(default_factory=list)   # LoadCombination
    diaphragms: list = field(default_factory=list)      # Diaphragm (slab plan S8)
    stages: list = field(default_factory=list)          # Stage (construction seq)
    stories: list = field(default_factory=list)         # Story (building levels, W4)
    grid_lines: list = field(default_factory=list)      # GridLine (W4)
    nonlinear_cases: list = field(default_factory=list)  # NonlinearCase (GUI-4)
    analysis_cases: list = field(default_factory=list)  # AnalysisCase (ACM plan)
    th_functions: list = field(default_factory=list)   # TimeHistoryFunction (ACM)
    # Saved nonlinear-run results (plan §16 G-S2) — the expensive, run-specific
    # exception to "results are recomputed": each is a lean ``nl_runs.RunRecord``
    # (curve + summary + ASCE 41 milestones). Not consumed by ``build_model``.
    runs: list = field(default_factory=list)

    def __post_init__(self):
        if not self.load_cases:
            self.load_cases.append(LoadCase(id=1, name="Dead", nature="dead"))

    # ------------------------------------------------------------- load cases
    def case(self, case_id):
        return next((c for c in self.load_cases if c.id == case_id), None)

    def combination(self, combo_id):
        return next((c for c in self.combinations if c.id == combo_id), None)

    # ------------------------------------------------------- shells / areas (S0)
    def shell_section(self, sec_id):
        return next((s for s in self.shell_sections if s.id == sec_id), None)

    def area(self, area_id):
        return next((a for a in self.areas if a.id == area_id), None)

    def next_shell_section_id(self) -> int:
        return max((s.id for s in self.shell_sections), default=0) + 1

    def next_area_id(self) -> int:
        return max((a.id for a in self.areas), default=0) + 1

    # --------------------------------------------------------- walls / piers (W0)
    def walls(self) -> list:
        """All area objects whose structural role is ``"wall"``."""
        return [a for a in self.areas if a.role == "wall"]

    def pier_names(self) -> list:
        """Sorted, de-duplicated pier labels currently in use (wall plan W0)."""
        return sorted({a.pier for a in self.areas
                       if a.role == "wall" and a.pier})

    def spandrel_names(self) -> list:
        """Sorted, de-duplicated spandrel labels currently in use."""
        return sorted({a.spandrel for a in self.areas
                       if a.role == "wall" and a.spandrel})

    def areas_in_pier(self, pier: str) -> list:
        """The wall areas grouped under a given pier label."""
        return [a for a in self.areas if a.role == "wall" and a.pier == pier]

    def areas_in_spandrel(self, spandrel: str) -> list:
        """The wall areas grouped under a given spandrel label."""
        return [a for a in self.areas
                if a.role == "wall" and a.spandrel == spandrel]

    # ------------------------------------------------------- stories / grid (W4)
    def stories_sorted(self) -> list:
        """Stories from the base up (ascending elevation)."""
        return sorted(self.stories, key=lambda s: s.elev)

    def story_elevations(self) -> list:
        """Sorted story level elevations (global Z)."""
        return [s.elev for s in self.stories_sorted()]

    def next_story_id(self) -> int:
        return max((s.id for s in self.stories), default=0) + 1

    def next_grid_id(self) -> int:
        return max((g.id for g in self.grid_lines), default=0) + 1

    def story_at(self, z: float, tol: float = 1e-6):
        """The story whose band contains elevation ``z`` — the lowest story
        whose level is at or above ``z`` (its floor-to-floor band ``(elev −
        height, elev]``). Returns ``None`` if ``z`` is above every level or no
        stories are defined."""
        for s in self.stories_sorted():
            if z <= s.elev + tol:
                return s
        return None

    def grid_lines_on(self, axis: str) -> list:
        """Grid lines of a given ``axis`` ('x' or 'y'), sorted by coordinate."""
        return sorted((g for g in self.grid_lines if g.axis == axis),
                      key=lambda g: g.coord)

    def diaphragm(self, dia_id):
        return next((d for d in self.diaphragms if d.id == dia_id), None)

    def next_diaphragm_id(self) -> int:
        return max((d.id for d in self.diaphragms), default=0) + 1

    def nonlinear_case(self, case_id):
        return next((c for c in self.nonlinear_cases if c.id == case_id), None)

    def analysis_case(self, case_id):
        return next((c for c in self.analysis_cases if c.id == case_id), None)

    def th_function(self, func_id):
        return next((f for f in self.th_functions if f.id == func_id), None)

    def default_case_id(self) -> int:
        return self.load_cases[0].id if self.load_cases else 1

    # -------------------------------------------------- run status (E5c)
    # Per-case session run status, keyed by ``(kind, id)`` (e.g.
    # ("analysis", 3), ("nonlinear", 7), ("stages",)). Transient: results are
    # recomputed each session, so this is NOT serialized (a plain attribute, not
    # a dataclass field, so ``asdict`` ignores it). Each entry is
    # ``{"status": str, "when": float epoch}``.
    def case_status(self, key) -> dict | None:
        return self.__dict__.get("_case_status", {}).get(tuple(key))

    def set_case_status(self, key, status: str, when: float | None = None) -> None:
        import time
        reg = self.__dict__.setdefault("_case_status", {})
        reg[tuple(key)] = {"status": status,
                           "when": time.time() if when is None else float(when)}

    # ------------------------------------------------------------- hinges (GUI-3)
    def hinge(self, hinge_id):
        return next((h for h in self.hinges if h.id == hinge_id), None)

    def member_length(self, member) -> float:
        """Geometric length of ``member`` from its end-node coordinates."""
        nodes = {n.id: n for n in self.nodes}
        a, b = nodes[member.n1], nodes[member.n2]
        d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2
        if self.ndm == 3:
            d2 += (a.z - b.z) ** 2
        return d2 ** 0.5

    def resolve_hinge_lengths(self, member, hinge) -> tuple:
        """Absolute (lp_i, lp_j) plastic-hinge lengths for ``member`` under
        ``hinge`` — ratios × member length when ``hinge.relative``, else the
        values as given — clamped positive and so lp_i + lp_j < the member
        length (the element requires lp_i + lp_j ≤ L)."""
        L = self.member_length(member) or 1.0
        raw_i = float(hinge.lp)
        raw_j = raw_i if hinge.lp_j is None else float(hinge.lp_j)
        lp_i = raw_i * L if hinge.relative else raw_i
        lp_j = raw_j * L if hinge.relative else raw_j
        lp_i = max(lp_i, 1.0e-9)
        lp_j = max(lp_j, 1.0e-9)
        if lp_i + lp_j >= L:                       # scale down to fit the member
            s = 0.98 * L / (lp_i + lp_j)
            lp_i *= s
            lp_j *= s
        return lp_i, lp_j

    def generate_asce7_combinations(self) -> list:
        """ASCE 7-22 LRFD strength combinations as project ``LoadCombination``s,
        mapping each case to its pattern key (D, L, W, …) via its nature. Combos
        that reduce to the same factor set — e.g. a project with no Lr/S/R cases
        — are de-duplicated. Returns the new combinations (caller adds them)."""
        from femsolver.analysis.load_combinations import asce7_lrfd_combinations
        key_to_cases: dict = {}
        for c in self.load_cases:
            key = NATURE_ASCE.get(c.nature)
            if key:
                key_to_cases.setdefault(key, []).append(c.id)
        out, seen = [], set()
        next_id = max((c.id for c in self.combinations), default=0) + 1
        for ec in asce7_lrfd_combinations():
            factors: dict = {}
            for key, f in ec.factors.items():
                for cid in key_to_cases.get(key, []):
                    factors[cid] = factors.get(cid, 0.0) + f
            if not factors:
                continue
            sig = frozenset(factors.items())
            if sig in seen:
                continue
            seen.add(sig)
            out.append(LoadCombination(id=next_id, name=ec.name,
                                       factors=dict(factors)))
            next_id += 1
        return out

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        cases = [LoadCase(**c) for c in d.get("load_cases", [])]
        if not cases:                       # migrate pre-load-case projects
            cases = [LoadCase(id=1, name="Dead", nature="dead")]
        default_id = cases[0].id
        return cls(
            name=d.get("name", "Untitled"),
            ndm=int(d.get("ndm", 2)),
            ndf=int(d.get("ndf", 3)),
            force_unit=d.get("force_unit", "N"),   # SI base default (matches the
            length_unit=d.get("length_unit", "m"),  # dataclass; was "kN" — a bug)
            design_code=d.get("design_code", "AISC 360"),
            materials=[Material(**m) for m in d.get("materials", [])],
            sections=[Section(**s) for s in d.get("sections", [])],
            shell_sections=[ShellSection(**s)
                            for s in d.get("shell_sections", [])],
            hinges=[Hinge(**h) for h in d.get("hinges", [])],
            nodes=[Node(**_coerce_node(n)) for n in d.get("nodes", [])],
            members=[Member(**m) for m in d.get("members", [])],
            areas=[Area(id=a["id"], nodes=list(a.get("nodes", [])),
                        shell_section=a["shell_section"],
                        material=a.get("material", 0),
                        mesh=tuple(a.get("mesh", (1, 1))),
                        local_axis=float(a.get("local_axis", 0.0)),
                        role=a.get("role", "slab"),   # migrate: absent -> slab
                        pier=a.get("pier"),
                        spandrel=a.get("spandrel"),
                        openings=[tuple(o) for o in a.get("openings", [])])
                   for a in d.get("areas", [])],
            load_cases=cases,
            loads=[Load(node=x["node"], values=tuple(x["values"]),
                        case=x.get("case", default_id))
                   for x in d.get("loads", [])],
            member_loads=[MemberLoad(member=x["member"], wy=x.get("wy", 0.0),
                                     wz=x.get("wz", 0.0),
                                     case=x.get("case", default_id))
                          for x in d.get("member_loads", [])],
            area_loads=[AreaLoad(area=x["area"], w=x.get("w", 0.0),
                                 kind=x.get("kind", "gravity"),
                                 case=x.get("case", default_id))
                        for x in d.get("area_loads", [])],
            combinations=[LoadCombination(
                id=c["id"], name=c["name"],
                factors={int(k): v for k, v in c.get("factors", {}).items()})
                for c in d.get("combinations", [])],
            diaphragms=[Diaphragm(id=x["id"], name=x.get("name", ""),
                                  nodes=list(x.get("nodes", [])),
                                  perp_dir=int(x.get("perp_dir", 2)),
                                  master=x.get("master"))
                        for x in d.get("diaphragms", [])],
            stages=[Stage(id=s["id"], name=s.get("name", ""),
                          add_members=list(s.get("add_members", [])))
                    for s in d.get("stages", [])],
            stories=[Story(id=s["id"], name=s.get("name", ""),
                           elev=s.get("elev", 0.0), height=s.get("height", 0.0),
                           master=s.get("master"))
                     for s in d.get("stories", [])],
            grid_lines=[GridLine(id=g["id"], name=g.get("name", ""),
                                 axis=g.get("axis", "x"),
                                 coord=g.get("coord", 0.0))
                        for g in d.get("grid_lines", [])],
            nonlinear_cases=[NonlinearCase(**c)
                             for c in d.get("nonlinear_cases", [])],
            analysis_cases=[AnalysisCase(
                id=c["id"], name=c.get("name", ""), type=c["type"],
                params=dict(c.get("params", {})), notes=c.get("notes", ""),
                initial_condition=_coerce_ic(c.get("initial_condition")))
                            for c in d.get("analysis_cases", [])],
            th_functions=[TimeHistoryFunction(
                id=f["id"], name=f.get("name", ""), dt=float(f.get("dt", 0.01)),
                values=[float(v) for v in f.get("values", [])],
                in_g=bool(f.get("in_g", False)), source=f.get("source", ""))
                for f in d.get("th_functions", [])],
            runs=_load_runs(d.get("runs", [])),
        )

    @classmethod
    def from_json(cls, text: str) -> "Project":
        return cls.from_dict(json.loads(text))

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Project":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -------------------------------------------------------- compile to solver
    def build_model(self, with_loads: bool = True):
        """Compile this declarative project into a transient femsolver.Model.

        With ``with_loads`` the unfactored sum of every load case is applied
        (ready to solve as-is). Pass ``with_loads=False`` to build geometry +
        supports only, then apply a specific case/combination via
        :meth:`apply_loads` — what the combination/envelope design does."""
        from femsolver import (BeamColumn2D, BeamColumn3D, ElasticIsotropic,
                               Model)

        m = Model(ndm=self.ndm, ndf=self.ndf)
        mats = {}
        for mat in self.materials:
            obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu,
                                   rho=getattr(mat, "rho", 0.0))
            m.add_material(obj)
            mats[mat.id] = obj
        secs = {s.id: s for s in self.sections}
        for nd in self.nodes:
            coords = (nd.x, nd.y) if self.ndm == 2 else (nd.x, nd.y, nd.z)
            m.add_node(nd.id, *coords)
        # One coordinate-keyed node registry shared by beam-splitting (BE0-BE2)
        # and slab meshing (S3), so a beam node created at a slab edge node and
        # the slab node itself collapse to one id (true compatibility). Seeded
        # with the project nodes already added above.
        registry, node_at = self._node_registry(m)
        # BE0: pre-create the slab edge nodes so the member loop can split
        # against them; the mesher below reuses them via the same registry.
        edge_nodes = {}
        if self.ndm == 3 and self.areas and getattr(
                self, "auto_connect_beams", True):
            edge_nodes = self._register_area_edge_nodes(node_at)
        gsd_mats: dict = {}          # section id -> concrete ElasticIsotropic
        for mb in self.members:
            sec = secs[mb.section]
            A, Iz, Iy, J = _resolve_section(sec)
            material = mats[mb.material]
            # a pin-ended cable / truss carries axial force only
            if getattr(mb, "kind", "") == "cable":
                from femsolver import Truss2D, Truss3D
                if self.ndm == 3:
                    m.add_element(Truss3D(mb.id, (mb.n1, mb.n2), material, A))
                else:
                    m.add_element(Truss2D(mb.id, (mb.n1, mb.n2), material, A))
                continue
            # a Section-Designer (concrete/PSC) section drives its own modulus,
            # so the member's stiffness reflects concrete E_c, not whatever
            # material was assigned in the model.
            if getattr(sec, "gsd_spec", None):
                cm = gsd_mats.get(sec.id)
                if cm is None:
                    try:
                        Ec = _gsd_modulus(sec.gsd_spec)
                    except Exception:
                        Ec = None
                    if Ec:
                        cm = ElasticIsotropic(10_000 + sec.id, E=Ec, nu=0.2,
                                              rho=getattr(material, "rho", 0.0))
                        m.add_material(cm)
                        gsd_mats[sec.id] = cm
                if cm is not None:
                    material = cm
            def _bc(tag, na, nb):
                if self.ndm == 3:
                    return BeamColumn3D(tag, (na, nb), material, A, Iy, Iz, J)
                return BeamColumn2D(tag, (na, nb), material, A, Iz)

            # BE1/BE2: if this beam lies along a meshed slab edge, split it at
            # the slab edge nodes so beam and slab share nodes. Otherwise emit
            # one element keyed by the member id (legacy behaviour).
            chain = self._member_split_chain(mb, edge_nodes) if edge_nodes \
                else None
            nodes = chain if (chain and len(chain) > 2) else [mb.n1, mb.n2]
            tags = ([member_element_tag(mb.id, j) for j in range(len(nodes) - 1)]
                    if len(nodes) > 2 else [mb.id])
            offset = float(getattr(mb, "z_offset", 0.0) or 0.0)
            if offset and self.ndm == 3:
                # BE6: build the beam at its offset centroid on phantom nodes,
                # each rigidly tied back to the drawn (slab) node — composite
                # T-beam action. Same tags as the centroidal case.
                self._add_offset_beam(m, node_at, nodes, tags, _bc, offset)
            else:
                for j in range(len(nodes) - 1):
                    m.add_element(_bc(tags[j], nodes[j], nodes[j + 1]))
        # ---- surface (shell / plate) area objects (slab plan S1 + S3 mesh) --
        # Areas are a 3-D feature — shells need ndf=6, so a 2-D model has no
        # surface elements. Each quad area is meshed into its ``mesh`` = (n1, n2)
        # sub-elements; coincident mesh nodes are merged so adjacent areas stay
        # compatible. Sub-element tags come from ``area_element_tag`` so the load
        # path can find them without build state.
        if self.ndm == 3 and self.areas:
            self._mesh_and_add_areas(m, mats, node_at)
        for nd in self.nodes:
            if nd.supports and any(nd.supports):
                m.fix(nd.id, list(nd.supports))
        # rigid diaphragms (slab S8) — 3-D only (they need ndf=6)
        if self.ndm == 3 and self.diaphragms:
            self._add_diaphragms(m)
        if with_loads:
            self.apply_loads(m, ("all", None))
        return m

    def _add_diaphragms(self, model) -> None:
        """Compile each :class:`Diaphragm` into a ``femsolver.RigidDiaphragm``
        MP-constraint (slab S8). When a diaphragm has no explicit master, a
        master node is auto-created at the centroid of its slaves and its
        out-of-plane DOFs are pinned (an unconnected master would otherwise be
        singular). Slaves not present in the model are dropped; <2 → skipped."""
        from femsolver import RigidDiaphragm
        for dia in self.diaphragms:
            slaves = [n for n in dia.nodes if n in model.nodes]
            master = dia.master
            if master is not None and master in slaves:
                slaves = [s for s in slaves if s != master]
            if len(slaves) < 2:
                continue
            perp = int(getattr(dia, "perp_dir", 2))
            if master is None or master not in model.nodes:
                import numpy as np
                c = np.mean([model.node(s).coords for s in slaves], axis=0)
                master = DIAPHRAGM_MASTER_NODE_BASE + dia.id
                if master in model.nodes:        # defensive: avoid a clash
                    master = DIAPHRAGM_MASTER_NODE_BASE + max(
                        d.id for d in self.diaphragms) + dia.id
                model.add_node(master, *[float(v) for v in c])
                # free only the in-plane translations + rotation about perp;
                # pin the rest so the unconnected master isn't singular.
                mask = [1, 1, 1, 1, 1, 1]
                in_plane = [k for k in range(3) if k != perp]
                for k in in_plane:
                    mask[k] = 0
                mask[3 + perp] = 0
                model.fix(master, mask)
            model.add_mp_constraint(
                RigidDiaphragm(master=master, slaves=slaves, perp_dir=perp))

    def _node_registry(self, model):
        """A coordinate-keyed node factory over ``model``: ``node_at(p)`` returns
        the id of the node at ``p``, creating (and adding) it on first sight and
        reusing it (merging coincident nodes) thereafter. Seeded with the project
        nodes already added to the model. Shared by beam-splitting (BE0-BE2) and
        slab meshing (S3) so their nodes collapse together."""
        ncoord = {nd.id: (nd.x, nd.y, nd.z) for nd in self.nodes}
        registry = {_coord_key(c): nid for nid, c in ncoord.items()}
        counter = [max(ncoord, default=0) + 1]

        def node_at(p):
            key = _coord_key(p)
            nid = registry.get(key)
            if nid is None:
                nid = counter[0]
                counter[0] += 1
                registry[key] = nid
                model.add_node(nid, *[float(v) for v in p])
            return nid

        return registry, node_at

    def _area_edge_coords(self) -> list:
        """The coordinates of every quad slab's *edge* subdivision points (the
        interior nodes along its four sides, per ``mesh`` = (n1, n2)) — the only
        nodes a beam lying on a slab edge could share. Polygon/triangle areas
        have no subdivided edges, so they contribute none; corners are project
        nodes already, so excluded. Shared by :meth:`_register_area_edge_nodes`
        (build) and :meth:`member_element_tags` (results/design, BE3), so the two
        can never disagree on where a member splits."""
        ncoord = {nd.id: (nd.x, nd.y, nd.z) for nd in self.nodes}
        coords = []
        for a in self.areas:
            if len(a.nodes) != 4 or any(nid not in ncoord for nid in a.nodes):
                continue
            corners = [ncoord[nid] for nid in a.nodes]
            n1 = max(1, int(a.mesh[0]))
            n2 = max(1, int(a.mesh[1]))
            # interior edge grid nodes, keeping only those a kept cell uses so an
            # opening on the boundary leaves no orphaned edge node (W5)
            needed = area_quad_needed_nodes(a)
            edge_ij = ([(i, 0) for i in range(1, n1)]            # P0->P1
                       + [(n1, j) for j in range(1, n2)]         # P1->P2
                       + [(i, n2) for i in range(1, n1)]         # P3->P2
                       + [(0, j) for j in range(1, n2)])         # P0->P3
            for i, j in edge_ij:
                if (i, j) not in needed:
                    continue
                coords.append(tuple(float(v)
                                    for v in _bilinear(corners, i / n1, j / n2)))
        return coords

    def _register_area_edge_nodes(self, node_at) -> dict:
        """BE0: create every quad slab edge subdivision node via ``node_at`` and
        return ``{node_id: (x, y, z)}`` for the member splitter to match against."""
        return {node_at(p): p for p in self._area_edge_coords()}

    @staticmethod
    def _interior_params(a, b, points):
        """Sorted, de-duplicated parameters ``t`` in (0, 1) of the ``points`` that
        lie *on the segment* a->b (collinear, strictly interior) within a
        scale-aware tolerance. Returns ``[(t, index)]`` so callers can recover
        which point matched. The single source of truth for "does this member
        pass through this slab node", shared by build and results/design."""
        import numpy as np
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 <= 1e-18:
            return []
        tol = 1e-6 * L2 ** 0.5 + 1e-9
        hits = []
        for idx, p in enumerate(points):
            pv = np.asarray(p, float)
            t = float((pv - a) @ ab) / L2
            if not (1e-6 < t < 1.0 - 1e-6):
                continue                        # not strictly interior
            if float(np.linalg.norm(a + t * ab - pv)) > tol:
                continue                        # off the line
            hits.append((t, idx))
        hits.sort()
        out = []
        for t, idx in hits:                     # drop coincident duplicates
            if not out or abs(t - out[-1][0]) > 1e-9:
                out.append((t, idx))
        return out

    @staticmethod
    def _splittable(mb) -> bool:
        """Whether a member may be auto-split at slab edges (BE5). Excluded:
        *cables* (pin-ended axial members — a mid-span node would add a spurious
        joint) and *hinged* members (a lumped end hinge assumes one element
        end-to-end; splitting would misplace it). Lumped hinges are 2-D today and
        splitting is 3-D, so these never actually co-occur — this keeps the two
        correct if 3-D lumped hinges are ever added."""
        return (getattr(mb, "kind", "") != "cable"
                and getattr(mb, "hinge", None) is None)

    def _member_split_chain(self, mb, edge_nodes: dict):
        """BE1: ordered node chain ``[n1, ...interior slab edge nodes..., n2]``
        for a member whose line passes through slab edge nodes, or ``None`` when
        none lie on it. Cables and hinged members are never split (:meth:`_splittable`)."""
        if not self._splittable(mb) or not edge_nodes:
            return None
        coord = {nd.id: (nd.x, nd.y, nd.z) for nd in self.nodes}
        if mb.n1 not in coord or mb.n2 not in coord:
            return None
        ids = list(edge_nodes.keys())
        pts = [edge_nodes[i] for i in ids]
        hits = self._interior_params(coord[mb.n1], coord[mb.n2], pts)
        if not hits:
            return None
        return [mb.n1] + [ids[idx] for _t, idx in hits] + [mb.n2]

    def member_element_tags(self, mb) -> list:
        """BE3: the engine element tag(s) a member owns — ``[mb.id]`` when it is a
        single element, or its split sub-element tags when it lies along a meshed
        slab edge (BE2). Recomputed from geometry (mirrors :meth:`_area_element_tags`)
        so results/design can map a member to its elements without build state.
        Kept in lock-step with build via the shared :meth:`_interior_params`."""
        if (self.ndm != 3 or not self.areas
                or not getattr(self, "auto_connect_beams", True)
                or not self._splittable(mb)):
            return [mb.id]
        coord = {nd.id: (nd.x, nd.y, nd.z) for nd in self.nodes}
        if mb.n1 not in coord or mb.n2 not in coord:
            return [mb.id]
        hits = self._interior_params(coord[mb.n1], coord[mb.n2],
                                     self._area_edge_coords())
        if not hits:
            return [mb.id]
        return [member_element_tag(mb.id, j) for j in range(len(hits) + 1)]

    def _add_offset_beam(self, model, node_at, nodes, tags, make_el,
                         offset: float) -> None:
        """BE6: build a composite (eccentric) beam. The beam runs on *phantom*
        nodes a vertical distance ``offset`` (m) from each drawn node in
        ``nodes``; every phantom node is rigidly tied back to its drawn node
        (which carries the slab shells + supports) by a ``RigidOffset`` with
        coupled rotations, so the beam's stiffness condenses onto the drawn nodes
        through the offset arm — full T-beam action (plane sections remain
        plane), with no support surgery. ``tags`` are the per-span element tags
        (identical to the centroidal case, so the results/design map is
        unchanged)."""
        from femsolver.constraints import RigidOffset
        phantom = []
        for c in nodes:
            x, y, z = (float(v) for v in model.node(c).coords[:3])
            p = node_at((x, y, z + offset))
            phantom.append(p)
            model.add_mp_constraint(
                RigidOffset(master=c, slave=p, couple_slave_rotations=True))
        for j in range(len(nodes) - 1):
            model.add_element(make_el(tags[j], phantom[j], phantom[j + 1]))

    def _mesh_and_add_areas(self, model, mats, node_at) -> None:
        """Mesh every ``Area`` into shell elements and add them to ``model``
        (slab S1 + S3). Quads are subdivided ``mesh`` = (n1, n2) times by a
        bilinear map; a polygon (≥5 nodes) is centroid-fan-triangulated into one
        shell triangle per edge; a triangle stays a single element. ``node_at``
        (shared with beam-splitting) merges coincident nodes by coordinate, so
        shared edges/corners across areas — and beam-split nodes — collapse to
        one, keeping meshes compatible."""
        shsecs = {s.id: s for s in self.shell_sections}
        ncoord = {nd.id: (nd.x, nd.y, nd.z) for nd in self.nodes}
        _node_at = node_at

        for a in self.areas:
            ss = shsecs.get(a.shell_section)
            mat = mats.get(a.material)
            if ss is None or mat is None:
                continue
            if any(nid not in ncoord for nid in a.nodes):
                continue                          # dangling node reference
            if len(a.nodes) == 4:
                corners = [ncoord[nid] for nid in a.nodes]
                n1 = max(1, int(a.mesh[0]))
                n2 = max(1, int(a.mesh[1]))
                cells = area_quad_cells(a)            # kept cells (W5 openings)
                grid = {}                             # only nodes a kept cell uses
                for (i, j) in area_quad_needed_nodes(a):
                    grid[(i, j)] = _node_at(_bilinear(corners, i / n1, j / n2))
                for (k, i, j) in cells:
                    quad = [grid[(i, j)], grid[(i + 1, j)],
                            grid[(i + 1, j + 1)], grid[(i, j + 1)]]
                    el = _build_shell_element(
                        area_element_tag(a.id, k), quad, mat, ss)
                    if el is not None:
                        model.add_element(el)
            elif len(a.nodes) >= 5:               # polygon → centroid fan of tris
                corners = [ncoord[nid] for nid in a.nodes]
                cx = sum(c[0] for c in corners) / len(corners)
                cy = sum(c[1] for c in corners) / len(corners)
                cz = sum(c[2] for c in corners) / len(corners)
                cnode = _node_at((cx, cy, cz))
                N = len(a.nodes)
                for k in range(N):
                    tri = [cnode, a.nodes[k], a.nodes[(k + 1) % N]]
                    el = _build_shell_element(area_element_tag(a.id, k), tri,
                                              mat, ss)
                    if el is not None:
                        model.add_element(el)
            else:                                 # triangle → single element
                el = _build_shell_element(area_element_tag(a.id, 0),
                                          a.nodes, mat, ss)
                if el is not None:
                    model.add_element(el)

    def build_buckling_model(self, selection=("all", None),
                             subdivisions: int = 6):
        """Compile a model for linear (eigenvalue) buckling: each member is
        meshed into ``subdivisions`` sub-elements so member (Euler) buckling
        between joints is captured, not just global sway. The reference load
        ``selection`` (``("all", None)`` / ``("case", id)`` /
        ``("combination", id)``) is applied — the buckling factor λ multiplies
        it. 2-D only for now (``BeamColumn2D`` carries the geometric stiffness).

        Returns ``(model, member_subelems)`` where ``member_subelems`` maps
        each original member id to its list of sub-element ids. Works for 2-D
        (``BeamColumn2D``) and 3-D (``BeamColumn3D``) — both carry the
        geometric stiffness eigenvalue buckling needs."""
        from femsolver import (BeamColumn2D, BeamColumn3D, ElasticIsotropic,
                               Model, Truss2D, Truss3D)
        import numpy as np

        d3 = self.ndm == 3
        k = max(1, int(subdivisions))

        m = Model(ndm=self.ndm, ndf=self.ndf)
        mats = {}
        for mat in self.materials:
            obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu,
                                   rho=getattr(mat, "rho", 0.0))
            m.add_material(obj)
            mats[mat.id] = obj
        secs = {s.id: s for s in self.sections}
        gsd_mats: dict = {}

        coord = {}
        for nd in self.nodes:
            c = (nd.x, nd.y, nd.z) if d3 else (nd.x, nd.y)
            m.add_node(nd.id, *c)
            coord[nd.id] = np.array(c, dtype=float)
        nid = max((nd.id for nd in self.nodes), default=0) + 1
        eid = max((mb.id for mb in self.members), default=0) + 1

        member_subelems: dict = {}
        for mb in self.members:
            sec = secs[mb.section]
            A, Iz, Iy, J = _resolve_section(sec)
            material = mats[mb.material]
            # a cable / truss stays a single pin-ended element (no sub-division,
            # no bending — carries axial only)
            if getattr(mb, "kind", "") == "cable":
                cls = Truss3D if d3 else Truss2D
                m.add_element(cls(eid, (mb.n1, mb.n2), material, A))
                member_subelems[mb.id] = [eid]
                eid += 1
                continue
            if getattr(sec, "gsd_spec", None):
                cm = gsd_mats.get(sec.id)
                if cm is None:
                    try:
                        Ec = _gsd_modulus(sec.gsd_spec)
                    except Exception:
                        Ec = None
                    if Ec:
                        cm = ElasticIsotropic(10_000 + sec.id, E=Ec, nu=0.2,
                                              rho=getattr(material, "rho", 0.0))
                        m.add_material(cm)
                        gsd_mats[sec.id] = cm
                if cm is not None:
                    material = cm
            p1, p2 = coord[mb.n1], coord[mb.n2]
            chain = [mb.n1]
            for j in range(1, k):
                t = j / k
                m.add_node(nid, *(p1 + t * (p2 - p1)))
                chain.append(nid)
                nid += 1
            chain.append(mb.n2)
            subs = []
            for a, b in zip(chain[:-1], chain[1:]):
                if d3:
                    m.add_element(BeamColumn3D(eid, (a, b), material,
                                               A, Iy, Iz, J))
                else:
                    m.add_element(BeamColumn2D(eid, (a, b), material, A, Iz))
                subs.append(eid)
                eid += 1
            member_subelems[mb.id] = subs

        for nd in self.nodes:
            if nd.supports and any(nd.supports):
                m.fix(nd.id, list(nd.supports))

        nodal, member = self.resolved_loads(selection)
        for node_id, comps in nodal.items():
            m.add_nodal_load(node_id, list(comps))
        for mid, (wy, wz) in member.items():
            for sid in member_subelems.get(mid, []):
                el = m.element(sid)
                if not hasattr(el, "add_uniform_load"):
                    continue
                if d3:
                    el.add_uniform_load(wy, wz)
                else:
                    el.add_uniform_load(wy)
        return m, member_subelems

    # ------------------------------------------------- load application
    def _selection_factors(self, selection) -> dict:
        """{case_id: factor} for a load selection (solving or display)."""
        kind = selection[0] if selection else "all"
        if kind == "case":
            return {selection[1]: 1.0}
        if kind == "combination":
            combo = self.combination(selection[1])
            return dict(combo.factors) if combo else {}
        return {c.id: 1.0 for c in self.load_cases}

    def _self_weight_member_nodal(self, factor: float) -> dict:
        """Lumped member self-weight as nodal gravity forces (global-down):
        each member's weight ρ·A·L·g is split equally to its two end nodes.
        Lumped rather than consistent so it is correct for any member
        orientation, including columns whose self-weight is purely axial (a
        local beam UDL cannot represent that). ``factor`` scales the whole
        contribution. Returns ``{node_id: [components]}``; empty when the factor
        is zero or no material carries a density."""
        if not factor:
            return {}
        mats = {m.id: m for m in self.materials}
        secs = {s.id: s for s in self.sections}
        down = 1 if self.ndm == 2 else 2      # gravity DOF: Y in 2-D, Z in 3-D
        out: dict = {}
        for mb in self.members:
            mat = mats.get(mb.material)
            sec = secs.get(mb.section)
            if mat is None or sec is None:
                continue
            rho = float(getattr(mat, "rho", 0.0))
            if rho <= 0.0:
                continue
            A = _resolve_section(sec)[0]
            L = self.member_length(mb)
            half = 0.5 * rho * A * L * GRAVITY * factor
            if not half:
                continue
            for nid in (mb.n1, mb.n2):
                acc = out.setdefault(nid, [0.0] * self.ndf)
                acc[down] -= half
        return out

    def resolved_loads(self, selection=("all", None)):
        """(nodal, member) factored applied-load maps for a selection —
        nodal = {node_id: [components]}, member = {member_id: (wy, wz)}."""
        factors = self._selection_factors(selection)
        nodal: dict = {}
        for ld in self.loads:
            f = factors.get(ld.case, 0.0)
            if not f:
                continue
            acc = nodal.setdefault(ld.node, [0.0] * self.ndf)
            for k, v in enumerate(ld.values):
                if k < len(acc):
                    acc[k] += v * f
        for c in self.load_cases:                 # member self-weight (lumped)
            sw = getattr(c, "self_weight_factor", 0.0)
            f = factors.get(c.id, 0.0)
            if not (sw and f):
                continue
            for nid, comps in self._self_weight_member_nodal(sw * f).items():
                acc = nodal.setdefault(nid, [0.0] * self.ndf)
                for k, v in enumerate(comps):
                    if k < len(acc):
                        acc[k] += v
        member: dict = {}
        for ml in self.member_loads:
            f = factors.get(ml.case, 0.0)
            if not f:
                continue
            wv = member.setdefault(ml.member, [0.0, 0.0])
            wv[0] += ml.wy * f
            wv[1] += ml.wz * f
        return nodal, member

    def apply_loads(self, model, selection=("all", None)) -> None:
        """Clear the model's loads and apply one selection:

        * ``("all", None)`` — every load pattern ×1 (the default build);
        * ``("case", id)`` — a single load pattern ×1;
        * ``("combination", id)`` — a saved :class:`LoadCombination`;
        * ``("applied", [(pattern_id, scale), …])`` — an explicit **Loads
          Applied** spec (SAP2000-style), each pattern scaled and summed
          (E3b). Zero-scale rows are skipped; a repeated pattern accumulates.

        The ``"applied"`` form lets an analysis case pin *which* patterns it
        loads (e.g. ``1.0 Dead + 0.5 Live``) without inventing a combination,
        reusing the same :meth:`apply_case` scaling machinery as the others."""
        model.clear_loads()
        kind = selection[0] if selection else "all"
        if kind == "case":
            self.apply_case(model, selection[1], 1.0)
        elif kind == "combination":
            combo = self.combination(selection[1])
            if combo:
                for cid, f in combo.factors.items():
                    if f:
                        self.apply_case(model, cid, f)
        elif kind == "applied":
            for pid, scale in normalize_loads_applied(selection[1]):
                self.apply_case(model, pid, scale)
        else:                                    # "all" (or unknown → all)
            for c in self.load_cases:
                self.apply_case(model, c.id, 1.0)

    def _apply_member_load(self, model, ml, factor: float) -> None:
        # BE4: a member split at a slab edge (BE2) is several collinear
        # sub-elements. ``add_uniform_load`` is an *intensity* (N/m) and each
        # element integrates it over its own length, so the same intensity on
        # every sub-element reproduces the whole member's UDL exactly — no
        # span-fraction weighting needed. Unsplit members resolve to [ml.member].
        mb = next((m for m in self.members if m.id == ml.member), None)
        tags = self.member_element_tags(mb) if mb is not None else [ml.member]
        for tag in tags:
            try:
                el = model.element(tag)
            except KeyError:
                continue
            if not hasattr(el, "add_uniform_load"):
                continue
            if self.ndm == 3:
                el.add_uniform_load(ml.wy * factor, ml.wz * factor)
            else:
                el.add_uniform_load(ml.wy * factor)

    def _area_element_tags(self, area) -> list:
        """Engine element tag(s) for an ``Area`` — every mesh sub-element (slab
        S3). A quad owns ``n1·n2`` sub-elements; a polygon (≥5 nodes) owns one
        centroid-fan triangle per edge; a triangle owns one. Matches the
        deterministic tags emitted by :meth:`_mesh_and_add_areas`."""
        n_nodes = len(area.nodes)
        if n_nodes == 4:                          # kept cells only (W5 openings)
            return [area_element_tag(area.id, k) for (k, _i, _j)
                    in area_quad_cells(area)]
        if n_nodes >= 5:
            n = n_nodes                           # centroid-fan: one tri per edge
        else:
            n = 1
        return [area_element_tag(area.id, k) for k in range(n)]

    def _area_selfweight(self, area) -> float:
        """Self-weight surface load ρ·t·g (Pa) for an area, from its material
        density and shell-section thickness (slab plan S5 self-weight). 0 if the
        material carries no density."""
        ss = self.shell_section(area.shell_section)
        mat = next((m for m in self.materials if m.id == area.material), None)
        rho = float(getattr(mat, "rho", 0.0)) if mat else 0.0
        t = float(getattr(ss, "thickness", 0.0)) if ss else 0.0
        return rho * t * GRAVITY

    def _apply_area_load(self, model, al, factor: float) -> None:
        a = self.area(al.area)
        if a is None:
            return
        pressure = al.w * factor if al.kind == "pressure" else None
        if al.kind == "selfweight":
            w = self._area_selfweight(a)              # ρ·t·g, computed here
        else:
            w = al.w
        for tag in self._area_element_tags(a):
            try:
                el = model.element(tag)
            except KeyError:
                continue
            if pressure is not None and hasattr(el, "add_pressure"):
                el.add_pressure(pressure)
            elif hasattr(el, "add_surface_load"):
                el.add_surface_load(0.0, 0.0, -w * factor)   # global −Z gravity

    def apply_case(self, model, case_id: int, factor: float = 1.0) -> None:
        """Add one case's nodal + line loads to a built model, scaled by
        ``factor`` (additive — pair with ``model.clear_loads()`` between
        combos). A case with a non-zero ``self_weight_factor`` also contributes
        the structure's own gravity weight (see :class:`LoadCase`)."""
        for ld in self.loads:
            if ld.case == case_id:
                model.add_nodal_load(ld.node, [v * factor for v in ld.values])
        for ml in self.member_loads:
            if ml.case == case_id:
                self._apply_member_load(model, ml, factor)
        for al in self.area_loads:
            if al.case == case_id:
                self._apply_area_load(model, al, factor)
        case = self.case(case_id)
        sw = getattr(case, "self_weight_factor", 0.0) if case else 0.0
        if sw:
            self._apply_self_weight(model, sw * factor)

    def _apply_self_weight(self, model, factor: float) -> None:
        """Apply the whole structure's self-weight to a built model, scaled by
        ``factor`` (global-down): members as lumped nodal gravity, areas (3-D
        shells) as a global −Z surface load ρ·t·g."""
        if not factor:
            return
        for nid, comps in self._self_weight_member_nodal(factor).items():
            try:
                model.add_nodal_load(nid, comps)
            except (KeyError, IndexError):
                continue                          # node absent from this build
        if self.ndm == 3:
            for a in self.areas:
                w = self._area_selfweight(a) * factor
                if not w:
                    continue
                for tag in self._area_element_tags(a):
                    try:
                        el = model.element(tag)
                    except KeyError:
                        continue
                    if hasattr(el, "add_surface_load"):
                        el.add_surface_load(0.0, 0.0, -w)

    def load_patterns(self) -> dict:
        """{str(case_id): LoadPattern} for the engine's combination/envelope
        drivers — one named, scalable pattern per load case."""
        from femsolver.analysis.load_combinations import LoadPattern
        return {str(c.id): LoadPattern(
            str(c.id),
            lambda model, factor=1.0, cid=c.id: self.apply_case(
                model, cid, factor))
            for c in self.load_cases}


def _load_runs(raw):
    """Deserialize stored run records (plan §16 G-S2) into ``RunRecord``s,
    tolerant of unknown keys. Imported lazily to keep the project model free of
    a hard dependency on the (desktop-only) results layer."""
    if not raw:
        return []
    from nl_runs import RunRecord
    return [RunRecord.from_dict(r) for r in raw]


def _coerce_ic(raw) -> tuple:
    """Normalize a stored initial condition (E2) to a canonical tuple. JSON
    round-trips a tuple to a list, and old projects have no field at all, so
    coerce here: ``("state", <int id>)`` when it names a source nonlinear case,
    else the default ``("zero",)`` (unstressed)."""
    try:
        if raw and raw[0] == "state" and raw[1] is not None:
            return ("state", int(raw[1]))
    except (TypeError, IndexError, ValueError):
        pass
    return ("zero",)


def normalize_loads_applied(rows) -> list:
    """Normalize a *Loads Applied* spec (E3b) — a list of ``(pattern_id,
    scale)`` rows — into clean ``(int, float)`` pairs.

    JSON round-trips each tuple to a list and hand-built / migrated specs may
    carry ``None`` ids or blank rows, so coerce defensively and drop anything
    with a missing id or a zero scale (a zero contributes nothing). Order is
    preserved and a repeated pattern is kept — :meth:`Project.apply_case` is
    additive, so duplicate rows accumulate, matching SAP's grid."""
    out = []
    for row in (rows or []):
        try:
            pid = int(row[0])
            scale = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if scale:
            out.append((pid, scale))
    return out


def _coerce_node(n: dict) -> dict:
    n = dict(n)
    if n.get("supports") is not None:
        n["supports"] = tuple(n["supports"])
    return n


def _norm_label(v) -> "str | None":
    """Normalize a pier/spandrel label: strip whitespace, and treat empty /
    ``None`` as unset (``None``). Keeps "no label" single-valued (wall plan W0)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _resolve_section(section):
    """(A, Iz, Iy, J) for analysis — from a General-Section-Designer section
    when the section carries a ``gsd_spec`` (concrete/PSC/composite: gross A and
    inertias from the built section), else from the AISC catalog when it names a
    W-shape (Iz = strong-axis Ix), else the section's own values with positive
    fallbacks for the 3-D-only Iy / J."""
    if getattr(section, "gsd_spec", None):
        try:
            return _gsd_section_props(section.gsd_spec)
        except Exception:
            pass
    if section.shape:
        try:
            from femsolver.design.steel.sections import get_section
            ss = get_section(section.shape.replace("X", "x"))
            return ss.A, ss.Ix, ss.Iy, ss.J
        except Exception:
            pass
    Iy = section.Iy or section.Iz
    J = section.J or (0.1 * section.Iz)
    return section.A, section.Iz, Iy, J


# ---------------------------------------------------- shell / area resolution (S1)

def _has_modifiers(mods) -> bool:
    """True if any modifier deviates from 1.0 (so the fast, section-free path
    can be used when they are all unity)."""
    if not mods:
        return False
    try:
        return any(abs(float(v) - 1.0) > 1e-12 for v in mods.values())
    except (TypeError, ValueError):
        return True


class _ModifiedShellSection:
    """Wrap an engine ``ShellSectionBase`` and scale its constitutive matrices
    by SAP-style stiffness modifiers (all default 1.0):

    * membrane ``f11 f22 f12`` scale ``D_membrane``,
    * bending  ``m11 m22 m12`` scale ``D_bending``,
    * shear    ``v13 v23``     scale ``D_shear``.

    Each 3×3 (or 2×2) matrix entry (i, j) is scaled by ``sqrt(s_i * s_j)`` so
    the result stays symmetric and reduces to uniform scaling when the factors
    are equal — the standard interpretation of area stiffness modifiers. Passed
    to ``ShellMITC4`` / ``ShellTri3`` via ``section=``. Mass/weight modifiers
    are applied later (loads), not here."""

    def __init__(self, base, modifiers: dict):
        self._base = base
        self._m = {k: float(v) for k, v in (modifiers or {}).items()}

    @property
    def thickness(self) -> float:
        return self._base.thickness

    @property
    def density(self) -> float:
        return self._base.density

    @property
    def k_shear(self) -> float:
        return float(getattr(self._base, "k_shear", 5.0 / 6.0))

    def _scale(self, D, keys):
        import numpy as np
        s = np.array([self._m.get(k, 1.0) for k in keys], dtype=float)
        return np.asarray(D) * np.sqrt(np.outer(s, s))

    def D_membrane(self):
        return self._scale(self._base.D_membrane(), ("f11", "f22", "f12"))

    def D_bending(self):
        return self._scale(self._base.D_bending(), ("m11", "m22", "m12"))

    def D_coupling(self):
        return self._base.D_coupling()

    def D_shear(self):
        return self._scale(self._base.D_shear(), ("v13", "v23"))


def _build_shell_element(tag: int, node_tags, material, shell_section):
    """Build one engine surface element for an ``Area`` (S1: one element per
    area; S3 meshes). Maps the ``ShellSection.kind`` to an element:

    * ``shell-*`` / ``membrane`` → ``ShellMITC4`` (quad) or ``ShellTri3`` (tri)
      — the general shells, which accept a ``section=`` so stiffness modifiers
      apply. (A dedicated 6-DOF membrane element is deferred; the general shell
      is used, its bending negligible for in-plane-loaded panels.)
    * ``plate-*`` → ``ShellDKMQ4`` (quad) or ``ShellDKT3`` (tri) — bending-only
      plates (material + thickness; modifiers not yet supported on these).

    Returns the element, or ``None`` for an unbuildable area (bad node count /
    non-positive thickness)."""
    nt = tuple(int(n) for n in node_tags)
    n = len(nt)
    if n not in (3, 4):
        return None
    kind = getattr(shell_section, "kind", "shell-thin")
    t = float(getattr(shell_section, "thickness", 0.0))
    if t <= 0.0:
        return None
    mods = getattr(shell_section, "modifiers", None)

    if kind.startswith("plate"):
        from femsolver import ShellDKMQ4, ShellDKT3
        cls = ShellDKMQ4 if n == 4 else ShellDKT3
        return cls(tag, nt, material, t)

    # general shell (default) and membrane fall-back
    from femsolver import ShellMITC4, ShellTri3
    cls = ShellMITC4 if n == 4 else ShellTri3
    if _has_modifiers(mods):
        from femsolver import ElasticShellSection
        base = ElasticShellSection(material, t)
        return cls(tag, nt, material, section=_ModifiedShellSection(base, mods))
    return cls(tag, nt, material, t)


def _spec_from_gsd(gsd_spec: dict):
    """Deserialize a stored ``gsd_spec`` dict into a ``section_gui_core.Spec``.
    Imported lazily so the core FEM model never hard-depends on the
    Section-Designer engine; repo root is put on sys.path since the SD engine
    lives one level above ``desktop/``."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import section_gui_core as core
    fields = set(core.Spec.__dataclass_fields__)
    return core.Spec(**{k: v for k, v in gsd_spec.items() if k in fields})


@lru_cache(maxsize=128)
def _gsd_built_section(spec):
    """The built ``section_gui_core`` Section for a Spec (cached; Spec is a
    frozen/hashable dataclass, so props + modulus share one build)."""
    import section_gui_core as core
    return core.build_case(spec).section


def _gsd_section_props(gsd_spec: dict):
    """(A, Iz, Iy, J) in SI for a General-Section-Designer section, built from
    its serialized ``section_gui_core.Spec``."""
    sec = _gsd_built_section(_spec_from_gsd(gsd_spec))
    Iz, Iy = sec.I_zz, sec.I_yy
    J = sec.J or (Iz + Iy)          # St-Venant fallback ~ polar for solid shapes
    return sec.area, Iz, Iy, J


def _gsd_modulus(gsd_spec: dict):
    """Concrete elastic modulus E_c [Pa] of a General-Section-Designer section
    (its section's ``primary_material.Ec`` — e.g. ACI 4700·√f'c), so a concrete
    frame member's stiffness reflects concrete, not a steel material assigned in
    the model. Returns None when no concrete modulus is available."""
    sec = _gsd_built_section(_spec_from_gsd(gsd_spec))
    Ec = getattr(getattr(sec, "primary_material", None), "Ec", None)
    return float(Ec) if Ec else None
