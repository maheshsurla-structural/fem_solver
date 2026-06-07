# RFC: Theme II — Unified Section Designer

**Status:** APPROVED 2026-05-31 — II.2 in progress
**Date:** 2026-05-31
**Author:** femsolver core
**Reviewer:** Hisham (MIDAS Soft)
**Decisions recorded:** Shapely (Q1) / Embedded dicts (Q2) / SI everywhere (Q3)

> **Approval gate.** No code refactoring begins until this RFC is
> approved. The proposed API and migration plan need a green light
> first so we do not regret the architecture during the wide-touch
> work in II.6 / II.7.

---

## 1. Executive summary

Today, "section" is represented by **eight different data types** in
seven directories of the codebase. They share no common base, no
common serialization, and no common visualization. The proposal is
to introduce a single canonical `Section` object that all subsystems
(analysis, design, bridges, reports, BOM, JSON) consume, plus a
user-facing `SectionDesigner` library that constructs every section
type from one entry point.

The existing `sections/` module already has the right ABC
(`SectionBase` with `get_response`, `commit_state`, `clone`,
`is_stateful`). We **grow** it; we do **not** replace it. All
existing user code keeps working through thin wrappers.

---

## 2. Audit — every section-like type in the codebase today

| # | Type | File | Inherits SectionBase? | Carries get_response? | Used by |
|---|---|---|---|---|---|
| 1 | `SectionBase` (ABC) | `sections/base.py` | (is the base) | (abstract) | -- |
| 2 | `ElasticSection2D` | `sections/elastic.py` | yes | yes (diag stiffness) | Beam elements |
| 3 | `ElasticSection3D` | `sections/elastic.py` | yes | yes (diag stiffness) | Beam elements |
| 4 | `Fiber` | `sections/fiber.py` | no (data class) | (held by section) | FiberSection |
| 5 | `FiberSection2D` | `sections/fiber.py` | yes | yes (per-fiber aggregate) | Beam elements + walls + bridges |
| 6 | `FiberSection3D` | `sections/fiber.py` | yes | yes (per-fiber aggregate + GJ) | Beam-3D elements + walls |
| 7 | `ShellSectionBase` | `sections/shell.py` | separate ABC | yes (D matrix) | Shell elements |
| 8 | `ElasticShellSection` | `sections/shell.py` | yes (shell-ABC) | yes | Shell elements |
| 9 | `LayeredShellSection` | `sections/shell.py` | yes (shell-ABC) | yes | Composite shells |
| 10 | `WallRegion` + factories | `sections/wall.py` | factories return FiberSection2D/3D | (delegates) | Tall-building walls |
| 11 | `CrackedSectionFactors` | `sections/wall_shear.py` | no (helpers) | -- | Wall stiffness mods |
| 12 | `PlyStrength` + indices | `sections/ply_failure.py` | no (helpers) | -- | Composite ply failure |
| 13 | `CompositeSectionProps` | `bridges/composite_section.py` | **no** | **no** (transformed props only) | Bridge girder+deck checks |
| 14 | `ConcreteSection` + `RebarLayout` + `ConcreteMaterial` | `design/concrete/section.py` | **no** | **no** (geometry + bars) | ACI 318 / EC2 / IS 456 design |
| 15 | `SteelSection` (W-shape) | `design/steel/sections.py` | **no** | **no** (catalogue lookup) | AISC 360 design |
| 16 | `SectionProperties` (EC) | `catalogs/sections_ec.py` | **no** | **no** (catalogue lookup) | EC3 design |
| 17 | `SectionProperties` (IS) | `catalogs/sections_is.py` | **no** (reuses EC type) | **no** | IS 800 design |
| 18 | `CompositeSection` (PT bridge) | `bridges/pt_tendon.py` | **no** | **no** (closed-form) | PSC bridges |

**Total: 18 distinct section-like types across 7 directories.** Only 8
of them (rows 1-10) live under `sections/`. Only 6 of them have a
`get_response()` method (so only those can drive a beam element).

---

## 3. What goes wrong because of this fragmentation

Three concrete failure modes that the codebase exhibits today:

**3.1 — Round-trip impossible.** A user designs a beam with
`ConcreteSection(b=300, h=600, RebarLayout(bottom_bars=("#8","#8","#8")))`.
The ACI module gives a beautiful DCR report. To run NLTHA on the same
beam, the user must independently build a `FiberSection2D` from
scratch -- there is no `ConcreteSection.to_fiber_section()` helper.
The two objects can drift out of sync, and they will.

**3.2 — JSON save/load loses design data.** The JSON model deck
(`io/json_deck.py`) round-trips `ElasticSection2D/3D` but not
`ConcreteSection.RebarLayout` or `SteelSection.designation`. Loading
a saved model loses the section identity.

**3.3 — Catalogue confusion.** A W14x90 lookup gives three different
objects depending on entry point: a `SteelSection` from the AISC
table, a `SectionProperties(family="W")` from the EC/IS catalogue
helpers (if mapped), and there is no way to get an `ElasticSection3D`
from it without manual construction. No single `Section` "is" a W14x90.

**3.4 (future) — Phase D landmines.** When we add timber CLT panels
(orthotropic plate stack), masonry wall sections (URM + RM with bond
beams), and aluminum extrusions (5xxx/6xxx with HAZ), each will
spawn its own ad-hoc section path if we do not unify first. **Three
new section paths from a single phase is exactly the kind of debt
that becomes permanent.**

---

## 4. Proposed unified architecture

### 4.1 Three orthogonal concepts

```
Section = (Geometry × Material(s) × Reinforcement?) + AnalysisStrategy
```

| Concept | Lives in | Purpose |
|---|---|---|
| `Geometry` | `sections/geometry/` | Shape only -- coordinates, holes, polygon outline. No materials, no analysis. Pure geometric properties (A, I_yy, I_zz, J, S, Z, ρ_paint area). |
| `Material` (already exists) | `materials/` | Constitutive behavior only. UniaxialMaterial, Plate, 3D continuum. |
| `Reinforcement` (new) | `sections/reinforcement/` | Rebar layout, prestressing tendons, FRP wrap, etc. Composes with geometry + material. |
| `AnalysisStrategy` (new) | inside `Section` | Picks `elastic` / `fiber` / `composite` / `custom` based on what user needs. The same physical section can yield an elastic response (for linear analysis) and a fiber response (for nonlinear), without duplication. |

### 4.2 The unified `Section` class

```python
class Section:
    """The unified, canonical cross-section object.

    Carries every piece of information any downstream subsystem needs:
    geometry, material(s), reinforcement, identity (catalogue or
    custom), and lazy-built analysis strategies. Concrete subclasses
    are `ParametricSection`, `CataloguedSection`, `CustomSection`,
    `CompositeSection`, `PrestressedSection`, `WallSection`, etc.
    """

    # ------------- identity --------------
    name: str                       # user-visible name, e.g. "B1 600x300"
    family: str                     # "I" | "rect" | "channel" | "T" | "polygon" | ...
    catalogue_ref: str | None       # "W14x90" | "IPE300" | "ISMB400" | None

    # ------------- geometry --------------
    geometry: Geometry              # has .area, .I_yy, .I_zz, .J, .S, .Z, .polygon

    # ------------- material(s) -----------
    material: Material | None       # primary material (or None for composite)
    composite: list[MaterialZone] | None  # for composite/PSC sections

    # ------------- reinforcement (optional) --
    rebar: RebarLayout | None       # for RC sections
    prestress: TendonLayout | None  # for PSC sections

    # ------------- analysis strategies (lazy) --
    def elastic_section_2d(self, *, transformed: bool = False) -> ElasticSection2D: ...
    def elastic_section_3d(self, *, transformed: bool = False) -> ElasticSection3D: ...
    def fiber_section_2d(self, *, n_fibers: int = 50) -> FiberSection2D: ...
    def fiber_section_3d(self, *, n_y: int = 20, n_z: int = 20) -> FiberSection3D: ...

    # ------------- design adapters (lazy) ----
    def as_aci_section(self) -> AciConcreteSection: ...
    def as_ec2_section(self) -> Ec2ConcreteSection: ...
    def as_is456_section(self) -> Is456ConcreteSection: ...
    def as_aisc_section(self) -> AiscSteelSection: ...
    def as_ec3_section(self) -> Ec3SteelSection: ...

    # ------------- serialization -------------
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, data: dict) -> "Section": ...

    # ------------- visualization -------------
    def to_polygon(self) -> Polygon: ...
    def to_dxf(self, path: str) -> None: ...
    def to_svg(self) -> str: ...
    def section_report(self) -> SectionReport: ...

    # ------------- BOM -----------------------
    def weight_per_length(self) -> float: ...
    def paint_area_per_length(self) -> float: ...
    def fabrication_data(self) -> FabricationData: ...
```

**Lazy strategies.** The fiber discretization is only built when
`fiber_section_2d()` is called, so a W14×90 lookup for an AISC
elastic check is still a one-microsecond catalogue read.

**Backward compatibility.** Existing `ElasticSection2D`,
`ElasticSection3D`, `FiberSection2D`, `FiberSection3D` keep working.
They are now *outputs* of `Section.elastic_section_*()` and
`Section.fiber_section_*()`, but you can still construct them
directly for low-level code paths.

### 4.3 The `SectionDesigner` (user-facing)

```python
from femsolver.sections import SectionDesigner, SectionLibrary

# 1. parametric primitive
sec = SectionDesigner.rectangle(b=0.3, h=0.6, material=concrete_C30)

# 2. catalogue lookup (one entry point for all standards)
sec = SectionDesigner.from_catalogue("W14x90", material=steel_A992)
sec = SectionDesigner.from_catalogue("IPE300", material=steel_S355)
sec = SectionDesigner.from_catalogue("ISMB400", material=steel_E250)

# 3. RC beam
sec = SectionDesigner.rc_rectangle(
    b=0.3, h=0.6, concrete=concrete_C30, rebar=RebarLayout(
        bottom_bars=("#8","#8","#8"),
        top_bars=("#6","#6"),
        bottom_cover=0.04,
        stirrup_designation="#3", stirrup_spacing=0.15,
    ),
)

# 4. composite girder + deck (replaces bridges/composite_section.py)
sec = SectionDesigner.composite_girder(
    girder=SectionDesigner.rectangle(b=0.6, h=1.5, material=concrete_girder),
    deck=SectionDesigner.rectangle(b=3.0, h=0.25, material=concrete_deck),
)

# 5. PSC girder with tendons
sec = SectionDesigner.psc_girder(
    girder=...,
    deck=...,
    tendons=TendonLayout(...),
)

# 6. custom polygon
sec = SectionDesigner.custom_polygon(
    outline=[(0,0),(0.5,0),(0.5,0.3),(0.4,0.3),(0.4,0.8),(0,0.8)],
    holes=[],
    material=steel_A992,
)

# 7. wall section (boundary + web)
sec = SectionDesigner.wall(
    L_w=4.0, L_be=0.4, t=0.25,
    confined_concrete=..., unconfined_concrete=..., rebar=...,
)

# 8. catalogue library (lookup by family)
lib = SectionLibrary.aisc()
w14x90 = lib["W14x90"]
ipe300 = SectionLibrary.eurocode()["IPE300"]

# Every one of the above returns the same `Section` type.
# Every one of them flows to:
sec.elastic_section_3d()        # for linear analysis
sec.fiber_section_3d()          # for nonlinear analysis
sec.as_aci_section()            # for ACI 318 design (where applicable)
sec.to_json()                   # for save/load
sec.to_svg()                    # for reports
sec.weight_per_length()         # for BOM
```

**This is the unique selling point.** A user builds the section
once; it flows everywhere.

### 4.4 What this is NOT

- It is **not** removing `ElasticSection2D/3D` or `FiberSection2D/3D`.
  Those become the *outputs* of the lazy adapters. Existing tests
  keep passing.
- It is **not** breaking the element API. Elements still receive
  `SectionBase` subclasses (now obtained via `sec.elastic_section_3d()`).
- It is **not** a new constitutive theory. The math is unchanged.
- It is **not** for the GUI. The GUI will *use* `SectionDesigner`, but
  Theme II is solver-only.

---

## 5. Migration plan (9 sub-phases)

| Sub-phase | Work | Tests added (est) | Touches | Risk |
|---|---|---|---|---|
| **II.1** | This RFC + audit | (doc only) | docs/ | low |
| **II.2** | `Geometry` + `Material` references + `Section` ABC + `SectionLibrary` (in-memory registry) | +15 | new files in `sections/` | low |
| **II.3** | Parametric primitives: `RectangularSection`, `ISection`, `ChannelSection`, `AngleSection`, `TSection`, `HollowRectSection`, `CircularSection`, `HollowCircularSection` -- each with gross-property formulas | +25 | new files in `sections/parametric/` | medium |
| **II.4** | Catalogued sections -- AISC + EC + IS now emit unified `Section` via `SectionLibrary.aisc() / .eurocode() / .indian()`; legacy `SteelSection` and `SectionProperties` become thin views over the new objects | +20 | `catalogs/`, `design/steel/sections.py` | medium |
| **II.5** | Custom polygon section: Shoelace formula for A, I_yy, I_zz; Green's-theorem for centroid; thin-wall approximation for J (or user-supplied) | +15 | new file `sections/custom.py` | medium |
| **II.6** | Adapter layer: `Section.elastic_section_2d/3d()`, `Section.fiber_section_2d/3d()`, `Section.as_aci_section()`, `Section.as_aisc_section()`, etc. -- **no consumer-side changes yet** | +20 | `sections/` only | low |
| **II.7** | Migrate consumers one subsystem at a time: bridges (composite + PSC), wall sections, design modules, calc sheets, BOM. Each migration step is a separate commit; tests stay green at every step. | +30 | wide-touch: `bridges/`, `design/`, `deliverables/`, `catalogs/` | **HIGH** |
| **II.8** | Section JSON round-trip (extend `io/json_deck.py`) + SVG sketcher + one-page PDF section report | +20 | `io/`, `sections/` | medium |
| **II.9** | Theme II capstone: `examples/68_section_designer_capstone.py` -- build 8 different section types, run each through analysis + design + report + JSON round-trip, demonstrate single-source-of-truth | (capstone) | examples/, docs/ | low |

**Tests added (target): +145.** Comparable to Theme HH (+153).

### 5.1 Wrapper strategy (II.7 risk mitigation)

The wide-touch migration in II.7 is the only high-risk step. The plan:

1. **Old APIs continue to work unchanged** through II.6. The new
   `Section` is *additive*.
2. **One consumer subsystem migrates per commit.** Order:
   - `bridges/composite_section.py` first (cleanest scope)
   - `design/concrete/section.py` second (well-tested)
   - `design/steel/sections.py` third
   - `catalogs/sections_ec.py` and `sections_is.py` fourth
   - `sections/wall.py` last (already uses FiberSection internally)
3. **Every commit runs the full test suite.** A single regression
   triggers a revert + re-think for that subsystem.
4. **Deprecation, not deletion.** Old constructors keep working for
   one full theme cycle (until after Phase C V&V). They emit
   `DeprecationWarning` so we know what still depends on them.
5. **One legacy-removal sweep** at the end of Phase C, after V&V has
   confirmed nothing critical depends on the old constructors.

### 5.2 What we explicitly defer to later phases

- **3D solid section integration** (for nonlinear shell-via-solid):
  out of Theme II scope. Section is still a beam concept.
- **Variable cross-section along element length** (tapered I-beams,
  haunched beams): out of scope. The element takes one `Section`
  per integration point already.
- **Section-level finite-element discretization** for warping
  torsion: deferred. We keep the uncoupled-torsion GJ assumption.
- **Heat transfer through cross-section** (for fire engineering):
  out of Theme II scope. Stays in `analysis/fire.py`.

---

## 6. Open questions for user decision

Three architectural decisions you should make before II.2 starts.

### Q1. Geometry: pure-Python polygon or accept a shapely dependency?

| Option | Pros | Cons |
|---|---|---|
| A. Pure Python polygons | Zero new deps; matches current "no NumPy beyond core" philosophy | We implement Shoelace + Green's theorem + polygon Boolean ops ourselves |
| **B. Shapely dependency** (DECIDED) | Battle-tested polygon ops; supports holes, Boolean ops, buffering | Adds a heavy GEOS dep; users must `pip install shapely` |

**Decision (Hisham, 2026-05-31): B.** Shapely. For composite sections
with multiple materials, holes in custom polygons, and the eventual
Boolean operations (girder + deck union, hollow box subtraction), the
maturity of the shapely/GEOS toolchain pays back the dependency cost.
Adds `shapely` to `pyproject.toml` runtime deps.

### Q2. Catalogue source — embed or import lazily?

| Option | Pros | Cons |
|---|---|---|
| **A. Embedded (current pattern)** (recommended) | Zero startup latency; deterministic | Source files are large dictionaries |
| B. Lazy JSON loader | Smaller binary footprint | First-call latency; harder to type-check |

**My recommendation: A.** Current pattern already works; don't fix
what isn't broken.

### Q3. Should `Section` carry units, or assume SI everywhere?

| Option | Pros | Cons |
|---|---|---|
| **A. SI everywhere** (recommended) | Matches the rest of the solver; no unit-conversion bugs | Users in US-customary land must convert at the boundary |
| B. Section carries units | Catalogues can store imperial natively | Risk of two `Section.area` values disagreeing in different unit systems |

**My recommendation: A.** Imperial → SI conversion happens once at
catalogue construction time, exactly as it does today in
`design/steel/sections.py`. The `Section` object is always SI.

---

## 7. Timeline (working estimate)

| Sub-phase | Estimate |
|---|---|
| II.1 audit + RFC (this) | done |
| II.2 ABC + library | 1 session |
| II.3 parametric primitives | 1-2 sessions |
| II.4 catalogues unified | 1 session |
| II.5 custom polygon | 1 session |
| II.6 adapters | 1 session |
| II.7 migration (5 commits) | 2-3 sessions |
| II.8 JSON + SVG + report | 1-2 sessions |
| II.9 capstone | 1 session |
| **Total** | **9-12 sessions** |

For comparison, Theme HH was 9 sub-phases in ~8 sessions and added
+153 tests. Theme II is similar in scope.

---

## 8. What happens after Theme II

With Section Designer in place:

- **Phase C (V&V)** is cleaner: every benchmark specifies its section
  in one canonical form; the V&V harness reads it via one API.
- **Phase D (new materials)** is *one extension point per material*,
  not three: timber CLT panels become a `CltSection`, masonry walls
  become a `MasonrySection`, glass laminates become a `GlassSection`
  -- each plugs into the same `Section` ABC.
- **Phase E (specialist analyses)** -- tunnels need lining sections,
  tanks need shell sections, all flow through the same path.
- **Tier 5 GUI** has one section-picker dialog. One.

The strategic gate (no GUI until the solver has nothing more to add)
still holds. Theme II makes "nothing more to add" achievable.

---

## 9. Action requested

Please review and respond with:

1. **Approve / reject / amend** the unified `Section` API (§4)
2. **Approve / reject / amend** the migration plan (§5)
3. **Answers to Q1-Q3** (§6) — or accept my recommendations as the
   default
4. **Anything missing** — sections types I haven't catalogued,
   downstream consumers I haven't accounted for, design decisions
   that need a different default

Once approved, II.2 (`Geometry` + `Section` ABC + `SectionLibrary`)
starts in the next session.
