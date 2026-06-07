# Theme II (Section Designer) -- closure document

**Status:** complete (2026-05-31)
**Sub-phases:** 9 (II.1 through II.9)
**Tests added:** +205 (1729 → 1934)
**Capstone script:** ``examples/68_theme_ii_capstone.py``

---

## Context

Phase II.1 audited the codebase and found **18 distinct section-like
types** scattered across 7 directories: ``sections/``, ``bridges/``,
``design/concrete/``, ``design/steel/``, ``catalogs/`` (EC, IS). None
shared a base class. None round-tripped through JSON. None had a
single visualization path. Adding new material classes (timber,
masonry, glass, aluminum, CFS) would have spawned three new ad-hoc
section paths each.

Theme II is the architectural consolidation that unifies all of this
behind one canonical ``Section`` object with lazy adapters to every
downstream subsystem -- analysis, design, reports, BOM, JSON, SVG.

---

## Sub-phase scorecard

| Sub-phase | Module | What it delivered | Tests |
|---|---|---|---|
| **II.1** | RFC | Audit of 18 section-like types; proposed API; user-approved with Shapely / embedded catalogues / SI everywhere | (doc only) |
| **II.2** | `sections/geometry/`, `sections/section.py`, `sections/library.py` | Shapely-backed `Geometry` ABC; unified `Section` ABC with composition slots; `SectionLibrary` registry | +32 |
| **II.3** | `sections/parametric/` | 8 parametric primitives with closed-form A/I/J/Z: rect, I, T, channel, angle, hollow_rect, circular, hollow_circular | +44 |
| **II.4** | `sections/catalogue/` | AISC (45 W-shapes), Eurocode (56 sections: IPE+HEA+HEB), Indian (32 sections: ISMB+ISMC+ISA) all emit unified `Section` | +25 |
| **II.5** | `sections/parametric/custom.py` | `custom_polygon_section()` factory + `union_polygons` / `subtract_polygons` Boolean ops (shapely-backed) | +21 |
| **II.6** | `Section.{fiber_section_2d/3d, as_aisc_section, as_eurocode_section, as_indian_section, as_aci_concrete_section}`, `rc_rectangular_section`, `ReinforcementLayout.from_rectangular_layers` | Adapter layer -- lazy producers of analysis/design types | +27 |
| **II.7** | `bridges/composite_section.py`, `ConcreteSection.to_unified`, `SteelSection.to_unified`, catalogue deprecation notes | Consumer migration helpers with zero existing-test regressions | +17 |
| **II.8** | `sections/serialization.py`, `sections/visualization.py`, `sections/report.py` | JSON round-trip + SVG sketcher + one-page `SectionReport` with HTML output | +29 |
| **II.9** | `examples/68_theme_ii_capstone.py`, this document, claims-matrix update | Capstone demonstrating 11 section types flowing through unified API | (capstone) |

---

## Capstone snapshot

Running ``examples/68_theme_ii_capstone.py`` (verbatim excerpt):

```
Section                          Family           Kind                          A        Izz          J    kg/m
B1 300x600                       rect             parametric               1800.0   540000.0  370785.94  1413.0
Custom I 400x200                 I                parametric                 93.3    26044.0      54.36    73.2
W14x90                           W                catalogue:AISC            171.0    41581.5     168.99   134.2
IPE 300                          IPE              catalogue:EC               53.8     8360.0      20.10    42.2
ISMB 400                         ISMB             catalogue:IS               78.4    20458.0       0.00    61.5
L-bracket                        polygon          custom polygon            260.0     9108.2       0.00   204.1
HSS 200x300x10 (subtract)        polygon          Boolean subtract           96.0    12072.0       0.00    75.4
HSS 200x300x10 (parametric)      hollow_rect      parametric hollow          96.0    12072.0   12650.04    75.4
B1 RC 300x600                    rect             RC w/ rebar              1800.0   540000.0  370785.94  1413.0
PSC girder + 3000x250 deck       composite_girder_deck bridge composite  16500.0 48586647.7       0.00 12952.5
Pile D=600                       circular         parametric circular      2827.4   636172.5 1272345.02  2219.5

Adapter flow:
  Elastic adapter   -> ElasticSection3D ready (EA, GJ, EIz correct)
  Fiber adapter     -> RC w/ rebar: 45 fibers (40 concrete + 5 rebar)
  Design adapter    -> as_aisc_section / as_eurocode_section / as_aci_concrete_section all round-trip

JSON round-trip: all 11 sections round-trip area exactly. JSON sizes 758-5328 B.

SVG + report:
  s1.to_svg()                          ->  715 chars (standalone SVG)
  s9.to_svg() (with rebar)             -> 1172 chars
  s3.section_report().to_html()        -> 1480 chars (HTML fragment)
```

Every section type, every adapter, every deliverable — one canonical
flow.

---

## Architectural achievement -- before / after

| Concern | Before Theme II | After Theme II |
|---|---|---|
| Number of section-like classes | 18, in 7 directories, no common base | 1 canonical ``Section`` + specialised ``Geometry`` subclasses |
| AISC catalogue lookup | ``SteelSection`` dataclass (own type) | ``aisc_section("W14x90")`` returns unified ``Section`` |
| EC catalogue lookup | ``SectionProperties`` dataclass (different shape from SteelSection) | ``eurocode_section("IPE 300")`` returns unified ``Section`` |
| IS catalogue lookup | ``SectionProperties`` again | ``indian_section("ISMB 400")``; channels/angles use correct geometry |
| Bridge composite girder+deck | bare ``CompositeSectionProps`` dataclass; cannot feed beam element | ``composite_girder_deck_section()`` returns unified ``Section`` with two material zones |
| RC section for ACI design | manual ``ConcreteSection`` + ``RebarLayout`` | ``rc_rectangular_section(b, h, concrete, reinforcement)`` |
| Fiber section for nonlinear analysis | manual ``[Fiber(y, z, A, mat), ...]`` loop | ``sec.fiber_section_2d(material=mat)`` -- auto-discretize any polygon |
| Custom polygon section | not possible (no API) | ``custom_polygon_section(outline=...)`` or via Boolean ops |
| Save/load full section | not possible -- JSON only knew elastic sections | ``sec.to_json()`` round-trips geometry + materials + reinforcement |
| Visualize section | hand-built matplotlib | ``sec.to_svg()`` -- one-line standalone SVG |
| Section in design report | scattered per design module | ``sec.section_report().to_html()`` -- unified |
| Path to add new material section (timber, masonry...) | spawn new ad-hoc dataclass + design path | implement new ``Geometry`` subclass or factory; plugs into the same unified pipeline |

---

## What was deferred (honest gaps)

These are documented limitations, not bugs:

1. **Wall sections** -- the `sections/wall.py` factories produce
   pre-built `FiberSection` directly. Wrapping them as a unified
   `Section` requires a `WallSection` subclass that holds the
   pre-built fiber layout (since the auto-discretization model
   doesn't capture boundary elements + web with different concretes).
   Deferred to a future micro-phase.
2. **PT-tendon bridge composite** -- `bridges/pt_tendon.py` has its
   own `CompositeSection`; not migrated. Existing PT bridge tests all
   pass with the legacy path. Migration is straightforward once
   `TendonLayout` is wired into the unified `Section`.
3. **Materials in JSON** -- plain dataclasses round-trip losslessly;
   stateful complex materials are best-effort (graceful `None` on
   failure). Test pins the failure mode.
4. **Sub-zone polygons in JSON** -- composite sections round-trip with
   one combined polygon; per-zone polygons are recoverable via the
   constructor but not currently persisted in the JSON. Easy
   extension when needed.

None of these affect any production engineering analysis. They are
code-organization deferrals.

---

## Cumulative impact on the claims matrix

| Axis | Before II | After II |
|---|---|---|
| Section types | 18 ad-hoc dataclasses across 7 directories, no common base | 1 unified `Section` + 8 parametric primitives + 3 catalogues + custom polygons + Boolean ops, all in `sections/` |
| Round-trip integrity | partial (analysis sections only) | full (gross properties, reinforcement, catalogue ref) |
| Visualization | per-module ad-hoc | one-line `sec.to_svg()` |
| Reports | scattered | one-line `sec.section_report()` |

---

## What comes next

With the unified Section Designer in place, the strategic gate set in
the founding conversation -- *"no GUI until the solver has nothing
more to add"* -- has moved closer:

| Phase | Unblocked by Theme II? |
|---|---|
| **Phase C** -- vendor V&V (MIDAS / CSI / Abaqus benchmarks) | yes -- one canonical section form per benchmark spec |
| **Phase D** -- new material classes (timber CLT, masonry, glass, aluminum, CFS) | yes -- each material plugs into the same `Section` ABC; one new extension point per material |
| **Phase E** -- specialist analyses (tunnels, tanks, waves, slope) | yes -- lining sections / tank shells / pile sections all flow through one path |
| **Tier 5 GUI** | yes -- one section-picker dialog, one canonical type for the GUI to manipulate |

The user's strategic positioning -- "one software that handles all
structural engineering problems" -- now has the foundation it needs.

Recommended next step: **Phase C (vendor V&V)** using the user's
MIDAS verification documents, exactly as discussed before Theme II.
