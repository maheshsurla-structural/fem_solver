# Loads & Analysis UX — Professional-Grade Redesign Plan

**Status:** Living document — the single source of truth for the Loads & Analysis GUI
redesign work stream (desktop app under `desktop/`).
**Owner rotation:** Multiple Claude sessions across multiple accounts.
**The git repo is the only shared state** (memory files do NOT cross accounts). Read this
file first, do **one** work item, update the [Status Tracker](#8-status-tracker), commit,
and stop.

**Goal in one line:** make the desktop app's Loads and Analysis surfaces look and behave
like serious commercial FEA software (CSiBridge / Midas Civil), reusing the design system we
already have — without changing the solver or the `project.py` data contracts.

**Reference screenshots (the target we are matching):** CSiBridge *Loads* ribbon; CSiBridge
*Load Case Data* dialogs for Static / Modal / Response Spectrum / Time History / Buckling;
Midas-style *Load cases* and *Add nonlinear case* dialogs. These are the visual language to
emulate: **titled group panels, two-column layout, a "Loads Applied" table with
Add/Modify/Delete, a Notes field, and a Type selector at the top of every case dialog.**

---

## 0. How to use this document (session protocol)

This work is done across many sessions on different Claude accounts. To stay consistent:

1. **Read this whole file before writing any code.** It freezes the widget names, the shared
   scaffold API, and the visual conventions so independent sessions converge on one look.
2. **Pick the lowest-numbered unchecked item** in the [Status Tracker](#8-status-tracker)
   whose dependencies are all checked. Phase **L1** (the shared scaffold) must be done first —
   most later items depend on it. One item ≈ one session.
3. **Code to the contracts in §5 (Shared scaffold API).** Do not rename the scaffold classes
   or change their signatures once L1 lands — later items import them. If a contract must
   change, edit §5 in the *same* commit and note it in §9 (Change log).
4. **Do not change `project.py` dataclasses or the solver.** This is a pure UI/UX work stream.
   The only data-model exception explicitly allowed is adding a `notes: str = ""` field to
   case dataclasses if a phase needs it — and only with a note in §9.
5. **Every item ships:** the widget/dialog + a headless smoke test under `tests/`
   (constructible under `QT_QPA_PLATFORM=offscreen`, per the GUI convention) + a screenshot
   pair (light + dark) saved under `phase21_outputs/loads_ux/` for the visual record.
6. **Update the Status Tracker** (check the box, add commit hash + date) and append a line to
   §9. Commit with a message tagged `feat(loads-ux Ln): …` or `feat(loads-ux An): …`.
7. **Reuse `style.py` and `widgets.py`. Never hand-code a colour, font size, radius, or
   spacing** — pull the token from `style` (e.g. `style.ACCENT`, `style.SP_MD`). If a token is
   missing, add it to `style.py` in the same commit.

Conventions in §4 (Design principles) are **binding**.

---

## 1. Why this work stream exists

The user's report: *"Particularly when it comes to Loads and Analysis, our GUI doesn't come up
to the professional software. Our windows have options but they don't look professional."*

The theme is **not** the problem. [`desktop/style.py`](../../desktop/style.py) is already a
mature design system: light/dark token palettes, and QSS that styles `QGroupBox` as cards,
plus KPI tiles, verdict pills, chips, segmented buttons, tables, tabs, a command palette. The
gap is that the Loads/Analysis dialogs were built as flat `QFormLayout` stacks and **do not use
any of it**. So this is an *arrangement + reuse* problem, not a *repaint* problem.

---

## 2. Current-state audit (as of 2026-09-13)

File references are `file:line` into `desktop/`.

### 2.1 Application shell — [`main_window.py`](../../desktop/main_window.py)
- Docked-panel window: left **Model** tree, right **Properties**, bottom **Output** log +
  **Analysis steps** scrubber. Good, keep it.
- Menus: File / Edit / Generate / Analysis / Select / Draw / View / Tools
  ([`main_window.py:226`](../../desktop/main_window.py:226)–`290`).
- **Problem — loads/analysis are scattered:** "Add load" lives under **Edit**
  (`:236`), "Generate loads" under **Generate** (`:247`), and everything else — "Load cases",
  "Generate combinations", "Run", "Nonlinear cases", "Pushover", "Time history", "Run history",
  "Check model", diagrams, Design — is piled into one flat **Analysis** menu (`:248`–`263`).
  There is no "Loads" home and no clear "define → run → view" flow.
- Toolbars are icon-only (`:292`) with terse tooltips; no captioned groups like a ribbon.

### 2.2 Loads dialogs — [`editing.py`](../../desktop/editing.py)
- `LoadDialog` (nodal load), `:158`–`190` — flat `QFormLayout`: Node, Load case, one force
  spin per DOF. No direction/sign diagram, no grouping, no help.
- `LoadCaseDialog`, `:193`–`271` — a bare 2-column table (Name, Nature) + `+ Case` / `Remove`.
  Functional but plain; 380×300; no icons, no analysis-type, no notes.
- `LoadGenDialog` (parametric gravity/lateral), `:463`–`493` — a 3-row form; no preview.
- **Gap:** `MemberLoad` (line/UDL loads) exists in the data model
  ([`project.py:156`](../../desktop/project.py:156)) but has **no add/edit dialog** anywhere.
- **Gap:** there is **no load-combination editor**. `LoadCombination`
  ([`project.py:166`](../../desktop/project.py:166)) is only ever produced by
  `Project.generate_asce7_combinations()` and edited by deletion. Every commercial tool has a
  combos table with a per-case factor grid; we have none.

### 2.3 Analysis / case dialogs
- `NonlinearCaseDialog` — [`nonlinear_cases.py:59`](../../desktop/nonlinear_cases.py:59)–`158`.
  **This is the "Add nonlinear case" screenshot.** ~14 rows in one flat single-column
  `QFormLayout`: Case id, Name, Control node, Control DOF, Protocol, Target, Steps, Axial
  preload, Axial node, Axial DOF, Continue from, Convergence tol, Max iterations. No panels, no
  two-column layout, no notes, no type header. This is the #1 "looks unprofessional" offender.
- `NonlinearCaseManagerDialog` — `:201`–`291`. A 4-column table (id/name/protocol/control) +
  Add/Edit/Delete. Fine bones, plain skin, no icons/status.
- `PushoverDialog` — [`pushover_dialog.py:78`](../../desktop/pushover_dialog.py:78). Already a
  left-form / right-live-plot split with a progress bar and convergence log (good!), but the
  left inputs are still a flat `QFormLayout` (`:91`).
- `TimeHistoryDialog` — [`timehistory_dialog.py`](../../desktop/timehistory_dialog.py). Same
  shape as pushover.
- "Run linear static" is a bare menu action with **no dialog** and no run-management surface
  ([`main_window.py:135`](../../desktop/main_window.py:135), `327`).

### 2.4 What we can build on (assets already in the repo)
- [`style.py`](../../desktop/style.py): tokens `SP_*`, `R_*`, `FS_*`, palette names
  (`ACCENT`, `PANEL`, `BORDER`, `MUTED`, `OK`, `BAD`, `WARN`, …); QSS for `QGroupBox` cards,
  `QLabel#eyebrow` / `#h2` / `#sub` / `#hintLabel`, `QToolButton#segbtn` / `#chip`, KPI tiles,
  pills. `style.apply(widget)` themes any top-level widget; light/dark + compact/comfortable.
- [`widgets.py`](../../desktop/widgets.py): `CollapsibleGroup` (titled card with a body layout).
- [`icons.py`](../../desktop/icons.py): `icons.icon(name)` vector icon factory.
- [`project.py`](../../desktop/project.py): `LoadCase`, `Load`, `MemberLoad`,
  `LoadCombination`, `NonlinearCase`, `NATURE_LABELS`, and `generate_asce7_combinations()`.

---

## 3. Target model — the CSiBridge "Load Case Data" pattern

Every case dialog we build should follow this recognizable structure (from the reference
screenshots), assembled from titled `QGroupBox` panels in a two-column grid:

```
┌─ Load Case Name ──────┐  ┌─ Notes ──────┐  ┌─ Load Case Type ───────────┐
│ [ ACASE1 ] [Set Def]  │  │ [Modify/Show]│  │ [ Static ▾ ]      [Design…]│
└───────────────────────┘  └──────────────┘  └────────────────────────────┘
┌─ Initial Conditions / Stiffness ─────────┐  ┌─ Analysis Type ────────────┐
│ ◉ Zero initial (unstressed)              │  │ ◉ Linear   ○ Nonlinear     │
│ ○ Continue from … [ case ▾ ]             │  └────────────────────────────┘
└───────────────────────────────────────────┘
┌─ Loads Applied ───────────────────────────────────────────────────────────┐
│  Load Type │ Load Name │ Function │ Scale │         [ Add ]                 │
│  ──────────┼───────────┼──────────┼───────│         [ Modify ]              │
│   …rows…                                   │         [ Delete ]              │
└─────────────────────────────────────────────────────────────────────────────┘
┌─ Other Parameters ────────────────────────┐              [ OK ]  [ Cancel ] │
│  Convergence tol │ Max iterations │ …      │                                 │
└───────────────────────────────────────────┘
```

The **"Loads Applied" table with Add/Modify/Delete** is the single most recognizable "serious
FEA" element. It is reused by combinations (factor grid), response spectrum, and time history.

---

## 4. Design principles (binding)

1. **Group everything.** No dialog with more than ~4 fields stays a flat form. Wrap each
   logical group in a `QGroupBox` (already styled as a card) or a `GroupCard` (§5). Titles are
   short nouns: *Control*, *Protocol*, *Initial Conditions*, *Solver*, *Loads Applied*.
2. **Two columns for dense dialogs.** Identity/type/controls on the left; the Loads-Applied
   table / options on the right. Use a `QGridLayout` of group cards, not one tall column.
3. **A consistent header block** on every case dialog: a **Name** field, a **Notes**
   (Modify/Show…) affordance, and a **Type** label/selector — top row, matching the reference.
4. **A "Loads Applied" table** (`LoadsAppliedTable`, §5) wherever a case references loads or
   factors — with right-aligned Add / Modify / Delete buttons.
5. **Icons + eyebrow captions.** Case-type icons on list rows; `QLabel#eyebrow` uppercase
   micro-captions above groups where it aids scanning.
6. **Inline help, not empty space.** Advanced fields get a `QLabel#hintLabel` one-liner
   (units, sign convention, what the field does). A sign/direction mini-diagram for load
   dialogs (small painted `QWidget` or an `icons` glyph).
7. **Reuse tokens only.** Spacing from `style.SP_*`, radius `style.R_*`, colours from the
   palette. Never a literal `#hex`, px font, or magic margin.
8. **Keep data contracts.** The `.edit()/.manage()/.get()` classmethods keep their current
   signatures and return types so `main_window.py` wiring is untouched unless a phase explicitly
   rewires the menu (S-phase).
9. **Headless-constructible.** No solver/OpenGL work in `__init__`; every dialog builds under
   `QT_QPA_PLATFORM=offscreen` so it has a smoke test.

---

## 5. Shared scaffold API (frozen once L1 lands)

Phase **L1** creates `desktop/analysis_ui.py` with these reusable pieces. Later phases import
them; **do not rename or resignature after L1**.

```python
# desktop/analysis_ui.py

class GroupCard(QGroupBox):
    """A titled card with an inner QFormLayout (default) or a caller layout.
    Reads as a CSiBridge panel. Usage:
        g = GroupCard("Control")
        g.add_row("Control node", node_combo)
        g.add_row("Control DOF", dof_combo)
    """
    def __init__(self, title: str, parent=None, *, form: bool = True): ...
    def add_row(self, label: str, widget) -> None: ...          # form mode
    def body_layout(self):  ...                                 # for custom layouts

class CaseHeader(QWidget):
    """The standard Name / Notes / Type header row for a case dialog.
    Emits nothing; read .name() / .notes() / .case_type() on accept."""
    def __init__(self, *, name: str, type_label: str, notes: str = "",
                 parent=None): ...
    def name(self) -> str: ...
    def notes(self) -> str: ...

class LoadsAppliedTable(QWidget):
    """A columns-configurable table + right-side Add / Modify / Delete column,
    matching the CSiBridge 'Loads Applied' box. Rows are plain dict payloads;
    the owner supplies an editor callback that returns/edits one row.
        t = LoadsAppliedTable(["Load Type", "Load Name", "Scale"],
                              on_add=..., on_edit=...)
        t.set_rows([...]); rows = t.rows()
    """
    def __init__(self, columns: list[str], *, on_add=None, on_edit=None,
                 parent=None): ...
    def rows(self) -> list[dict]: ...
    def set_rows(self, rows: list[dict]) -> None: ...

def two_column(*cards) -> QWidget:
    """Lay out GroupCards into a responsive two-column grid with OK/Cancel
    handled by the caller's QDialogButtonBox."""

def dialog_buttons(dialog) -> QDialogButtonBox:   # Ok|Cancel, wired
    ...

def direction_glyph(dof_label: str) -> QWidget:   # small sign/axis hint
    ...
```

Acceptance for L1: a throwaway demo dialog assembled purely from these pieces renders as a
grouped, two-column, boxed panel under both themes, and a smoke test constructs each class
offscreen.

---

## 6. Work plan (phased)

Each item is one session. **L1 first.** Within a phase, lower numbers first. `→` marks the
file(s) touched.

### Phase L — Loads
- **L1 — Shared scaffold** `→ desktop/analysis_ui.py` (+ test). Build §5. *No deps.* **Do first.**
- **L2 — Load-combination editor (NEW)** `→ desktop/combinations_dialog.py`, wire into
  `main_window.py`. A combos list on the left, a per-combo **factor grid over load cases** on
  the right (`LoadsAppliedTable` or a small grid), plus a "Generate ASCE 7-22 LRFD" button that
  folds in the existing `generate_asce7_combinations()`. Returns `list[LoadCombination]` via
  `.manage()`. *Deps: L1.* **Highest user value — this surface is entirely missing.**
- **L3 — Redesign `LoadCaseDialog`** `→ editing.py`. Grouped card; per-nature icon in the
  table; nature combo styled; a hint line explaining natures drive ASCE combos; keep the
  `.edit()` contract + return type. *Deps: L1.*
- **L4 — Redesign nodal `LoadDialog`** `→ editing.py`. GroupCards (Target / Load case /
  Components), a `direction_glyph` sign-convention hint per DOF, units from the project. Keep
  `.edit()`. *Deps: L1.*
- **L5 — Member/line-load dialog (NEW)** `→ desktop/member_load_dialog.py`. Add/edit a
  `MemberLoad` (member picker, `wy`, `wz` in 3-D, case) in the grouped style; wire an "Add line
  load…" action into the Loads menu + toolbar and into the tree double-click. *Deps: L1.*
- **L6 — Redesign `LoadGenDialog`** `→ editing.py`. Grouped, with a one-line summary/preview of
  what will be generated (n nodes affected, total force). Keep `.get()`. *Deps: L1.*

### Phase A — Analysis / cases
- **A1 — Unified "Load Cases / Analysis Cases" manager (NEW)** `→ desktop/analysis_cases_dialog.py`.
  One list of **all** analysis cases with a **Type** column + type icon (Linear Static,
  Nonlinear Static / Pushover, Time History, and greyed *Modal / Response Spectrum / Buckling /
  Moving Load* placeholders to signal the roadmap, matching CSiBridge's list). Add / Modify /
  Delete / **Run**. The per-type Add opens the matching dialog (A2/A3). This becomes the
  Analysis home. *Deps: L1, A2.*
- **A2 — Redesign `NonlinearCaseDialog`** `→ nonlinear_cases.py`. **The screenshot fix.**
  Rebuild as CSiBridge panels: `CaseHeader` (Name/Notes/Type) top; *Control* card
  (node/DOF/target); *Protocol* card (monotonic↔cyclic fields swap in place); *Initial
  Conditions* card (Continue-from + Axial preload/node/DOF); *Solver* card (tol/max-iter);
  two-column via `two_column(...)`. Keep `NonlinearCase` fields + `.edit()` return. *Deps: L1.*
- **A3 — Redesign Pushover + Time-History input panels** `→ pushover_dialog.py`,
  `timehistory_dialog.py`. Wrap the left inputs in GroupCards (*Control*, *Protocol*, *Solver*);
  keep the right-side live plot / progress / log exactly as-is. *Deps: L1.*
- **A4 — "Run Analysis" control (NEW)** `→ desktop/run_analysis_dialog.py`. A case table with
  Run / Do-not-run toggles + a status column (Not run / Running / Done / Failed), a Run-now
  button, replacing the bare "Run linear static" action. Mirrors CSiBridge *Run Analysis*.
  *Deps: L1, A1.*
- **A5 — Polish `NonlinearCaseManagerDialog`** `→ nonlinear_cases.py`. Type/status icons, a
  status column, and reflow buttons; or fold it into A1 as the "nonlinear" filter. *Deps: A1.*

### Phase S — Shell / information architecture
- **S1 — Menu reorg** `→ main_window.py`. Add a dedicated **Loads** menu (Load cases, Nodal
  load, Line load, Generate loads, **Combinations**) and slim the **Analysis** menu to
  (Load/Analysis cases…, Run…, Results & diagrams, Design, Check model). Consistent icons.
  *Deps: L2, L5, A1, A4 (so the targets exist).* 
- **S2 — Loads & Analysis toolbar groups** `→ main_window.py`, maybe `style.py`. Captioned,
  text-under-icon tool-button groups (ribbon feel) for the common load + analysis actions,
  with group separators. *Deps: S1.*
- **S3 — (Optional) ribbon-style top bar** — a grouped, captioned toolbar band evoking the
  CSiBridge/Midas ribbon without a full ribbon framework. Evaluate after S2; may be deferred.

### Phase Q — Cross-cutting quality
- **Q1 — Smoke-test sweep** `→ tests/`. One offscreen construction test per new/redesigned
  dialog (mirrors the existing GUI smoke-test convention).
- **Q2 — Visual record** `→ phase21_outputs/loads_ux/`. Light+dark, compact+comfortable
  screenshots of every redesigned surface; a short before/after note in this file's §10.

---

## 7. Definition of done (the whole stream)

- Every Loads/Analysis dialog is grouped, boxed, and reads as one product with the rest of the
  app in both themes and both densities.
- A load-combination editor and a member-load dialog exist (the two missing surfaces).
- A single Analysis-cases home lists all case types with icons and a Run path.
- Menus/toolbars have a clear **Loads** area and a clean **Analysis** area.
- No regressions: all `.edit()/.manage()/.get()` contracts and `project.py` unchanged (except
  an allowed optional `notes` field, if used).

---

## 8. Status Tracker

Legend: `[ ]` todo · `[~]` in progress · `[x]` done (add `commit` + date).

| Item | Title | Deps | Status | Commit / date |
|------|-------|------|--------|---------------|
| L1 | Shared scaffold `analysis_ui.py` | — | [x] | `feat(loads-ux L1)` · 2026-09-13 |
| L2 | Load-combination editor (NEW) | L1 | [x] | `feat(loads-ux L2)` · 2026-09-13 |
| L3 | Redesign `LoadCaseDialog` | L1 | [x] | `feat(loads-ux L3)` · 2026-09-13 |
| L4 | Redesign nodal `LoadDialog` | L1 | [x] | `feat(loads-ux L4)` · 2026-09-13 |
| L5 | Member/line-load dialog (NEW) | L1 | [x] | `feat(loads-ux L5)` · 2026-09-13 |
| L6 | Redesign `LoadGenDialog` | L1 | [x] | `feat(loads-ux L6)` · 2026-09-13 |
| A1 | Unified analysis-cases manager (NEW) | L1, A2 | [x] | `feat(loads-ux A1)` · 2026-09-13 |
| A2 | Redesign `NonlinearCaseDialog` (screenshot fix) | L1 | [x] | `feat(loads-ux A2)` · 2026-09-13 |
| A3 | Redesign Pushover + Time-History panels | L1 | [x] | `feat(loads-ux A3)` · 2026-09-13 |
| A4 | "Run Analysis" control (NEW) | L1, A1 | [x] | `feat(loads-ux A4)` · 2026-09-13 |
| A5 | Fold `NonlinearCaseManagerDialog` into A1 | A1 | [x] | `feat(loads-ux A5)` · 2026-09-13 |
| S1 | Menu reorg (Loads menu) | L2, L5, A1, A4 | [ ] | |
| S2 | Loads & Analysis toolbar groups | S1 | [ ] | |
| S3 | (Optional) ribbon-style top bar | S2 | [ ] | |
| Q1 | Smoke-test sweep | all above | [ ] | |
| Q2 | Visual record (screenshots) | all above | [ ] | |

**Suggested order for the first three sessions:** L1 → A2 (fixes the screenshot everyone sees)
→ L2 (the biggest missing surface).

---

## 9. Change log

- 2026-09-13 — Plan created (audit of `desktop/` Loads & Analysis surfaces; scaffold API
  frozen in §5; 16-item tracker). No code yet.
- 2026-09-13 — **L1 done.** `desktop/analysis_ui.py` implements the frozen §5 API
  (`GroupCard`, `CaseHeader`, `LoadsAppliedTable`, `two_column`, `dialog_buttons`,
  `direction_glyph`) plus a `_demo_dialog` for the visual record. 10 offscreen smoke tests in
  `tests/test_desktop_analysis_ui.py` (all green). `GroupCard` wraps a themed `QGroupBox`;
  `two_column` honours a `full_width` attribute so a Loads-Applied table spans both columns.
  Note for later phases: `CaseHeader` exposes `type_label()` (additive getter) alongside the
  §5 `name()`/`notes()`. No `project.py` / solver changes.
- 2026-09-13 — **A2 done.** `NonlinearCaseDialog` (`desktop/nonlinear_cases.py`) rebuilt from
  the L1 scaffold: a `CaseHeader` (Name/Notes/Type = "Nonlinear Static") over
  `two_column(Control, Protocol, Initial conditions, Solver)`. Monotonic/cyclic fields still
  swap in place via `mono_host`/`cyc_host`; the Control card carries a live `direction_glyph`
  that updates with the DOF. `.data()`/`.edit()` and every test-referenced attribute preserved.
  **§4.4 exception used:** added `notes: str = ""` to `NonlinearCase` (`project.py`) so the
  header's Notes control persists — backward/forward compatible (`NonlinearCase(**c)` + `asdict`).
  Tests: 2 new in `test_desktop_nlcases.py` (scaffold panels + notes round-trip); full
  nlcases + scaffold + pushover/nonlinear/runs suites green (62 tests). Screenshots under
  `phase21_outputs/loads_ux/A2_nonlinear_case_*.png`.
- 2026-09-13 — **L2 done.** New `desktop/combinations_dialog.py` — the missing
  load-combination editor: a combinations list (＋ Combo / Delete) on the left, and for the
  selected combo a **factor grid over every load case** (Load case · Nature · Scale-factor spin)
  plus a "Generate ASCE 7-22 LRFD" button that folds in `generate_asce7_combinations()` via a
  proxy project (so generated ids continue from the working list). Factor 0 ⇒ case excluded.
  `.manage()` returns `list[LoadCombination]`. Wired into Analysis ▸ **Load combinations…**
  (`main_window.manage_combinations`, undoable). The generate *logic* is split into a UI-free
  `_append_generated()` so it is testable without the modal. 6 offscreen tests in
  `test_desktop_combinations.py` (all green; 29 across the scaffold-dependent suites).
  Screenshots `phase21_outputs/loads_ux/L2_combinations_*.png`. No solver changes.
- 2026-09-13 — **A3 done.** Pushover (`pushover_dialog.py`) and Time-History
  (`timehistory_dialog.py`) left input panels regrouped into scaffold GroupCards —
  pushover: *Control* (case/node/DOF/target/steps + live `direction_glyph`) · *Initial
  conditions* (axial) · *Solver* (tol/max-iter/record-fibers); time-history: *Monitor* ·
  *Ground motion* (record/dt/scale/units) · *Damping & mass*. The right-side live plot,
  tabs, step scrubber, progress bar, log, export row and every run control are unchanged, as
  are all test-referenced attributes (`case_combo`, `node`, `n_steps`, `_kwargs`, `_accel`,
  `in_g`, `scale`, `_selected_case`, …). Pure layout refactor: pushover + timehistory +
  nlcases suites green (29 tests). Screenshots `phase21_outputs/loads_ux/A3_*.png`.
- 2026-09-13 — **L4 done.** Nodal `LoadDialog` (`editing.py`) rebuilt from the scaffold: an
  *Applied to* card (node + load case) over a *Components [force-unit]* card whose rows each
  pair the force/moment spin with a per-DOF `direction_glyph` sign hint (→ +X, ↑ +Y, ⊙ +Z,
  ↻ about axes). `.edit()`/`.data()` and `node`/`case`/`vals` preserved. 4 offscreen tests
  in `test_desktop_load_dialog.py` (2-D + 3-D). Screenshots
  `phase21_outputs/loads_ux/L4_load_*.png`. No `project.py`/solver changes.
- 2026-09-13 — **L5 done.** New `desktop/member_load_dialog.py` — the previously UI-less
  `MemberLoad` (uniform line/UDL load in local axes) now has a grouped add/edit dialog
  (*Applied to* + *Uniform line load [force/length]* with per-component local-axis hints;
  `wz` in 3-D only). Wired into `main_window`: an **Add line load…** action (Edit menu +
  toolbar), `add_line_load` / `_edit_member_load` handlers (undoable), tree double-click, and
  delete. Line loads now also appear in the model tree under a **Line loads** group (they were
  invisible before). 4 offscreen tests in `test_desktop_member_load.py` (2-D/3-D + wiring +
  tree). Screenshots `phase21_outputs/loads_ux/L5_lineload_*.png`. No `project.py`/solver
  changes.
- 2026-09-13 — **A1 done.** New `desktop/analysis_cases_dialog.py` — the unified analysis-cases
  home (CSiBridge *Define ▸ Load Cases*): one table (Case · Type · Details) listing the
  built-in **Linear Static** launcher, every saved **Nonlinear Static** case, a **Time History**
  launcher, and greyed roadmap rows (Modal / Response Spectrum / Buckling / Moving Load). Add /
  Modify / Delete operate on nonlinear cases (delegating to `NonlinearCaseDialog`); **Run**
  records a request and closes. Wired into Analysis ▸ **Analysis cases…**
  (`main_window.manage_analysis_cases`): commits case edits, then dispatches the run — linear →
  `run_linear_static`, nonlinear → `run_pushover_dialog(preselect_case=…)` (new optional arg),
  time-history → `run_timehistory_dialog`. Decoupled: the dialog never runs anything itself.
  6 offscreen tests in `test_desktop_analysis_cases.py` (list composition, button gating, Run
  dispatch, delete-guard, wiring). The existing `Nonlinear cases…` action stays until A5 folds
  it in. Screenshots `phase21_outputs/loads_ux/A1_analysis_cases_*.png`. No solver changes.
- 2026-09-13 — **A4 done.** New `desktop/run_analysis_dialog.py` — CSiBridge *Run Analysis*
  control: a table (Case · Type · Action · Status) of the runnable cases, each with a
  **Run / Do not run** combo and a live **Status**. **Run Now** runs the batchable Linear
  Static inline (Running → Done / Failed / No model) via an injected `run_linear` callback, and
  *queues* flagged nonlinear / time-history cases (they need their interactive dialogs). Wired
  into Analysis ▸ **Run analysis…** (`main_window.run_analysis`): opens the control, then opens
  the pushover / time-history dialogs for the queued requests on close. `run_linear_static` now
  returns its info dict (was `None`) so the control can show status; callers that ignored the
  return are unaffected. 6 offscreen tests in `test_desktop_run_analysis.py`. Screenshots
  `phase21_outputs/loads_ux/A4_run_analysis_*.png`. No solver changes.
- 2026-09-13 — **L3 done.** `LoadCaseDialog` (`editing.py`) rebuilt into a scaffold GroupCard:
  the Name/Nature table now carries a per-nature **ASCE-key badge** (D / L / Lr / S / R / W / E)
  on each name cell — a new `icons.letter_icon(text, color)` helper renders the badge — updated
  live when the nature combo changes; plus a hint line explaining natures drive the ASCE 7-22
  generator. `.edit()`/`result_cases`, id preservation and add/remove unchanged. 4 offscreen
  tests in `test_desktop_load_cases.py`. Screenshots `phase21_outputs/loads_ux/L3_load_cases_*.png`.
  No `project.py`/solver changes.
- 2026-09-13 — **L6 done.** `LoadGenDialog` (`editing.py`) rebuilt into a *Load pattern*
  GroupCard with a **live preview** line that runs the actual generator
  (`generators.gravity_loads` / `lateral_loads`) to report how many loads will be created and
  the total force / base shear, updating as pattern or magnitude changes (empty-state message
  when no nodes are above the base). `.get()`/`params()` unchanged. 4 offscreen tests in
  `test_desktop_load_gen.py`. Screenshots `phase21_outputs/loads_ux/L6_loadgen_*.png`. No
  `project.py`/solver changes. **All six Loads-phase (L) items complete.**
- 2026-09-13 — **A5 done (fold, not polish).** Retired the standalone
  `NonlinearCaseManagerDialog` — the A1 Analysis-cases home already lists and does
  Add/Modify/Delete for nonlinear cases, so a second manager was redundant. Removed the class
  from `nonlinear_cases.py` (pruned its now-unused imports; `NonlinearCaseDialog` stays and is
  driven by the home), and removed the `Nonlinear cases…` action + `manage_nonlinear_cases`
  from `main_window.py`. Tests updated: dropped the manager test, and
  `test_nonlinear_cases_folded_into_analysis_home` now asserts the home is wired and the old
  manager/action are gone. nlcases + analysis-cases + pushover suites green (27 tests). No new
  screenshot (the A1 home is the surface). **All five Analysis-phase (A) items complete.**

---

## 10. Before / after notes

_(Fill in as items land — one or two lines + a screenshot path per redesigned surface.)_

- **L1 scaffold demo** — a dialog assembled purely from the scaffold renders as grouped,
  two-column, boxed panels with a full-width Loads-Applied table and OK/Cancel, in both themes.
  `phase21_outputs/loads_ux/L1_scaffold_light.png`, `…_dark.png`. (Text glyphs show as boxes in
  the headless `offscreen` renderer — a screenshot artifact only; the app renders text
  normally on Windows.)
- **A2 nonlinear-case dialog** — was a flat 14-row form; now a Name/Notes/Type header over a
  2×2 grid of Control / Protocol / Initial-conditions / Solver cards, matching CSiBridge *Load
  Case Data*. `phase21_outputs/loads_ux/A2_nonlinear_case_light.png`, `…_dark.png`,
  `…_cyclic_light.png` (cyclic shows the in-place protocol-field swap).
- **L2 combination editor** — a surface that did not exist before: combinations list on the
  left, a per-case factor grid + ASCE-7 generator on the right.
  `phase21_outputs/loads_ux/L2_combinations_light.png`, `…_dark.png`.
- **A3 pushover / time-history** — flat left forms became Control / Initial-conditions / Solver
  (pushover) and Monitor / Ground-motion / Damping-&-mass (time-history) cards; the live
  right-side plot and run controls are unchanged.
  `phase21_outputs/loads_ux/A3_pushover_light.png`, `…_dark.png`,
  `A3_timehistory_light.png`, `…_dark.png`.
- **L4 nodal load dialog** — flat form became an *Applied to* card + a *Components* card with a
  sign-convention hint on every DOF row. `phase21_outputs/loads_ux/L4_load_light.png`,
  `…_dark.png`, `L4_load_3d_light.png`.
- **L5 member/line-load dialog** — a surface with no UI before: *Applied to* + *Uniform line
  load* cards with local-axis hints; line loads now also show in the model tree.
  `phase21_outputs/loads_ux/L5_lineload_light.png`, `…_dark.png`, `L5_lineload_3d_light.png`.
- **A1 analysis-cases home** — one CSiBridge-style list of every analysis case: Linear Static +
  nonlinear cases + Time History launchers, and greyed Modal / Response-Spectrum / Buckling /
  Moving-Load roadmap rows. `phase21_outputs/loads_ux/A1_analysis_cases_light.png`, `…_dark.png`.
- **A4 run-analysis control** — a Case/Type/Action/Status table with Run/Do-not-run toggles;
  linear static runs inline with live status, nonlinear/time-history are queued.
  `phase21_outputs/loads_ux/A4_run_analysis_light.png`, `…_dark.png`.
- **L3 load-cases dialog** — bare table became a titled card with a per-nature ASCE-key badge on
  each row and a hint line. `phase21_outputs/loads_ux/L3_load_cases_light.png`, `…_dark.png`.
- **L6 load-generator dialog** — grouped card with a live preview of how many loads / total
  force will be generated. `phase21_outputs/loads_ux/L6_loadgen_light.png`, `…_dark.png`.
