# Model Navigation (Model Tree) — Charter & Plan

**Status:** Living document — the single source of truth for the desktop app's **left Model tree**
(the docked outline of everything in the model). Sibling to
[`ribbon_top_chrome_plan.md`](ribbon_top_chrome_plan.md) (the top chrome),
[`gui_professional_polish_plan.md`](gui_professional_polish_plan.md) (the app-wide skin) and
[`loads_analysis_ux_plan.md`](loads_analysis_ux_plan.md) (the dialogs). This one owns **how the
model's contents are summarised and reached in the left panel**.
**Owner rotation:** multiple Claude sessions across multiple accounts.
**The git repo is the only shared state** (memory files do NOT cross accounts). Read this file
first, do **one** tracker item, update §5, commit `feat(nav <ID>): …`, and stop.

**Goal in one line:** the Model tree is a **compact, complete outline** of the model — every
category that exists (materials · sections · hinges · nodes · elements-by-type · supports · load
cases · loads · combinations · analysis cases · results), each with a **count**, collapsed by
default so the panel reads like a *table of contents*; **right-click → Show Table…** on any
category opens the full spreadsheet of that category's rows. Exactly the Midas / CSiBridge model
tree idiom, in **light *and* dark**, **comfortable *and* compact**.

---

## 0. Session protocol (binding)

1. **Read this whole file before writing code.** It freezes the charter (§2), the tree taxonomy
   (§3), and the tracker (§5) so independent sessions converge on one arrangement.
2. **Pick the lowest-numbered unchecked item** whose deps are all checked. One item ≈ one session.
3. **Charter §2 rules are binding.** Never hand-code a colour, font size, radius, or spacing —
   pull the token from `style` (`style.ACCENT`, `style.MUTED`, `style.SP_MD`, `style.R_SM`, …).
   Add missing tokens to `style.py` in the *same* commit and note them in §6.
4. **Do not change `project.py` dataclasses or the solver.** Pure UI/UX. The tree only *reads*
   the `Project`; it never invents new persisted state.
5. **Selection is sacred.** The tree is the app's selection model: `_selected_refs()` drives
   move / copy / delete / the Properties panel, and the viewport syncs to it through
   `_on_pick` / `_select` / `_find_item`. Any restructure MUST keep `("node", id)`,
   `("member", id)`, `("section", id)`, `("load", i)`, `("member_load", i)` refs locatable by
   `_find_item` (it is recursive — keep it so) and selectable by `_select` (which expands
   ancestors). Guarded by `tests/test_desktop_member_load.py` and `tests/test_desktop_nav.py`.
6. **Every item ships:** the change + headless smoke coverage in `tests/test_desktop_nav.py`
   (constructible under `QT_QPA_PLATFORM=offscreen`) + a light+dark screenshot pair under
   `phase21_outputs/gui_polish/` (`N<ID>_*_light.png` / `_dark.png`). Capture on the **native**
   platform, not offscreen — offscreen has no font glyphs (renders as ▯).
7. **Update §5** (check the box, add commit + date) and append a line to §6 (Change log).

---

## 1. Why this stream exists

The tree dumped **every** node, member, section and load as an individual row under five flat
groups (`Nodes (n)` / `Members (n)` / `Sections (n)` / `Loads (n)` / `Line loads (n)`). Two
problems, both raised by the user:

1. **It's too much.** A real model floods the panel with hundreds of leaf rows, so it stops being
   a *navigation* aid — you can't see at a glance *what is in the model*. Midas/CSi solve this by
   showing the tree as a **summary** (categories + type + counts) and moving the row-by-row detail
   into an on-demand **table** (right-click → Show Table…).
2. **It's incomplete.** Materials, hinges, load cases, combinations, analysis cases and results
   all exist in the `Project` but never appeared in the tree. "Show everything that exists in the
   model" — the tree must be the model's table of contents.

This is an **information-architecture** problem, not a repaint. The charter makes the rules
explicit so it stays that way.

---

## 2. Charter (binding rules)

- **B1 — Summary first.** The default (freshly-built) tree shows **category headers with counts**,
  collapsed. Structural groups (Materials/Sections, Nodes, Elements, …) are expanded to reveal the
  category rows; the long **leaf** lists (individual nodes/elements/loads) stay **collapsed** so
  the default view is a compact outline. Expanding a category is opt-in.
- **B2 — Completeness.** Every category the `Project` can hold has a row, **even when empty**
  (count `0`, muted) — so the tree tells you what the model *doesn't* have too.
- **B3 — Elements by type.** `Elements (n)` groups its members under a child row per element type
  (`Beam`, `Truss`, `Fiber hinge`, …) carrying that type's count — the reference behaviour.
- **B4 — Counts as a second column.** Counts render right-aligned in a narrow second tree column
  (`style.MUTED`), like the badges in Midas — not baked into the label text.
- **B5 — Right-click → tables.** Every category row (and the element-type rows) has a context menu
  whose primary action is **Show Table…**, opening `model_tables.ModelTableDialog` with the full,
  columned list of that category's rows. Editable categories also offer their manager/editor
  (Manage materials…, New section…, …). Double-clicking a category header does its primary action.
- **B6 — Tokens only.** No hand-coded colour/size/radius/spacing (see §0.3).
- **B7 — Selection preserved (§0.5).** Never regress move/copy/delete/Properties/viewport-sync.

---

## 3. Tree taxonomy (the frozen outline)

Top-level super-groups, each holding category rows. `‹n›` = count column. Leaf rows (individual
items, collapsed) hang under the category that owns their selection ref.

```
Properties
  Materials            ‹n›   → Show Table · Manage materials…
  Sections             ‹n›   → Show Table · New section…            (leaves: section items)
  Hinge properties     ‹n›   → Show Table · Manage hinges…
Structures
  Nodes                ‹n›   → Show Table · New node…               (leaves: node items)
  Elements             ‹n›   → Show Table
    ‹Type›             ‹n›   (Beam / Truss / Fiber hinge / …)       (leaves: member items)
  Supports             ‹n›   → Show Table   (nodes with any fixity; informational)
Loads
  Load cases           ‹n›   → Show Table · Manage load cases…
  Nodal loads          ‹n›   → Show Table · New load…               (leaves: load items)
  Line loads           ‹n›   → Show Table · New line load…          (leaves: member_load items)
  Load combinations    ‹n›   → Show Table · Manage combinations…
Analysis
  Analysis cases       ‹n›   → Show Table · Manage analysis cases…  (leaves: case rows)
  Results              ‹n›   → Show Table   (saved nonlinear runs)
```

Refs (frozen, per §0.5): node→`("node", id)`, member→`("member", id)`, section→`("section", id)`,
nodal load→`("load", index)`, line load→`("member_load", index)`. Category headers carry a
`("cat", key)` tag in `Qt.UserRole+1` (key ∈ the taxonomy above) so the context menu and
double-click know which table/manager to open; they carry **no** `UserRole` ref (selecting a
header selects nothing).

---

## 4. Non-goals / out of scope (for now)

- Replacing the viewport as the primary graphical selection surface (the tree keeps its leaves;
  we do **not** yet strip individual items — that's N6, sequenced last).
- Editing directly inside the tables (tables are read + drill-down to the existing editors first;
  in-place editing is a later item).
- Drag-and-drop reordering, grouping/named selection sets, and search/filter (tracked, later).

---

## 5. Tracker

Legend: `[x]` done · `[ ]` open. Do the lowest open item whose deps are met.

- [x] **N1 — Summary tree.** Rebuilt `_populate_tree` to the §3 taxonomy: super-groups + category
  rows for **every** category (incl. empty ones), counts in a second column (B4), Elements grouped
  by type (B3), leaves preserved but collapsed by default (B1/B7). `_find_item` now recursive and
  `_select` expands ancestors. Icons per category from `icons.py`. — `feat(nav N1)`, 2026-09-13.
- [x] **N2 — Right-click → Show Table….** Context menu (`_on_tree_menu`) on every category/type
  row; new `desktop/model_tables.py::ModelTableDialog` renders the full columned list per category
  (materials/sections/hinges/nodes/elements/supports/load-cases/loads/line-loads/combinations/
  analysis-cases/results). Double-click a header runs its primary action (`_category_primary`);
  double-click a table row drills to the existing editor via `_table_activate`. — `feat(nav N2)`,
  2026-09-13.
- [x] **N3 — Expand/collapse state persistence.** Each branch carries a stable `KEY_ROLE` key
  (`grp:…` / `cat:…`); `_populate_tree` restores expansion from `self._expanded`, kept in sync by
  `_on_branch_expanded/collapsed` and persisted to `QSettings("nav/expanded")` (seeded with
  `_DEFAULT_EXPANDED` on first run). Nav panel gains an **Expand all / Collapse all** header
  (`_build_nav_panel`, `expand_all_tree` / `collapse_all_tree`). — `feat(nav N3)`, 2026-09-13.
- [ ] **N4 — Search / filter box.** A filter field above the tree that live-hides non-matching
  categories/leaves (by id, name, type). Deps: N1.
- [ ] **N5 — Live counts / partial refresh.** Update counts + affected branch in place after an
  edit instead of a full `clear()`+rebuild, preserving expansion and scroll. Deps: N1, N3.
- [ ] **N6 — Pure-summary option.** A toggle that drops individual leaves entirely (tree = counts
  only), moving all item selection to the viewport + tables — the full Midas "clean tree". Requires
  routing move/copy/delete/Properties off the tree's leaves onto a viewport-owned selection set.
  Deps: N2, N5.
- [ ] **N7 — In-table editing.** Let `ModelTableDialog` edit values in place (with undo through
  `_apply_edit`) rather than only drilling to dialogs. Deps: N2.

---

## 6. Change log

_(prepend newest)_

- **2026-09-13 — N3.** Expand/collapse state now survives edit-rebuilds and sessions. Every
  expandable branch carries a stable key in `KEY_ROLE` (`UserRole+2`): `grp:<title>` for
  super-groups, `cat:<key>` for categories/element-types (`_cat_key_str` flattens the tuple keys).
  `self._expanded` (a set) is loaded from `QSettings("nav/expanded")` at init — seeded with
  `_DEFAULT_EXPANDED` (the four super-groups + Elements) on first run — restored in `_populate_tree`
  under a `self._building_tree` guard so the programmatic pass doesn't churn the signals;
  `_on_branch_expanded/_on_branch_collapsed` keep the set in sync and `_persist_expanded` writes it
  back. The tree is now wrapped by `_build_nav_panel`, which adds an **Expand all / Collapse all**
  header (nav-panel-local actions `_nav_expand_act` / `_nav_collapse_act`, deliberately off the
  `act_*` namespace the ribbon "homed once" test guards); `collapse_all_tree` keeps the four
  super-groups open so the category list stays visible. Tests: +3 in `tests/test_desktop_nav.py`
  (temp-`QSettings` isolated). Screenshot `N3_nav_panel.png`. Desktop suite: 391 passed, 1 skipped.

- **2026-09-13 — N1 + N2.** The model tree became a compact table-of-contents. `_populate_tree`
  (in `desktop/main_window.py`) now builds four super-groups (Properties / Structures / Loads /
  Analysis) whose category rows carry a right-aligned **count** in a new second column
  (`setColumnCount(2)`, header hidden, col-1 = ResizeToContents); empty categories are muted so
  the tree shows what the model *lacks* too. Elements group under a child row per element type
  (`_element_type` → Beam / Truss / Fiber hinge). Every category — materials, sections, hinge
  properties, nodes, elements, supports, load cases, nodal loads, line loads, load combinations,
  analysis cases, results — is present. Leaves are preserved (selection intact) but collapsed by
  default; `_find_item` is now recursive (`_iter_tree_items`) and `_select` expands ancestors +
  scrolls. Category rows tag `("cat", key)` in `CAT_ROLE` (`UserRole+1`). New helpers `_super`,
  `_category`, `_leaf`. Right-click (`_on_tree_menu`) offers **Show Table…** + the category's
  manager/new action; double-click a header runs `_category_primary`. New `desktop/model_tables.py`
  (`ModelTableDialog`) renders a read-only spreadsheet per category and drills a double-clicked row
  back to its editor (`_table_activate`). Imports: `QHeaderView`, `QBrush/QColor/QFont`. Tests:
  new `tests/test_desktop_nav.py` (6 cases). Screenshots `N1_nav_tree_light/dark.png`,
  `N1_elements_by_type.png`, `N2_elements_table.png`. Desktop suite: 389 passed, 1 skipped.
