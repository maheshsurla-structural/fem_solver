# GUI Professional-Grade Polish — Charter & Plan

**Status:** Living document — the single source of truth for making the whole desktop app
(`desktop/`) read as one polished commercial product. Successor/companion to
[`loads_analysis_ux_plan.md`](loads_analysis_ux_plan.md) (that stream, 16/16 done, fixed the
Loads & Analysis dialogs; this one covers **everything else + the global rules**).
**Owner rotation:** multiple Claude sessions across multiple accounts.
**The git repo is the only shared state** (memory files do NOT cross accounts). Read this file
first, do **one** tracker item, update §5, commit `feat(gui-polish <ID>): …`, and stop.

**Goal in one line:** every window, dialog, panel, icon, and the 3-D viewport look and behave
like serious commercial CAD/FEA software (CSiBridge / Midas / SAP2000 / AdSec) — coherent in
**light *and* dark**, **comfortable *and* compact**, with no raw/native-looking surface left —
**without changing the solver or the `project.py` data contracts**.

---

## 0. Session protocol (binding)

1. **Read this whole file before writing code.** It freezes the design charter (§2) and the
   tracker (§5) so independent sessions converge on one look.
2. **Pick the lowest-numbered unchecked item** whose deps are all checked. One item ≈ one session.
3. **Charter §2 rules are binding.** Never hand-code a colour, font size, radius, or spacing —
   pull the token from `style` (`style.ACCENT`, `style.SP_MD`, `style.R_MD`, …). If a token is
   missing, add it to `style.py` in the *same* commit and note it in §6.
4. **Do not change `project.py` dataclasses or the solver.** Pure UI/UX. (The `notes` field
   exception from the loads stream still stands; no new data-model changes without a §6 note.)
5. **Every item ships:** the change + a headless smoke test under `tests/` (constructible under
   `QT_QPA_PLATFORM=offscreen`, per the GUI convention) + a light+dark screenshot pair saved
   under `phase21_outputs/gui_polish/` for the visual record.
6. **Update §5** (check the box, add commit + date) and append a line to §6 (Change log).
7. **Preserve every public contract** (`.edit()/.manage()/.get()/.data()` signatures, action
   objects, test-referenced attributes). Layout/skin only unless the item says otherwise.

---

## 1. Why this stream exists

The design *system* is already excellent — [`style.py`](../../desktop/style.py) has light/dark
token palettes, density modes, card/pill/KPI/segmented-button QSS, and matplotlib helpers; the
Loads/Analysis dialogs were rebuilt onto it. **But the app still doesn't read as fully
professional, because the shell and the surfaces the eye lands on first don't use the system:**

- The **3-D viewport** — the literal centre of the app — is hard-coded white with a
  matplotlib-style `show_grid()` ("X Axis / Y Axis" bounds labels) and fixed hex colours. It
  **never themes**: in dark mode the middle of the window stays a bright white rectangle.
- The **shell's icons don't re-theme.** `main_window.py` is the *only* file still calling
  `icons.icon("name")` with no colour (hard-coded `#3a3a3a`); every newer file already passes
  `style.ICON`. In dark mode the toolbar/menu icons are near-invisible dark-grey on dark panel.
- **Dark mode & compact density are unreachable from the FEM shell.** `toggle_theme` /
  `set_density` have **zero** wiring in `main_window.py` — only the Section Designer exposes a
  theme button. The whole app supports two themes and two densities; the user can't switch.
- **No window/taskbar brand icon.** `icons.monogram_icon()` (the product mark) is authored but
  `setWindowIcon` is never called, so the app shows a generic OS icon.
- **A tail of pre-scaffold dialogs** (material editor, hinge editor, model checks, section
  import, run history, design, drawings) predate `analysis_ui.py` and are still flat forms.
- **No codified global window rules** — sizing, centring, remembered geometry, button order,
  keyboard defaults, title casing, empty states, status-bar content — so each surface drifts.

This is an *arrangement + reuse + reach* problem, **not a repaint**. The charter below makes the
rules explicit; the plan applies them to every remaining surface.

---

## 2. The GUI Design Charter (binding rules)

Any surface — existing or new — must satisfy these. They are the "rules of our GUI."

### A. Theming & tokens
1. **No literals, ever.** Colours, spacing, radius, and font sizes come from `style` tokens.
   A raw `#hex`, a `px` font size, or a magic margin in a widget is a bug.
2. **Every surface renders correctly in all four combinations:** light/dark × comfortable/compact.
   If a surface can't be shown in dark mode, it isn't done.
3. **Read tokens at call time** (`style.ACCENT`), never cache a colour at import — the palette is
   swapped in place on theme change, and surfaces must be able to restyle live.
4. **Icons are themed ink.** Toolbar/menu/inline icons render in `style.ICON` (or a semantic
   token: `OK`/`BAD`/`WARN`/`ACCENT`), and **re-colour on theme change** — never left at the
   `icon()` default. Semantic colour (red=fail, green=pass, amber=warn) is reserved for meaning.

### B. The 3-D viewport is a first-class themed surface
5. Background, grid, and the reference/ghost greys come from `style` tokens (a `VIEW_BG`,
   `VIEW_GRID`, `VIEW_AXIS` family), not `"white"` / literal hex. It repaints on theme change.
6. The grid reads as a **CAD floor/axes**, not a scientific plot — no "X Axis / Y Axis" bounding
   labels dominating the frame; use a restrained ground grid + a small origin axis triad.
7. Model entity colours (member/node/support/selection/deformed/diagram) live in one themed
   place and stay legible on both grounds.

### C. Layout & grouping
8. **Group everything.** No dialog with more than ~4 fields is a flat form — wrap logical groups
   in `GroupCard` / `QGroupBox` (already styled as cards). Titles are short nouns.
9. **Two columns for dense dialogs** (`analysis_ui.two_column`), not one tall stack.
10. **8px rhythm.** Margins/spacing from `SP_XS…SP_XL`. Content insets ≥ `SP_MD`; no field
    touching a card border.
11. **Reuse the scaffold** (`desktop/analysis_ui.py`: `GroupCard`, `CaseHeader`,
    `LoadsAppliedTable`, `two_column`, `dialog_buttons`) and `widgets.py` before hand-rolling.

### D. Windows & dialogs
12. **Consistent chrome:** every top-level window sets the product `setWindowIcon`, Title-Case
    `setWindowTitle` (`"Load combinations"`, not `"load combos"`), and a sensible min-size.
13. **Buttons:** OK/Cancel via `analysis_ui.dialog_buttons` — primary (accent) on the right,
    Enter accepts, Esc rejects. Destructive actions confirm and are never the default button.
14. **Open centred on the parent**, not at an OS-default corner; remember size where it helps
    (managers, designers). Modal only when the flow requires it.
15. **A header block on case/entity dialogs** (`CaseHeader` or an `#h2` title + `#sub` line) so
    every dialog states what it is.

### E. Typography & iconography
16. **One type ladder** (`FS_DISPLAY…FS_MICRO`, set via the `#display/#h1/#h2/#h3/#sub/#eyebrow`
    object names). Uppercase eyebrow captions use `LS_LABEL` tracking.
17. **One icon family** — the in-house 24px line set in `icons.py`, themed per rule 4. No mixed
    icon styles, no emoji as UI chrome.
18. Numbers/coords/results in the mono stack (`MONO_STACK`) so columns align.

### F. Feedback & state
19. **A real status bar:** persistent context (active units, selection count, cursor coords,
    theme/density), not just "Ready". Transient results use the existing toast/busy widgets.
20. **Empty states, not blank panels.** No model / no results / no sections shows a centred hint
    (`#canvasHint`) with the next action — never an empty grey void.
21. **Progress for anything >~200 ms** (solve, generate, import) via the themed busy bar; the app
    never looks frozen.
22. **Confirm the irreversible** (delete, overwrite, discard) with a themed `QMessageBox`.

### G. Restraint
23. Motion is functional and quick (hover/expand ≤150 ms); no gratuitous animation.
24. One accent (`ACCENT`). Semantic colours only for status. Borders/dividers are the quiet
    `BORDER`/`BORDER_STRONG` tokens — the UI is calm, the model is the hero.

---

## 3. Current-state audit (as of 2026-09-13)

`file:line` into `desktop/`.

- **Viewport — [`model_view.py`](../../desktop/model_view.py):** `set_background("white")`
  (`:91`); module-level hard-coded entity hex (`:17`–`25`); `show_grid()` at every render
  (`:337`, `:396`, `:466`, `:490`, `:536`) draws the plot-style axes. No theme hook. **→ V1, V2.**
- **Shell icons — [`main_window.py`](../../desktop/main_window.py):** bare `icons.icon(...)`
  throughout `_build_menu` (`:96`+) and `_action` (`icon` default `#3a3a3a`). **→ T1.**
- **No theme/density reach — `main_window.py`:** no `toggle_theme`/`set_density` action; no View
  menu entry; no header chip. Section Designer has `_theme_btn`
  ([`section_designer.py:627`](../../desktop/section_designer.py:627)) to mirror. **→ T2.**
- **No window icon — `main_window.py`:** no `setWindowIcon`; `icons.monogram_icon()` unused as
  chrome. **→ T3.**
- **Bare status bar — `main_window.py:87`:** `showMessage("Ready")` only. **→ T4.**
- **Pre-scaffold dialogs (flat forms, tokens partial):** `material_editor.py`, `hinge_editor.py`,
  `model_checks_dialog.py`, `section_import.py`, `run_history_dialog.py`, `design.py`,
  `drawing_window.py`. **→ D1–D3.**
- **Empty states:** on `new_project`/no results the viewport + Properties show blank. **→ T5.**

Assets to build on: `style.py` tokens + QSS; `analysis_ui.py` scaffold; `widgets.py`
`CollapsibleGroup`; `icons.py` `icon()/letter_icon()/monogram_icon()`.

---

## 4. How the phases fit together

- **Phase T — the shell & theming reach** (highest visible impact, mostly `main_window.py` +
  `style.py`): make the app themeable end-to-end and the chrome branded. Do these first — they
  light up every screenshot.
- **Phase V — the viewport** as a themed CAD surface (the centre of the app).
- **Phase D — the pre-scaffold dialog tail** onto the charter (mechanical, well-trodden by the
  loads stream).
- **Phase G — global guarantees** (empty states, status bar content, a sweep test that every
  top-level surface obeys the charter in all four theme×density combos).

---

## 5. Status Tracker

Legend: `[ ]` todo · `[~]` in progress · `[x]` done (add `commit` + date).
Pick the lowest-numbered unchecked item whose deps are met.

| ID | Title | Deps | Status | Commit / date |
|----|-------|------|--------|---------------|
| T1 | Theme the shell's icons (`main_window` → `style.ICON`, re-colour on theme change) | — | [x] | `feat(gui-polish T1-T3)` · 2026-09-13 |
| T2 | Wire **theme + density toggles** into the shell (View menu + status-bar chip; persist choice) | T1 | [x] | `feat(gui-polish T1-T3)` · 2026-09-13 |
| T3 | Product **window icon** (`setWindowIcon(monogram_icon())`) + title-case audit | — | [x] | `feat(gui-polish T1-T3)` · 2026-09-13 |
| T4 | **Status bar** with real context (units · selection · coords · theme/density) | T2 | [x] | `feat(gui-polish T4-T5)` · 2026-09-13 |
| T5 | **Empty states** for viewport / tree / properties (no model · no results) | — | [x] | `feat(gui-polish T4-T5)` · 2026-09-13 |
| V1 | Viewport **themes** (bg/grid/axis + entity colours from tokens; repaint on theme change) | — | [x] | `feat(gui-polish V1)` · 2026-09-13 |
| V2 | Replace plot-style `show_grid()` with a **CAD ground grid + origin triad** | V1 | [x] | `feat(gui-polish V2)` · 2026-09-13 |
| D1 | Redesign **Material editor** onto the scaffold | — | [ ] | |
| D2 | Redesign **Hinge editor / assignment** onto the scaffold | — | [ ] | |
| D3 | Polish **Model-checks · Section-import · Run-history · Design · Drawings** dialogs | — | [ ] | |
| G1 | **Charter sweep test** — every top-level surface builds + themes in all 4 combos | T1–T5, V1, D1–D3 | [ ] | |
| G2 | **Visual record** — light/dark (+ density) screenshots of every polished surface | most above | [ ] | |

**Suggested first three sessions:** T1 (icons theme) → T2 (reach dark mode / compact from the
shell) → V1 (viewport themes) — after these three, opening the app in dark mode looks like a
different, finished product.

---

## 6. Change log

- 2026-09-13 — Charter created. Audit of `desktop/` shell + viewport + pre-scaffold dialogs;
  design charter (§2) frozen; 12-item tracker. No code yet.
- 2026-09-13 — **V1 done** (taken ahead of T2 by user choice — "viewport first"). The 3-D
  viewport (`desktop/model_view.py`) now themes: 14 new `style.V_*` / `VIEW_*` tokens in both
  palettes drive the background, grid/axis colour and every model-entity ink (member / node /
  support / reference / deformed / selection / hinge / diagram N-V-M), read at render time — the
  module-level hard-coded hex constants are gone. `set_background("white")` → `style.VIEW_BG`;
  the plot-style `show_grid()` is wrapped in `_draw_grid()` which colours the axes from
  `VIEW_AXIS` and drops the "X Axis / Y Axis" titles (V2 replaces it with a real CAD ground grid
  + origin triad). A new `apply_theme()` (mirroring `SectionCanvas.apply_theme`) replays the last
  scene with fresh tokens and restores the camera, ready for T2 to call on a theme switch; each
  draw method captures a `functools.partial` replay. The DCR legend/label backgrounds and the
  lasso overlay now theme too; the semantic DCR ramp (green→red) is intentionally left literal.
  4 offscreen tests in `test_desktop_viewport_theme.py` (tokens swap across themes, diagram
  colour tracks theme, no `color="#hex"` in a draw call, and the real viewport builds + restyles
  light↔dark over a demo model with the renderer background measurably darkening). Full desktop
  suite green (192 passed, 1 skipped). Visual record `phase21_outputs/gui_polish/V1_before_white`,
  `V1_after_light`, `V1_after_dark` (off_screen pyvista render of the same scene, since the
  Qt `offscreen` platform can push meshes but can't read pixels back). No `project.py` / solver
  changes; `ModelView` is imported only by `main_window.py`, whose call sites are untouched.
- 2026-09-13 — **V2 done.** `_draw_grid()` (`desktop/model_view.py`) no longer draws PyVista's
  plot-style bounds box. It now lays a **CAD ground grid** — line segments in the model's ground
  plane (`z = base`), spanning the model bounds at a "nice" 1/2/5×10ⁿ step (`_nice_step`), a hair
  behind the model plane so it reads as a floor — plus a **corner XYZ orientation triad**
  (`add_axes`, label ink themed). Grid colour from `style.VIEW_GRID`. The builder is a pure
  module function `_build_ground_grid(model)` (returns a lines `PolyData`, `None` for an empty
  model) so it's testable without GL and reusable by the screenshot harness. 4 offscreen tests in
  `test_desktop_viewport_grid.py` (`_nice_step` snapping + line-count sanity, the grid brackets
  the frame in x/y and sits at z ≤ 0, empty-model guard, and a live `set_model`→`_draw_grid`
  rebuild leaving a stable `groundgrid` actor). Full desktop suite green (196 passed, 1 skipped).
  Visual record `phase21_outputs/gui_polish/V2_ground_grid_{light,dark}` (2-D) and
  `V2_ground_grid_3d_{light,dark}` (the floor reads correctly under an isometric 3-D frame). No
  `project.py` / solver changes. **Phase V (viewport) complete.**
- 2026-09-13 — **T1–T3 done** (shipped together — T1 is the infra T2 rides on, T3 is a one-liner).
  The desktop **shell now themes end-to-end and dark mode is finally reachable** (`main_window.py`).
  **T1:** the shell was the last place calling bare `icons.icon(name)` at the hard-coded `#3a3a3a`
  (near-invisible on dark). A new `_set_icon(act, name)` helper inks every action from `style.ICON`
  and stores the icon name on the action; `_action`, the ~10 direct `QAction`s and the Results
  submenu all route through it. `_retheme_icons()` walks `findChildren(QAction)` and re-inks by the
  stored name (the ribbon buttons mirror their action, so they follow). **T2:** `QSettings`
  ("MidasStructural"/"Desktop", mirroring the Section Designer) restores theme + density in
  `__init__` *before* the viewport is built (it reads `VIEW_BG` at construction); a **Toggle theme**
  + **Compact density** pair in the **View menu** and an always-visible **status-bar theme chip**
  (`_theme_btn`) call `toggle_theme`/`toggle_density`, which persist the choice and run
  `_apply_theme_density()` — `style.apply(self)` (QSS cascade) + `_retheme_icons()` +
  `self.view.apply_theme()` (viewport repaint) + `_sync_theme_ui()`. **T3:** `setWindowIcon` with a
  product monogram (`FS`, stable brand blue) — the mark was authored but never used as chrome.
  Title-case audit: every `setWindowTitle` already follows a consistent sentence-case convention
  (lowercase "femsolver" is the deliberate brand name) — no changes. 8 offscreen tests in
  `test_desktop_shell_theme.py` (icon-name coverage + no theme-blind `icons.icon()` left; retheme
  re-inks; theme/density toggle flips + persists + syncs the action; controls reachable in View
  menu + status bar; persisted theme restored on construction; branded window icon). Full desktop
  suite green (204 passed, 1 skipped). Visual record
  `phase21_outputs/gui_polish/T_shell_{light,dark}` (native grab — the whole window, viewport
  included, in both themes). No `project.py` / solver changes. **Phase T's theming-reach items
  (T1–T3) complete; T4 status-bar content + T5 empty states remain.**
- 2026-09-13 — **T4–T5 done** (`main_window.py` + `model_view.py`). **Phase T complete.**
  **T4:** the status bar used to only `showMessage` a transient model count that any other
  message clobbered. It now carries persistent right-side context as permanent widgets —
  **model summary** (`N nodes · M members · L loads`), **selection** count, live **cursor
  coords**, and **units** — beside the T2 theme chip, with `showMessage` freed for transient
  hints. A `_build_status_items()` builds the labels; `_refresh_status()` (called from `_rebuild`)
  fills model + units and re-derives the selection count so it stays consistent;
  `_on_selection_changed` updates it live. The viewport gained `set_coord_callback` +
  `_world_on_ground(pos)` — a cursor-ray/`z=0`-plane intersection (works ortho *and* iso, guarded)
  — feeding `_on_cursor_coords`. **T5:** `ModelView` now shows a centred `#canvasHint` overlay
  ("No model yet — draw a node, or open a project (Ctrl+O)") whenever the model is absent or has
  no nodes (e.g. right after File ▸ New, which makes a project with a section but no geometry),
  hiding once a model with nodes is drawn; it's click-through (`WA_TransparentForMouseEvents`) and
  themes via QSS. (The Properties panel already had its own "select…" empty state.) 9 offscreen
  tests (`test_desktop_statusbar.py` ×6, `test_desktop_empty_state.py` ×3). Full desktop suite
  green (213 passed, 1 skipped). Visual record `phase21_outputs/gui_polish/T4_statusbar_light`,
  `T5_empty_{light,dark}`. No `project.py` / solver changes. **All of Phase T (T1–T5) done; V + T
  complete — remaining: D1–D3 dialog tail, G1–G2 sweep + record.**
