# Ribbon & Top-Chrome — Charter & Plan

**Status:** Living document — the single source of truth for the desktop app's **top chrome**
(the strip above the workspace: navigation + command surface). Sibling to
[`gui_professional_polish_plan.md`](gui_professional_polish_plan.md) (12/12 done — the app-wide
skin) and [`loads_analysis_ux_plan.md`](loads_analysis_ux_plan.md) (16/16 — the dialogs). This
one owns **how commands are arranged and reached at the top of the window**.
**Owner rotation:** multiple Claude sessions across multiple accounts.
**The git repo is the only shared state** (memory files do NOT cross accounts). Read this file
first, do **one** tracker item, update §5, commit `feat(ribbon <ID>): …`, and stop.

**Goal in one line:** the top of the window is **one compact strip** — a *File* backstage
button + a tab strip whose active tab swaps a **single** row of captioned tool-groups — exactly
the CSiBridge / Midas / SAP2000 model, so the workspace (tree · 3-D viewport · properties) gets
the vertical space, in **light *and* dark**, **comfortable *and* compact**.

---

## 0. Session protocol (binding)

1. **Read this whole file before writing code.** It freezes the charter (§2), the tab/group
   taxonomy (§3), and the tracker (§5) so independent sessions converge on one arrangement.
2. **Pick the lowest-numbered unchecked item** whose deps are all checked. One item ≈ one session.
3. **Charter §2 rules are binding.** Never hand-code a colour, font size, radius, or spacing —
   pull the token from `style` (`style.ACCENT`, `style.SP_MD`, `style.R_SM`, …). Add missing
   tokens to `style.py` in the *same* commit and note them in §6.
4. **Do not change `project.py` dataclasses or the solver.** Pure UI/UX.
5. **Preserve every command.** Every `QAction` on the window must be reachable from **exactly
   one** ribbon button or the File backstage (the two dialog-only launchers — `act_pushover`,
   `act_timehistory` — are the sole exceptions; they open from the Analysis-cases home). This is
   guarded by `tests/test_desktop_ribbon.py::test_every_action_is_homed_exactly_once` — keep it
   green.
6. **Every item ships:** the change + headless smoke coverage in `tests/test_desktop_ribbon.py`
   (constructible under `QT_QPA_PLATFORM=offscreen`) + a light+dark screenshot pair under
   `phase21_outputs/gui_polish/` (`R<ID>_*_light.png` / `_dark.png`). Capture on the **native**
   platform, not offscreen — offscreen has no font glyphs (renders as ▯).
7. **Update §5** (check the box, add commit + date) and append a line to §6 (Change log).
8. **Preserve public contracts:** action objects, `RibbonBar` API (`add_tab` / `set_current` /
   `page` / `file_menu` / `tabs` / `stack`), and every `act_*` / test-referenced attribute.

---

## 1. Why this stream exists

The command surface was **four-plus stacked strips**: a classic menu bar (File…Tools, the full
set) **over** three always-visible ribbon rows (Home / Loads / Analysis, added with
`addToolBarBreak()`) **over** a modal draw/select palette. All of it was on screen at once,
eating ~500 px before the model tree began — see `phase21_outputs/gui_polish/T_shell_light.png`
(the "before"). The old S1/S2/S3 comment even admitted the rows were stacked because there was
"no tab framework."

The reference tools the user targets (CSiBridge v27, Midas, SAP2000) don't stack — they show
**one** ribbon row and **swap** it with tabs (`Components / Loads / Analysis / …`). Same buttons,
one-third the height. **Ribbon R1** brought that model here: the menu bar and the three rows are
gone; one `RibbonBar` (a `QTabBar` + a `QStackedWidget` of group rows + a File backstage button)
takes their place. Top chrome: ~5 rows → **2**.

This is an **arrangement + reach** problem, not a repaint. The charter makes the rules explicit
so it stays that way.

---

## 2. The Top-Chrome Charter (binding rules)

Any change to the top of the window must satisfy these — "the rules of our ribbon."

### A. One strip, always
1. **No stacked command rows.** Exactly one `RibbonBar` hosts navigation + commands. Never call
   `addToolBarBreak()` to stack a second command row, and never re-introduce a `QMenuBar` as the
   primary navigation. (The menu bar stays constructed but `hidden` — do not populate or show it.)
2. **Tabs swap; they don't add height.** A tab click is a pure view swap of the single group row
   (`QStackedWidget`). All pages are the same height (one button row + caption).
3. **File is a backstage**, not a tab page — a distinct accent-filled button (`ribbonFile`) at the
   left of the strip that opens the File menu (New / Open / Save / …).

### B. Groups & buttons
4. **Every button mirrors a `QAction`** via `setDefaultAction` — so enabled state, re-inked icons
   (§C) and Ctrl-shortcuts stay live no matter which tab is showing.
5. **Text-under-icon, short label.** Ribbon buttons are `ToolButtonTextUnderIcon`; the *short*
   label is the action's `iconText` (e.g. `"Cases"`), while the action keeps its full `text()`
   (e.g. `"Load &cases…"`). One action → one button (no duplicates, so `iconText` is unambiguous).
6. **Captioned groups.** Related buttons sit in a `_ribbon_group(caption, items)` with an
   uppercase eyebrow caption (`ribbonCap`); groups are divided by a thin `ribbonVSep` rule.
   An item may be a bare widget (e.g. the grid-snap spin box) — added as-is.
7. **Every ribbon button should carry a themed icon.** A text-only ribbon button is a smell
   (see R6). Icons come from `icons.icon(name, style.ICON)` via `_set_icon`, never hard-coded.

### C. Theming & tokens
8. **No literals, ever.** Colours / spacing / radius / font sizes come from `style` tokens.
   The strip must render correctly in light **and** dark and in comfortable **and** compact — the
   `QToolButton#ribbonFile`, `QTabBar#ribbonTabs`, `#ribbonStrip/#ribbonPage/#ribbonHost` rules
   all read palette tokens, and the compact overlay tightens `ribbonBtn` / `ribbonFile` padding.
9. **The body owns the space it frees.** Freed vertical space goes to the workspace, not to
   padding. Keep the host toolbar flush (`padding: 0`) with a single bottom rule.

---

## 3. Tab & group taxonomy (authoritative)

The canonical map of every command. Keep new commands consistent with it; if a command doesn't
fit, add a group — don't add a tab lightly (target ≤ 8 tabs). Built in
[`desktop/main_window.py`](../../desktop/main_window.py) `_build_menu`.

| Tab | Group | Buttons (`act_*` → short label) |
|-----|-------|--------------------------------|
| **File** *(backstage)* | — | new · new3d · open · save · saveas |
| **Home** | Model | add_node·Node · add_member·Member · add_section·Section · materials·Materials |
| | Edit | undo·Undo · redo·Redo · delete·Delete |
| | Modify | move·Move · copy·Copy · mirror·Mirror · rotate·Rotate · extrude·Extrude |
| | Generate | gen·Frame |
| | Tools | sectiondesigner·Designer |
| **Draw** | Draw | draw_node·Node · draw_member·Member · snap·Snap · *grid-snap spin* |
| | Select | select·Select · sel_window·Window · sel_poly·Poly · deselect·Deselect |
| | Select by | sel_all_nodes·Nodes · sel_all_members·Members · sel_all·All · sel_by_section·Section |
| **Loads** | Loads | loadcases·Cases · add_load·Nodal · add_lineload·Line · genloads·Generate |
| | Combinations | editcombos·Combos · gencombos·ASCE-7 |
| **Analysis** | Analyse | analysiscases·Cases · runanalysis·Run · run·Linear |
| | Hinges | hinges·Define · assign_hinges·Assign |
| **Results** | Diagrams | undef·Undeformed · diag_n·Axial · diag_v·Shear · diag_m·Moment |
| | Reports | runhistory·History |
| | Design | design·Design · checkmodel·Check |
| **View** | Navigate | fit·Fit |
| | Orient | v_iso·Iso · v_top·Top · v_front·Front · v_right·Right · v_left·Left · v_back·Back · v_bottom·Bottom |
| | Display | drawings·Drawings |
| | Appearance | theme·Theme · density·Compact |

*Dialog-only (no ribbon button):* `act_pushover`, `act_timehistory` (launched from the
Analysis-cases home).

---

## 4. Anatomy (where things live)

- `RibbonBar` (class, `desktop/main_window.py`) — the widget: `file_btn` + `file_menu`, `tabs`
  (`QTabBar#ribbonTabs`), `stack` (`QStackedWidget#ribbonStack`). API in §0.8.
- `_ribbon_page(groups)` — one tab body: a left-aligned row of groups + `ribbonVSep` rules.
- `_ribbon_group(caption, items)` — one captioned cluster; items are `(action, label)` or a
  bare widget.
- Host: a single non-movable `QToolBar#ribbonHost` added in `_build_menu`; the `QMenuBar` is
  hidden.
- QSS: `desktop/style.py` — "tabbed ribbon shell (plan ribbon R1)" block + compact overlay
  entries. Tab look reuses the base `QTabBar::tab` underline-selected style.

---

## 5. Tracker

Legend: `[x]` done · `[ ]` open. Do the lowest open item whose deps are met.

**Core is done (R1–R6): the top chrome is one compact strip, iconed, keyboard-driven,
persistent, contextual and collapsible.** Remaining items are polish/robustness, not
load-bearing — **R7 + R8 done; only R9 left** (extends the model to the second window — largest,
least urgent) plus the R10–R12 backlog.

- [x] **R1 — Tabbed ribbon shell.** Replace the menu bar + 3 stacked rows + draw palette with one
  `RibbonBar` (File backstage + tabs Home/Draw/Loads/Analysis/Results/View, one swapping group
  row). Re-home every menu-only command per §3. Rewrite chrome tests. — `feat(ribbon R1)`,
  2026-09-13.
- [x] **R2 — Persist & restore the active tab** across sessions via `QSettings` key `ribbon/tab`
  (default Home; unknown name falls back to Home). Persisted on every switch. — `feat(ribbon R2)`,
  2026-09-14.
- [x] **R3 — Keyboard access.** Alt+H/D/L/A/R/V raise a tab by its initial, Ctrl+Tab /
  Ctrl+Shift+Tab cycle (wrapping), and the tab strip is `TabFocus`-able with a soft hover/focus
  fill (`outline:none` kills the native dotted rect). — `feat(ribbon R3)`, 2026-09-14.
- [x] **R4 — Contextual tab raising.** `run_linear_static` success raises **Results** (covers
  Ctrl+R, Run-analysis, Analysis-cases); `_set_mode` raises **Draw** when a draw/select tool
  activates. Both no-op while collapsed (R5) so nothing pops up unbidden. — `feat(ribbon R4)`,
  2026-09-14.
- [x] **R5 — Collapse / expand the ribbon.** Double-click the active tab or Ctrl+F1 collapses the
  group row to just the tab strip; while collapsed a single tab click reveals it transiently until
  a click outside. State persisted (`ribbon/collapsed`). — `feat(ribbon R5)`, 2026-09-14.
- [x] **R6 — Icon coverage.** Every ribbon button now carries a themed glyph (was text-only:
  Materials, Undo, Redo, the *Select by* group, Hinges Define/Assign, History, Check, Compact).
  Guarded by `test_every_ribbon_button_has_an_icon`. — `feat(ribbon R6)`, 2026-09-14.
- [x] **R7 — Width overflow.** `_ribbon_page` became `RibbonPage`: on resize it measures the
  groups and collapses the lowest-priority (rightmost) ones into a single `»` popup (their actions
  become a grouped menu) instead of clipping; reverses when width returns, always keeping ≥1 group
  inline. Guarded by `test_narrow_width_collapses_groups_into_overflow`. — code swept into
  `0b2c86e`; closed 2026-09-15.
- [x] **R8 — File backstage panel.** The File button now opens a full-window backstage
  (`desktop/backstage.py`): a command rail (New · New 3-D · Open · Save · Save As), a persisted
  **Recent** list (MRU in `recent/files`, filtered to existing files), and an **About** card, on
  the card scaffold. Back arrow / Esc closes. *(Export deferred — no export action exists yet; see
  R12.)* — `feat(ribbon R8)`, 2026-09-14.
- [ ] **R9 — Section Designer parity.** Bring the separate Section Designer window
  (`section_designer.py`, still its own `QMenuBar`) onto the same ribbon model, or explicitly
  scope it out here with a rationale. Deps: R1.
- [ ] **R10 — Broaden contextual raising (from R4).** R4 only raises **Results** after
  *linear-static*; the modal / response-spectrum / buckling / moving-load runs open their own
  result surfaces without surfacing the ribbon. Route those through a single post-run hook so
  every analysis type raises the right tab. The **Draw**-raise in `_set_mode` is effectively a
  no-op today (those tools already live on the Draw tab) — revisit if a command palette / shortcut
  can activate a tool from elsewhere. Deps: R4.
- [ ] **R11 — Quick Access Toolbar (optional).** A small, user-pinnable row of common actions
  (Save · Undo · Redo · Run) beside the File button, independent of the active tab — the last piece
  of the CSi/Office ribbon idiom. Deps: R1. *(Nice-to-have; only if the strip still feels sparse.)*
- [ ] **R12 — Export (from R8).** The backstage has no **Export** command because the app has no
  export action yet (only project save/open). When an export path exists (results CSV, drawing/PDF,
  model interchange), add it as a backstage command + a Home/Results button. Deps: R8.

---

## 6. Change log

- **2026-09-15 — R7.** Width overflow. `_ribbon_page` → `RibbonPage(QWidget)`: keeps its groups +
  a hidden `»` `QToolButton#ribbonMore`; `resizeEvent`→`_relayout` measures group sizeHints against
  the available width and hides the rightmost groups that don't fit (min one stays), toggling `»`;
  `_fill_overflow` (on `aboutToShow`) rebuilds the popup menu from `hidden_captions()` as captioned
  sections of the hidden groups' actions. QSS `ribbonMore` in `style.py`. Tests: +3 in
  `test_desktop_ribbon.py`. NB: the code was swept into another session's `0b2c86e` (shared
  working tree, `git add -A`); this entry + screenshots close it out. Ribbon+backstage suite 33
  passed. Screenshots `R7_overflow_{light,dark}.png`.
- **2026-09-14 — R8.** File backstage. New `desktop/backstage.py` (`Backstage(QWidget)`,
  `WA_StyledBackground` so the overlay fully occludes): a command rail built from the ribbon's file
  actions, a Recent list, and an About card on `GroupCard`. The ribbon File button drops its popup
  menu and its `clicked` opens the overlay (`_open_backstage`); `file_menu` is kept as the canonical
  action list. MRU: `_recent_files`/`_remember_recent`/`_open_recent` (QSettings `recent/files`,
  cap 8, existence-filtered), hooked in `load_project`/`_write`; `resizeEvent` keeps the overlay
  sized. QSS block in `style.py`. Tests: new `tests/test_desktop_backstage.py` (5). Desktop suite
  326 passed. Screenshots `R8_backstage_{light,dark}.png`.
- **2026-09-14 — R4 + R5.** R4 (contextual tabs): `MainWindow._show_results_tab` called at the end
  of `run_linear_static` raises **Results**; `_set_mode` raises **Draw** on any draw/select tool.
  Both guarded to no-op while collapsed. R5 (collapse): `RibbonBar` gains `collapsedChanged` signal
  + `set_collapsed`/`toggle_collapsed`/`is_collapsed`, `tabBarDoubleClicked`→toggle, and a transient
  reveal (`_on_tab_clicked` shows the row + an app event filter re-collapses on an outside click);
  Ctrl+F1 toggles; state persists in `ribbon/collapsed`. Imports `QEvent`, `Signal`. Tests: +7 in
  `test_desktop_ribbon.py` (Results/Draw raise, suppressed-when-collapsed, collapse/toggle, persist,
  transient, Ctrl+F1). Deterministic desktop suite 321 passed (the lone order-dependent
  `test_main_window_wires_line_loads` failure is the nav/summary QSettings leak — see below —
  unrelated). Screenshot `R5_collapsed_light.png`.
- **2026-09-14 — R2 + R3.** Ribbon navigation. R2: `MainWindow._persist_ribbon_tab` saves the
  active tab to `QSettings("MidasStructural","Desktop")` key `ribbon/tab` on every switch;
  construction restores it (unknown name → Home). R3: `_install_ribbon_shortcuts` adds Alt+initial
  accelerators (H/D/L/A/R/V) and Ctrl+Tab / Ctrl+Shift+Tab cycling via `_cycle_ribbon_tab`; the tab
  strip is now `TabFocus` with `QTabBar#ribbonTabs` hover/focus fill in `style.py` (`outline:none`).
  Imports `QShortcut`, `QKeySequence`. Tests: +4 in `test_desktop_ribbon.py` (persist/restore,
  fallback, accelerators, cycle-wrap), all resetting `ribbon/tab` in teardown. Desktop suite: 314
  passed. Behavioural items — no new screenshots (the R1/R6 record still holds).
- **2026-09-14 — R6.** Icon coverage. Added 10 in-house glyphs to `desktop/icons.py`
  (`materials`, `selnodes`, `selmembers`, `selall`, `selsection`, `hinge`, `assignhinge`,
  `history`, `checkmodel`, `density`) and wired the 12 previously text-only ribbon buttons via
  `_set_icon` / the `_action(..., icon_name)` arg (`undo`/`redo` glyphs already existed). New
  regression guard `test_every_ribbon_button_has_an_icon`. Screenshots `R6_home_light.png`,
  `R6_draw_light.png`, `R6_view_dark.png`. Desktop suite: 310 passed. (Note: a stale shared
  `nav/summary=True` QSettings value — left by the model-navigation suite — makes the unrelated
  `test_desktop_member_load::test_main_window_wires_line_loads` fail until reset; not an R6
  regression.)
- **2026-09-13 — R1.** New `RibbonBar` (tab strip + `QStackedWidget` + File backstage) in
  `desktop/main_window.py` replaces the classic menu bar, the three `ribbonBar` toolbars, and the
  draw/select palette; `_ribbon_group` now accepts bare widgets and `_ribbon_page` builds a tab
  body. Tokens/QSS added in `desktop/style.py` ("tabbed ribbon shell" block + compact entries:
  `ribbonHost`, `ribbonRoot/Strip/Page/Stack`, `ribbonFile`, `ribbonTabs`, `ribbonVSep`). Menu
  bar is hidden; every command re-homed per §3. Tests: rewrote `tests/test_desktop_ribbon.py`
  (13 cases incl. the "homed exactly once" guard), updated the reachability test in
  `test_desktop_shell_theme.py`, and removed the superseded `test_desktop_menus.py` /
  `test_desktop_toolbars.py` (S1/S2/S3 contracts). Screenshots `R1_ribbon_light.png`,
  `R1_ribbon_dark.png`. Desktop suite: 223 passed.
