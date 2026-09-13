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

- [x] **R1 — Tabbed ribbon shell.** Replace the menu bar + 3 stacked rows + draw palette with one
  `RibbonBar` (File backstage + tabs Home/Draw/Loads/Analysis/Results/View, one swapping group
  row). Re-home every menu-only command per §3. Rewrite chrome tests. — `feat(ribbon R1)`,
  2026-09-13.
- [ ] **R2 — Persist & restore the active tab** across sessions via `QSettings` (like theme /
  density); default to Home on first run. Deps: R1.
- [ ] **R3 — Keyboard access.** Alt-accelerators to raise each tab (Alt+H/D/L/A/R/V), Ctrl+Tab /
  Ctrl+Shift+Tab to cycle, and a visible focus ring on `ribbonTabs`. Deps: R1.
- [ ] **R4 — Contextual tab raising.** Auto-raise **Results** after a successful analysis run and
  **Draw** when a draw/select tool becomes active (CSi contextual-ribbon behaviour), without
  stealing focus mid-edit. Deps: R1.
- [ ] **R5 — Collapse / expand the ribbon.** Double-click the active tab (and Ctrl+F1) collapses
  the group row to just the tab strip; next tab click shows it transiently. Reclaims the last row
  on demand. Deps: R1.
- [ ] **R6 — Icon coverage.** Give every ribbon button a themed icon (currently text-only:
  Materials, Undo, Redo, the *Select by* group, some *Orient* views). Add glyphs to `icons.py`
  as needed; keep `_set_icon` / `style.ICON`. Deps: R1.
- [ ] **R7 — Width overflow.** When a tab's groups exceed the window width, collapse the
  lowest-priority group(s) to a single popup button (Office/CSi behaviour) instead of clipping.
  Deps: R1.
- [ ] **R8 — File backstage panel.** Promote the File popup menu to a proper backstage (recent
  files, New/Open/Save/Save As/Export, About) styled on the card scaffold. Deps: R1.
- [ ] **R9 — Section Designer parity.** Bring the separate Section Designer window
  (`section_designer.py`, still its own `QMenuBar`) onto the same ribbon model, or explicitly
  scope it out here with a rationale. Deps: R1.

---

## 6. Change log

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
