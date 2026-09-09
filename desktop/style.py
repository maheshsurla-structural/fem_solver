"""Visual design system for the desktop GUI — a light, blue-accent theme.

A single Qt stylesheet (``QSS``) plus matplotlib chart helpers, so the widgets,
the section drawing and the plots read as one professional product rather than
raw native controls. "Comfortable" density: generous spacing and clear
grouping. Applied per-window with ``apply(widget)``; charts call
``beautify_axes(ax)``.
"""
from __future__ import annotations

# ---- design tokens (theme-independent) -------------------------------------
FONT_STACK = "'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif"
MONO_STACK = "'Cascadia Mono', 'Consolas', 'SF Mono', monospace"
# Spacing — a 4px base on an 8px rhythm. Reach for these, not ad-hoc pixels.
SP_XS, SP_SM, SP_MD, SP_LG, SP_XL = 4, 8, 12, 16, 24
# Corner radius by role: controls / cards / large surfaces.
R_SM, R_MD, R_LG = 6, 8, 12
# Type scale (px) — one ladder shared across every screen.
FS_DISPLAY, FS_H1, FS_H2, FS_H3 = 26, 18, 15, 14
FS_BODY, FS_SMALL, FS_MICRO = 13, 12, 11
LS_LABEL = "0.06em"          # tracking for uppercase captions / eyebrows

# ---- palettes (light / dark) -----------------------------------------------
# Every colour token lives here so the whole app can swap themes at runtime.
# The active palette is installed onto the module namespace (style.BG, …) so
# call-time reads of ``style.X`` always see the current theme.
_LIGHT = dict(
    BG="#f4f6f9", PANEL="#ffffff", BORDER="#d9dee6", BORDER_STRONG="#c3cad4",
    TEXT="#1e2430", MUTED="#6b7482", ACCENT="#2563eb", ACCENT_HOVER="#1d4ed8",
    ACCENT_SOFT="#eaf1fe", OK="#1a7f37", BAD="#cf222e", ZEBRA="#f7f9fc",
    WARN="#9a6a12", WARN_SOFT="#faf0da", OK_SOFT="#e4f4e9", BAD_SOFT="#fbe6e9",
    ICON="#44506a",
    # chart series + axes
    C_PRIMARY="#2563eb", C_SECONDARY="#d1462f", C_MILESTONE="#d1462f",
    C_DEMAND="#e3a008", GRID="#e6eaf0", AX_SPINE="#c3cad4", AX_TEXT="#3a4150",
    # named curve roles (C2): exact=green · fibre=red · nominal=blue ·
    # design=neutral · demand=amber (C_DEMAND above) — legible on both grounds
    C_EXACT="#1a7f37", C_FIBRE="#d1462f", C_NOMINAL="#2563eb", C_DESIGN="#64748b",
    # 3-D interaction surface: green wireframe + magenta current-slice highlight
    C_WIRE="#16a34a", C_SLICE="#c026d3",
    # section canvas
    CANVAS_BG="#ffffff", CANVAS_GRID="#eef2f6", BODY_FILL="#cfe8ff",
    BODY_STROKE="#1f4f73",
)
_DARK = dict(
    BG="#151922", PANEL="#1e232d", BORDER="#2c333f", BORDER_STRONG="#3b4553",
    TEXT="#e7ecf3", MUTED="#93a0b2", ACCENT="#5b93f7", ACCENT_HOVER="#79a7ff",
    ACCENT_SOFT="#243350", OK="#46b768", BAD="#f0616d", ZEBRA="#232936",
    WARN="#d9a441", WARN_SOFT="#39301c", OK_SOFT="#1c3326", BAD_SOFT="#3a2329",
    ICON="#aab4c2",
    C_PRIMARY="#5b93f7", C_SECONDARY="#f2795f", C_MILESTONE="#f2795f",
    C_DEMAND="#e3b341", GRID="#2a313d", AX_SPINE="#3b4553", AX_TEXT="#aab4c2",
    # named curve roles (C2) — brightened for the dark ground
    C_EXACT="#46b768", C_FIBRE="#f2795f", C_NOMINAL="#5b93f7", C_DESIGN="#94a3b8",
    C_WIRE="#4ade80", C_SLICE="#e879f9",
    CANVAS_BG="#1a1f28", CANVAS_GRID="#262d38", BODY_FILL="#26374f",
    BODY_STROKE="#5b93f7",
)
_PALETTES = {"light": _LIGHT, "dark": _DARK}
_theme = "light"

# The stylesheet is a str.format template: ``{{`` / ``}}`` are literal CSS
# braces and ``{TOKEN}`` fields are filled from the active palette + tokens.
_QSS_TEMPLATE = """
* {{
    font-family: {FONT_STACK};
    font-size: {FS_BODY}px;
    color: {TEXT};
}}
QMainWindow, QDialog, QScrollArea, QWidget#sd_inner {{
    background: {BG};
}}
QScrollArea {{ border: none; }}

/* ---- grouping cards ---- */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 16px;
    padding: 12px 12px 10px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 0 4px;
    color: {MUTED};
    font-size: 12px;
}}

/* ---- inputs ---- */
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border-color: {MUTED};
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    width: 16px; border: none; background: transparent;
}}

/* ---- buttons ---- */
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 22px;
    color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_SOFT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {BORDER}; }}

/* a menu tool-button that should read like a push button (e.g. "＋ New") */
QToolButton#navNew {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: {R_SM}px;
    padding: 5px 12px;
    color: {TEXT};
}}
QToolButton#navNew:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton#navNew::menu-indicator {{ image: none; width: 0; }}

/* ---- tabs ---- */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
    background: {PANEL};
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 7px 16px;
    margin-right: 2px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* ---- tables ---- */
QTableWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    alternate-background-color: {ZEBRA};
}}
QHeaderView::section {{
    background: {ACCENT_SOFT};
    color: {MUTED};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QTableWidget::item {{ padding: 3px 6px; }}
QListWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
}}
QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}

/* ---- toolbar / menu / status ---- */
QToolBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px 8px;
}}
QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item:selected {{ background: {ACCENT_SOFT}; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER};
             color: {MUTED}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG}; border-radius: 5px; min-width: 30px;
}}

/* ---- collapsible group (widgets.CollapsibleGroup) ---- */
QWidget#collGroup {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QToolButton#collHeader {{
    background: transparent;
    border: none;
    color: {MUTED};
    font-size: 12px;
    font-weight: 600;
    padding: 8px 10px;
    text-align: left;
}}
QToolButton#collHeader:hover {{ color: {TEXT}; }}
QWidget#collBody {{ background: transparent; }}

/* ---- trees / logs / docks (FEM main window) ---- */
QTreeWidget, QTreeView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
    alternate-background-color: {ZEBRA};
}}
QTreeWidget::item, QTreeView::item {{ padding: 3px 4px; }}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: {ACCENT_SOFT}; color: {TEXT};
}}
QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QDockWidget {{ titlebar-close-icon: none; color: {MUTED}; }}
QDockWidget::title {{
    background: {ACCENT_SOFT};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    text-align: left;
}}

/* ---- type scale (set via setObjectName) ---- */
QLabel#display {{ font-size: {FS_DISPLAY}px; font-weight: 700; color: {TEXT};
                 letter-spacing: -0.02em; }}
QLabel#h1 {{ font-size: {FS_H1}px; font-weight: 700; color: {TEXT};
            letter-spacing: -0.01em; }}
QLabel#h2 {{ font-size: {FS_H2}px; font-weight: 600; color: {TEXT}; }}
QLabel#h3 {{ font-size: {FS_H3}px; font-weight: 600; color: {TEXT}; }}
QLabel#sub {{ color: {MUTED}; font-size: {FS_SMALL}px; }}
QLabel#caption {{ color: {MUTED}; font-size: {FS_SMALL}px; }}
QLabel#hintLabel {{ color: {MUTED}; font-size: {FS_MICRO}px; }}
QLabel#warnText {{ color: {BAD}; font-size: {FS_SMALL}px; }}
QLabel#eyebrow {{ color: {MUTED}; font-size: {FS_MICRO}px; font-weight: 600;
                 letter-spacing: {LS_LABEL}; }}

/* ---- KPI + verdict pills (for result headers) ---- */
QLabel#kpiValue {{ font-size: 22px; font-weight: 700; color: {TEXT};
                  font-family: {MONO_STACK}; }}
QLabel#kpiLabel {{ color: {MUTED}; font-size: {FS_MICRO}px; font-weight: 600;
                  letter-spacing: {LS_LABEL}; }}
QLabel#pillPass {{ background: {OK_SOFT}; color: {OK}; font-weight: 700;
                  border-radius: {R_SM}px; padding: 3px 12px; }}
QLabel#pillFail {{ background: {BAD_SOFT}; color: {BAD}; font-weight: 700;
                  border-radius: {R_SM}px; padding: 3px 12px; }}
QLabel#pillWarn {{ background: {WARN_SOFT}; color: {WARN}; font-weight: 700;
                  border-radius: {R_SM}px; padding: 3px 12px; }}
QWidget#verdictStrip {{ background: {ZEBRA}; border: 1px solid {BORDER};
                       border-radius: {R_MD}px; }}
QFrame#kpiTile {{ background: {PANEL}; border: 1px solid {BORDER};
                 border-radius: {R_MD}px; }}
QFrame#kpiTile QLabel#kpiNum {{ font-size: 19px; font-weight: 700; color: {TEXT};
                               font-family: {MONO_STACK}; }}
QFrame#kpiTile QLabel#kpiCap {{ color: {MUTED}; font-size: {FS_MICRO}px;
                               font-weight: 600; letter-spacing: {LS_LABEL}; }}
QProgressBar#utilBar {{ background: {BORDER}; border: none; border-radius: 3px; }}
QProgressBar#utilBar::chunk {{ border-radius: 3px; }}

/* ---- application header (brand + active section + setting chips) ---- */
QToolBar#hdrToolbar {{ background: {PANEL}; border: none; padding: 0;
                      spacing: 0; }}
QToolBar#hdrToolbar::separator {{ width: 0; }}
QFrame#appHeader {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QLabel#brandWord {{ font-size: {FS_H2}px; font-weight: 700; color: {TEXT};
                   letter-spacing: -0.01em; }}
QLabel#brandDot {{ color: {ACCENT}; font-size: {FS_H1}px; font-weight: 700; }}
QFrame#hdrRule {{ background: {BORDER}; }}
QLabel#hdrTitle {{ font-size: {FS_H3}px; font-weight: 600; color: {TEXT}; }}
QFrame#chip {{ background: {BG}; border: 1px solid {BORDER}; border-radius: {R_MD}px; }}
QFrame#chip QComboBox {{ border: none; background: transparent; padding: 1px 4px;
                        min-height: 18px; font-weight: 600; }}
QFrame#chip QComboBox:hover {{ color: {ACCENT}; }}
QFrame#chip QComboBox::drop-down {{ width: 16px; }}
QLabel#chipLabel {{ color: {MUTED}; font-size: {FS_MICRO}px; font-weight: 600;
                   letter-spacing: {LS_LABEL}; }}

/* live cursor coordinate readout under an analysis chart */
QLabel#chartCursor {{ color: {MUTED}; font-family: {MONO_STACK};
                     font-size: {FS_MICRO}px; }}
/* control bar beneath the P-M-M graphs (CSi-style) */
QFrame#chartCtrlBar {{ background: {PANEL}; border: 1px solid {BORDER};
                      border-radius: 8px; }}

/* ---- constant [ controls | view ] workspace shell ---- */
QWidget#ctrlRail {{ background: {BG}; border-right: 1px solid {BORDER}; }}
QLabel#ctrlHead {{ color: {MUTED}; font-size: {FS_MICRO}px; font-weight: 600;
                  letter-spacing: {LS_LABEL}; }}

/* ---- flattened canvas chrome (single toolbar + properties drawer) ---- */
QFrame#canvasBar {{ background: {PANEL}; border: 1px solid {BORDER};
                   border-radius: {R_MD}px; }}
QFrame#canvasBar QToolButton {{ border: none; border-radius: {R_SM}px;
                               padding: 2px; color: {TEXT}; }}
QFrame#canvasBar QToolButton:hover {{ background: {ACCENT_SOFT}; }}
QFrame#canvasBar QToolButton:checked {{ background: {ACCENT_SOFT};
                                       color: {ACCENT}; }}
QFrame#barSep {{ background: {BORDER}; }}
QFrame#propsDrawer {{ background: {PANEL}; border: 1px solid {BORDER};
                     border-radius: {R_MD}px; }}

/* ---- transient toast notifications ---- */
QLabel#toastOk, QLabel#toastErr {{ color: #ffffff; font-weight: 600;
    padding: 9px 18px; border-radius: {R_LG}px; }}
QLabel#toastOk {{ background: {OK}; }}
QLabel#toastErr {{ background: {BAD}; }}
QProgressBar#busyBar {{ background: {BORDER}; border: none; border-radius: 3px;
    max-height: 6px; }}
QProgressBar#busyBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

/* ---- section navigator rows ---- */
QFrame#navThumb {{ background: {CANVAS_BG}; border: 1px solid {BORDER};
                  border-radius: {R_SM}px; }}
QLabel#navName {{ font-weight: 600; color: {TEXT}; background: transparent; }}
QLabel#navSub {{ color: {MUTED}; font-size: {FS_SMALL}px;
                background: transparent; }}
QLineEdit#navEdit {{ padding: 1px 4px; }}

/* ---- empty-state canvas hint ---- */
QLabel#canvasHint {{ color: {MUTED}; font-size: {FS_H3}px;
    background: transparent; }}

/* ---- template gallery ---- */
QFrame#tplCard {{ background: {PANEL}; border: 1px solid {BORDER};
    border-radius: {R_MD}px; }}
QFrame#tplCard:hover {{ border-color: {ACCENT}; }}
QLabel#tplName {{ font-weight: 600; color: {TEXT}; background: transparent; }}

/* ---- command palette (Ctrl+K) ---- */
QDialog#cmdPalette {{ background: {PANEL}; border: 1px solid {BORDER_STRONG};
    border-radius: {R_LG}px; }}
QLineEdit#cmdEdit {{ font-size: {FS_H3}px; padding: 8px 10px; }}
QListWidget#cmdList {{ border: none; background: transparent; }}
QListWidget#cmdList::item {{ padding: 7px 10px; border-radius: {R_SM}px; }}
QListWidget#cmdList::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
"""


# Compact density: a thin QSS overlay that tightens paddings / heights. Plain
# CSS (single braces) — appended after the themed base sheet, not formatted.
_density = "comfortable"
_COMPACT_QSS = """
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { padding: 2px 6px;
    min-height: 17px; }
QPushButton { padding: 3px 10px; min-height: 17px; }
QToolButton#navNew { padding: 3px 9px; }
QTableWidget::item { padding: 1px 5px; }
QHeaderView::section { padding: 3px 6px; }
QListWidget::item { padding: 2px 6px; }
QTabBar::tab { padding: 5px 12px; }
QGroupBox { margin-top: 13px; padding: 8px 8px 6px 8px; }
"""


def _install(theme: str) -> None:
    """Copy the chosen palette onto the module namespace and rebuild QSS."""
    global _theme, QSS
    _theme = "dark" if theme == "dark" else "light"
    globals().update(_PALETTES[_theme])
    QSS = _QSS_TEMPLATE.format(**globals())


def set_density(mode: str) -> str:
    global _density
    _density = "compact" if mode == "compact" else "comfortable"
    return _density


def current_density() -> str:
    return _density


def current_theme() -> str:
    return _theme


def set_theme(theme: str) -> str:
    """Switch the active palette; returns the resolved theme name."""
    _install(theme)
    return _theme


def toggle_theme() -> str:
    return set_theme("light" if _theme == "dark" else "dark")


_install("light")            # seed the module with the light palette + QSS


def apply(widget) -> None:
    """Apply the current theme + density stylesheet to a top-level widget."""
    widget.setStyleSheet(QSS + (_COMPACT_QSS if _density == "compact" else ""))


def beautify_axes(ax, *, title_color=None) -> None:
    """Style a matplotlib Axes to match the app: light grid, soft spines, the
    app's text color, no top/right spines."""
    ax.set_facecolor(PANEL)
    if ax.figure is not None:
        ax.figure.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AX_SPINE)
    ax.tick_params(colors=AX_TEXT, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(AX_TEXT)
        lbl.set_fontsize(10)
    if ax.title is not None:
        ax.title.set_color(title_color or TEXT)
        ax.title.set_fontsize(11)
        ax.title.set_fontweight("bold")
