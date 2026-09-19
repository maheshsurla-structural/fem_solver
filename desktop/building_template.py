"""Building quick-template — uniform grid + simple story stack (wall plan W8c).

The ETABS "New Model Quick Templates" idiom: a uniform rectangular grid plus a
simple story stack (N stories, a typical height, a possibly-different bottom
story), generated in one shot. Pure — returns ``(stories, grid_lines)`` so the
dialog can preview it live and the caller applies it.

Grid convention matches ``model_geometry``/``StoryGridDialog``: an *X-direction*
grid line is a line of **constant X** (``axis="x"``, runs in Y), labelled from
``x_label`` (A, B, C, …); a *Y-direction* line is **constant Y** (``axis="y"``),
labelled from ``y_label`` (1, 2, 3, …).
"""
from __future__ import annotations

from project import GridLine, Story


def _label_seq(start: str, n: int) -> list:
    """``n`` labels starting at ``start``: letters roll A→B→…→Z→AA; a numeric
    start counts 1, 2, 3, …."""
    start = str(start).strip() or "A"
    if start.isdigit():
        base = int(start)
        return [str(base + i) for i in range(n)]

    def _alpha(k: int) -> str:                       # 0→A, 25→Z, 26→AA
        s = ""
        k += 1
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(ord("A") + r) + s
        return s

    off = 0
    su = start.upper()
    if su.isalpha():
        off = 0
        for ch in su:
            off = off * 26 + (ord(ch) - ord("A") + 1)
        off -= 1
    return [_alpha(off + i) for i in range(n)]


def build_grid_stories(nx: int, ny: int, sx: float, sy: float,
                       n_stories: int, typ_h: float, bot_h: float | None = None,
                       *, x_label: str = "A", y_label: str = "1",
                       base_elev: float = 0.0):
    """Generate ``(stories, grid_lines)`` for a uniform grid + simple story
    stack. All lengths SI. ``nx``/``ny`` grid lines each way at ``sx``/``sy``
    spacing; ``n_stories`` above the base, typical height ``typ_h``, bottom
    (ground) story ``bot_h`` (defaults to ``typ_h``)."""
    nx, ny = max(int(nx), 0), max(int(ny), 0)
    n_stories = max(int(n_stories), 0)
    typ_h = float(typ_h)
    bot_h = float(bot_h) if bot_h is not None else typ_h

    grids: list = []
    gid = 1
    for i, name in enumerate(_label_seq(x_label, nx)):
        grids.append(GridLine(id=gid, name=name, axis="x", coord=i * float(sx)))
        gid += 1
    for j, name in enumerate(_label_seq(y_label, ny)):
        grids.append(GridLine(id=gid, name=name, axis="y", coord=j * float(sy)))
        gid += 1

    stories = [Story(id=1, name="Base", elev=float(base_elev), height=0.0)]
    z = float(base_elev)
    for i in range(1, n_stories + 1):
        h = bot_h if i == 1 else typ_h
        z += h
        stories.append(Story(id=i + 1, name=f"Story {i}", elev=z, height=h))
    return stories, grids
