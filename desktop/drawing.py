"""General-arrangement drawing of a Project — an on-screen matplotlib sheet
and a DXF export, both drawn from the same geometry (2-D frames for now).

No Qt / OpenGL here: ``draw_sheet`` renders onto any matplotlib Figure (so it
works headless via the Agg backend), and ``export_dxf`` uses the engine's
``femsolver.results.dxf`` writer.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
from matplotlib.patches import Polygon

MEMBER_C = "#1a1a1a"
NODE_C = "#1a1a1a"
LABEL_C = "#1d4ed8"
NODELABEL_C = "#666666"
DIM_C = "#a3282b"


def _node_xy(project) -> dict:
    return {n.id: (float(n.x), float(n.y)) for n in project.nodes}


def _bbox(project):
    xs = [n.x for n in project.nodes]
    ys = [n.y for n in project.nodes]
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def _span(project) -> float:
    x0, y0, x1, y1 = _bbox(project)
    return float(np.hypot(x1 - x0, y1 - y0)) or 1.0


def member_schedule(project):
    """Rows of (member_id, node_i, node_j, section label, length_m)."""
    xy = _node_xy(project)
    secs = {s.id: s for s in project.sections}
    rows = []
    for m in project.members:
        p1, p2 = xy.get(m.n1), xy.get(m.n2)
        L = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) if p1 and p2 else 0.0
        s = secs.get(m.section)
        label = (s.shape or s.name) if s else "?"
        rows.append((m.id, m.n1, m.n2, label, L))
    return rows


# ------------------------------------------------------------------ matplotlib

def draw_sheet(project, fig):
    """Render the GA drawing + member schedule onto a matplotlib Figure."""
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[7.0, 3.0], hspace=0.28)
    _draw_ga(project, fig.add_subplot(gs[0]))
    _draw_schedule(project, fig.add_subplot(gs[1]))
    fig.suptitle(f"{project.name}  —  General Arrangement",
                 fontsize=12, x=0.5, y=0.98)
    fig.text(0.99, 0.01,
             f"femsolver desktop · {_dt.date.today().isoformat()} · "
             "preliminary — not for construction",
             ha="right", va="bottom", fontsize=6.5, color="#888")
    return fig


def _draw_ga(project, ax) -> None:
    xy = _node_xy(project)
    secs = {s.id: s for s in project.sections}
    for m in project.members:
        p1, p2 = xy[m.n1], xy[m.n2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=MEMBER_C, lw=2, zorder=2)
        s = secs.get(m.section)
        lab = f"M{m.id}" + (f" · {s.shape or s.name}" if s else "")
        ax.annotate(lab, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
                    textcoords="offset points", xytext=(4, 4), fontsize=7,
                    color=LABEL_C, zorder=4)
    for n in project.nodes:
        ax.plot(n.x, n.y, "o", ms=5, color=NODE_C, zorder=3)
        ax.annotate(f"N{n.id}", (n.x, n.y), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color=NODELABEL_C, zorder=4)
        if n.supports and any(n.supports):
            _support_symbol(ax, n.x, n.y, _span(project))
    _dimensions(project, ax, xy)
    x0, y0, x1, y1 = _bbox(project)
    off = max(_span(project) * 0.08, 0.3)
    ax.set_xlim(x0 - off * 3.2, x1 + off)
    ax.set_ylim(y0 - off * 3.2, y1 + off)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _support_symbol(ax, x, y, span) -> None:
    s = max(span * 0.03, 0.15)
    ax.add_patch(Polygon([[x, y], [x - s / 2, y - s], [x + s / 2, y - s]],
                         closed=True, fill=False, edgecolor=MEMBER_C, lw=1,
                         zorder=3))
    for i in range(4):
        xx = x - s / 2 + i * s / 3
        ax.plot([xx, xx - s / 4], [y - s, y - s - s / 3], color=MEMBER_C,
                lw=0.6, zorder=3)


def _dimensions(project, ax, xy) -> None:
    x0, y0, x1, y1 = _bbox(project)
    if x1 == x0 and y1 == y0:
        return
    off = max(_span(project) * 0.08, 0.3)
    yb = y0 - off * 2.5
    ax.annotate("", xy=(x1, yb), xytext=(x0, yb),
                arrowprops=dict(arrowstyle="<->", color=DIM_C, lw=0.8))
    ax.text((x0 + x1) / 2, yb - off * 0.3, f"{x1 - x0:g} m", ha="center",
            va="top", fontsize=8, color=DIM_C)
    xl = x0 - off * 2.5
    ax.annotate("", xy=(xl, y1), xytext=(xl, y0),
                arrowprops=dict(arrowstyle="<->", color=DIM_C, lw=0.8))
    ax.text(xl - off * 0.3, (y0 + y1) / 2, f"{y1 - y0:g} m", ha="right",
            va="center", rotation=90, fontsize=8, color=DIM_C)


def _draw_schedule(project, ax) -> None:
    ax.axis("off")
    ax.text(0.0, 1.04, "Member schedule", transform=ax.transAxes,
            va="bottom", fontsize=9, fontweight="bold")
    rows = member_schedule(project)
    if not rows:
        ax.text(0.0, 0.5, "(no members)", fontsize=8, color="#888")
        return
    data = [[f"M{r[0]}", r[1], r[2], r[3], f"{r[4]:.3f}"] for r in rows]
    tbl = ax.table(cellText=data,
                   colLabels=["Member", "Node i", "Node j", "Section",
                              "Length (m)"],
                   cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)


# ------------------------------------------------------------------------ DXF

def build_ga_dxf(project):
    """Build a DxfDocument of the GA drawing (members, nodes, labels,
    supports, overall dimensions)."""
    from femsolver.results.dxf import DxfDocument

    doc = DxfDocument()
    for name, color in (("MEMBERS", 5), ("NODES", 7), ("TEXT", 3),
                        ("DIMS", 1), ("SUPPORTS", 7)):
        doc.add_layer(name, color)

    xy = _node_xy(project)
    secs = {s.id: s for s in project.sections}
    span = _span(project)
    r = max(span * 0.01, 0.02)
    h = max(span * 0.02, 0.1)

    for m in project.members:
        p1, p2 = xy[m.n1], xy[m.n2]
        doc.add_line((p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0), layer="MEMBERS")
        s = secs.get(m.section)
        lab = f"M{m.id} " + ((s.shape or s.name) if s else "")
        doc.add_text(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + r * 1.5, 0.0),
                     lab, height=h, layer="TEXT")
    for n in project.nodes:
        doc.add_circle((n.x, n.y, 0.0), r, layer="NODES")
        doc.add_text((n.x + r, n.y - r * 3, 0.0), f"N{n.id}", height=h,
                     layer="TEXT")
        if n.supports and any(n.supports):
            s = max(span * 0.03, 0.15)
            doc.add_polyline([(n.x, n.y), (n.x - s / 2, n.y - s),
                              (n.x + s / 2, n.y - s)], closed=True,
                             layer="SUPPORTS")
    _dxf_dims(doc, project, span, h)
    return doc


def _dxf_dims(doc, project, span, h) -> None:
    x0, y0, x1, y1 = _bbox(project)
    if x1 == x0 and y1 == y0:
        return
    off = max(span * 0.08, 0.3)
    yb = y0 - off * 2.5
    doc.add_line((x0, yb, 0.0), (x1, yb, 0.0), layer="DIMS")
    doc.add_text(((x0 + x1) / 2, yb - off * 0.6, 0.0), f"{x1 - x0:g} m",
                 height=h, layer="DIMS")
    xl = x0 - off * 2.5
    doc.add_line((xl, y0, 0.0), (xl, y1, 0.0), layer="DIMS")
    doc.add_text((xl - off * 1.6, (y0 + y1) / 2, 0.0), f"{y1 - y0:g} m",
                 height=h, layer="DIMS")


def export_dxf(project, path) -> None:
    build_ga_dxf(project).write(str(path))
