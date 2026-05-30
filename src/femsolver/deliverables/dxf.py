"""Minimal native DXF writer (no third-party dependency).

DXF is an ASCII format of *group-code / value* pairs. To emit a
useful drawing-style plan of an FE model we need:

* A HEADER section with one variable (``$ACADVER``).
* A TABLES section with at least the LAYER table.
* An ENTITIES section containing LINE / CIRCLE / TEXT / POLYLINE
  entities.
* The mandatory closing ``ENDSEC`` + ``EOF``.

This module produces clean R12-compatible DXF that AutoCAD,
BricsCAD, and LibreCAD all import without warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# Group codes (canonical):
#   0   entity type
#   2   name (block, section, layer)
#   8   layer name
#   10  primary x
#   20  primary y
#   30  primary z
#   11  secondary x (LINE endpoint)
#   21  secondary y
#   31  secondary z
#   40  scalar (radius / height)
#   1   text value
#   62  colour (ACI 1..255)
#   70  flag


@dataclass
class DxfDocument:
    """Container for layered 2-D / 2.5-D entities."""

    layers: dict = field(default_factory=lambda: {"0": 7})
    entities: list = field(default_factory=list)

    def add_layer(self, name: str, color: int = 7) -> None:
        if not name:
            raise ValueError("layer name must be non-empty")
        if not 1 <= color <= 255:
            raise ValueError(f"color must be in [1, 255], got {color}")
        self.layers[name] = color

    def add_line(
        self, p1, p2, *, layer: str = "0", color: int | None = None,
    ) -> None:
        self.entities.append(("LINE", layer, color, p1, p2))

    def add_circle(
        self, center, radius: float, *,
        layer: str = "0", color: int | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.entities.append(("CIRCLE", layer, color, center, radius))

    def add_text(
        self, anchor, text: str, height: float = 0.2, *,
        layer: str = "0", color: int | None = None,
    ) -> None:
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        self.entities.append(
            ("TEXT", layer, color, anchor, height, text),
        )

    def add_polyline(
        self, points: Sequence, *, closed: bool = False,
        layer: str = "0", color: int | None = None,
    ) -> None:
        if len(points) < 2:
            raise ValueError("polyline needs at least 2 points")
        self.entities.append(
            ("LWPOLYLINE", layer, color, list(points), closed),
        )

    # ---------------------------------------------------------------- writer

    def write(self, path: str) -> None:
        """Write the document to ``path`` as an ASCII DXF."""
        lines: list[str] = []

        def w(code: int, value):
            lines.append(f"{code:>3d}")
            lines.append(f"{value}")

        # Header
        w(0, "SECTION"); w(2, "HEADER")
        w(9, "$ACADVER"); w(1, "AC1009")     # R12
        w(0, "ENDSEC")

        # Tables (just LAYER)
        w(0, "SECTION"); w(2, "TABLES")
        w(0, "TABLE"); w(2, "LAYER"); w(70, len(self.layers))
        for name, color in self.layers.items():
            w(0, "LAYER"); w(2, name); w(70, 64)
            w(62, color); w(6, "CONTINUOUS")
        w(0, "ENDTAB")
        w(0, "ENDSEC")

        # Entities
        w(0, "SECTION"); w(2, "ENTITIES")
        for ent in self.entities:
            kind = ent[0]
            layer = ent[1]
            color = ent[2]
            if kind == "LINE":
                p1, p2 = ent[3], ent[4]
                w(0, "LINE"); w(8, layer)
                if color is not None:
                    w(62, color)
                w(10, p1[0]); w(20, p1[1])
                w(30, p1[2] if len(p1) >= 3 else 0.0)
                w(11, p2[0]); w(21, p2[1])
                w(31, p2[2] if len(p2) >= 3 else 0.0)
            elif kind == "CIRCLE":
                c, r = ent[3], ent[4]
                w(0, "CIRCLE"); w(8, layer)
                if color is not None:
                    w(62, color)
                w(10, c[0]); w(20, c[1])
                w(30, c[2] if len(c) >= 3 else 0.0)
                w(40, r)
            elif kind == "TEXT":
                anchor, h, text = ent[3], ent[4], ent[5]
                w(0, "TEXT"); w(8, layer)
                if color is not None:
                    w(62, color)
                w(10, anchor[0]); w(20, anchor[1])
                w(30, anchor[2] if len(anchor) >= 3 else 0.0)
                w(40, h)
                w(1, text)
            elif kind == "LWPOLYLINE":
                pts, closed = ent[3], ent[4]
                w(0, "LWPOLYLINE"); w(8, layer)
                if color is not None:
                    w(62, color)
                w(90, len(pts))
                w(70, 1 if closed else 0)
                for p in pts:
                    w(10, p[0]); w(20, p[1])
        w(0, "ENDSEC")
        w(0, "EOF")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def write_model_plan_dxf(
    model,
    path: str,
    *,
    label_nodes: bool = True,
    label_elements: bool = False,
    node_color: int = 1,         # red
    element_color: int = 5,      # blue
    text_height: float = 0.1,
) -> DxfDocument:
    """Export a 2-D plan view of an FE :class:`Model` as DXF.

    Each line element becomes a DXF LINE; each node becomes a small
    CIRCLE optionally labelled with its tag.
    """
    doc = DxfDocument()
    doc.add_layer("NODES", color=node_color)
    doc.add_layer("ELEMENTS", color=element_color)
    doc.add_layer("LABELS", color=2)

    # Lines first (so the order is element under node markers)
    for elem in model.elements.values():
        tags = elem.node_tags
        if len(tags) == 2:
            n1 = model.node(tags[0]).coords
            n2 = model.node(tags[1]).coords
            doc.add_line(
                (float(n1[0]), float(n1[1])),
                (float(n2[0]), float(n2[1])),
                layer="ELEMENTS",
            )
            if label_elements:
                midx = 0.5 * (n1[0] + n2[0])
                midy = 0.5 * (n1[1] + n2[1])
                doc.add_text(
                    (float(midx), float(midy)),
                    f"E{elem.tag}",
                    height=text_height, layer="LABELS",
                )
    # Nodes
    for n in model.nodes.values():
        x, y = float(n.coords[0]), float(n.coords[1])
        doc.add_circle((x, y), text_height / 2, layer="NODES")
        if label_nodes:
            doc.add_text(
                (x + text_height, y + text_height),
                f"N{n.tag}", height=text_height, layer="LABELS",
            )
    doc.write(path)
    return doc
