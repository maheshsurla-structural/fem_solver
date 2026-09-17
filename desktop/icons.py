"""Bundled toolbar/menu icon set — compact 24px line icons authored in-house
(no external dependency or licensing), rendered from inline SVG to themable
QIcons. ``icon(name)`` returns a QIcon; unknown names yield an empty icon.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_WRAP = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
         'fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round">{inner}</svg>')

_ICONS = {
    "new": '<rect x="5" y="3" width="14" height="18" rx="2"/>'
           '<path d="M9 9h6M9 13h6M9 17h4"/>',
    "new3d": '<path d="M12 3l7 4v10l-7 4-7-4V7z"/><path d="M12 12v9M5 7l7 5 7-5"/>',
    "open": '<path d="M3 8a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v7a2 2 0 0 1-2 '
            '2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v5h7M8 21v-6h8v6"/>',
    "undo": '<path d="M9 7L4 12l5 5"/><path d="M4 12h11a5 5 0 0 1 0 10"/>',
    "redo": '<path d="M15 7l5 5-5 5"/><path d="M20 12H9a5 5 0 0 0 0 10"/>',
    "node": '<circle cx="12" cy="12" r="4" fill="{c}" stroke="none"/>',
    "member": '<line x1="5" y1="19" x2="19" y2="5"/>'
              '<circle cx="5" cy="19" r="2.3" fill="{c}" stroke="none"/>'
              '<circle cx="19" cy="5" r="2.3" fill="{c}" stroke="none"/>',
    "section": '<path d="M7 5h10M7 19h10M12 5v14"/>',
    # slab / shell area — a plate panel drawn in perspective with a mesh line
    "slab": '<path d="M2 9l8-4 12 4-8 4z"/>'
            '<path d="M2 9v4l8 4 12-4V9"/><path d="M6 7l12 4"/>',
    # contour — nested iso-bands (a filled-contour / result map glyph)
    "contour": '<path d="M4 18h16"/><path d="M6 18a6 6 0 0 1 12 0"/>'
               '<path d="M9 18a3 3 0 0 1 6 0"/>',
    # shell forces / moments — a plate with a curved bending arrow
    "shellforce": '<path d="M3 8l7-3 11 3-7 3z"/>'
                  '<path d="M5 14q7 4 14 0" stroke-dasharray="2 2"/>'
                  '<path d="M17 12l2 2-2 2"/>',
    "function": '<path d="M3 12q3-7 6 0t6 0t6 0"/>',   # time-history waveform
    "load": '<path d="M12 4v13M6 11l6 6 6-6"/>',
    "delete": '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
    "move": '<path d="M12 3v18M3 12h18M12 3l-2.5 2.5M12 3l2.5 2.5'
            'M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5'
            'M21 12l-2.5-2.5M21 12l-2.5 2.5"/>',
    "copy": '<rect x="4" y="4" width="12" height="12" rx="2"/>'
            '<rect x="9" y="9" width="12" height="12" rx="2"/>',
    "mirror": '<path d="M12 3v18" stroke-dasharray="3 2.5"/>'
              '<path d="M9 8l-4 4 4 4z" fill="{c}" stroke="none"/>'
              '<path d="M15 8l4 4-4 4z" fill="{c}" stroke="none"/>',
    "rotate": '<path d="M20 12a8 8 0 1 1-2.4-5.7"/><path d="M20 4v4h-4"/>',
    "extrude": '<rect x="3" y="13" width="8" height="8"/>'
               '<rect x="13" y="3" width="8" height="8"/>'
               '<path d="M9 13l4-4" stroke-dasharray="2 2"/>',
    "single": '<path d="M6 4l6 16 2-6 6-2z" fill="{c}" stroke="none"/>',
    "window": '<rect x="4" y="6" width="16" height="12" rx="1" '
              'stroke-dasharray="3 2.5"/>',
    "deselect": '<rect x="4" y="6" width="16" height="12" rx="1" '
                'stroke-dasharray="3 2.5"/><path d="M5 19L19 5"/>',
    "polygon": '<path d="M4 9l5-5 8 2 3 8-6 5-9-3z" stroke-dasharray="3 2.5"/>',
    "drawnode": '<circle cx="7" cy="17" r="2.5" fill="{c}" stroke="none"/>'
                '<path d="M17 4v6M14 7h6"/>',
    "drawmember": '<path d="M4 20L14 10"/>'
                  '<circle cx="4" cy="20" r="2" fill="{c}" stroke="none"/>'
                  '<circle cx="14" cy="10" r="2" fill="{c}" stroke="none"/>'
                  '<path d="M18 5v5M15.5 7.5h5"/>',
    "snap": '<path d="M4 9h16M4 15h16M9 4v16M15 4v16"/>',
    "frame": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
             '<path d="M4 12h16M12 4v16"/>',
    "loadsgen": '<path d="M4 18h16M8 6v7M8 13l-2-2M8 13l2-2M14 6v7'
                'M14 13l-2-2M14 13l2-2"/>',
    "run": '<path d="M7 4l13 8-13 8z" fill="{c}" stroke="none"/>',
    "undeformed": '<path d="M6 20V6h12v14M6 13h12"/>',
    "design": '<path d="M12 3l7 3v6c0 5-3 7.5-7 9-4-1.5-7-4-7-9V6z"/>'
              '<path d="M9 12l2 2 4-4"/>',
    "fit": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    # --- viewport navigation tools ---
    "pan": '<path d="M9 11V6.5a1.5 1.5 0 0 1 3 0V11m0-.5V5.5a1.5 1.5 0 0 1 3 0'
           'V11m0-.5a1.5 1.5 0 0 1 3 0V15a5 5 0 0 1-5 5h-1.5a5 5 0 0 1-3.6-1.6'
           'L4 14.5a1.5 1.5 0 0 1 2.3-1.9L8 14V8a1.5 1.5 0 0 1 1-1.4"/>',
    "orbit": '<ellipse cx="12" cy="12" rx="9" ry="3.6"/>'
             '<ellipse cx="12" cy="12" rx="3.6" ry="9"/>'
             '<circle cx="12" cy="12" r="1.6" fill="{c}" stroke="none"/>',
    "zoomwin": '<rect x="3" y="4" width="13" height="10" rx="1" '
               'stroke-dasharray="3 2.4"/><circle cx="14.5" cy="15.5" r="4.2"/>'
               '<path d="M17.6 18.6l2.9 2.9"/>',
    "zoomin": '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>'
              '<path d="M8 10.5h5M10.5 8v5"/>',
    "zoomout": '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>'
               '<path d="M8 10.5h5"/>',
    "fitsel": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'
              '<circle cx="12" cy="12" r="2.4" fill="{c}" stroke="none"/>',
    "customize": '<path d="M4 7h9M4 12h5M4 17h11"/>'
                 '<circle cx="16" cy="7" r="2.3"/>'
                 '<circle cx="12" cy="12" r="2.3"/>'
                 '<circle cx="18" cy="17" r="2.3"/>',
    "lock2d": '<rect x="5" y="11" width="14" height="9" rx="2"/>'
              '<path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    "unlock2d": '<rect x="5" y="11" width="14" height="9" rx="2"/>'
                '<path d="M8 11V8a4 4 0 0 1 7.5-2"/>',
    "iso": '<path d="M12 3l7 4v10l-7 4-7-4V7z"/><path d="M12 12v9M5 7l7 5 7-5"/>',
    "top": '<rect x="5" y="5" width="14" height="14" rx="1"/>'
           '<circle cx="12" cy="12" r="1.6" fill="{c}" stroke="none"/>',
    "front": '<rect x="5" y="5" width="14" height="14" rx="1"/><path d="M5 15h14"/>',
    "drawings": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
                '<path d="M4 15l4-4 3 3 5-6 4 4"/>',
    "sectiondesigner": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
                       '<circle cx="9" cy="9" r="1.4" fill="{c}" stroke="none"/>'
                       '<circle cx="15" cy="9" r="1.4" fill="{c}" stroke="none"/>'
                       '<circle cx="9" cy="15" r="1.4" fill="{c}" stroke="none"/>'
                       '<circle cx="15" cy="15" r="1.4" fill="{c}" stroke="none"/>',
    # --- Section Designer canvas tools ---
    "sd_select": '<path d="M6 4l11 7-4.6 1.4 2.7 5.4-2 1-2.7-5.4L7 17z" '
                 'fill="{c}" stroke="none"/>',
    "sd_point": '<circle cx="9.5" cy="14" r="2.8" fill="{c}" stroke="none"/>'
                '<path d="M17.5 5v5M15 7.5h5"/>',
    "sd_rebar": '<circle cx="9.5" cy="14" r="4" fill="{c}" stroke="none"/>'
                '<circle cx="9.5" cy="14" r="1.5" fill="#fff" stroke="none"/>'
                '<path d="M17.5 5v5M15 7.5h5"/>',
    "sd_void": '<path d="M3.5 6.5h11v11h-11z"/>'
               '<path d="M6.5 9.5h5v5h-5z" stroke-dasharray="2 1.6"/>'
               '<path d="M18 5v5M15.5 7.5h5"/>',
    "sd_dims": '<path d="M4 7v10M20 7v10M4 12h16"/>'
               '<path d="M4 12l3-2.4M4 12l3 2.4M20 12l-3-2.4M20 12l-3 2.4" '
               'fill="none"/>',
    "sd_fibres": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
                 '<path d="M4 9h16M4 14h16M9 4v16M14 4v16"/>',
    "sd_centroids": '<rect x="4" y="4" width="16" height="16" rx="1" '
                    'opacity=".55"/><g fill="{c}" stroke="none">'
                    '<circle cx="8.5" cy="8.5" r="1.3"/>'
                    '<circle cx="15.5" cy="8.5" r="1.3"/>'
                    '<circle cx="12" cy="12" r="1.3"/>'
                    '<circle cx="8.5" cy="15.5" r="1.3"/>'
                    '<circle cx="15.5" cy="15.5" r="1.3"/></g>',
    "sd_editfree": '<path d="M4 13l3.5-7.5 8 1 3.5 6.5-5.5 5.5z"/>'
                   '<circle cx="15.5" cy="6.5" r="2.3" fill="{c}" stroke="none"/>',
    "sd_props": '<rect x="4" y="4" width="16" height="16" rx="1.5"/>'
                '<path d="M8 9h8M8 13h8M8 17h5"/>',
    "theme_dark": '<path d="M20 13.5A8 8 0 1 1 10.5 4a6.5 6.5 0 0 0 9.5 9.5z"/>',
    "theme_light": '<circle cx="12" cy="12" r="4"/>'
                   '<path d="M12 2v2M12 20v2M4 12H2M22 12h-2'
                   'M5.6 5.6L4.2 4.2M19.8 19.8l-1.4-1.4'
                   'M18.4 5.6l1.4-1.4M4.2 19.8l1.4-1.4"/>',
    # --- ribbon R6: glyphs for buttons that were text-only ---
    "materials": '<path d="M12 3l9 4.5-9 4.5-9-4.5z"/>'
                 '<path d="M3 12l9 4.5 9-4.5"/>'
                 '<path d="M3 16.5l9 4.5 9-4.5"/>',
    "selnodes": '<rect x="4" y="5" width="16" height="14" rx="1" '
                'stroke-dasharray="3 2.5"/>'
                '<circle cx="9" cy="10" r="1.7" fill="{c}" stroke="none"/>'
                '<circle cx="15" cy="10" r="1.7" fill="{c}" stroke="none"/>'
                '<circle cx="9" cy="15" r="1.7" fill="{c}" stroke="none"/>'
                '<circle cx="15" cy="15" r="1.7" fill="{c}" stroke="none"/>',
    "selmembers": '<rect x="4" y="5" width="16" height="14" rx="1" '
                  'stroke-dasharray="3 2.5"/><line x1="8" y1="16" x2="16" y2="8"/>'
                  '<circle cx="8" cy="16" r="1.6" fill="{c}" stroke="none"/>'
                  '<circle cx="16" cy="8" r="1.6" fill="{c}" stroke="none"/>',
    "selall": '<rect x="3.5" y="5" width="17" height="14" rx="1" '
              'stroke-dasharray="3 2.5"/><line x1="7" y1="16" x2="17" y2="8"/>'
              '<circle cx="7" cy="16" r="1.5" fill="{c}" stroke="none"/>'
              '<circle cx="12" cy="12" r="1.5" fill="{c}" stroke="none"/>'
              '<circle cx="17" cy="8" r="1.5" fill="{c}" stroke="none"/>',
    "selsection": '<rect x="4" y="5" width="16" height="14" rx="1" '
                  'stroke-dasharray="3 2.5"/><path d="M9 9h6M9 15h6M12 9v6"/>',
    "hinge": '<path d="M3 12h6M15 12h6"/><circle cx="12" cy="12" r="3"/>',
    "assignhinge": '<path d="M3 15h6M14 15h5"/>'
                   '<circle cx="11.5" cy="15" r="2.6"/>'
                   '<path d="M18 5h4M20 3v4"/>',
    "history": '<path d="M3.5 12a8.5 8.5 0 1 0 2.7-6.2"/><path d="M3 3v4h4"/>'
               '<path d="M12 8v4l3 2"/>',
    "checkmodel": '<rect x="5" y="4" width="14" height="17" rx="2"/>'
                  '<path d="M9 4h6v2H9z" fill="{c}" stroke="none"/>'
                  '<path d="M8.5 13l2.5 2.5 4.5-5"/>',
    "density": '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6v12"/>'
               '<path d="M4 6l-1.6 2.2M4 6l1.6 2.2M4 18l-1.6-2.2M4 18l1.6-2.2"/>',
}
_LETTERS = {"axial": "N", "shear": "V", "moment": "M"}


def monogram_svg(text: str = "ASA", fg: str = "#ffffff",
                 bg: str = "#c0392b") -> str:
    """A rounded-square monogram badge as raw SVG (the product mark)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<rect x="3" y="3" width="94" height="94" rx="22" fill="{bg}"/>'
        f'<text x="50" y="54" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="34" font-weight="700" fill="{fg}" text-anchor="middle" '
        f'dominant-baseline="middle" letter-spacing="1">{text}</text></svg>')


@lru_cache(maxsize=None)
def monogram_icon(text: str = "ASA", fg: str = "#ffffff",
                  bg: str = "#c0392b") -> QIcon:
    """The product monogram rendered to a crisp QIcon (window / app icon)."""
    renderer = QSvgRenderer(QByteArray(monogram_svg(text, fg, bg).encode()))
    scale = 8
    pm = QPixmap(64 * scale, 64 * scale)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


@lru_cache(maxsize=None)
def letter_icon(text: str, color: str = "#3a3a3a") -> QIcon:
    """A crisp badge rendering a short label (1-2 chars, e.g. an ASCE load key
    'D' / 'L' / 'Lr') as a themable QIcon — for list/table rows."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           f'<text x="12" y="18" font-size="13" font-family="sans-serif" '
           f'font-weight="700" text-anchor="middle" fill="{color}">'
           f'{text}</text></svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    scale = 4
    pm = QPixmap(24 * scale, 24 * scale)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def svg_markup(name: str, color: str = "#3a3a3a") -> str:
    """The raw ``<svg>…</svg>`` string for an icon, for embedding in HTML
    (e.g. the calc report brand mark). Empty string for an unknown name."""
    inner = _ICONS.get(name)
    if inner is None:
        return ""
    return _WRAP.format(c=color, inner=inner.replace("{c}", color))


@lru_cache(maxsize=None)
def icon(name: str, color: str = "#3a3a3a") -> QIcon:
    if name in _LETTERS:
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               f'<text x="12" y="18" font-size="17" font-family="sans-serif" '
               f'font-weight="700" text-anchor="middle" fill="{color}">'
               f'{_LETTERS[name]}</text></svg>')
    else:
        inner = _ICONS.get(name)
        if inner is None:
            return QIcon()
        svg = _WRAP.format(c=color, inner=inner.replace("{c}", color))
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    scale = 4                                    # render hi-res for crisp icons
    pm = QPixmap(24 * scale, 24 * scale)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)
