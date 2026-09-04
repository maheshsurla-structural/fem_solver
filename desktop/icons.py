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
    "iso": '<path d="M12 3l7 4v10l-7 4-7-4V7z"/><path d="M12 12v9M5 7l7 5 7-5"/>',
    "top": '<rect x="5" y="5" width="14" height="14" rx="1"/>'
           '<circle cx="12" cy="12" r="1.6" fill="{c}" stroke="none"/>',
    "front": '<rect x="5" y="5" width="14" height="14" rx="1"/><path d="M5 15h14"/>',
    "drawings": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
                '<path d="M4 15l4-4 3 3 5-6 4 4"/>',
}
_LETTERS = {"axial": "N", "shear": "V", "moment": "M"}


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
    pm = QPixmap(24, 24)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return QIcon(pm)
