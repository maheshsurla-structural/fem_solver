"""CataloguedGeometry -- wrap a parametric Geometry but override gross
properties with catalogue-exact values."""
from __future__ import annotations

from typing import Optional

from femsolver.sections.geometry.base import Geometry


class CataloguedGeometry(Geometry):
    """Wraps a base :class:`Geometry` (typically built from a parametric
    primitive sized to the catalogue dimensions) and overrides ``area``,
    ``I_zz``, ``I_yy``, ``J``, and optionally ``Z_zz`` / ``Z_yy`` with
    the catalogue-exact values.

    Why this layer exists
    ---------------------
    A parametric I-section built from raw ``h``, ``b``, ``t_f``, ``t_w``
    matches an AISC W-shape to within ~1% on area and ~5% on J (the
    parametric outline has no fillets; the actual section does). When
    the user looks up ``W14x90``, they want the AISC-tabulated value,
    not the parametric approximation. ``CataloguedGeometry`` provides
    the override without throwing away the polygon outline (which is
    still useful for visualization).

    Parameters
    ----------
    base : Geometry
        The underlying outline (e.g. an :class:`ISectionGeometry`).
        Provides ``polygon``, ``centroid``, ``c_top/c_bot``, etc.
    area, I_zz, I_yy, J : float
        Catalogue-exact gross properties (m^2 / m^4).
    Z_zz, Z_yy : float, optional
        Catalogue plastic moduli (m^3). If omitted, the base
        geometry's value is used.
    S_zz_top, S_zz_bot, S_yy_left, S_yy_right : float, optional
        Catalogue elastic section moduli (m^3). If omitted, computed
        from ``I / c``.
    """

    def __init__(
        self,
        base: Geometry,
        *,
        area: float,
        I_zz: float,
        I_yy: float,
        J: float,
        Z_zz: Optional[float] = None,
        Z_yy: Optional[float] = None,
        S_zz_top: Optional[float] = None,
        S_zz_bot: Optional[float] = None,
        S_yy_left: Optional[float] = None,
        S_yy_right: Optional[float] = None,
    ):
        if min(area, I_zz, I_yy, J) <= 0:
            raise ValueError(
                "all catalogue gross properties must be positive"
            )
        self._base = base
        self._area = float(area)
        self._I_zz = float(I_zz)
        self._I_yy = float(I_yy)
        self._J = float(J)
        self._Z_zz = Z_zz
        self._Z_yy = Z_yy
        self._S_zz_top_override = S_zz_top
        self._S_zz_bot_override = S_zz_bot
        self._S_yy_left_override = S_yy_left
        self._S_yy_right_override = S_yy_right

    # -------------------------------------------------- pass-through
    @property
    def polygon(self):
        return self._base.polygon

    @property
    def centroid(self) -> tuple[float, float]:
        return self._base.centroid

    # -------------------------------------------------- overrides
    @property
    def area(self) -> float:
        return self._area

    @property
    def I_zz(self) -> float:
        return self._I_zz

    @property
    def I_yy(self) -> float:
        return self._I_yy

    @property
    def I_yz(self) -> float:
        # Doubly-symmetric catalogue sections; channels and angles
        # could override but we currently assume zero for I_yz.
        return 0.0

    @property
    def J(self) -> float:
        return self._J

    @property
    def Z_zz(self) -> float:
        if self._Z_zz is not None:
            return self._Z_zz
        return self._base.Z_zz

    @property
    def Z_yy(self) -> float:
        if self._Z_yy is not None:
            return self._Z_yy
        return self._base.Z_yy

    @property
    def S_zz_top(self) -> float:
        if self._S_zz_top_override is not None:
            return self._S_zz_top_override
        return super().S_zz_top

    @property
    def S_zz_bot(self) -> float:
        if self._S_zz_bot_override is not None:
            return self._S_zz_bot_override
        return super().S_zz_bot
