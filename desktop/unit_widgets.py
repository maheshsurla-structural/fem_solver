"""Unit-aware Qt widgets (plan U3) — the bridge between the SI-base model and
the display-unit UI.

``UnitSpin`` is a ``QDoubleSpinBox`` that *shows* a value in the project's
chosen display unit for some :class:`units.Quantity` while its true value stays
SI. Dialogs seed it with ``set_si`` and read it back with ``si_value`` — so the
stored model is always SI regardless of what units the user is typing in.

Kept separate from :mod:`units` (which is deliberately Qt-free and unit-tested
in isolation); this module is the thin GUI layer on top.
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox

from units import Quantity, UnitSystem


class UnitSpin(QDoubleSpinBox):
    """A spin box whose *displayed* number is in ``units``' display unit for
    ``quantity`` but whose stored value is SI base.

    Read/write SI through :meth:`si_value` / :meth:`set_si`; ``value()`` /
    ``setValue()`` still work in display units (that is what the widget shows,
    and what existing headless tests poke at)."""

    def __init__(self, quantity: Quantity, units: UnitSystem | None = None, *,
                 si: float = 0.0, decimals: int = 4, step: float | None = None,
                 rng: tuple[float, float] = (-1.0e12, 1.0e12), parent=None):
        super().__init__(parent)
        self._quantity = quantity
        self._units = units or UnitSystem()
        self.setRange(*rng)
        self.setDecimals(decimals)
        if step is not None:
            self.setSingleStep(step)
        self.set_si(si)

    def set_si(self, si_value: float) -> None:
        """Seed the widget from a stored SI value (converts to display)."""
        self.setValue(self._units.to_display(float(si_value), self._quantity))

    def si_value(self) -> float:
        """The current value converted back to SI base (what the model stores)."""
        return self._units.to_si(self.value(), self._quantity)

    def unit_label(self) -> str:
        return self._units.label(self._quantity)


def labeled(base: str, spin: UnitSpin) -> str:
    """``'base [unit]'`` form-row label for a ``UnitSpin`` (e.g. ``x [m]``)."""
    return f"{base} [{spin.unit_label()}]"
