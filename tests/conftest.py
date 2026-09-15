"""Shared pytest fixtures / test-session setup.

Desktop (PySide6) tests construct ``MainWindow`` and other widgets that read and
write persisted UI state through ``QSettings("MidasStructural", <app>)`` — the
*real* per-user store (the native Windows registry / config). Left unisolated
this bites in two ways:

* a test flips a persisted flag and it survives the run, poisoning *future* runs
  ("prior sessions") — the store is shared across every session on the machine;
* the concrete offender: the model-navigation summary flag ``nav/summary``. With
  it ``True`` the model tree shows counts only and drops the per-item leaves, so
  ``MainWindow._find_item(("member_load", 0))`` returns ``None`` and e.g.
  ``test_desktop_member_load.py::test_main_window_wires_line_loads`` fails
  ``assert found is not None`` — but only when an earlier session happened to
  persist ``nav/summary=True``, hence the order-dependent flakiness.

Fix: redirect the app's ``QSettings(org, app)`` stores to a throwaway directory
that lives only for this test session. It starts empty (so no stale flag from a
prior session reaches the app) and is discarded at exit (so nothing this run does
reaches the next). A *single* session-wide directory — rather than one per test —
is deliberate: the suite already relied on the store being shared within a run
(that is what the real registry was), including module-scoped widgets that
persist a value one test then reads back; per-test isolation would break that
without touching the actual bug, which is purely cross-session.

On Windows the two-argument ``QSettings(org, app)`` constructor is hard-wired to
the registry (``setDefaultFormat``/``setPath`` do not redirect it), so we swap
the class the code imports for a thin shim: an ``(org, app)`` construction is
rerouted to ``<session-dir>/<org>.<app>.ini``; every other form (an explicit
``QSettings(fileName, format)`` — as the nav tests' own isolation fixture uses —
or the no-arg form) passes straight through untouched.

Pure test hygiene: the application code is unchanged, and when PySide6 is absent
(the non-GUI test runs) the shim is never installed.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

try:
    from PySide6 import QtCore as _qtcore

    _REAL_QSETTINGS = _qtcore.QSettings
except Exception:  # pragma: no cover - PySide6 not installed (non-GUI run)
    _qtcore = None
    _REAL_QSETTINGS = None


if _REAL_QSETTINGS is not None:
    # Throwaway store for the whole session; empty at start, gone at exit.
    _SESSION_DIR = tempfile.mkdtemp(prefix="femsolver-qsettings-")
    atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)

    class _ShimMeta(type):
        # Proxy class-level access (Format, Scope, any classmethod) to the real
        # QSettings, so e.g. ``QSettings.Format.IniFormat`` keeps working.
        def __getattr__(cls, name):
            return getattr(_REAL_QSETTINGS, name)

    class _IsolatedQSettings(metaclass=_ShimMeta):
        """Drop-in for :class:`PySide6.QtCore.QSettings` used only under test.

        Constructing one always returns a *real* ``QSettings`` instance; only the
        backing location of the ``(org, app)`` form is rerouted into the session
        directory.
        """

        def __new__(cls, *args, **kwargs):
            if (
                len(args) >= 2
                and isinstance(args[0], str)
                and isinstance(args[1], str)
            ):
                # QSettings(organization, application[, parent]) → session ini.
                path = os.path.join(_SESSION_DIR, f"{args[0]}.{args[1]}.ini")
                return _REAL_QSETTINGS(path, _REAL_QSETTINGS.Format.IniFormat)
            # Any other constructor form (explicit file/format, no args, …).
            return _REAL_QSETTINGS(*args, **kwargs)

    # Install the shim. Desktop app/test modules import QSettings lazily (inside
    # functions/fixtures), so doing this at conftest import time — before any of
    # those run — means every ``from PySide6.QtCore import QSettings`` picks it up.
    _qtcore.QSettings = _IsolatedQSettings
