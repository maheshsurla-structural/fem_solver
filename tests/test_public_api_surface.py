"""Guardrail: the public ``import femsolver`` surface must not change silently.

This is a *reorganization* safety net (see ``docs/source/architecture.md``).
The planned package reshuffle moves modules around internally; the cast-iron
rule is that the top-level public API -- the classes and functions you reach
as ``femsolver.<name>`` -- stays identical. A pure relocation (with
re-export shims) leaves this set untouched, so this test passing is a
*mechanical proof* that no public name was dropped, renamed, or accidentally
added during a move.

What we snapshot
---------------
We track the **non-module** public names (classes, functions, constants).
We deliberately *exclude* sub-package attributes (``femsolver.analysis``,
``femsolver.sections``, ...): those get bound onto the package as a side
effect of whichever submodules happen to have been imported, so the raw
``dir(femsolver)`` is sensitive to test execution order. The classes and
functions, by contrast, are all bound by ``femsolver/__init__.py`` itself
and are therefore deterministic.

The expected surface lives in ``tests/data/public_api.txt`` (one name per
line, sorted). If you change the public API **on purpose**, regenerate it::

    python -c "import femsolver,pathlib,types; \
      p=pathlib.Path('tests/data/public_api.txt'); \
      p.write_text(chr(10).join(sorted(n for n in dir(femsolver) \
      if not n.startswith('_') and not isinstance(getattr(femsolver,n), \
      types.ModuleType)))+chr(10), encoding='utf-8')"

and commit the diff in the *same* change, so the intent is reviewable.
"""
from __future__ import annotations

import pathlib
import types

import femsolver

_SNAPSHOT = pathlib.Path(__file__).parent / "data" / "public_api.txt"


def _current_surface() -> list[str]:
    return sorted(
        n for n in dir(femsolver)
        if not n.startswith("_")
        and not isinstance(getattr(femsolver, n), types.ModuleType)
    )


def _expected_surface() -> list[str]:
    text = _SNAPSHOT.read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def test_public_api_surface_unchanged() -> None:
    expected = set(_expected_surface())
    current = set(_current_surface())

    removed = sorted(expected - current)
    added = sorted(current - expected)

    msg = []
    if removed:
        msg.append(
            "REMOVED from public API (breaks `from femsolver import ...`):\n  "
            + "\n  ".join(removed)
            + "\n  -> add a re-export shim, or intentionally update "
            "tests/data/public_api.txt"
        )
    if added:
        msg.append(
            "ADDED to public API (was a new export intended?):\n  "
            + "\n  ".join(added)
            + "\n  -> if intentional, regenerate tests/data/public_api.txt"
        )
    assert not removed and not added, "\n".join(msg)


def test_all_advertised_names_are_reachable() -> None:
    """Every name in ``femsolver.__all__`` must actually resolve.

    Catches a move that renames/drops an export while leaving it listed in
    ``__all__`` (which would break ``from femsolver import *``).
    """
    missing = sorted(n for n in femsolver.__all__ if not hasattr(femsolver, n))
    assert not missing, (
        "names in femsolver.__all__ that no longer resolve:\n  "
        + "\n  ".join(missing)
    )


def test_snapshot_is_sorted_and_unique() -> None:
    """The snapshot file itself stays canonical (sorted, no dupes)."""
    expected = _expected_surface()
    assert expected == sorted(expected), "tests/data/public_api.txt is not sorted"
    assert len(expected) == len(set(expected)), "duplicate names in snapshot"
