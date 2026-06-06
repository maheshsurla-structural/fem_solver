"""Guardrail: every ``femsolver`` submodule must import cleanly.

This is the second reorganization safety net (see
``docs/source/architecture.md``). The most likely way a package move goes
wrong is a **circular import** -- module A is relocated next to B, and now
``import femsolver.A`` triggers a partially-initialised ``femsolver.B``.
``pytest`` collecting the normal test suite does *not* reliably exercise
every module, so a cycle can hide until much later.

Here we walk the whole package and import each submodule individually.
A circular import (or any genuinely broken internal import / syntax error)
fails the test loudly and names the module. Failures caused purely by a
missing **third-party optional dependency** (matplotlib, pyvista, meshio,
h5py, ...) are reported as skips, so the guardrail is meaningful regardless
of which optional extras happen to be installed.
"""
from __future__ import annotations

import importlib
import pkgutil

import femsolver


def _walk_module_names() -> tuple[list[str], list[str]]:
    """Return (module_names, discovery_errors)."""
    names: list[str] = []
    discovery_errors: list[str] = []

    def _on_error(name: str) -> None:
        discovery_errors.append(name)

    for info in pkgutil.walk_packages(
        femsolver.__path__, prefix="femsolver.", onerror=_on_error
    ):
        names.append(info.name)
    return sorted(names), discovery_errors


def test_all_submodules_import() -> None:
    names, discovery_errors = _walk_module_names()

    assert names, "discovered no femsolver submodules -- walk failed"

    imported: list[str] = []
    failures: list[str] = []
    skipped_optional: list[str] = []

    pending = list(discovery_errors)

    for name in names:
        try:
            importlib.import_module(name)
            imported.append(name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            # A missing *femsolver* module is a real breakage (e.g. a botched
            # move left a dangling import). A missing third-party package is
            # just an optional dependency this environment doesn't have.
            if missing == "femsolver" or missing.startswith("femsolver."):
                failures.append(f"{name}: {exc!r}")
            else:
                skipped_optional.append(f"{name} (needs {missing!r})")
        except Exception as exc:  # circular import, syntax error, etc.
            failures.append(f"{name}: {exc!r}")

    for name in pending:
        if name not in imported and not any(
            s.startswith(name + " ") or s.startswith(name + ":") for s in
            skipped_optional + failures
        ):
            failures.append(f"{name}: failed during package discovery")

    assert not failures, (
        "femsolver submodules failed to import (circular import or broken "
        "internal import?):\n  " + "\n  ".join(failures)
    )

    # Sanity floor: the package is large; importing only a handful would mean
    # the walk silently collapsed.
    assert len(imported) >= 100, (
        f"only {len(imported)} modules imported -- expected the full package; "
        f"skipped(optional)={len(skipped_optional)}"
    )
