"""Phase 39 tests -- documentation source-tree sanity.

These tests do NOT require Sphinx; they only verify that the
documentation source tree contains the expected files and that key
cross-references (toctree entries) point at files that exist.

A separate ``test_sphinx_build`` test runs the actual ``sphinx-build``
if Sphinx is importable; otherwise it is skipped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]
_DOCS_SRC = _REPO / "docs" / "source"


# ============================================================ source tree

def test_docs_source_root_exists():
    assert _DOCS_SRC.is_dir(), f"missing {_DOCS_SRC}"
    assert (_DOCS_SRC / "conf.py").is_file()
    assert (_DOCS_SRC / "index.rst").is_file()


@pytest.mark.parametrize("page", [
    "guide/getting_started.md",
    "guide/modeling_primer.md",
    "guide/analysis_types.md",
    "guide/materials_catalog.md",
    "guide/design_workflows.md",
    "guide/pbe_pipeline.md",
    "guide/bridges.md",
    "guide/connections.md",
    "guide/vv_benchmarks.md",
])
def test_user_guide_pages_exist(page):
    p = _DOCS_SRC / page
    assert p.is_file(), f"missing user-guide page: {page}"
    body = p.read_text(encoding="utf-8")
    assert len(body.strip()) > 100, f"page too short: {page}"


@pytest.mark.parametrize("page", [
    "api/core.rst",
    "api/materials.rst",
    "api/sections.rst",
    "api/elements.rst",
    "api/analysis.rst",
    "api/design.rst",
    "api/bridges.rst",
    "api/benchmarks.rst",
    "api/io.rst",
])
def test_api_reference_pages_exist(page):
    p = _DOCS_SRC / page
    assert p.is_file(), f"missing API page: {page}"
    body = p.read_text(encoding="utf-8")
    assert "automodule::" in body, (
        f"API page {page} should contain at least one automodule directive"
    )


def test_tutorial_index_exists():
    p = _DOCS_SRC / "tutorials" / "index.rst"
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    # Should reference several capstone scripts
    for cap in [
        "43_pbe_full_workflow.py",
        "47_indian_codes_frame_design.py",
        "49_psc_girder_bridge.py",
        "53_full_design_report.py",
    ]:
        assert cap in body, f"tutorial index missing {cap}"


def test_theory_overview_exists():
    assert (_DOCS_SRC / "theory" / "overview.md").is_file()
    assert (_DOCS_SRC / "theory" / "vv_report.md").is_file()
    assert (_REPO / "docs" / "theory" / "theory.tex").is_file()


# ============================================================ toctree integrity

def _parse_toctree_entries(rst_text: str) -> list[str]:
    """Return all explicit toctree entries (one per line) from an RST file."""
    out: list[str] = []
    in_toctree = False
    for line in rst_text.splitlines():
        if line.strip().startswith(".. toctree::"):
            in_toctree = True
            continue
        if in_toctree:
            stripped = line.strip()
            if not stripped:
                # Blank line: still inside toctree if options follow
                continue
            if line.startswith("   :"):
                continue            # option, e.g. :maxdepth: 2
            if not line.startswith("   "):
                in_toctree = False  # exited
                continue
            out.append(stripped)
    return out


def test_index_toctree_entries_resolve():
    """Every toctree entry in index.rst must point to an existing file."""
    index = (_DOCS_SRC / "index.rst").read_text(encoding="utf-8")
    entries = _parse_toctree_entries(index)
    assert entries, "no toctree entries parsed"
    for entry in entries:
        # Look for .md or .rst
        md_path = _DOCS_SRC / f"{entry}.md"
        rst_path = _DOCS_SRC / f"{entry}.rst"
        assert md_path.is_file() or rst_path.is_file(), (
            f"toctree entry '{entry}' resolves to neither "
            f"{md_path} nor {rst_path}"
        )


# ============================================================ README

def test_readme_has_design_codes_section():
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    for code in ["ACI 318", "AISC 360", "IS 456", "IS 800",
                  "IS 1893", "AASHTO"]:
        assert code in readme, f"README missing reference to {code}"
    assert "examples/43_pbe_full_workflow.py" in readme
    assert "Build the docs" in readme


# ============================================================ Sphinx build

@pytest.mark.skipif(
    shutil.which("sphinx-build") is None,
    reason="sphinx-build not installed",
)
def test_sphinx_build_runs(tmp_path):
    """Run sphinx-build into a temporary output dir.

    Skipped if Sphinx is not installed in the current environment.
    Failure here means a syntax error in the docs source.
    """
    out_dir = tmp_path / "html"
    cmd = [
        "sphinx-build", "-b", "html",
        "-W", "--keep-going",            # treat warnings as errors but finish
        str(_DOCS_SRC), str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (
            f"sphinx-build failed with code {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
        pytest.fail(msg)
    assert (out_dir / "index.html").is_file()
