"""Sphinx configuration for femsolver documentation.

To build::

    pip install sphinx myst-parser sphinx-rtd-theme sphinx-autodoc-typehints
    sphinx-build -b html docs/source docs/build/html

The HTML site will then live in ``docs/build/html/index.html``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Allow autodoc to import the package from the source tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# -- Project information -----------------------------------------------------
project = "femsolver"
author = "femsolver contributors"
copyright = "2026, femsolver contributors"

# Pull the version from the package itself.
try:
    import femsolver
    version = femsolver.__version__
    release = femsolver.__version__
except Exception:
    version = "0.1.0"
    release = "0.1.0"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy / Google docstring parsing
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_parser",                   # Markdown support
]

# Source file types accepted
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

# Files / directories to ignore
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
]

# -- autodoc / autosummary ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autosummary_generate = True

# Suppress autodoc warnings about overloaded methods or `__init__` not found,
# plus duplicate object descriptions caused by autosummary stubs that re-list
# attributes already documented under their parent dataclass in api/*.rst.
suppress_warnings = [
    "autodoc.import_object",
    "ref.python",   # duplicate object descriptions are harmless here
    # Many of our math-heavy docstrings use ``|x|`` for absolute-value
    # which docutils tries to parse as a substitution reference. They
    # render correctly as plain text -- silence the cosmetic noise.
    "docutils",
]


# -- Napoleon (NumPy docstrings) ---------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# Use :ivar: (instance variable) for Attributes sections instead of
# generating duplicate :attribute: directives that conflict with
# autodoc's automatic attribute discovery.
napoleon_use_ivar = True


# -- HTML output -------------------------------------------------------------
# Fall back gracefully if the chosen theme is not installed.
try:
    import sphinx_rtd_theme    # noqa: F401
    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_title = "femsolver"
html_short_title = "femsolver"
html_show_sphinx = False


# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

# -- MyST ---------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "deflist",
    "dollarmath",
    "html_image",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Custom -------------------------------------------------------------------
todo_include_todos = True
