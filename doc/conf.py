# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
from datetime import datetime

project = "whitematteranalysis"
_author = "Fan Zhang and Lauren O'Donnell"
_email = "odonnell@bwh.harvard.edu"
_version = "0.0.0"

_init_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "whitematteranalysis", "__init__.py"))
with open(_init_path, "r") as f:
    _match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE)
    if _match:
        _version = _match.group(1)

# -- Project information -----------------------------------------------------

copyright = f"2013-{datetime.now().year}, {_author} <{_email}>"
author = f"{_author}s"

# The full version, including alpha/beta/rc tags
release = _version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Sources
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
