# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Antupy'
copyright = '2026, David Saldivia'
author = 'David Saldivia'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../../'))

# Mock heavy dependencies for ReadTheDocs
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['CoolProp', 'CoolProp.CoolProp', 'pvlib', 'scipy', 'matplotlib', 'xarray']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

release = "0.6.0"  # Update this when you bump version
version = ".".join(release.split(".")[:2])

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []


# --  Napoleon options--------------------------------------------------------
# use the :param directive
napoleon_use_param = True
napoleon_attr_attributes = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Autodoc options ---------------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "both"

# Only insert class docstring
autoclass_content = "class"

# Generate autosummary pages
autosummary_generate = True

# Include __init__ method if it has a docstring
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# PyData Sphinx Theme options
html_theme_options = {
    "github_url": "https://github.com/DavidSaldivia/antupy",
    "show_toc_level": 2,
    "navbar_align": "left",
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_prev_next": True,
}

# ReadTheDocs-specific
html_context = {
    "display_github": True,
    "github_user": "DavidSaldivia",
    "github_repo": "antupy",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
