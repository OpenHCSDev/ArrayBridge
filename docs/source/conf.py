# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'arraybridge'
copyright = '2025, Tristan Simas'
author = 'Tristan Simas'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Mock imports for optional dependencies ----------------------------------

# Mock imports for frameworks that might not be installed
# This ensures documentation builds without requiring all optional dependencies
autodoc_mock_imports = [
    'cupy',
    'torch',
    'tensorflow',
    'jax',
    'jaxlib',
    'pyclesperanto_prototype',
    'pyclesperanto',
    'cupyx',  # CuPy extensions
]

# -- Additional configuration -----------------------------------------------

# The master toctree document
master_doc = 'index'

# Custom sidebar templates
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# Theme options for sphinx_rtd_theme
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': False,
}

# Output file base name for HTML help builder
htmlhelp_basename = 'arraybridgedoc'

# If true, links to the reST sources are added to the pages
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer
html_show_sphinx = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::)
add_module_names = False
