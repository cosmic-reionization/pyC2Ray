# --- Project information ---
project = 'pyc2ray'
copyright = '2024, Michele Bianco'
author = 'Michele Bianco'

# --- Option for HTML output ---
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_logo = 'fig/logo.png'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "nbsphinx"
]
