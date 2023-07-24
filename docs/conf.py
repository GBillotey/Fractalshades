# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import fractalshades
fs_dir = os.path.abspath(os.path.dirname(fractalshades.__file__))
sys.path.insert(0, fs_dir)


# -- Project information -----------------------------------------------------

project = 'Fractalshades'
copyright = '2023, Fractalshades development team'
author = 'G. Billotey'

# The short X.Y version
version = fractalshades.__version__
# The full version, including alpha/beta/rc tags
release = fractalshades.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.inheritance_diagram',
    # 'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.napoleon',  # for NumPy style docstrings
    'sphinx.ext.githubpages',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
    #'sphinxext.github',
    'sphinxcontrib.youtube'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx' #None

# used by sphinxcontrib.bibtex
bibtex_bibfiles = ['refs.bib']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' # 'sphinx_rtd_theme' # 'alabaster' # 'groundwork'
# extensions = ["sphinx_rtd_dark_mode"]

# https://github.com/python-pillow/Pillow/pull/4968/files
def setup(app):
    app.add_css_file("rtd_dark.css")
    app.add_css_file("math.css")

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
# https://stackoverflow.com/questions/32079200/how-do-i-set-up-custom-styles-for-restructuredtext-sphinx-readthedocs-etc
# https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs?noredirect=1&lq=1


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Fractalshadesdoc'
mathjax3_config = {'chtml': {'displayAlign': 'left',
                             'displayIndent': '2em'}}


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Fractalshades.tex', 'Fractalshades Documentation',
     'Geoffroy Billotey', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'fractalshades', 'Fractalshades Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Fractalshades', 'Fractalshades Documentation',
     author, 'Fractalshades', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# The base URL which points to the root of the HTML documentation.
# It is used to indicate the location of document using The Canonical Link
# Relation. Default: ''.
# Used by also by sphinx.ext.githubpages CNAME
html_baseurl = ""
html_logo = "_static/logo3.jpg"
html_theme_options = {
    "collapse_navigation": False,
    #"navbar_center": ["mpl_nav_bar.html"],
}



default_role = 'py:obj'

# Not sure how this works... to be investigated
# extlinks = {'numpy': ('https://numpy.org/', '%s')}

# numpydoc config
numpydoc_show_class_members = False

# Automatic build of the gallery
import sphinx_gallery
from sphinx_gallery.sorting import FileNameSortKey
# We will not write to disk but feed fractalshades_scraper
fractalshades.settings.output_context["doc"] = True 
fractalshades.settings.output_context["doc_max_width"] = 800 
source_dir = os.path.dirname(os.path.realpath(__file__))
fractalshades.settings.output_context["doc_data_dir"] = (
    os.path.join(source_dir, html_static_path[0])
)

# https://sphinx-gallery.github.io/stable/advanced.html?highlight=scrapers#example-2-detecting-image-files-on-disk
def fractalshades_scraper(block, block_vars, gallery_conf):
    # We use a list to collect references to image names
    image_names = list()
    # The `image_path_iterator` is created by Sphinx-Gallery, it will yield
    # a path to a file name that adheres to Sphinx-Gallery naming convention.
    image_path_iterator = block_vars['image_path_iterator']

    # Define a list of our already-created figure objects.
    list_of_my_figures = fractalshades.get_figures()

    # Iterate through figure objects, save to disk, and keep track of paths.
    for fig, image_path in zip(list_of_my_figures, image_path_iterator):
        fig.save_png(image_path)
        image_names.append(image_path)

    # Close all references to figures so they aren't used later.
    fractalshades.close('all')

    # Use the `figure_rst` helper function to generate the rST for this
    # code block's figures. Alternatively you can define your own rST.
    return sphinx_gallery.scrapers.figure_rst(image_names,
                                              gallery_conf['src_dir'])

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'examples',       # path to where to save gallery generated output
     'within_subsection_order': FileNameSortKey,
     'filename_pattern': r'\.py',     # all python files r'\.py',
     # except those ending with skip or in "movies" subdir
     'ignore_pattern': r'(skip\.py)|(movies' + os.path.sep + ')', 
     'image_scrapers': (fractalshades_scraper),
     # 'plot_gallery': 'False',  # Activate this to skip the calculation
}

####
# GitHub extension
github_project_url = "https://github.com/matplotlib/matplotlib/"

