from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("pspy").version
except DistributionNotFound:
    __version__ = "unknown version"


# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = ".rst"
master_doc = "index"

project = "pspy"
copyright = "2019, Simons Observatory Collaboration Analysis Library Task Force"
author = "T. Louis, S. Choi, DW Han, X. Garrido"
language = "en"
version = __version__
release = __version__

exclude_patterns = ["_build"]


# HTML theme
html_theme = "sphinx_book_theme"
# Add paths to extra static html files from notebook conversion
html_extra_path = ["latex"]
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "pspy"
# html_favicon = "_static/favicon.png"
# html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/simonsobs/pspy",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "off"
nb_execution_timeout = -1
