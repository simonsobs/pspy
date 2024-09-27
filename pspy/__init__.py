from importlib.metadata import PackageNotFoundError, version

from pspy import (
    flat_tools,
    pspy_utils,
    so_config,
    so_cov,
    so_dict,
    so_map,
    so_map_preprocessing,
    so_mcm,
    so_misc,
    so_mpi,
    so_spectra,
    so_window,
    sph_tools,
)

try:
    __version__ = version("pspy")
except PackageNotFoundError:
    __version__ = "unknown"
