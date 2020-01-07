from pspy import (mcm_fortran, pspy_utils, so_config, so_cov, so_dict, so_map,
                  so_map_preprocessing, so_mcm, so_misc, so_mpi, so_spectra,
                  so_window, sph_tools)

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
