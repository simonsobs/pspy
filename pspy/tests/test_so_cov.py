"""Tests for `pspy.so_cov`."""

import tempfile, unittest, os
import numpy as np
from pspy import  so_cov, so_map, so_mcm, pspy_utils, so_window

TEST_DATA_PREFIX = os.path.join(os.path.dirname(__file__), "data", "example_newcov")

class SOCovmatTests(unittest.TestCase):

    def test_custom_couplings(self, verbose=False):
        ra0, ra1, dec0, dec1 = -25, 25, -15, 15
        res = 2
        lmax = 500
        l_exact, l_band, l_toep = 50, 100, 200

        binary_car = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
        binary_car.data[:] = 0
        binary_car.data[1:-1, 1:-1] = 1
        window = so_window.create_apodization(binary_car, apo_type="C1", apo_radius_degree=2)
        window2 = so_window.create_apodization(binary_car, apo_type="C2", apo_radius_degree=4)

        default_coup_02 = so_mcm.mcm_and_bbl_spin0and2(
            (window, window2), "", lmax=lmax, niter=0, l_exact=l_exact, l_band=l_band, l_toep=l_toep, return_coupling_only=True)

        xi = so_mcm.coupling_block("00", window, lmax, niter=0, l_exact=l_exact, l_band=l_band, l_toep=l_toep)
        np.testing.assert_almost_equal( default_coup_02[0,:,:], xi[2:-1,2:-1])

        xi = so_mcm.coupling_block("02", window, lmax, niter=0, win2=window2, l_exact=l_exact, l_band=l_band, l_toep=l_toep)
        np.testing.assert_almost_equal( default_coup_02[1,:,:], xi[2:-1,2:-1])

        xi = so_mcm.coupling_block("02", window2, lmax, niter=0, win2=window, l_exact=l_exact, l_band=l_band, l_toep=l_toep)
        np.testing.assert_almost_equal( default_coup_02[2,:,:], xi[2:-1,2:-1])

        xi = so_mcm.coupling_block("++", window2, lmax, niter=0, win2=window2, l_exact=l_exact, l_band=l_band, l_toep=l_toep)
        np.testing.assert_almost_equal( default_coup_02[3,:,:], xi[2:-1,2:-1])

        xi = so_mcm.coupling_block("--", window2, lmax, niter=0, win2=window2, l_exact=l_exact, l_band=l_band, l_toep=l_toep)
        np.testing.assert_almost_equal( default_coup_02[4,:,:], xi[2:-1,2:-1])


if __name__ == "__main__":
    unittest.main()
