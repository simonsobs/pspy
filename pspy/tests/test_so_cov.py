"""Tests for `pspy.so_cov`."""

import tempfile, unittest, os
import numpy as np
from pspy import  so_cov, so_map, so_mcm, pspy_utils, so_window

TEST_DATA_PREFIX = os.path.join(os.path.dirname(__file__), "data", "example_newcov")

class SOCovmatTests(unittest.TestCase):
    def setUp(self, verbose=False):

        # read in downgraded windows and ivar
        self.lmax = 1349
        self.window = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "window_adjusted.fits"))
        var0 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "ivar1_adjusted.fits"))
        var1 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "ivar2_adjusted.fits"))
        self.cl_dict = np.load(os.path.join(TEST_DATA_PREFIX, "cl_dict.npy"))
        self.nl_dict = np.load(os.path.join(TEST_DATA_PREFIX, "nl_dict.npy"))
        self.cov_ref_TTTT = np.load(os.path.join(TEST_DATA_PREFIX, "cov_TTTT.npy")) 
        self.cov_ref_EEEE = np.load(os.path.join(TEST_DATA_PREFIX, "cov_EEEE.npy")) 

        var0.data = 1 / var0.data
        var1.data = 1 / var1.data
        var0.data[np.isinf(var0.data)] = 0.
        var1.data[np.isinf(var1.data)] = 0.
        self.var0, self.var1 = var0, var1

        self.binning_file = os.path.join(TEST_DATA_PREFIX, "binning.dat")  # ignored
        # pspy_utils.create_binning_file(bin_size=40, n_bins=100, file_name=self.binning_file)

        

    def test_covmat_TTTT(self, verbose=False):
        lmax, binning_file = self.lmax, self.binning_file
        window, var0, var1 = self.window, self.var0, self.var1
        cl_dict, nl_dict = self.cl_dict, self.nl_dict

        # set up scenario
        survey_id = ["Ta", "Tb", "Tc", "Td"]
        survey_name = ["s1", "s4", "s1", "s4"]
        win = {'Ta': window, 'Tb': window, 'Tc': window, 'Td': window}
        var = {
            'Ta': so_cov.make_weighted_variance_map(var0, window),
            'Tb': so_cov.make_weighted_variance_map(var1, window),
            'Tc': so_cov.make_weighted_variance_map(var0, window),
            'Td': so_cov.make_weighted_variance_map(var1, window)
        }

        couplings = so_cov.generate_aniso_couplings_TTTT(
            survey_id, survey_name, win, var, lmax)

        id2spec = {'Ta': 'dr6&pa6_f150_s1', 'Tb': 'dr6&pa6_f150_s4', 
                'Tc': 'dr6&pa6_f150_s1', 'Td': 'dr6&pa6_f150_s4'}

        white_noise = {
            'Ta': so_cov.measure_white_noise_level(var0.data, window.data),
            'Tb': so_cov.measure_white_noise_level(var1.data, window.data),
            'Tc': so_cov.measure_white_noise_level(var0.data, window.data),
            'Td': so_cov.measure_white_noise_level(var1.data, window.data)
        }

        # compute Rl, could put this in a function maybe
        Clth_dict = {}
        Rl_dict = {}
        for name1, id1 in zip(survey_name, survey_id):
            Rl_dict[id1] = np.sqrt(nl_dict[id1[0] + id1[0] + id2spec[id1] + id2spec[id1]][:(lmax+1)]
                / white_noise[id1])
            for name2, id2 in zip(survey_name, survey_id):
                spec = id1[0] + id2[0]
                key12 = spec + id2spec[id1] + id2spec[id2]
                if name1 == name2:
                    Clth_dict[id1 + id2] = cl_dict[key12][:(lmax+1)]
                else:
                    Clth_dict[id1 + id2] = cl_dict[key12][:(lmax+1)]

        cov_e = so_cov.coupled_cov_aniso_same_pol(survey_id, Clth_dict, Rl_dict, couplings)

        num_diag = 30 # check first 30 off-diagonals
        diag_max_errors = [
            np.max(np.abs(np.diag(self.cov_ref_TTTT / cov_e, k) - 1))
            for k in range(0,num_diag)
        ]
        np.testing.assert_almost_equal(np.zeros(num_diag), 
                                       diag_max_errors, 7, 
                                       err_msg="aniso covmats don't match")

    def test_covmat_EEEE(self, verbose=False):
        lmax, binning_file = self.lmax, self.binning_file
        window, var0, var1 = self.window, self.var0, self.var1
        cl_dict, nl_dict = self.cl_dict, self.nl_dict

        # set up scenario
        survey_id = ["Ea", "Eb", "Ec", "Ed"]
        survey_name = ["s1", "s4", "s1", "s4"]
        win = {'Ea': window, 'Eb': window, 'Ec': window, 'Ed': window}
        var = {
            'Ea': so_cov.make_weighted_variance_map(var0, window),
            'Eb': so_cov.make_weighted_variance_map(var1, window),
            'Ec': so_cov.make_weighted_variance_map(var0, window),
            'Ed': so_cov.make_weighted_variance_map(var1, window)
        }

        couplings = so_cov.generate_aniso_couplings_EEEE(
            survey_id, survey_name, win, var, lmax)

        id2spec = {'Ea': 'dr6&pa6_f150_s1', 'Eb': 'dr6&pa6_f150_s4', 
                'Ec': 'dr6&pa6_f150_s1', 'Ed': 'dr6&pa6_f150_s4'}

        white_noise = {
            'Ea': so_cov.measure_white_noise_level(var0.data, window.data),
            'Eb': so_cov.measure_white_noise_level(var1.data, window.data),
            'Ec': so_cov.measure_white_noise_level(var0.data, window.data),
            'Ed': so_cov.measure_white_noise_level(var1.data, window.data)
        }

        # compute Rl, could put this in a function maybe
        Clth_dict = {}
        Rl_dict = {}
        for name1, id1 in zip(survey_name, survey_id):
            Rl_dict[id1] = np.sqrt(nl_dict[id1[0] + id1[0] + id2spec[id1] + id2spec[id1]][:(lmax+1)]
                / white_noise[id1])
            for name2, id2 in zip(survey_name, survey_id):
                spec = id1[0] + id2[0]
                key12 = spec + id2spec[id1] + id2spec[id2]
                if name1 == name2:
                    Clth_dict[id1 + id2] = cl_dict[key12][:(lmax+1)]
                else:
                    Clth_dict[id1 + id2] = cl_dict[key12][:(lmax+1)]

        cov_e = so_cov.coupled_cov_aniso_same_pol(survey_id, Clth_dict, Rl_dict, couplings)

        num_diag = 30 # check first 30 off-diagonals
        diag_max_errors = [
            np.max(np.abs(np.diag(self.cov_ref_EEEE[2:,2:] / cov_e[2:,2:], k) - 1))
            for k in range(0,num_diag)
        ]
        np.testing.assert_almost_equal(np.zeros(num_diag), 
                                       diag_max_errors, 7, 
                                       err_msg="aniso covmats don't match")

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
