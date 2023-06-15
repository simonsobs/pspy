"""Tests for `pspy.so_cov`."""

import tempfile, unittest, os
import numpy as np
from pspy import  so_cov, so_map, so_mcm, pspy_utils

TEST_DATA_PREFIX = os.path.join(os.path.dirname(__file__), "data", "example_newcov")

class SOCovmatTests(unittest.TestCase):
    def setUp(self, verbose=False):

        # read in downgraded windows and ivar
        self.lmax = 1350
        self.window = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "window_adjusted.fits"))
        var0 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "ivar1_adjusted.fits"))
        var1 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "ivar2_adjusted.fits"))
        self.cl_dict = np.load(os.path.join(TEST_DATA_PREFIX, "cl_dict.npy"))
        self.nl_dict = np.load(os.path.join(TEST_DATA_PREFIX, "nl_dict.npy"))
        self.cov_ref = np.load(os.path.join(TEST_DATA_PREFIX, "analytic_cov_ll2.npy"))

        var0.data = 1 / var0.data
        var1.data = 1 / var1.data
        var0.data[np.isinf(var0.data)] = 0.
        var1.data[np.isinf(var1.data)] = 0.
        self.var0, self.var1 = var0, var1

        self.binning_file = os.path.join(TEST_DATA_PREFIX, "binning.dat")  # ignored
        # pspy_utils.create_binning_file(bin_size=40, n_bins=100, file_name=self.binning_file)

        

    def test_covmat(self, verbose=False):
        lmax, binning_file = self.lmax, self.binning_file
        window, var0, var1 = self.window, self.var0, self.var1
        cl_dict, nl_dict = self.cl_dict, self.nl_dict

        # set up scenario
        survey_id = ["Ta", "Tb", "Tc", "Td"]
        survey_name = ["split_0", "split_1", "split_0", "split_1"]
        win = {'Ta': window, 'Tb': window, 'Tc': window, 'Td': window}
        var = {'Ta': var0, 'Tb': var1, 'Tc': var0, 'Td': var1}

        # generate mcm
        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, 
                                                type="Dl", niter=0, binned_mcm = False)

        couplings = so_cov.generate_aniso_couplings(survey_name, win, var, lmax)

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
            Rl_dict[id1] = np.sqrt(nl_dict[id1[0] + id1[0] + id2spec[id1] + id2spec[id1]][:lmax]
                / white_noise[id1])[2:] 
            for name2, id2 in zip(survey_name, survey_id):
                spec = id1[0] + id2[0]
                key12 = spec + id2spec[id1] + id2spec[id2]
                if name1 == name2:
                    Clth_dict[id1 + id2] = cl_dict[key12][:lmax]
                else:
                    Clth_dict[id1 + id2] = cl_dict[key12][:lmax]
                Clth_dict[id1 + id2] = Clth_dict[id1 + id2][2:]

        cov_e = so_cov.cov_spin0_aniso1(
            Clth_dict, Rl_dict, couplings,
            binning_file, lmax, mbb_inv, mbb_inv, binned_mcm=False)

        num_diag = 30 # check first 30 off-diagonals
        diag_max_errors = [
            np.max(np.abs(np.diag(self.cov_ref[2:,2:] / cov_e[0], k) - 1))
            for k in range(0,num_diag)
        ]
        np.testing.assert_almost_equal(np.zeros(num_diag), 
                                       diag_max_errors, 7, 
                                       err_msg="aniso covmats don't match")


if __name__ == "__main__":
    unittest.main()
