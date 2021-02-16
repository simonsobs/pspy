"""Tests for spectra generation."""

import os
import pickle
import tempfile
import unittest
from itertools import combinations_with_replacement as cwr

import healpy as hp
import numpy as np
from pspy import pspy_utils, so_map, so_mcm, so_spectra, so_window, sph_tools


class SOSpectraTests(unittest.TestCase):
    def organic_test_spectra(self, template, window, healpix=False, lmax=1000):
        cl_file = "./data/cl_camb.dat"
        cmb = template.synfast(cl_file)

        nsplits = 2
        splits = [cmb.copy() for i in range(nsplits)]
        for i in range(nsplits):
            noise = so_map.white_noise(cmb, rms_uKarcmin_T=5, rms_uKarcmin_pol=np.sqrt(2) * 5)
            splits[i].data += noise.data

        binning_file = os.path.join(tempfile.gettempdir(), "binning.dat")
        pspy_utils.create_binning_file(bin_size=40, n_bins=100, file_name=binning_file)
        window = (window, window)
        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(
            window, binning_file, lmax=lmax, type="Dl", niter=0
        )
        alms = [sph_tools.get_alms(split, window, niter=0, lmax=lmax) for split in splits]
        spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

        Db_dict = {}
        for (i1, alm1), (i2, alm2) in cwr(enumerate(alms), 2):
            l, ps = so_spectra.get_spectra(alm1, alm2, spectra=spectra)
            lb, Db = so_spectra.bin_spectra(
                l, ps, binning_file, lmax, type="Dl", mbb_inv=mbb_inv, spectra=spectra
            )
            Db_dict[f"split{i1}xsplit{i2}"] = Db

        # l, ps_theory = pspy_utils.ps_lensed_theory_to_dict(cl_file, "Dl", lmax=lmax)
        # ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory, spectra=spectra)

        with open("./data/unit_test_spectra_spin0and2.pkl", "rb") as f:
            Db_ref = pickle.load(f)["healpix" if healpix else "car"]
            for k, v in Db_dict.items():
                for spec in spectra:
                    np.testing.assert_almost_equal(v[spec], Db_ref[k][spec], decimal=2)

    def test_spectra_spin0and2_car(self):
        np.random.seed(14)
        ra0, ra1, dec0, dec1 = -25, 25, -25, 25
        res = 1
        ncomp = 3
        template_car = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)

        binary_car = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
        binary_car.data[:] = 0
        binary_car.data[1:-1, 1:-1] = 1

        window = so_window.create_apodization(binary_car, apo_type="Rectangle", apo_radius_degree=1)
        mask = so_map.simulate_source_mask(binary_car, n_holes=100, hole_radius_arcmin=10)
        mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=0.3)
        window.data *= mask.data

        self.organic_test_spectra(template_car, window)

    def test_spectra_spin0and2_healpix(self):
        np.random.seed(14)
        lon, lat = 30, 50
        radius = 25
        nside = 1024
        ncomp = 3
        template_healpix = so_map.healpix_template(ncomp, nside=nside)

        binary_healpix = so_map.healpix_template(ncomp=1, nside=nside)
        vec = hp.ang2vec(lon, lat, lonlat=True)
        disc = hp.query_disc(nside, vec, radius=radius * np.pi / 180)
        binary_healpix.data[disc] = 1

        window = so_window.create_apodization(binary_healpix, apo_type="C1", apo_radius_degree=1)
        mask = so_map.simulate_source_mask(binary_healpix, n_holes=100, hole_radius_arcmin=10)
        mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=0.3)
        window.data *= mask.data

        self.organic_test_spectra(template_healpix, window, healpix=True)


if __name__ == "__main__":
    unittest.main()
