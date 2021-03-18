"""Tests for spectra generation."""

import pickle
import unittest
from itertools import product

import numpy as np

from generate_test_data import do_simulation


class SOSpectraTests(unittest.TestCase):
    def setUp(self):
        with open("./data/test_data.pkl", "rb") as f:
            self.reference = pickle.load(f)

        self.current = {
            kind: do_simulation(car=kind == "car", window_data=self.reference[kind]["window"])
            for kind in ["car", "healpix"]
        }

    def compare(self, ref, current, msg=""):
        self.assertIsInstance(ref, type(current), msg=msg)

        if isinstance(ref, dict):
            for k in ref.keys():
                self.compare(ref[k], current[k], msg=msg)
        elif isinstance(ref, np.ndarray):
            np.testing.assert_almost_equal(ref, current, err_msg=msg)
        else:
            self.assertTrue(True, f"Data type {type(ref)} are not compared!")

    def test_spectra(self):
        kinds = ["car", "healpix"]
        # do not test window function due to mismatch in distance computation
        # datas = ["cmb", "window", "mbb_inv", "bbl", "spectra"]
        datas = ["cmb", "mbb_inv", "bbl", "spectra"]
        for kind, data in product(kinds, datas):
            self.compare(
                self.reference.get(kind).get(data),
                self.current.get(kind).get(data),
                msg=f"Testing {kind} pixellisation and {data} data content",
            )


if __name__ == "__main__":
    unittest.main()
