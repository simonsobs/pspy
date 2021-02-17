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

        self.current = {"car": do_simulation(car=True), "healpix": do_simulation(car=False)}

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
        datas = ["cmb", "window", "mbb_inv", "bbl", "spectra"]
        for kind, data in product(kinds, datas):
            self.compare(
                self.reference.get(kind).get(data),
                self.current.get(kind).get(data),
                msg=f"Testing {kind} pixellisation and {data} data content",
            )


if __name__ == "__main__":
    unittest.main()
