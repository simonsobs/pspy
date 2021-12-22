"""Tests for `pspy.so_map` object."""

import os
import unittest

import healpy as hp
import numpy as np
from pspy import so_map

TEST_DATA_PREFIX = os.path.join(os.path.dirname(__file__), "data")


def maps_are_equal(so_map1, so_map2):
    assert so_map1.pixel == so_map2.pixel
    assert so_map1.nside == so_map2.nside
    assert so_map1.ncomp == so_map2.ncomp
    # assert so_map1.geometry[0] == so_map2.geometry[0]
    assert so_map1.coordinate == so_map2.coordinate
    assert np.allclose(so_map1.data, so_map2.data, rtol=0, atol=1e-4)


class SOMapTests(unittest.TestCase):
    def test_box(self):
        ra0, ra1, dec0, dec1 = -1, 2, -3, 4
        box = so_map.get_box(ra0, ra1, dec0, dec1)
        box = np.rad2deg(box)
        self.assertEqual(box[0, 0], dec0)
        self.assertEqual(box[0, 1], ra1)
        self.assertEqual(box[1, 0], dec1)
        self.assertEqual(box[1, 1], ra0)

    def test_copy(self):
        ncomp, res = 3, 1
        ra0, ra1, dec0, dec1 = -1, 2, -3, 4
        so_map1 = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
        so_map2 = so_map1.copy()
        maps_are_equal(so_map1, so_map2)

    # def test_car_synfast(self):
    #     # Fix seed
    #     np.random.seed(14)
    #     ra0, ra1, dec0, dec1 = -5, 5, -5, 5
    #     res = 1
    #     ncomp = 3
    #     so_map1 = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
    #     so_map2 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "map_car.fits"))
    #     test_maps_equality(so_map1.synfast(os.path.join(TEST_DATA_PREFIX, "cl_camb.dat")), so_map2)

    # def test_healpix_synfast(self):
    #     # Fix seed
    #     np.random.seed(14)
    #     nside = 256
    #     ncomp = 3
    #     so_map1 = so_map.healpix_template(ncomp, nside)
    #     so_map2 = so_map.read_map(os.path.join(TEST_DATA_PREFIX, "map_healpix.fits"))
    #     test_maps_equality(so_map1.synfast(os.path.join(TEST_DATA_PREFIX, "cl_camb.dat")), so_map2)

    def test_subtract_mono_dipole_healpix(self):
        monopole = 0.5
        dipole = np.array([0.0, 1.0, 0.0])
        nside = 32
        m = so_map.healpix_template(ncomp=1, nside=nside)
        mask = m.copy()
        m.data = np.full_like(m.data, monopole)
        for ipix in range(hp.nside2npix(nside)):
            vec = hp.pix2vec(nside, ipix)
            m.data[ipix] += np.sum(vec * dipole)
            mask.data[ipix] = 1 if vec[0] < 0.5 else 0

        # Test raw function
        results = so_map.subtract_mono_dipole(m.data, mask.data, return_values=True)
        np.testing.assert_almost_equal(results[0], 0.0)
        self.assertAlmostEqual(results[1], monopole)
        np.testing.assert_almost_equal(results[2], dipole)

        # Test so_map method
        m.ncomp = 3
        m.data = np.tile(m.data, (3, 1))
        m.subtract_mono_dipole(mask=(mask, mask))
        np.testing.assert_almost_equal(m.data, 0.0)

    def test_subtract_mono_dipole_car(self):
        monopole = 0.5
        dipole = np.array([0.0, 1.0, 0.0])
        m = so_map.car_template(ncomp=1, res=10, ra0=-180, ra1=180, dec0=-90, dec1=90)
        mask = m.copy()
        m.data = np.full_like(m.data, monopole)

        dec, ra = m.data.posmap()
        theta = np.pi / 2 - dec
        x = np.sin(theta) * np.cos(ra)
        y = np.sin(theta) * np.sin(ra)
        z = np.cos(theta)
        mask.data = np.where(x < 0.5, 1, 0)

        m.data += np.sum(np.array([x, y, z]) * dipole[:, None, None], axis=0)

        # Test raw function
        results = so_map.subtract_mono_dipole(m.data, mask.data, healpix=False, return_values=True)
        np.testing.assert_almost_equal(results[0], 0.0)
        self.assertAlmostEqual(results[1], monopole)
        np.testing.assert_almost_equal(results[2], dipole)

        # Test so_map method
        m.ncomp = 3
        m.data = np.tile(m.data, (3, 1, 1))
        m.subtract_mono_dipole(mask=(mask, mask))
        np.testing.assert_almost_equal(m.data, 0.0)


if __name__ == "__main__":
    unittest.main()
