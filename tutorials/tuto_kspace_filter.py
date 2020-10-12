"""
This script tests the kspace filter used to remove systematics in the ACT data
"""
import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_map_preprocessing
import numpy as np
import healpy as hp
import pylab as plt
import os


ra0, ra1, dec0, dec1 = -25, 25, -25, 25
res = 1
ncomp = 3

clfile = "../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"

template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
cmb = template.synfast(clfile)

vk_mask = [-90,90]
filtered_cmb = so_map_preprocessing.kspace_filter(cmb, vk_mask=vk_mask)

filtered_cmb.plot()
