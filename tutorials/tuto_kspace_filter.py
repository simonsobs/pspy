"""
This script tests the kspace filter used to remove systematics in the ACT data
"""
#import matplotlib
#matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_map_preprocessing
import numpy as np
import pylab as plt
import os


ra0, ra1, dec0, dec1 = -20, 20, -20, 20
res = 1
ncomp = 3
apo_type = "C1"
apo_radius_degree = 1
niter = 0
lmax = 2000
nsims = 30
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
vk_mask = [-90, 90]
hk_mask = None

test_dir = "result_kspace"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=40, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir

template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)

binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1
window = so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree)
window = (window,window)
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window, binning_file, lmax=lmax, type="Dl", niter=niter)

clfile = "../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"

Db_list = {}
for spec in spectra:
    Db_list["standard", spec] = []
    Db_list["filtered", spec] = []

for iii in range(nsims):
    print("sim number %03d" % iii)
    for run in ["standard", "filtered"]:
        cmb = template.synfast(clfile)
        if run == "filtered":
            cmb = so_map_preprocessing.kspace_filter(cmb, vk_mask=vk_mask, hk_mask=hk_mask)

        alm_cmb = sph_tools.get_alms(cmb, window, niter, lmax)
        l, ps = so_spectra.get_spectra_pixell(alm_cmb, alm_cmb, spectra=spectra)
        lb, Db_dict = so_spectra.bin_spectra(l,
                                            ps,
                                            binning_file,
                                            lmax,
                                            type="Dl",
                                            mbb_inv=mbb_inv,
                                            spectra=spectra)
                                            
        for spec in spectra:
            Db_list[run, spec] += [Db_dict[spec]]

for spec in spectra:
    for run in ["standard", "filtered"]:
        Db = np.mean(Db_list[run, spec], axis=0)
        std = np.std(Db_list[run, spec], axis=0)

        plt.errorbar(lb, Db, std, fmt=".", label="%s" % run)
    plt.legend()
    plt.show()
