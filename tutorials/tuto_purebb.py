"""
This script illustrates the spectra computation for standard and pure B modes.
It is done in HEALPIX pixellisation.
The HEALPIX survey mask is a disk centered on longitude 'lon' and latitude 'lat' in degree  with a given 'radius' the resolution is given by 'nside'.
We appodize the survey mask with an apodisation 'apo_radius_degree_survey'
The script generate 'nsims' simulation and compare the result from the pure and standard method.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
import os

#We specify the HEALPIX survey parameter, it will be a disk of radius 25 degree centered on longitude 30 degree and latitude 50 degree
# It will have a resolution nside=512
lon, lat = 30, 50
radius = 25
nside = 512
# ncomp=3 mean that we are going to use spin0 and 2 field
ncomp = 3
# specify the order of the spectra, this will be the order used in pspy
# note that if you are doing cross correlation between galaxy and kappa for example, you should follow a similar structure
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
# clfile are the camb lensed power spectra
clfile = "../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
# the maximum multipole to consider
lmax = 3 * nside - 1
# the number of iteration in map2alm
niter = 3
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 5
# spectra type
type = "Cl"
# the templates for the CMB splits
template = so_map.healpix_template(ncomp,nside=nside)

#  we set pixel inside the disk at 1 and pixel outside at zero
binary = so_map.healpix_template(ncomp=1, nside=nside)
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary.data[disc] = 1

test_dir = "result_pureBB"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=50, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir

window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=apo_radius_degree_survey)
window.plot(file_name="%s/window"%(test_dir), hp_gnomv=(lon, lat, 3500, 1))

w1_plus, w1_minus, w2_plus, w2_minus =  so_window.get_spinned_windows(window, lmax, niter)


w1_plus.plot(file_name="%s/win_spin1_a"%(test_dir), hp_gnomv=(lon, lat, 3500, 1))
w1_minus.plot(file_name="%s/win_spin1_b"%(test_dir), hp_gnomv=(lon, lat, 3500, 1))
w2_plus.plot(file_name="%s/win_spin2_a"%(test_dir), hp_gnomv=(lon, lat, 3500, 1))
w2_minus.plot(file_name="%s/win_spin2_b"%(test_dir), hp_gnomv=(lon, lat, 3500, 1))


window_tuple = (window,window)

print("computing standard mode coupling matrix")
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                            binning_file,
                                            lmax=lmax,
                                            niter=niter,
                                            type=type)

print("computing pure mode coupling matrix")
mbb_inv_pure, Bbl_pure = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                                      binning_file,
                                                      lmax=lmax,
                                                      niter=niter,
                                                      type=type,
                                                      pure=True)

n_sims = 30

bb = []
bb_pure = []

for iii in range(n_sims):
    print("sim number %04d" % iii)
    
    cmb=template.synfast(clfile)

    alm = sph_tools.get_alms(cmb, window_tuple, niter, lmax)
    alm_pure = sph_tools.get_pure_alms(cmb, window_tuple, niter, lmax)

    l, ps = so_spectra.get_spectra(alm, spectra=spectra)
    l, ps_pure = so_spectra.get_spectra(alm_pure, spectra=spectra)

    lb, ps_dict = so_spectra.bin_spectra(l,
                                         ps,
                                         binning_file,
                                         lmax,
                                         type=type,
                                         mbb_inv=mbb_inv,
                                         spectra=spectra)

    lb, ps_dict_pure = so_spectra.bin_spectra(l,
                                              ps_pure,
                                              binning_file,
                                              lmax,
                                              type=type,
                                              mbb_inv=mbb_inv_pure,
                                              spectra=spectra)
                                              
    bb += [ps_dict["BB"]]
    bb_pure += [ps_dict_pure["BB"]]

mean_bb, std_bb = np.mean(bb, axis=0), np.std(bb, axis=0)
mean_bb_pure, std_bb_pure = np.mean(bb_pure, axis=0), np.std(bb_pure, axis=0)

l_th, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, type, lmax=lmax)
ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory, spectra=spectra)
ps_theory_b_pure = so_mcm.apply_Bbl(Bbl_pure, ps_theory, spectra=spectra)

fac = lb * (lb + 1) / (2 * np.pi)
facth = l_th * (l_th + 1) / (2 * np.pi)

plt.figure(figsize=(14, 12))
plt.plot(l_th[:lmax], ps_theory["BB"][:lmax] * facth[:lmax], color="grey")
plt.errorbar(lb - 5, ps_theory_b["BB"] * fac, color="red", label = "binned theory BB")
plt.errorbar(lb + 5, ps_theory_b_pure["BB"] * fac, color="blue", label = "binned theory BB pure")
plt.errorbar(lb - 5, mean_bb * fac, std_bb * fac, fmt=".", color="red", label = "mean BB ")
plt.errorbar(lb + 5, mean_bb_pure * fac, std_bb_pure * fac, fmt=".", color="blue", label = "mean BB pure")
plt.ylim(-0.07, 0.17)
plt.xlim(0, 2 * nside)
plt.legend()
plt.ylabel(r"$D^{BB}_{\ell}$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.savefig("%s/BB.png" % test_dir)
plt.clf()
plt.close()

plt.figure(figsize=(10, 12))
plt.plot(lb, std_bb_pure / std_bb)
plt.ylabel(r"$\sigma^{\rm pure}_\ell/ \sigma_\ell$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.savefig("%s/error_ratio.png" % test_dir)
plt.clf()
plt.close()
