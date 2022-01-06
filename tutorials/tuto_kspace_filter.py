"""
This script display the effect of the kspace filter used to remove systematics in the ACT data.
We show the transfer function (or its effect) in 1d and in 2d.
You can play with the survey parameter and the number of simulations.
"""
#import matplotlib
#matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_map_preprocessing, flat_tools
import numpy as np
import pylab as plt
import os

# survey and computation specifications
ra0, ra1, dec0, dec1 = -20, 20, -20, 20
res = 1
ncomp = 3
apo_type = "C1"
apo_radius_degree = 2
niter = 0
lmax = 2000
nsims = 30


test_dir = "result_kspace"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=40, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir


# we start by creating a template, a window function and compute the associated mode coupling matrix
# it will be used to form unbiased 1d power spectrum

template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1
window = so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree)
window = (window,window)
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window, binning_file, lmax=lmax, type="Dl", niter=niter)

# we build two different filters and compute the analytical 1d binned spectra they correspond to

vk_mask = [-90, 90]
hk_mask = [-50, 50]
filter_std = so_map_preprocessing.build_std_filter(template.data.shape, template.data.wcs, vk_mask, hk_mask, dtype=np.float64)

lbounds = [4000, 5]
filter_sig = so_map_preprocessing.build_sigurd_filter(template.data.shape, template.data.wcs, lbounds, dtype=np.float64)


lb, analytic_tf_std = so_map_preprocessing.analytical_tf(template, filter_std, binning_file, lmax)
lb, analytic_tf_sig = so_map_preprocessing.analytical_tf(template, filter_sig, binning_file, lmax)




# Let's run nsims, and apply the transfer function, we will compute both the SPHT and the FFT of the initial and filtered maps.
# Then we compute the associated 1d and 2d power spectra, we will store the value of the resulting
# computation for each simulation in a list (to inspect the mean later).

clfile = "./data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra_2d = ["II", "IQ", "IU", "QI", "QQ", "QU", "UI", "UQ", "UU"]
Db_list = {}
p2d_list = {}

runs = ["sig_filtered", "std_filtered", "no_filter"]

for run in runs:
    for (spec, spec2d) in zip(spectra, spectra_2d):
        Db_list[run, spec] = []
        p2d_list[run, spec2d] = []

    
for iii in range(nsims):
    print("sim number %03d" % iii)
    for run in runs:
        cmb = template.synfast(clfile)
        
        if iii == 0: cmb.plot(file_name="%s/cmb"%(test_dir))
        if run == "std_filtered":
            cmb = so_map_preprocessing.apply_std_filter(cmb, filter_std)
            if iii == 0: cmb.plot(file_name="%s/cmb_filter"%(test_dir))
        elif run == "sig_filtered":
            cmb = so_map_preprocessing.apply_sigurd_filter(cmb, binary.data, filter_sig, tol=1e-4, ref=0.9)
            if iii == 0: cmb.plot(file_name="%s/cmb_sigurd_filter"%(test_dir))

        alm_cmb = sph_tools.get_alms(cmb, window, niter, lmax)
        fft_cmb = flat_tools.get_ffts(cmb, window, lmax)
        
        
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
            
        ells, p2d_dict = flat_tools.power_from_fft(fft_cmb, fft_cmb, type=type)
        
        for spec in spectra_2d:
            p2d_list[run, spec] += [p2d_dict.powermap[spec]]


# First the 1d part with the associated transfer functions


mean = {}
std = {}
for spec in spectra:
    for run in runs:
        mean[run, spec]= np.mean(Db_list[run, spec], axis=0)
        std[run, spec]= np.std(Db_list[run, spec], axis=0)
        plt.errorbar(lb, mean[run, spec], std[run, spec], fmt=".", label=run)
        
    plt.legend()
    plt.savefig("%s/spectra_%s.png" % (test_dir, spec))
    plt.clf()
    plt.close()
    

    if spec in ["TT", "EE", "TE", "BB"] :
        plt.plot(lb, analytic_tf_std, label = "std analytic")
        plt.plot(lb, analytic_tf_sig, label = "sigurd analytic")

        plt.errorbar(lb, mean["std_filtered", spec]/mean["no_filter", spec], label="std filter", fmt=".")
        plt.errorbar(lb, mean["sig_filtered", spec]/mean["no_filter", spec], label="sigurd filter", fmt=".")

        plt.legend()
        plt.savefig("%s/tf_%s.png" % (test_dir, spec))
        plt.clf()
        plt.close()


plt.plot(lb, analytic_tf_std, label = "std analytic")
plt.plot(lb, analytic_tf_sig, label = "sigurd analytic")

for spec in ["TT", "EE"] :
    plt.plot(lb, mean["std_filtered", spec]/mean["no_filter", spec], ".", label = spec + "std")
    plt.plot(lb, mean["sig_filtered", spec]/mean["no_filter", spec], ".", label = spec + "sigurd")

plt.legend()
plt.savefig("%s/tf_TT_and_EE.png" % (test_dir))
plt.clf()
plt.close()

# Now the 2d part with the plot of the effect of the filter


for run in runs:
    mean[run] = p2d_dict.copy()
for spec in spectra_2d:
    for run in runs:
        mean[run].powermap[spec] = np.mean(p2d_list[run, spec], axis=0)
for run in runs:
    mean[run].plot(power_of_ell=2, png_file="%s/%s" % (test_dir, run))
