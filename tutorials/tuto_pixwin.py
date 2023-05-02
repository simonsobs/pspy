"""
This test the convolution of so_map with a pixel window function
we also test the approximation of deconvolving on a cut sky
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, sph_tools, so_spectra, pspy_utils
import pylab as plt
import numpy as np

test_dir = "result_pixwin"
pspy_utils.create_directory(test_dir)

clfile = "./data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
ncomp = 3
res = 5
nside = 512
niter = 0
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

niters = [0, 3]

binaries = [so_map.full_sky_car_template(1, res), so_map.healpix_template(1, nside=nside)]
templates = [so_map.full_sky_car_template(ncomp, res), so_map.healpix_template(ncomp, nside=nside)]
run_name = ["run_CAR", "run_HEALPIX"]

# Let's first test the pixel window function convolution

for run, niter, template, binary in zip(run_name, niters, templates, binaries):
    lmax = template.get_lmax_limit()
    print(f"lmax: {lmax}")
    
    cmb = template.synfast(clfile)
    cmb_pixwin = cmb.copy()
    cmb_pixwin = cmb_pixwin.convolve_with_pixwin(niter=niter)
    
    diff = cmb_pixwin.copy()
    diff.data -= cmb.data
    diff.plot(file_name=f"{test_dir}/diff_{run}")

    window = binary.copy()
    window.data[:] = 1
    
    window = (window, window)

    alms = sph_tools.get_alms(cmb, window, niter, lmax)
    l, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)

    alms = sph_tools.get_alms(cmb_pixwin, window, niter, lmax)
    l, ps_pixwin = so_spectra.get_spectra(alms, alms, spectra=spectra)

    id = np.where(l>=2)
    l = l[id]
    fac = l * (l + 1) / (2 * np.pi)
    
    plt.figure(figsize=(12,8))
    for spec in ["TT", "EE", "BB"]:
        plt.plot(l, ps_pixwin[spec][id] / ps[spec][id], label=spec)
        
    plt.ylabel(r"$C^{pixwin}_{\ell}/C_{\ell}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.legend(fontsize=16)
    plt.savefig(f"{test_dir}/spectra_{run}.png", bbox_inches="tight")
    plt.clf()
    plt.close()


# In practice in CAR we often have to do the deconvolution in the cut sky let's test this
# in order to do so we will do an exact convolution of the pixel window function,
# so we will convolve a full sky map with it
# and we will do an inexact deconvolution, but deconvolving the pix win on the cut sky


# let's generate the cmb on the full sky
template = so_map.full_sky_car_template(ncomp, res)
cmb = template.synfast(clfile)

# let's convolve it with the pixel window function
cmb_pixwin = cmb.copy()
cmb_pixwin = cmb_pixwin.convolve_with_pixwin(niter=0)

# let's cut both the original cmb map (our reference) and the one with the pixwin convolution
ra0, ra1, dec0, dec1 = -50, 50, -60, 10
box = so_map.get_box(ra0, ra1, dec0, dec1)

cmb_cut = so_map.get_submap_car(cmb, box, mode="round")
cmb_cut_pixwin = so_map.get_submap_car(cmb_pixwin, box, mode="round")

# now we need to deconvolve the pixwin on the cut sky
pixwin = cmb_cut_pixwin.get_pixwin()
cmb_cut_pixwin_deconvolved = cmb_cut_pixwin.convolve_with_pixwin(niter=0, pixwin=pixwin ** -1)

diff = cmb_cut_pixwin_deconvolved.copy()
diff.data -= cmb_cut.data

diff.plot(file_name=f"{test_dir}/approximation_cutsky", color_range=[2, 0.2, 0.2])


# we then estimate pseudo power spectrum and compare them
binary_car = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary_car.data[:] = 0
binary_car.data[1:-1, 1:-1] = 1
window = so_window.create_apodization(binary_car, apo_type="C1", apo_radius_degree=1)
window = (window, window)

lmax = template.get_lmax_limit()
alms = sph_tools.get_alms(cmb_cut, window, niter, lmax)
l, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)

alms_pixwin_deconvolved = sph_tools.get_alms(cmb_cut_pixwin_deconvolved, window, niter, lmax)
l, ps_pixwin_deconvolved = so_spectra.get_spectra(alms_pixwin_deconvolved,
                                                  alms_pixwin_deconvolved,
                                                  spectra=spectra)
id = np.where(l>=2)
l = l[id]
fac = l * (l + 1) / (2 * np.pi)

for spec in ["TT", "EE", "BB"]:
    plt.figure(figsize=(12,8))

    plt.plot(l, l * 0 + 1)
    plt.plot(l, (ps_pixwin_deconvolved[spec][id] - ps[spec][id])/ps[spec][id], label=spec)
    plt.ylabel(r"$(C^{pixWin deconv}_{\ell} - C_{\ell})/C_{\ell}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)

    plt.ylim( -10 ** -4, 10 ** -4)
    plt.legend(fontsize=16)
    plt.savefig(f"{test_dir}/cut_spectra_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

