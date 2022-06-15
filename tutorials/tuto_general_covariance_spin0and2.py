"""
This is by far the most complicated tutorial of pspy, this is basically a end-to-end test of the pipeline.
This goal is to emulate as close as possible what we are doing in real ACT data analysis
for the analytic covariance computation (but with a much smaller test case).
Let's imagine an experiment with different 'surveys', for example a set
of observation in year 2019: sv_1 and a set of observation in year 2020: sv_2.
The experiment observe the sky with one dichroic array,
the two groups of detectors within the array will be called ar_1 and ar_2.
Each set of observation (e.g (sv_a)) is divided into a number of uniform data splits.
We use cross power spectrum, i.e, we discard power spectrum that are obtained  for the same split of the same survey
Cl^{sv_a&ar_b, sv_c&ar_d} = 1 / norm \sum_{i=1}^{Ns} \sum_{j=1}^{Ns} 1/(2 * ell + 1) alm_{i, sv_a&ar_b} alm_{j, sv_c&ar_d}(1-\delta(i,j)\delta(sv_a,sv_b))
Note the role of the delta functions, if we consider the same split and the same survey the power spectrum will be discarded,
this ensures that correlated noise doesn't bias our estimate of the power spectrum,
note also that power spectrum from the same split, same survey but between ar_1 and ar_2 is also discarded,
this is because the two wafers of the dichroic array observe the same atm fluctuation, therefore they can be impacted by correlated noise.
The question is what is the general covariance between  Cl^{sv_a&ar_b, sv_c&ar_d} and Cl^{sv_e&ar_f, sv_g&ar_h},
where a, b, c, d, e, f, g, h can take values between 1 and 2
"""
import os
import time
import healpy as hp
import numpy as np
import pylab as plt
from pspy import pspy_utils, so_cov, so_map, so_mcm, so_spectra, so_window, sph_tools
from pixell import curvedsky

##########################################################################
# Let's start by specifying the general parameters of the analysis
##########################################################################
ncomp = 3
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
run = "HEALPIX"
niter = 0

# We assume a common survey area for the  different survey and the different detectors wafers
if run == "CAR":
    ra0, ra1, dec0, dec1 = -15, 15, -8, 8
    res = 5
    apo_type_survey = "Rectangle"
    template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
    binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
    binary.data[:] = 0
    binary.data[1:-1, 1:-1] = 1
    binned_mcm = True
    lmax = 1000
    bin_size = 60
    source_mask_radius = 20
    apo_radius_degree_mask = 0.5

elif run == "HEALPIX":
    lon, lat = 30, 50
    radius = 120
    nside = 512
    apo_type_survey = "C1"
    template = so_map.healpix_template(ncomp=ncomp, nside=nside)
    binary = so_map.healpix_template(ncomp=1, nside=nside)
    vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
    disc = hp.query_disc(nside, vec, radius=radius * np.pi / 180)
    binary.data[disc] = 1
    binned_mcm = False
    lmax = 600
    bin_size = 10
    source_mask_radius = 60
    apo_radius_degree_mask = 1


# The type of power spectra we want to compute
type = "Dl"
# a binningfile with format, lmin,lmax,lmean
# the maximum multipole to consider

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237, "omch2": 0.1200, "ns": 0.9649, "Alens": 1.0, "tau": 0.0544}
l, ps_theory = pspy_utils.ps_from_params(cosmo_params, "Cl", 3 * lmax, start_at_zero=True)
ps_mat = np.zeros((ncomp, ncomp, 3 * lmax))
for i, f1 in enumerate("TEB"):
    for j, f2 in enumerate("TEB"):
        ps_mat[i, j] = ps_theory[f1+f2].copy()
        ps_theory[f1+f2] *= l * (l + 1) / (2 * np.pi)
        ps_theory[f1+f2] = ps_theory[f1+f2][2:lmax]

# for the CAR survey we will use an apodisation type designed for rectangle maps
n_sims = 500

test_dir = "result_generalized_cov_spin0and2"
pspy_utils.create_directory(test_dir)

binning_file = "%s/binning.dat" % test_dir
pspy_utils.create_binning_file(bin_size=bin_size, n_bins=300, file_name=binning_file)

##########################################################################
# Now we specify the caracteristic of the two different surveys and wafers
##########################################################################

surveys = ["sv_1", "sv_2"]
arrays = {}
arrays["sv_1"] = ["pa_1"]
arrays["sv_2"] = ["pa_1", "pa_2"]

n_splits, rms_uKarcmin_T, apo_radius_degree_survey, source_mask_nholes, bl = {}, {}, {}, {}, {}

print("sim parameters:")
for sv in surveys:
    n_splits[sv] = 3#np.random.randint(2, high=8)
    for ar in arrays[sv]:
        rms_uKarcmin_T[sv, ar] = np.random.randint(10, high=40)
        apo_radius_degree_survey[sv, ar] = np.random.uniform(0.5, high=3)
        source_mask_nholes[sv, ar] = np.random.randint(10, high=30)
        fwhm = np.random.randint(3, high=8)
        l, bl[sv, ar] = pspy_utils.beam_from_fwhm(fwhm, 2 * lmax)
        
        print("")
        print("nsplit %s: %d" % (sv, n_splits[sv] ))
        print("rms %s %s: %d" % (sv, ar, rms_uKarcmin_T[sv, ar]))
        print("apo survey %s %s: %.02f" % (sv, ar, apo_radius_degree_survey[sv, ar]))
        print("mask nholes %s %s: %d" % (sv, ar, source_mask_nholes[sv, ar]))
        print("fwhm %s %s: %d" % (sv, ar, fwhm))
        print("")

##########################################################################
# Let's generate the different window functions
##########################################################################


window = {}
for sv in surveys:
    for ar in arrays[sv]:
        window[sv, ar] = so_window.create_apodization(binary,
                                              apo_type=apo_type_survey,
                                              apo_radius_degree=apo_radius_degree_survey[sv, ar])

        mask = so_map.simulate_source_mask(binary,
                                           n_holes=source_mask_nholes[sv, ar],
                                           hole_radius_arcmin=source_mask_radius)
                                           
                                           
        mask = so_window.create_apodization(mask,
                                            apo_type="C1",
                                            apo_radius_degree=apo_radius_degree_mask)
                                            
        window[sv, ar].data *= mask.data
        window[sv, ar].plot(file_name="%s/window_%s_%s" % (test_dir, sv, ar))

##########################################################################
# Let's generate all the necessary mode coupling matrices
##########################################################################
spec_name_all = []
mbb_inv_dict = {}
Bbl_dict = {}
for id_sv_a, sv_a in enumerate(surveys):
    for id_ar_a, ar_a in enumerate(arrays[sv_a]):
        # we need both the window for T and pol, here we assume they are the same
        window_tuple_a = (window[sv_a, ar_a], window[sv_a, ar_a])
        # we need both the beam for T and pol, here we assume they are the same
        bl_a = (bl[sv_a, ar_a], bl[sv_a, ar_a])

        for id_sv_b, sv_b in enumerate(surveys):
            for id_ar_b, ar_b in enumerate(arrays[sv_b]):
            
                if  (id_sv_a == id_sv_b) & (id_ar_a > id_ar_b) : continue
                if  (id_sv_a > id_sv_b) : continue
                # the if here help avoiding redondant computation
                # the mode coupling matrices for sv1 x sv2 are the same as the one for sv2 x sv1
                # identically, within a survey, the mode coupling matrix for ar1 x ar2 =  ar2 x ar1
                window_tuple_b = (window[sv_b, ar_b], window[sv_b, ar_b])
                bl_b = (bl[sv_b, ar_b], bl[sv_b, ar_b])


                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple_a,
                                                            win2 = window_tuple_b,
                                                            bl1 = bl_a,
                                                            bl2 = bl_b,
                                                            binning_file = binning_file,
                                                            lmax=lmax,
                                                            type="Dl",
                                                            niter=niter,
                                                            binned_mcm=binned_mcm)

                mbb_inv_dict[sv_a, ar_a, sv_b, ar_b] = mbb_inv
                Bbl_dict[sv_a, ar_a, sv_b, ar_b] = Bbl
                
                spec_name = "%s&%sx%s&%s" % (sv_a, ar_a, sv_b, ar_b)
                spec_name_all += [spec_name]


#################################################################################################
# Let's prepare the model and precompute the alms fot the covariance matrix analytic computation
#################################################################################################

my_spectra = ["TT", "TE", "ET", "EE"]

ps_all_th = {}
nl_all_th = {}

for spec in spec_name_all:
    na, nb = spec.split("x")
    sv_a, ar_a = na.split("&")
    sv_b, ar_b = nb.split("&")
    
    if (sv_a == sv_b) & (ar_a == ar_b):
        nlth = pspy_utils.get_nlth_dict(rms_uKarcmin_T[sv_a, ar_a],  "Dl", lmax, spectra=spectra)
        
    for spec in my_spectra:
    
        ps_all_th["%s&%s" % (sv_a, ar_a), "%s&%s" % (sv_b, ar_b), spec] = bl[sv_a, ar_a][2:lmax] * bl[sv_b, ar_b][2:lmax] * ps_theory[spec]
        if (sv_a == sv_b) & (ar_a == ar_b):
            nl_all_th["%s&%s" % (sv_a, ar_a), "%s&%s" % (sv_b, ar_b), spec] = nlth[spec][2:lmax] * n_splits[sv_a]
        else:
            nl_all_th["%s&%s" % (sv_a, ar_a), "%s&%s" % (sv_b, ar_b), spec] = 0

        ps_all_th["%s&%s" % (sv_b, ar_b), "%s&%s" % (sv_a, ar_a), spec] = ps_all_th["%s&%s" % (sv_a, ar_a), "%s&%s" % (sv_b, ar_b), spec]
        nl_all_th["%s&%s" % (sv_b, ar_b), "%s&%s" % (sv_a, ar_a), spec] = nl_all_th["%s&%s" % (sv_a, ar_a), "%s&%s" % (sv_b, ar_b), spec]

    sq_win = window[sv_a, ar_a].copy()
    sq_win.data *= window[sv_b, ar_b].data
    sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
    np.save("%s/alms_%s&%sx%s&%s.npy" % (test_dir, sv_a, ar_a, sv_b, ar_b), sqwin_alm)


##########################################################################
# Let's compute the generalized analytic covariance matrix
##########################################################################

analytic_cov = {}

for sid1, spec1 in enumerate(spec_name_all):
   for sid2, spec2 in enumerate(spec_name_all):
       if sid1 > sid2: continue
       
       na, nb = spec1.split("x")
       nc, nd = spec2.split("x")

       coupling = so_cov.fast_cov_coupling_spin0and2(test_dir,
                                                    [na, nb, nc, nd],
                                                    lmax)
                                           
       sv_a, ar_a = na.split("&")
       sv_b, ar_b = nb.split("&")
       sv_c, ar_c = nc.split("&")
       sv_d, ar_d = nd.split("&")

       # These objects are symmetric in (sv_a, ar_a) and (sv_b, ar_b)
       try: mbb_inv_ab, Bbl_ab = mbb_inv_dict[sv_a, ar_a, sv_b, ar_b], Bbl_dict[sv_a, ar_a, sv_b, ar_b]
       except: mbb_inv_ab, Bbl_ab = mbb_inv_dict[sv_b, ar_b, sv_a, ar_a], Bbl_dict[sv_b, ar_b, sv_a, ar_a]

       try: mbb_inv_cd, Bbl_cd = mbb_inv_dict[sv_c, ar_c, sv_d, ar_d], Bbl_dict[sv_c, ar_c, sv_d, ar_d]
       except: mbb_inv_cd, Bbl_cd =  mbb_inv_dict[sv_d, ar_d, sv_c, ar_c], Bbl_dict[sv_d, ar_d, sv_c, ar_c]


       analytic_cov[spec1, spec2] = so_cov.generalized_cov_spin0and2(coupling,
                                                                     [na, nb, nc, nd],
                                                                     n_splits,
                                                                     ps_all_th,
                                                                     nl_all_th,
                                                                     lmax,
                                                                     binning_file,
                                                                     mbb_inv_ab,
                                                                     mbb_inv_cd,
                                                                     binned_mcm=binned_mcm)

       np.save("%s/analytic_cov_%s_%s.npy" % (test_dir, spec1, spec2), analytic_cov[spec1, spec2])

##########################################################################
# Let's generate the simulation and compute the corresponding spectra
##########################################################################

ps_all = {}

for iii in range(n_sims):
    t = time.time()

    sim_alm = {}
    cmb_alms = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    for sv in surveys:
        for ar in arrays[sv]:
            alms_beamed = cmb_alms.copy()
            for i in range(3):
                alms_beamed[i] = curvedsky.almxfl(alms_beamed[i], bl[sv, ar])
            cmb = sph_tools.alm2map(alms_beamed, template)
            for k in range(n_splits[sv]):
                split = cmb.copy()
                noise = so_map.white_noise(template, rms_uKarcmin_T=rms_uKarcmin_T[sv, ar])
                split.data += noise.data * np.sqrt(n_splits[sv])
                sim_alm[sv, ar, k] = sph_tools.get_alms(split, (window[sv, ar], window[sv, ar]), niter, lmax)

    for id_sv1, sv1 in enumerate(surveys):
        for id_ar1, ar1 in enumerate(arrays[sv1]):
            for id_sv2, sv2 in enumerate(surveys):
                for id_ar2, ar2 in enumerate(arrays[sv2]):
            
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue
                    
                    ps_dict = {}
                    for spec in spectra:
                        ps_dict[spec] = []
                    
                    for s1 in range(n_splits[sv1]):
                        for s2 in range(n_splits[sv2]):
                            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

                            l, ps_master = so_spectra.get_spectra_pixell(sim_alm[sv1, ar1, s1],
                                                                         sim_alm[sv2, ar2, s2],
                                                                         spectra=spectra)

                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv_dict[sv1, ar1, sv2, ar2],
                                                            spectra=spectra,
                                                            binned_mcm=binned_mcm)

                            for count, spec in enumerate(spectra):
                                if (s1 == s2) & (sv1 == sv2): continue #discard the auto
                                else: ps_dict[spec] += [ps[spec]]

                    ps_dict_cross_mean = {}
                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec], axis=0)
                        if ar1 == ar2 and sv1 == sv2:
                            # Average TE / ET so that for same array same season TE = ET
                            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec], axis=0) + np.mean(ps_dict[spec[::-1]], axis=0)) / 2.
                    spec_name = "%s&%sx%s&%s" % (sv1, ar1, sv2, ar2)
                    if iii == 0:
                        ps_all[spec_name] = []
                         
                    ps_all[spec_name] += [ps_dict_cross_mean]
    print("sim %05d/%05d took %.02f s to compute" % (iii, n_sims, time.time() - t))


##########################################################################
# Let's compare MC and analytic errorbars
##########################################################################

n_bins = len(lb)
for sid1, spec1 in enumerate(spec_name_all):
    for sid2, spec2 in enumerate(spec_name_all):
        if sid1 > sid2: continue
        
        mean_a, mean_b, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[spec1], ps_all[spec2], spectra=my_spectra)
        
        np.save("%s/mc_cov_%s_%s.npy" % (test_dir, spec1, spec2), mc_cov)

        mc_corr = so_cov.cov2corr(mc_cov)
        analytic_corr = so_cov.cov2corr(analytic_cov[spec1, spec2])

        plt.figure(figsize=(18,12))
        plt.subplot(2, 2, 1)
        plt.semilogy()
        plt.ylabel(r"$Cov_{%sx%s}$" % (spec1, spec2), fontsize=22)
        plt.plot(np.abs(mc_cov.diagonal()), ".")
        plt.plot(np.abs(analytic_cov[spec1, spec2].diagonal()))
        plt.subplot(2, 2, 2)
        plt.plot(mc_cov.diagonal()/analytic_cov[spec1, spec2].diagonal(), "o", color="red")
        plt.plot(mc_cov.diagonal() * 0 + 1, "--", color="black")
        plt.ylabel(r"$Cov_{%s %s, MC/analytic} $" % (spec1, spec2), fontsize=22)
        plt.subplot(2, 2, 3)
        plt.imshow(mc_corr, cmap="seismic", vmin=-0.5, vmax=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(2, 2, 4)
        plt.imshow(analytic_corr, cmap="seismic", vmin=-0.5, vmax=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig("%s/cov_element_comparison_%s_%s.png" % (test_dir, spec1, spec2), bbox_inches="tight")
        plt.clf()
        plt.close()
