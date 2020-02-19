"""
This script tests the generation of simulation of spin0and2 spectra, including an option to use mpi
"""
import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_cov, so_mpi
import healpy as hp
import numpy as np
import pylab as plt
import os,time

#We start by specifying the CAR survey parameters, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
# It will have a resolution of 1 arcminute
ra0, ra1, dec0, dec1 = -10, 10, -8, 8
res = 2.5
# ncomp=3 mean that we are going to use spin0 and 2 field
ncomp = 3
# specify the order of the spectra, this will be the order used in pspy
# note that if you are doing cross correlation between galaxy and kappa for example, you should follow a similar structure
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
# clfile are the camb lensed power spectra
clfile = "../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
# n_splits stands for the number of splits we want to simulate
n_splits = 2
# The type of power spectra we want to compute
type = "Dl"
# a binningfile with format, lmin,lmax,lmean
binning_file = "../data/BIN_ACTPOL_50_4_SC_low_ell_startAt2"
# the maximum multipole to consider
lmax = 2000
# the number of iteration in map2alm
niter = 0
# the noise on the spin0 component
rms_uKarcmin_T = 1
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 1
# the number of holes in the point source mask
source_mask_nholes = 10
# the radius of the holes (in arcminutes)
source_mask_radius = 10
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask = 0.3
# for the CAR survey we will use an apodisation type designed for rectangle maps
apo_type = "Rectangle"
# parameter for the monte-carlo simulation
DoMonteCarlo = True
mpi = True
iStart = 0
iStop = 30
n_sims = iStop - iStart + 1


test_dir = "mc_spin0and2"
pspy_utils.create_directory(test_dir)


template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
# the binary template for the window functionpixels
# for CAR we set pixels inside the survey at 1 and  at the border to be zero
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

#we then apodize the survey mask
window = so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree_survey)
#we create a point source mask
mask = so_map.simulate_source_mask(binary, n_holes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
#... and we apodize it
mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree_mask)
#the window is given by the product of the survey window and the mask window
window.data *= mask.data

#for spin0 and 2 the window need to be a tuple made of two objects
#the window used for spin0 and the one used for spin 2
window_tuple = (window, window)


#the window is going to couple mode together, we compute a mode coupling matrix in order to undo this effect
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                            binning_file,
                                            lmax=lmax,
                                            type="Dl",
                                            niter=niter)


nameList = []
specList = []
for i in range(n_splits):
    nameList += ["split_%d" % i]
for c1, name1 in  enumerate(nameList):
    for c2, name2 in enumerate(nameList):
        if c1 > c2: continue
        spec_name = "%sx%s" % (name1, name2)
        specList += [spec_name]

if mpi == True:
    so_mpi.init(True)
    subtasks = so_mpi.taskrange(imin=iStart, imax=iStop)
else:
    subtasks = np.arange(iStart, iStop + 1)

# loop over the simulation index and compute the corresponding spectra

for iii in subtasks:
    t = time.time()
    cmb = template.synfast(clfile)
    splitlist = []
    for i in range(n_splits):
        split = cmb.copy()
        noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
        split.data += noise.data
        splitlist += [split]

    almList = []
    for s in splitlist:
        almList += [sph_tools.get_alms(s, window_tuple, niter, lmax)]

    for name1, alm1, c1  in zip(nameList, almList, np.arange(n_splits)):
        for name2, alm2, c2  in zip(nameList, almList, np.arange(n_splits)):
            if c1 > c2: continue
            ls, ps = so_spectra.get_spectra(alm1, alm2, spectra=spectra)
            spec_name = "%sx%s" % (name1, name2)
            lb, Db = so_spectra.bin_spectra(ls,
                                            ps,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mbb_inv,
                                            spectra=spectra)
                                                
            so_spectra.write_ps("%s/sim_spectra_%s_%04d.dat"%(test_dir, spec_name, iii),
                                lb,
                                Db,
                                type=type,
                                spectra=spectra)
    print("sim number %04d took %s second to compute" % (iii, time.time()-t))


