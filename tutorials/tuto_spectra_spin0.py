"""
This script tests the spectra computation for spin0 fields.
The first part is done on CAR pixellisation and the second in HEALPIX pixellisation.
This illustrates how pspy function are blind to pixellisation.
The CAR survey mask is defined with equatorial coordinates 'ra0', 'ra1', 'dec0', 'dec1' and a resolution of 'res' arcminute.
The HEALPIX survey mask is a disk centered on longitude 'lon' and latitude 'lat' in degree  with a given 'radius' the resolution is given by 'nside'.
For each pixellisation we simulate 2 splits with 'rms_uKarcmin_T' noise rms in temperature.
We also include a point source mask with 'source_mask_nholes' holes of size 'source_mask_radius' in arcminute
We appodize the survey mask with an apodisation 'apo_radius_degree_survey' and the point source mask with an apodisation 'apo_radius_degree_mask'
We compute the spectra between the 2 splits of data up to 'lmax'.
"""
import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
import numpy as np
import healpy as hp
import pylab as plt
import os

#We start by specifying the CAR survey parameters, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
# It will have a resolution of 'res' arcminute
ra0, ra1, dec0, dec1 = -25, 25, -25, 25
res = 1
#We then specify the HEALPIX survey parameter, it will be a disk of radius 'radius' degree centered on longitude 'lon' degree and latitude 'lat' degree
# It will have a resolution 'nside'
lon, lat = 30, 50
radius = 25
nside = 1024
# ncomp = 1 mean that we are going to use only spin0 field (ncomp = 3 for spin 0 and 2 fields
ncomp = 1
# clfile are the camb lensed power spectra
clfile = "./data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
# n_splits stand for the number of splits we want to simulate
n_splits = 2
# the maximum multipole to consider
lmax = 1000
# the number of iteration in map2alm
niter = 0
# the noise on the spin0 component, if not specified, the noise in polarisation wil be sqrt(2)x that
rms_uKarcmin_T = 20
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 1
# the number of holes in the point source mask
source_mask_nholes = 100
# the radius of the holes (in arcminutes)
source_mask_radius = 10
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask = 0.3
# if you want to write down the data file
write_data = False

test_dir = "result_spectra_spin0"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=40, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir

# the templates for the CMB splits
template_car = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
template_healpix = so_map.healpix_template(ncomp, nside=nside)

# the binary template for the window functionpixels
# for CAR we set pixels inside the survey at 1 and  at the border to be zero
binary_car = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary_car.data[:] = 0
binary_car.data[1:-1, 1:-1] = 1

# for Healpix we set pixel inside the disk at 1 and pixel outside at zero
binary_healpix = so_map.healpix_template(ncomp=1, nside=nside)
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius * np.pi / 180)
binary_healpix.data[disc] = 1

# for the CAR survey we will use an apodisation type designed for rectangle maps, and a C1 apodisation for the healpix maps
apo_type_list = ["C1", "C1"]
binary_list = [binary_car, binary_healpix]
template_list = [template_car, template_healpix]
run_name = ["run_CAR", "run_HEALPIX"]

# Ok let's loop on the CAR and HEALPIX case, starting with CAR
for run, apo_type, binary,template in zip(run_name, apo_type_list, binary_list, template_list):
    print("start %s" % run)
    
    #First let's generate a CMB realisation
    cmb = template.synfast(clfile)
    splitlist = []
    nameList = []
    
    print("generate sims")
    #let's make 2 splits out of it, each with 'rms_uKarcmin_T' rms in T and 'rms_uKarcmin_T'xsqrt(2) rms in pol
    for i in range(n_splits):
        split = cmb.copy()
        noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
        split.data += noise.data
        splitlist += [split]
        nameList += ["split_%d" % i]
        split.plot(file_name="%s/split_%d_%s"%(test_dir, i, run))
        if write_data==True:
            split.write_map("%s/split_%d_%s.fits"%(test_dir, i, run))

    print("generate window")
    #we then apodize the survey mask
    window = so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree_survey)
    #we create a point source mask
    mask = so_map.simulate_source_mask(binary, n_holes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
    #... and we apodize it
    mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree_mask)
    #the window is given by the product of the survey window and the mask window
    window.data *= mask.data

    if write_data == True:
        window.write_map("%s/window_%s.fits"%(test_dir, run))
    
    #let's look at it
    
    if run == "run_CAR":
        window.plot(file_name="%s/window_%s"%(test_dir, run))
    else:
        window.plot(file_name="%s/window_%s"%(test_dir, run), hp_gnomv=(lon, lat, 3500, 1))

    print("compute mcm")

    #the window is going to couple mode together, we compute a mode coupling matrix in order to undo this effect
    mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, type="Dl", niter=niter)

    if write_data == True:
        np.savetxt("%s/mbb_inv.dat" % (test_dir), mbb_inv)

    #then we compute the spherical transform of the splits X window

    print("compute alms and bin spectra")

    almList = []
    for s in splitlist:
        almList += [sph_tools.get_alms(s, window, niter, lmax)]

    Db_dict = {}
    spec_name_list = []
    for name1, alm1, c1  in zip(nameList, almList, np.arange(n_splits)):
        for name2, alm2, c2  in zip(nameList, almList, np.arange(n_splits)):
            if c1 > c2: continue
            
            #we get the unbin spectra, it has modes correlation due to the window function
            l, ps = so_spectra.get_spectra(alm1, alm2)
            spec_name = "%sx%s" % (name1, name2)
            #we bin the spectra and apply the inverse of the mode coupling matrix
            lb, Db_dict[spec_name] = so_spectra.bin_spectra(l,
                                                            ps,
                                                            binning_file,
                                                            lmax,
                                                            type="Dl",
                                                            mbb_inv=mbb_inv)
            spec_name_list += [spec_name]
            #we write the spectra in a file
            if write_data == True:
                so_spectra.write_ps("%s/spectra_%s.dat" % (test_dir, spec_name),
                                    lb,
                                    Db_dict[spec_name],
                                    type="Dl")

    print("compute spectra")
    #read a theory spectra, bin it, and compare to the data

    l, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)
    ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory["TT"])

    for spec_name in spec_name_list:
        plt.plot(lb,Db_dict[spec_name], "o", label="%s" % spec_name)
    plt.plot(lb, ps_theory_b, color="lightblue", label="binned theory")
    plt.plot(l, ps_theory["TT"], color="grey", label="theory")
    plt.ylabel(r"$D_{\ell}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.legend()
    plt.savefig("%s/spectra_%s.png" % (test_dir, run))
    plt.clf()
    plt.close()

