"""
Utils for pspy.
"""
import os

import numpy as np


def ps_lensed_theory_to_dict(filename, output_type, lmax=None, start_at_zero=False):
    """Read a lensed power spectrum from CAMB and return a dictionnary

    Parameters
    ----------
    filename : string
      the name of the CAMB lensed power spectrum you want to read
    lmax : integer
      the maximum multipole (spectra will be cut at)
    output_type :  string
      'Cl' or 'Dl'
    start_at_zero : boolean
      if True, ps start at l=0 and cl(l=0) and cl(l=1) are set to 0

    """

    fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    ps = {}
    l, ps["TT"], ps["EE"], ps["BB"], ps["TE"] = np.loadtxt(filename, unpack=True)
    ps["ET"] = ps["TE"].copy()
    ps["TB"], ps["BT"], ps["EB"], ps["BE"] = np.zeros((4, len(l)))

    if lmax is not None:
        l = l[:lmax]
    scale = l * (l + 1) / (2 * np.pi)
    for f in fields:
        if lmax is not None:
            ps[f] = ps[f][:lmax]
        if output_type == "Cl":
            ps[f] /= scale
        if start_at_zero:
            ps[f] = np.append(np.array([0, 0]), ps[f])
    if start_at_zero:
        l = np.append(np.array([0, 1]), l)
    return l, ps


def ps_from_params(cosmo_params, output_type, lmax, start_at_zero=False):

    """Given a set of cosmological parameters compute the corresponding lensed power spectrum
       You need to have camb installed to use this function
      ----------
      cosmo_params: dict
        dictionnary of cosmological parameters
        # e.g cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237, "omch2": 0.1200, "ns": 0.9649, "Alens": 1.0, "tau": 0.0544}
      output_type :  string
        'Cl' or 'Dl'
      lmax: integer
        the maximum multipole to consider
      start_at_zero : boolean
        if True, ps start at l=0 and cl(l=0) and cl(l=1) are set to 0
        else, start at l=2
    """
    try:
        import camb
    except ModuleNotFoundError:
        raise ModuleNotFoundError("you need to install camb to use this function")

    if start_at_zero:
        lmin = 0
    else:
        lmin = 2

    camb_cosmo = {k: v for k, v in cosmo_params.items() if k not in ["logA", "As"]}
    camb_cosmo.update({"As": 1e-10*np.exp(cosmo_params["logA"]), "lmax": lmax, "lens_potential_accuracy": 1})
    pars = camb.set_params(**camb_cosmo)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=output_type == "Cl")
    l = np.arange(lmin, lmax)
    ps = {spec: powers["total"][l][:, count] for count, spec in enumerate(["TT", "EE", "BB", "TE" ])}
    ps["ET"] = ps["TE"].copy()
    for spec in ["TB", "BT", "EB", "BE" ]:
        ps[spec] = ps["TT"] * 0

    return l, ps


def get_nlth_dict(rms_uKarcmin_T, type, lmax, spectra=None, rms_uKarcmin_pol=None, beamfile=None):
    """Return the effective noise power spectrum Nl/bl^2 given a beam file and a noise rms

    Parameters
    ----------
    rms_uKarcmin_T: float
      the temperature noise rms in uK.arcmin
    type: string
      'Cl' or 'Dl'
    lmax: integer
      the maximum multipole to consider
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    rms_uKarcmin_pol: float
      the polarisation noise rms in uK.arcmin
    beamfile: string
      the name of the beam transfer function (assuming it's given as a two column file l,bl)
    """

    if beamfile is not None:
        l, bl = np.loadtxt(beamfile, unpack=True)
    else:
        bl = np.ones(lmax + 2)

    l_th = np.arange(2, lmax + 2)
    nl_th = {}
    if spectra is None:
        nl_th["TT"] = (
            np.ones(lmax) * (rms_uKarcmin_T * np.pi / (60 * 180)) ** 2 / bl[2 : lmax + 2] ** 2
        )
        if type == "Dl":
            nl_th["TT"] *= l_th * (l_th + 1) / (2 * np.pi)
        return l_th, nl_th
    else:
        if rms_uKarcmin_pol is None:
            rms_uKarcmin_pol = rms_uKarcmin_T * np.sqrt(2)
        for spec in spectra:
            nl_th[spec] = np.zeros(lmax)
        nl_th["TT"] = np.ones(lmax) * (rms_uKarcmin_T * np.pi / (60 * 180)) ** 2 / bl[2 :lmax + 2] ** 2
        nl_th["EE"] = np.ones(lmax) * (rms_uKarcmin_pol * np.pi / (60 * 180)) ** 2 / bl[2 :lmax + 2] ** 2
        nl_th["BB"] = np.ones(lmax) * (rms_uKarcmin_pol * np.pi / (60 * 180)) ** 2 / bl[2 :lmax + 2] ** 2
        if type == "Dl":
            for spec in spectra:
                nl_th[spec] *= l_th * (l_th + 1) / (2 * np.pi)
    return l_th, nl_th


def read_beam_file(beamfile, lmax=None):
    """Read beam file with formal, l, bl, stuff and normalize it

    Parameters
    __________
    beamfile: string
      the name of the beam file
    lmax: integer
      the maximum multipole to consider
    """

    beam = np.loadtxt(beamfile)
    l, bl = beam[:, 0], beam[:, 1]
    if lmax is not None:
        l, bl = l[:lmax], bl[:lmax]

    return l, bl / bl[0]


def create_binning_file(bin_size, n_bins, lmax=None, file_name=None):
    """Create a (constant) binning file, and optionnaly write it to disk

    Parameters
    ----------
    bin_size: float
      the size of the bins
    n_bins: integer
      the number of  bins
    lmax: integer
      the maximum multipole to consider
    file_name: string
      the name of the binning file
    """
    bins = np.arange(n_bins)
    bin_low = bins * bin_size + 2
    bin_hi = (bins + 1) * bin_size + 1
    bin_cent = (bin_low + bin_hi) / 2

    if lmax is not None:
        id = np.where(bin_hi < lmax)
        bin_low, bin_hi, bin_cent = bin_low[id], bin_hi[id], bin_cent[id]

    if file_name is None:
        return bin_low, bin_hi, bin_cent
    else:
        f = open("%s" % file_name, mode="w")
        for i in range(len(bin_low)):
            f.write("%0.2f %0.2f %0.2f\n" % (bin_low[i], bin_hi[i], bin_cent[i]))
        f.close()


def read_binning_file(file_name, lmax):
    """Read a binningFile and truncate it to lmax, if bin_low lower than 2, set it to 2.
    format is bin_low, bin_high, bin_mean

    Parameters
    ----------
    binningfile: string
      the name of the binning file
    lmax: integer
      the maximum multipole to consider
    """
    bin_low, bin_hi, bin_cent = np.loadtxt(file_name, unpack=True)
    id = np.where(bin_hi < lmax)
    bin_low, bin_hi, bin_cent = bin_low[id], bin_hi[id], bin_cent[id]
    if bin_low[0] < 2:
        bin_low[0] = 2
    bin_hi = bin_hi.astype(int)
    bin_low = bin_low.astype(int)
    bin_size = bin_hi - bin_low + 1
    return bin_low, bin_hi, bin_cent, bin_size


def create_directory(name):
    """Create a directory

    Parameters
    ----------
    name: string
      the name of the directory
    """
    os.makedirs(name, exist_ok=True)


def naive_binning(l, fl, binning_file, lmax):
    """Bin a function of l given a binning file and lmax

    Parameters
    ----------
    l: 1d integer array
      the multipoles
    fl: 1d float array
      the 1-dimensional function to bin
    binning_file: string
      the name of the binning file
    lmax: integer
      the maximum multipole to consider

    """

    bin_low, bin_hi, bin_cent, bin_size = read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    fl_bin = np.zeros(len(bin_cent))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_low[ibin]) & (l <= bin_hi[ibin]))
        fl_bin[ibin] = (fl[loc]).mean()

    return bin_cent, fl_bin


def beam_from_fwhm(fwhm_arcminute, lmax):
    """Compute the harmonic transform of the beam
    given the beam full width half maximum in arcminute

    Parameters
    ----------
    fwhm_arcminute: float
      full width half maximum in arcminute
    lmax: integer
      the maximum multipole to consider

    """
    beam_fwhm_rad = np.deg2rad(fwhm_arcminute) / 60
    fac = beam_fwhm_rad / np.sqrt(8 * np.log(2))

    ell = np.arange(2, lmax)
    bl = np.exp(-ell * (ell + 1) * fac ** 2 / 2.0)
    return ell, bl


def create_arbitrary_binning_file(delta_l_list, l_bound_list, binning_file=None):
    """
    This function return a binning scheme with non constant bin size
    you specify a list of bin size: delta_l_list, and the corresponding bound at
    which this bin size ceases to apply: l_bound_list
    optionnaly write to disk a binning file in format, lmin, lmax, lmean

    Parameters
    ----------
    delta_l_list: list of integers
        the different binsize we want to consider
    l_bound_list: list of integers
        the multipole at which the binsize ceases to apply
    write_binning_file: string
        if you want to write to disk, pass a string with the name of the file
    """
 
    lmin_list = []
    lmax_list = []
    lmean_list = []
    
    lmin = 2
    for count, ells in enumerate(l_bound_list):
        while True:
            lmax = lmin +  delta_l_list[count]

            if lmax > l_bound_list[count]:
                break

            lmean = (lmax + lmin) / 2

            lmin_list += [lmin]
            lmax_list += [lmax]
            lmean_list += [lmean]
            
            lmin = lmax + 1
    if binning_file is not None:
        np.savetxt(binning_file, np.transpose([lmin_list, lmax_list, lmean_list]))
    
    return lmin_list, lmax_list, lmean_list

def maximum_likelihood_combination(cov_mat, P_mat, data_vec, test_matrix=False):
    """
    This function solve for the maximum likelihood data_vec and covariance for a problem of the form
    chi2 = (data_vec - P ML_data_vec).T cov_mat ** -1 data_vec - P ML_data_vec)


    Parameters
    ----------
    cov_mat: 2d array
        the covariance matrix of data_vec, of size N_ini x N_ini
    P_mat: 2d array
        the "pointing matrix" that project the final data vector of size N_f into the initial data vector space N_ini
        shape is (N_ini, N_f)
    data_vec: 1d array
        the initial data vector
    """
    
    inv_cov_mat = np.linalg.inv(cov_mat)

    ML_cov_mat = np.linalg.inv(np.dot(np.dot(P_mat, inv_cov_mat), P_mat.T))
    ML_data_vec = np.dot(ML_cov_mat, np.dot(P_mat, np.dot(inv_cov_mat, data_vec)))
    
    if test_matrix:
        is_symmetric(ML_cov_mat, tol=1e-7)
        is_pos_def(ML_cov_mat)

    return ML_cov_mat, ML_data_vec


def is_symmetric(mat, tol=1e-8):
    """
    This function check if a matrix is symmetric
    
    Parameters
    ----------
    mat: 2d array
        the matrix tested
    tol:
        the absolute tolerance on assymetry
    """
    if np.all(np.abs(mat-mat.T) < tol):
        print("the tested matrix is symmetric")
    else:
        print("the tested matrix is not symmetric")
    return


def is_pos_def(mat):
    """
    This function check if a matrix is positive definite
    
    Parameters
    ----------
    mat: 2d array
        the matrix tested
    """
    if np.all(np.linalg.eigvals(mat) > 0):
        print("the tested matrix is positive definite")
    else:
        print("the tested matrix is not positive definite")
        print(np.linalg.eigvals(mat))
    return


    
