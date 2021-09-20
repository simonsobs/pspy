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
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    l = np.arange(lmin, lmax)
    ps = {spec: powers["total"][l][:, count] for count, spec in enumerate(["TT", "EE", "BB", "TE" ])}
    ps["ET"] = ps["TE"]
    for spec in ["TB", "BT", "EB", "BE" ]:
        ps[spec] = ps["TT"] * 0
    
    scale = l * (l + 1) / (2 * np.pi)
    if output_type == "Cl":
        if start_at_zero:
            ps[2:] /= scale[2:]
        else:
            ps[:] /= scale[:]

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

    lth = np.arange(2, lmax + 2)
    nl_th = {}
    if spectra is None:
        nl_th["TT"] = (
            np.ones(lmax) * (rms_uKarcmin_T * np.pi / (60 * 180)) ** 2 / bl[2 : lmax + 2] ** 2
        )
        if type == "Dl":
            nl_th["TT"] *= lth * (lth + 1) / (2 * np.pi)
        return nl_th
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
                nl_th[spec] *= lth * (lth + 1) / (2 * np.pi)
    return nl_th


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


