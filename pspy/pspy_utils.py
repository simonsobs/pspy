"""
Utils for pspy.
"""
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
    ell, ps["TT"], ps["EE"], ps["BB"], ps["TE"] = np.loadtxt(filename, unpack=True)
    ps["ET"] = ps["TE"].copy()
    ps["TB"], ps["BT"], ps["EB"], ps["BE"] = np.zeros((4, len(ell)))

    if lmax is not None:
        ell = ell[:lmax]
    for field in fields:
        if lmax is not None:
            ps[field] = ps[field][:lmax]
        if output_type == "Cl":
            scale = ell * (ell + 1) / (2 * np.pi)
            ps[field] /= scale
        if start_at_zero:
            ps[field] = np.append(np.array([0, 0]), ps[field])
    if start_at_zero:
        ell = np.append(np.array([0, 1]), ell)
    return ell, ps


def get_nlth_dict(rms_uKarcmin_T,
                  ps_type,
                  lmax,
                  spectra=None,
                  rms_uKarcmin_pol=None,
                  beamfile=None):
    """ Return the effective noise power spectrum Nl/bl^2 given a beam file and a noise rms

    Parameters
    ----------
    rms_uKarcmin_T: float
      the temperature noise rms in uK.arcmin
    ps_type: string
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
        ell, bl = np.loadtxt(beamfile, unpack=True)
    else:
        bl = np.ones(lmax + 2)

    lth = np.arange(2, lmax + 2)
    nl_th = {}
    if spectra is None:
        nl_th["TT"] = np.ones(lmax) * (rms_uKarcmin_T * np.pi / (60 * 180))**2 / bl[2:lmax + 2]**2
        if ps_type == "Dl":
            nl_th["TT"] *= lth * (lth + 1) / (2 * np.pi)
    else:
        if rms_uKarcmin_pol is None:
            rms_uKarcmin_pol = rms_uKarcmin_T * np.sqrt(2)
        for spec in spectra:
            nl_th[spec] = np.zeros(lmax)
        nl_th["TT"] = np.ones(lmax) * (rms_uKarcmin_T * np.pi / (60 * 180))**2 / bl[:lmax]**2
        nl_th["EE"] = np.ones(lmax) * (rms_uKarcmin_pol * np.pi / (60 * 180))**2 / bl[:lmax]**2
        nl_th["BB"] = np.ones(lmax) * (rms_uKarcmin_pol * np.pi / (60 * 180))**2 / bl[:lmax]**2
        if ps_type == "Dl":
            for spec in spectra:
                nl_th[spec] *= lth * (lth + 1) / (2 * np.pi)
    return nl_th


def create_binning_file(bin_size, n_bins, lmax=None, file_name=None):
    """ Create a (constant) binning file, and optionnaly write it to disk

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
        idx = np.where(bin_hi < lmax)
        bin_low, bin_hi, bin_cent = bin_low[idx], bin_hi[idx], bin_cent[idx]

    if file_name is not None:
        output = open("%s" % file_name, mode="w")
        for i in range(n_bins):
            output.write("%0.2f %0.2f %0.2f\n" % (bin_low[i], bin_hi[i], bin_cent[i]))
        output.close()
    return bin_low, bin_hi, bin_cent


def read_binning_file(binning_file, lmax):
    """Read a binningFile and truncate it to lmax, if bin_low lower than 2, set it to 2.
    format is bin_low, bin_high, bin_mean

    Parameters
    ----------
    binning_file: string
      the name of the binning file
    lmax: integer
      the maximum multipole to consider
    """

    bin_low, bin_hi, bin_cent = np.loadtxt(binning_file, unpack=True)
    idx = np.where(bin_hi < lmax)
    bin_low, bin_hi, bin_cent = bin_low[idx], bin_hi[idx], bin_cent[idx]
    if bin_low[0] < 2:
        bin_low[0] = 2
    bin_hi = bin_hi.astype(np.int)
    bin_low = bin_low.astype(np.int)
    bin_size = bin_hi - bin_low + 1
    return bin_low, bin_hi, bin_cent, bin_size


def naive_binning(ell, fcn, binning_file, lmax):
    """bin a function of l given a binning file and lmax

    Parameters
    ----------
    ell: 1d integer array
      the multipoles
    fcn: 1d float array
      the 1-dimensional function to bin
    binning_file: string
      the name of the binning file
    lmax: integer
      the maximum multipole to consider

    """

    bin_lo, bin_hi, bin_c, bin_size = read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    fcn_bin = np.zeros(len(bin_c))
    for ibin in range(n_bins):
        loc = np.where((ell >= bin_lo[ibin]) & (ell <= bin_hi[ibin]))
        fcn_bin[ibin] = (fcn[loc]).mean()
    return bin_c, fcn_bin
