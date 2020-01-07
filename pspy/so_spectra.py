"""
Routines for power spectra estimation and debiasing.
"""
import healpy as hp
import numpy as np

from pspy import pspy_utils, so_mcm


def get_spectra(alm1, alm2=None, spectra=None):
    """Get the power spectrum of alm1 and alm2, we use healpy.alm2cl for doing this.
    for the spin0 and spin2 case it is a bit ugly as we have to deal with healpix convention.
    Our  convention for spectra is:  ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    while healpix convention is to take alm1,alm2 and return ['TT','EE','BB','TE','EB','TB']

    Parameters
    ----------
    alm1: 1d array
      the spherical harmonic transform of map1
    alm2: 1d array
      the spherical harmonic transform of map2
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra

    Returns
    -------

    The function returns the multipole array l and cl a 1d power spectrum array or
    cl_dict (for spin0 and spin2) a dictionnary of cl with entry spectra
    """

    if spectra is None:
        if alm2 is None:
            cls = hp.alm2cl(alm1)
        else:
            cls = hp.alm2cl(alm1, alm2)
        ell = np.arange(len(cls))
        return ell, cls

    cls = hp.alm2cl(alm1, alm2)
    ell = np.arange(len(cls[0]))
    # spectra_healpix=[TT,EE,BB,TE,EB,TB]
    spectra_healpix = [spectra[0], spectra[5], spectra[8], spectra[1], spectra[6], spectra[2]]
    cl_dict = {spec: cls[i] for i, spec in enumerate(spectra_healpix)}

    if alm2 is None:
        #here we set ET=TE, BE=EB and BT=TB
        cl_dict[spectra[3]] = cl_dict[spectra[1]]
        cl_dict[spectra[7]] = cl_dict[spectra[6]]
        cl_dict[spectra[4]] = cl_dict[spectra[2]]
    else:
        #here we need to recompute cls inverting the order of the alm to get ET,BT and BE
        cls = hp.alm2cl(alm2, alm1)
        # spectra_healpix=[TT,EE,BB,ET,BE,BT]
        spectra_healpix = [spectra[0], spectra[5], spectra[8], spectra[3], spectra[7], spectra[4]]
        for i, spec in enumerate(spectra_healpix):
            cl_dict[spec] = cls[i]

    return ell, cl_dict


def bin_spectra(ell, cl, binning_file, lmax, ps_type, spectra=None, mbb_inv=None, mcm_inv=None):
    """Bin the power spectra according to a binning file and optionnaly deconvolve the mode coupling
       matrix

    Parameters
    ----------
    l: 1d array
      the multipoles
    cl: 1d array or dict of 1d array
      the power spectra to bin, can be a 1d array (spin0) or a dictionnary (spin0 and spin2)
    binning_file: data file
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    ps_type: string
      the type of binning, either bin Cl or bin Dl
    spectra: list of string
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    mbb_inv: 2d array
      the inverse of the binned mode coupling matrix to debiais the spectra
    mcm_inv: 2d array
      the inverse of the raw mode coupling matrix to debiais the spectra
      note that you have to choose between debiasing with the binned or the
      raw matrix

    Returns
    -------

    The function return the binned multipole array bin_c and a 1d power spectrum
    array (or dictionnary of 1d power spectra if spectra is not None).

    """

    #You don't want to deconvolve the binned mcm and the unbinned mcm
    #You have to choose one of the other
    if mbb_inv is not None and mcm_inv is not None:
        raise ValueError("You have to choose between binned or raw mcm")

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    if ps_type == "Dl":
        fac = (ell * (ell + 1) / (2 * np.pi))
    if ps_type == "Cl":
        fac = ell * 0 + 1

    if spectra is None:
        if mcm_inv is not None:
            cl = np.dot(mcm_inv, cl)
        binned_power = np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((ell >= bin_lo[ibin]) & (ell <= bin_hi[ibin]))
            binned_power[ibin] = (cl[loc] * fac[loc]).mean()
        if mbb_inv is None:
            return bin_c, binned_power

        return bin_c, np.dot(mbb_inv, binned_power)

    if mcm_inv is not None:
        unbin_vec = []
        mcm_inv = so_mcm.coupling_dict_to_array(mcm_inv)
        for spec in spectra:
            unbin_vec = np.append(unbin_vec, cl[spec][2:lmax])
        cl = vec2spec_dict(lmax - 2, np.dot(mcm_inv, unbin_vec), spectra)
        ell = np.arange(2, lmax)

    vec = []
    for spec in spectra:
        binned_power = np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((ell >= bin_lo[ibin]) & (ell <= bin_hi[ibin]))
            binned_power[ibin] = (cl[spec][loc] * fac[loc]).mean()
        vec = np.append(vec, binned_power)

    if mbb_inv is None:
        return bin_c, vec2spec_dict(n_bins, vec, spectra)

    mbb_inv = so_mcm.coupling_dict_to_array(mbb_inv)
    return bin_c, vec2spec_dict(n_bins, np.dot(mbb_inv, vec), spectra)


def vec2spec_dict(n_bins, vec, spectra):
    """Take a vector of power spectra and return a power spectra dictionnary.
    vec should be of the form [spectra[0], spectra[1], ... ].
    For example [cl_TT,cl_TE,cl_ET, ...., cl_BB]

    Parameters
    ----------
    n_bins: int
      the number of bins per spectrum
    vec: 1d array
      an array containing a vector of spectra
    spectra: list of strings
      the arrangement of the spectra for example:
      ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    """

    return {spec: vec[i * n_bins:(i + 1) * n_bins] for i, spec in enumerate(spectra)}


def write_ps(file_name, ell, ps, ps_type, spectra=None):
    """Write down the power spectra to disk.

    Parameters
    ----------
    file_name: str
      the name of the file to write the spectra
    ell: 1d array
      the multipoles (or binned multipoles)
    ps: 1d array or dict of 1d array
      the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    ps_type:  string
      'Cl' or 'Dl'
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    if spectra is None:
        ps_list = [ps]
        ps_list[0:0] = [ell]
        ps_list = np.array(ps_list)
        header = "l %s" % ps_type
        np.savetxt(file_name, np.transpose(ps_list), header=header)
    else:
        ps_list = [ps[spec] for spec in spectra]
        ps_list[0:0] = [ell]
        header = "l"
        for spec in spectra:
            header += " %s_%s" % (ps_type, spec)
        ps_list = np.array(ps_list)
        np.savetxt(file_name, np.transpose(ps_list), header=header)


def read_ps(file_name, spectra=None):
    """Read the power spectra.

    Parameters
    ----------
    file_name: str
      the name of the file to read the spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra

    Returns
    -------

    The function return the multipole l (or binned multipoles) and a 1d power spectrum
    array or a dictionnary of power spectra (if spectra is not None).
    """

    data = np.loadtxt(file_name)
    if spectra is None:
        return data[:, 0], data[:, 1]

    ps = {spec: data[:, i + 1] for i, spec in enumerate(spectra)}
    ell = data[:, 0]
    return ell, ps


def write_ps_hdf5(file, spec_name, ell, ps, spectra=None):
    """Write down the power spectra in a hdf5 file.

    Parameters
    ----------
    file: hdf5
      the name of the hdf5 file
    spec_name: string
      the name of the group in the hdf5 file
    ell: 1d array
      the multipoles (or binned multipoles)
    ps: 1d array or dict of 1d array
      the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """
    def array_from_dict(ell, ps, spectra=None):
        array = [ell]
        if spectra is None:
            array += [ps]
        else:
            for spec in spectra:
                array += [ps[spec]]
        return array

    group = file.create_group(spec_name)
    array = array_from_dict(ell, ps, spectra=spectra)
    group.create_dataset(name="data", data=array, dtype="float")


def read_ps_hdf5(file, spec_name, spectra=None):
    """Read the power spectra in a hdf5 file.

    Parameters
    ----------
    file: hdf5
      the name of the hdf5 file
    spec_name: string
      the name of the group in the hdf5 file
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra

    Returns
    -------

    The function returns the multipole l (or binned multipoles) and a 1d power spectrum
    array (or a dictionnary of power spectra if spectra is not None).

    """

    spec = file[spec_name]
    data = np.array(spec["data"]).T

    ell = data[:, 0]
    if spectra is None:
        ps = data[:, 1]
    else:
        ps = {spec: data[:, i + 1] for i, spec in enumerate(spectra)}

    return ell, ps
