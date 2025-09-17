"""
Routines for mode coupling calculation. For more details on computation of the matrix see
https://pspy.readthedocs.io/en/latest/scientific_doc.pdf.
"""

from copy import deepcopy

import healpy as hp
import numpy as np

from pspy import pspy_utils, so_cov, sph_tools
from ._mcm_fortran import mcm_compute as mcm_fortran

def mcm_and_bbl_spin0(win1,
                      binning_file,
                      lmax,
                      niter,
                      type="Dl",
                      win2=None,
                      bl1=None,
                      bl2=None,
                      input_alm=False,
                      binned_mcm=True,
                      save_file=None,
                      l_exact=None,
                      l_toep=None,
                      l_band=None,
                      l3_pad=2000,
                      return_coupling_only=False):


    """Get the mode coupling matrix and the binning matrix for spin0 fields

     Parameters
     ----------
     win1: so_map (or alm)
       the window function of survey 1, if input_alm=True, expect wlm1
     binning_file: text file
       a binning file with three columns bin low, bin high, bin mean
     lmax: integer
       the maximum multipole to consider for the spectra computation
     type: string
       the type of binning, either bin Cl or bin Dl
     win2: so_map (or alm)
       the window function of survey 2, if input_alm=True, expect wlm2
     bl1: 1d array
       the beam of survey 1, expected to start at l=0
     bl2: 1d array
       the beam of survey 2, expected to start at l=0
     niter: int
       specify the number of iteration in map2alm
     binned_mcm: boolean
       wether to bin or not the mcm
     save_file: boolean
       save the mcm and bbl to disk
     l_toep: int
     l_band: int
     l_exact: int

     """

    if type == "Dl": doDl = 1
    if type == "Cl": doDl = 0
    if type not in ["Dl", "Cl"]:
        raise ValueError("Unkown 'type' value! Must be either 'Dl' or 'Cl'")

    if input_alm == False:
        l_max_limit = win1.get_lmax_limit()
        if lmax > l_max_limit: raise ValueError("the requested lmax is too high with respect to the map pixellisation")
        maxl = np.minimum(lmax + l3_pad, l_max_limit)

        win1 = sph_tools.map2alm(win1, niter=niter, lmax=maxl)
        if win2 is not None:
            win2 = sph_tools.map2alm(win2, niter=niter, lmax=maxl)

    if win2 is None:
        wcl = hp.alm2cl(win1)
    else:
        wcl = hp.alm2cl(win1, win2)

    l = np.arange(len(wcl))
    wcl *= (2 * l + 1)

    if bl1 is None: bl1 = np.ones(len(l)+2)
    if bl2 is None: bl2 = bl1.copy()

    mcm = np.zeros((lmax, lmax))

    if l_toep is None: l_toep = lmax
    if l_band is None: l_band = lmax
    if l_exact is None: l_exact = lmax

    mcm_fortran.calc_coupling_spin0(wcl,
                                    l_exact,
                                    l_band,
                                    l_toep,
                                    mcm.T)

    if l_toep < lmax:
        mcm = format_toepliz_fortran2(mcm, l_toep, l_exact, lmax)

    mcm_fortran.fill_upper(mcm.T)

    if return_coupling_only == True:
        return mcm[:lmax - 2, :lmax - 2]

    fac = (2 * np.arange(2, lmax + 2) + 1) / (4 * np.pi) * bl1[2:lmax + 2] * bl2[2:lmax + 2]
    mcm *= fac

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    Bbl = np.zeros((n_bins, lmax))

    if binned_mcm == True:
        mbb = np.zeros((n_bins, n_bins))
        mcm_fortran.bin_mcm(mcm.T, bin_lo, bin_hi, bin_size, mbb.T, doDl)
        mcm_fortran.binning_matrix(mcm.T, bin_lo, bin_hi, bin_size, Bbl.T, doDl)
        mbb_inv = np.linalg.inv(mbb)
        Bbl = np.dot(mbb_inv, Bbl)
        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl)
        return mbb_inv, Bbl
    else:
        identity_matrix = np.eye(mcm.shape[0])
        mcm_fortran.binning_matrix(identity_matrix.T, bin_lo, bin_hi, bin_size, Bbl.T, doDl)
        mcm = mcm[:lmax - 2, :lmax - 2]
        mcm_inv = np.linalg.inv(mcm)

        if save_file is not None:
            save_coupling(save_file, mcm_inv, Bbl)
        return mcm_inv, Bbl


def mcm_and_bbl_spin0and2(win1,
                          binning_file,
                          lmax,
                          niter,
                          type="Dl",
                          win2=None,
                          bl1=None,
                          bl2=None,
                          input_alm=False,
                          pure=False,
                          binned_mcm=True,
                          save_file=None,
                          l3_pad=2000,
                          l_exact=None,
                          l_toep=None,
                          l_band=None,
                          return_coupling_only=False):

    """Get the mode coupling matrix and the binning matrix for spin 0 and 2 fields

    Parameters
    ----------

    win1: python tuple of so_map or alms (if input_alm=True)
      a python tuple (win_spin0,win_spin2) with the window functions of survey 1, if input_alm=True, expect (wlm_spin0, wlm_spin2)
    binning_file: text file
      a binning file with three columns bin low, bin high, bin mean
    lmax: integer
      the maximum multipole to consider
    type: string
      the type of binning, either bin Cl or bin Dl
    win2: python tuple of so_map or alms (if input_alm=True)
      a python tuple (win_spin0,win_spin2) with the window functions of survey 1, if input_alm=True, expect (wlm_spin0, wlm_spin2)
    bl1: python tuple of 1d array
      a python tuple (beam_spin0,beam_spin2) with the beam of survey 1, expected to start at l=0
    bl2: python tuple of 1d array
      a python tuple (beam_spin0,beam_spin2) with the beam of survey 2, expected to start at l=0
    niter: int
      specify the number of iteration in map2alm
    pureB: boolean
      do B mode purification
    binned_mcm: boolean
      wether to bin or not the mcm
    save_file: boolean
      save the mcm and bbl to disk
    l_toep: int
    l_band: int
    l_exact: int

    save_coupling: str
    """

    def get_coupling_dict(array):
        ncomp, dim1, dim2 = array.shape
        dict = {}
        dict["spin0xspin0"] = array[0, :, :]
        dict["spin0xspin2"] = array[1, :, :]
        dict["spin2xspin0"] = array[2, :, :]
        dict["spin2xspin2"] = np.zeros((4 * dim1, 4 * dim2))
        for i in range(4):
            dict["spin2xspin2"][i * dim1:(i + 1) * dim1, i * dim2:(i + 1) * dim2] = array[3, :, :]
        dict["spin2xspin2"][2 * dim1:3 * dim1, dim2:2 * dim2] = -array[4, :, :]
        dict["spin2xspin2"][dim1:2 * dim1, 2 * dim2:3 * dim2] = -array[4, :, :]
        dict["spin2xspin2"][3 * dim1:4 * dim1, :dim2] = array[4, :, :]
        dict["spin2xspin2"][:dim1, 3 * dim2:4 * dim2] = array[4, :, :]
        return dict

    if type == "Dl": doDl = 1
    if type == "Cl": doDl = 0
    if type not in ["Dl", "Cl"]:
        raise ValueError("Unkown 'type' value! Must be either 'Dl' or 'Cl'")

    if input_alm == False:
        l_max_limit = win1[0].get_lmax_limit()
        if lmax > l_max_limit: raise ValueError("the requested lmax is too high with respect to the map pixellisation")
        maxl = np.minimum(lmax + l3_pad, l_max_limit)

        win1_alm_T = sph_tools.map2alm(win1[0], niter=niter, lmax=maxl)
        win1_alm_P = sph_tools.map2alm(win1[1], niter=niter, lmax=maxl)
        win1 = (win1_alm_T, win1_alm_P)
        if win2 is not None:
            win2_alm_T = sph_tools.map2alm(win2[0], niter=niter, lmax=maxl)
            win2_alm_P = sph_tools.map2alm(win2[1], niter=niter, lmax=maxl)
            win2 = (win2_alm_T, win2_alm_P)

    if win2 is None: win2 = deepcopy(win1)
    if bl1 is None: bl1 = (np.ones(2 + lmax), np.ones(2 + lmax))
    if bl2 is None: bl2 = deepcopy(bl1)

    wcl, wbl = {}, {}
    spins = ["spin0", "spin2"]
    spin_pairs = []

    for i, spin1 in enumerate(spins):
        for j, spin2 in enumerate(spins):
            wcl["%sx%s" % (spin1, spin2)] = hp.alm2cl(win1[i], win2[j])
            wcl_lmax = len(wcl["%sx%s" % (spin1, spin2)])
            wcl["%sx%s" % (spin1, spin2)] *= (2 * np.arange(wcl_lmax) + 1)
            wbl["%sx%s" % (spin1, spin2)] = bl1[i][2:lmax + 2] * bl2[j][2:lmax + 2]
            spin_pairs += ["%sx%s" % (spin1, spin2)]

    mcm = np.zeros((5, lmax, lmax))

    if pure == False:
        if l_toep is None: l_toep = lmax
        if l_band is None: l_band = lmax
        if l_exact is None: l_exact = lmax

        mcm_fortran.calc_coupling_spin0and2(wcl["spin0xspin0"],
                                            wcl["spin0xspin2"],
                                            wcl["spin2xspin0"],
                                            wcl["spin2xspin2"],
                                            l_exact,
                                            l_band,
                                            l_toep,
                                            mcm.T)

        for id_mcm in range(5):
            if l_toep < lmax:
                mcm[id_mcm] = format_toepliz_fortran2(mcm[id_mcm], l_toep, l_exact, lmax)
            mcm_fortran.fill_upper(mcm[id_mcm].T)
    else:
        mcm_fortran.calc_mcm_spin0and2_pure(wcl["spin0xspin0"],
                                            wcl["spin0xspin2"],
                                            wcl["spin2xspin0"],
                                            wcl["spin2xspin2"],
                                            mcm.T)

    if return_coupling_only == True:
        return mcm[:, :lmax - 2, :lmax - 2]

    for id_mcm, spairs in enumerate(["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2", "spin2xspin2"]):
        fac = (2 * np.arange(2, lmax + 2) + 1) / (4 * np.pi) *  wbl[spairs]
        mcm[id_mcm] *= fac

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    if binned_mcm == True:
        mbb_array = np.zeros((5, n_bins, n_bins))
        Bbl_array = np.zeros((5, n_bins, lmax))

        for id_mcm in range(5):

            mcm_fortran.bin_mcm((mcm[id_mcm, :, :]).T,
                                 bin_lo,
                                 bin_hi,
                                 bin_size,
                                 (mbb_array[id_mcm, :, :]).T,
                                 doDl)

            mcm_fortran.binning_matrix((mcm[id_mcm, :, :]).T,
                                        bin_lo,
                                        bin_hi,
                                        bin_size,
                                        (Bbl_array[id_mcm, :, :]).T,
                                        doDl)


        mbb = get_coupling_dict(mbb_array)
        Bbl = get_coupling_dict(Bbl_array)

        mbb_inv = {}
        for s in spin_pairs:
            mbb_inv[s] = np.linalg.inv(mbb[s])
            Bbl[s] = np.dot(mbb_inv[s], Bbl[s])

        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl, spin_pairs=spin_pairs)
        return mbb_inv, Bbl

    else:
        Bbl_array = np.zeros((n_bins, lmax))
        identity_matrix = np.eye(mcm[0].shape[0])
        mcm_fortran.binning_matrix(identity_matrix.T, bin_lo, bin_hi, bin_size, Bbl_array.T, doDl)

        mcm = get_coupling_dict(mcm[:, :lmax - 2, :lmax - 2])
        mcm_inv = {}
        for s in spin_pairs:
            mcm_inv[s] = np.linalg.inv(mcm[s])


        # if we deconvolve the full mcm, the binning matrix is the same for all spectra
        Bbl = {}
        Bbl["spin0xspin0"] = Bbl_array
        Bbl["spin0xspin2"] = Bbl_array
        Bbl["spin2xspin0"] = Bbl_array
        Bbl["spin2xspin2"] = np.zeros((4 * n_bins, 4 * lmax))
        for i in range(4):
            Bbl["spin2xspin2"][i * n_bins:(i + 1) * n_bins, i * lmax:(i + 1) * lmax] = Bbl_array
        if save_file is not None:
            save_coupling(save_file, mcm_inv, Bbl, spin_pairs=spin_pairs)

        return mcm_inv, Bbl


def format_toepliz_fortran(coupling, l_toep, lmax):
    """Take a matrix and apply the toepliz appoximation (fortran)

    Parameters
    ----------

    coupling: array
      consist of an array where the upper part is the exact matrix and
      the lower part is the diagonal. We will feed the off diagonal
      of the lower part using the measurement of the correlation from the exact computatio
    l_toep: integer
      the l at which we start the approx
    lmax: integer
      the maximum multipole of the array
    """
    toepliz_array = np.zeros(coupling.shape)
    mcm_fortran.toepliz_array_fortran(toepliz_array.T, coupling.T, l_toep)
    toepliz_array[coupling != 0] = coupling[coupling != 0]
    return toepliz_array

def format_toepliz_fortran2(coupling, l_toep, l_exact, lmax):
    """Take a matrix and apply the toepliz appoximation (fortran)

    Parameters
    ----------

    coupling: array
      consist of an array where the upper part is the exact matrix and
      the lower part is the diagonal. We will feed the off diagonal
      of the lower part using the measurement of the correlation from the exact computatio
    l_toep: integer
      the l at which we start the approx
    l_exact: integer
      the l until which we do the exact computation
    lmax: integer
      the maximum multipole of the array
    """
    toepliz_array = np.zeros(coupling.shape)
    mcm_fortran.toepliz_array_fortran2(toepliz_array.T, coupling.T, l_toep, l_exact)
    toepliz_array[coupling != 0] = coupling[coupling != 0]
    return toepliz_array


def coupling_dict_to_array(dict):
    """Take a mcm or Bbl dictionnary with entries:

    -  (spin0xspin0), (spin0xspin2), (spin2xspin0), (spin2xspin2)

    and return a 9 x dim1, 9 x dim2 array.
    dim1 and dim2 are the dimensions of the spin0 object
    """

    dim1, dim2 = dict["spin0xspin0"].shape
    array = np.zeros((9 * dim1, 9 * dim2))
    array[0:dim1, 0:dim2] = dict["spin0xspin0"]
    array[dim1:2 * dim1, dim2:2 * dim2] = dict["spin0xspin2"]
    array[2 * dim1:3 * dim1, 2 * dim2:3 * dim2] = dict["spin0xspin2"]
    array[3 * dim1:4 * dim1, 3 * dim2:4 * dim2] = dict["spin2xspin0"]
    array[4 * dim1:5 * dim1, 4 * dim2:5 * dim2] = dict["spin2xspin0"]
    array[5 * dim1:9 * dim1, 5 * dim2:9 * dim2] = dict["spin2xspin2"]

    return array

def apply_Bbl(Bbl, ps, spectra=None):
    """Bin theoretical power spectra

    Parameters
    ----------

    Bbl: 2d array (or dict of 2d array)
        a binning matrix, if spectra is not None will be a Bbl dictionnary for spin0 and 2 fields,
        otherwise a (n_bins,lmax) matrix
    ps: 1d array or dict of 1d array
      theory ps, if spectra is not None it will be a ps dictionnary, otherwise a (lmax) vector
    spectra: list of string
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    if spectra is not None:
        Bbl_array = coupling_dict_to_array(Bbl)
        ps_vec = ps[spectra[0]]
        for spec in spectra[1:]:
            ps_vec = np.append(ps_vec, ps[spec])
        ps_b = np.dot(Bbl_array, ps_vec)
        n_bins = int(Bbl_array.shape[0] / 9)
        ps_th = {}
        for i, spec in enumerate(spectra):
            ps_th[spec] = ps_b[i * n_bins:(i + 1) * n_bins]
    else:
        ps_th = np.dot(Bbl, ps)
    return ps_th

def save_coupling(prefix, mode_coupling_inv, Bbl, spin_pairs=None):
    """Save the inverse of the mode coupling matrix and the binning matrix in npy format

    Parameters
    ----------

    prefix: string
      the prefix for the name of the file
    mode_coupling_inv: 2d array (or dict of 2d array)
      the inverse of the mode coupling matrix, if spin pairs is not none, should be a dictionnary with entries
      -  spin0xspin0, spin0xspin2, spin2xspin0, spin2xspin2
    Bbl: 2d array (or dict of 2d array)
      the binning matrix,if spin pairs is not none, should be a dictionnary with entries
      -  spin0xspin0, spin0xspin2, spin2xspin0, spin2xspin2
      otherwise, it will be a single matrix.
    spin_pairs: list of strings
      needed for spin0 and 2 fields.
    """

    if spin_pairs is not None:
        for spin in spin_pairs:
            np.save(prefix + "_mode_coupling_inv_%s.npy" % spin, mode_coupling_inv[spin])
            np.save(prefix + "_Bbl_%s.npy" % spin, Bbl[spin])
    else:
        np.save(prefix + "_mode_coupling_inv.npy", mode_coupling_inv)
        np.save(prefix + "_Bbl.npy", Bbl)



def read_coupling(prefix, spin_pairs=None):
    """Read the inverse of the mode coupling matrix and the binning matrix
    Parameters
    ----------

    prefix: string
      the prefix for the name of the file
    spin_pairs: list of strings
      needed for spin0 and 2 fields.
    unbin: boolean
      also read the unbin matrix
    """

    if spin_pairs is not None:
        Bbl = {}
        mode_coupling_inv = {}
        for spin in spin_pairs:
            mode_coupling_inv[spin] = np.load(prefix + "_mode_coupling_inv_%s.npy" % spin)
            Bbl[spin] = np.load(prefix + "_Bbl_%s.npy" % spin)
    else:
        mode_coupling_inv = np.load(prefix + "_mode_coupling_inv.npy")
        Bbl = np.load(prefix + "_Bbl.npy")

    return mode_coupling_inv, Bbl


def coupling_block(
                mode,
                win1,
                lmax,
                niter=None,
                win2=None,
                input_alm=False,
                l_exact=None,
                l_toep=None,
                l_band=None,
                l3_pad=2000,
                legacy_ell_range=False):
 
    """Get the coupling matrix for two fields
     Parameters
     ----------
     mode: string
        one of "00", "02", "++" or "--"
     win1: so_map (or alm)
       the window function of survey 1, if input_alm=True, expect wlm1
     lmax: integer
       the maximum multipole to consider for the spectra computation
     win2: so_map (or alm)
       the window function of survey 2, if input_alm=True, expect wlm2
     niter: int
       specify the number of iteration in map2alm
     l_toep: int
     l_band: int
     l_exact: int
     """

    if input_alm == False:
        if niter is None:
            raise ValueError("niter required")
        l_max_limit = win1.get_lmax_limit()
        if lmax > l_max_limit: raise ValueError("the requested lmax is too high with respect to the map pixellisation")
        maxl = np.minimum(lmax + l3_pad, l_max_limit)

        win1 = sph_tools.map2alm(win1, niter=niter, lmax=maxl)
        if win2 is not None:
            win2 = sph_tools.map2alm(win2, niter=niter, lmax=maxl)

    if win2 is None:
        wcl = hp.alm2cl(win1)
    else:
        wcl = hp.alm2cl(win1, win2)

    l = np.arange(len(wcl))
    wcl *= (2 * l + 1)

    mcm = np.zeros((lmax + 1, lmax + 1))

    if l_toep is None: l_toep = lmax + 1
    if l_band is None: l_band = lmax + 1
    if l_exact is None: l_exact = lmax + 1

    if mode == "00":
        mcm_fortran.calc_coupling_block_spin0(wcl, l_exact, l_band, l_toep, mcm.T)
    elif mode == "02":
        mcm_fortran.calc_coupling_block_spin02(wcl, l_exact, l_band, l_toep, mcm.T)
    elif mode == "++":
        mcm_fortran.calc_coupling_block_spin2_pp(wcl, l_exact, l_band, l_toep, mcm.T)
    elif mode == "--":
        mcm_fortran.calc_coupling_block_spin2_mm(wcl, l_exact, l_band, l_toep, mcm.T)
    else:
        raise ValueError(f"mode \"{mode}\" not one of (\"00\", \"02\", \"++\", \"--\")")
      
    if l_toep < lmax:
        mcm = format_toepliz_fortran2(mcm, l_toep, l_exact, lmax)

    mcm_fortran.fill_upper(mcm.T)
    
    mcm_full = np.zeros((lmax + 1, lmax + 1))
    mcm_full[2:, 2:] = mcm[:-2, :-2]
    mcm_full[0, 0] = 1.0
    mcm_full[1, 1] = 1.0

    if legacy_ell_range:
        return mcm_full[2:-1, 2:-1]
    else:
        return mcm_full
