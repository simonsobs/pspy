"""
Routines for mode coupling calculation. For more details on computation of the matrix see
https://pspy.readthedocs.io/en/latest/scientific_doc.pdf.
"""

from copy import deepcopy

import healpy as hp
import numpy as np
from pspy import pspy_utils, sph_tools
from pspy.mcm_fortran import mcm_fortran


def mcm_and_bbl_spin0(win1,
                      binning_file,
                      lmax,
                      niter,
                      type="Dl",
                      win2=None,
                      bl1=None,
                      bl2=None,
                      input_alm=False,
                      unbin=None,
                      save_file=None,
                      lmax_pad=None,
                      threshold=None):
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
    unbin: boolean
      return the unbinned mode coupling matrix
    save_file: boolean
      save the mcm and bbl to disk
    lmax_pad: integer
      the maximum multipole to consider for the mcm computation
      lmax_pad should always be greater than lmax
    threshold: integer
      for approximating the mcm computation, threshold is the max |l1-l2| consider
      in the calculation
      
    """

    if type == "Dl":
        doDl = 1
    if type == "Cl":
        doDl = 0

    maxl = lmax
    if lmax_pad is not None:
        maxl = lmax_pad

    if input_alm == False:
        win1 = sph_tools.map2alm(win1, niter=niter, lmax=maxl)
        if win2 is not None:
            win2 = sph_tools.map2alm(win2, niter=niter, lmax=maxl)

    if win2 is None:
        wcl = hp.alm2cl(win1)
    else:
        wcl = hp.alm2cl(win1, win2)

    l = np.arange(len(wcl))
    wcl *= (2 * l + 1)

    if bl1 is None:
        bl1 = np.ones(len(l)+2)
    if bl2 is None:
        bl2 = bl1.copy()
    
    mcm = np.zeros((maxl, maxl))
    
    if threshold is None:
        mcm_fortran.calc_mcm_spin0(wcl, mcm.T)
    else:
        mcm_fortran.calc_mcm_spin0_threshold(wcl, threshold, mcm.T)
        

    mcm = mcm + mcm.T - np.diag(np.diag(mcm))
    
    fac = (2 * np.arange(2, maxl + 2) + 1) / (4 * np.pi) * bl1[2:maxl + 2] * bl2[2:maxl + 2]
    mcm *= fac

    mcm = mcm[:lmax, :lmax]
    
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    mbb = np.zeros((n_bins, n_bins))
    mcm_fortran.bin_mcm(mcm.T, bin_lo, bin_hi, bin_size, mbb.T, doDl)
    
    
    Bbl = np.zeros((n_bins, lmax))
    mcm_fortran.binning_matrix(mcm.T, bin_lo, bin_hi, bin_size, Bbl.T, doDl)
    mbb_inv = np.linalg.inv(mbb)
    Bbl = np.dot(mbb_inv, Bbl)

    if unbin:
        mcm_inv = np.linalg.inv(mcm)
        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl, mcm_inv=mcm_inv)
        return mcm_inv, mbb_inv, Bbl
    else:
        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl)
        return mbb_inv, Bbl


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
                          unbin=None,
                          save_file=None,
                          lmax_pad=None,
                          threshold=None):

                          
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
    unbin: boolean
      return the unbinned mode coupling matrix
    save_file: boolean
      save the mcm and bbl to disk
    lmax_pad: integer
      the maximum multipole to consider for the mcm computation
      lmax_pad should always be greater than lmax
    threshold: integer
      for approximating the mcm computation, threshold is the max |l1-l2| consider
      in the calculation

    """
    def get_coupling_dict(array, fac=1.0):
        ncomp, dim1, dim2 = array.shape
        dict = {}
        dict["spin0xspin0"] = array[0, :, :]
        dict["spin0xspin2"] = array[1, :, :]
        dict["spin2xspin0"] = array[2, :, :]
        dict["spin2xspin2"] = np.zeros((4 * dim1, 4 * dim2))
        for i in range(4):
            dict["spin2xspin2"][i * dim1:(i + 1) * dim1, i * dim2:(i + 1) * dim2] = array[3, :, :]
        dict["spin2xspin2"][2 * dim1:3 * dim1, dim2:2 * dim2] = array[4, :, :] * fac
        dict["spin2xspin2"][dim1:2 * dim1, 2 * dim2:3 * dim2] = array[4, :, :] * fac
        dict["spin2xspin2"][3 * dim1:4 * dim1, :dim2] = array[4, :, :]
        dict["spin2xspin2"][:dim1, 3 * dim2:4 * dim2] = array[4, :, :]
        return dict

    if type == "Dl":
        doDl = 1
    if type == "Cl":
        doDl = 0

    maxl = lmax
    if lmax_pad is not None:
        maxl = lmax_pad

    if input_alm == False:
        win1 = (sph_tools.map2alm(win1[0], niter=niter,
                                  lmax=maxl), sph_tools.map2alm(win1[1], niter=niter, lmax=maxl))
        if win2 is not None:
            win2 = (sph_tools.map2alm(win2[0], niter=niter,
                                      lmax=maxl), sph_tools.map2alm(win2[1], niter=niter,
                                                                    lmax=maxl))
    if win2 is None:
        win2 = deepcopy(win1)

    if bl1 is None:
        bl1 = (np.ones(2+maxl), np.ones(2+maxl))
    if bl2 is None:
        bl2 = deepcopy(bl1)

    wcl = {}
    wbl = {}
    spins = ["0", "2"]

    for i, spin1 in enumerate(spins):
        for j, spin2 in enumerate(spins):
            wcl[spin1 + spin2] = hp.alm2cl(win1[i], win2[j])
            wcl[spin1 + spin2] *= (2 * np.arange(len(wcl[spin1 + spin2])) + 1)
            wbl[spin1 + spin2] = bl1[i] * bl2[j]

    mcm = np.zeros((5, maxl, maxl))

    if pure == False:
    
        if threshold is None:
            mcm_fortran.calc_mcm_spin0and2(wcl["00"], wcl["02"], wcl["20"], wcl["22"], mcm.T)
        else:
            mcm_fortran.calc_mcm_spin0and2_threshold(wcl["00"], wcl["02"], wcl["20"], wcl["22"], threshold, mcm.T)

        for id_mcm in range(5):
            mcm[id_mcm] = mcm[id_mcm] + mcm[id_mcm].T - np.diag(np.diag(mcm[id_mcm]))
            
    else:
        mcm_fortran.calc_mcm_spin0and2_pure(wcl["00"], wcl["02"], wcl["20"], wcl["22"], mcm.T)
        

    for id_mcm, spairs in enumerate(["00", "02", "20", "22", "22"]):
        fac = (2 * np.arange(2, maxl + 2) + 1) / (4 * np.pi) *  wbl[spairs][2:maxl + 2]
        mcm[id_mcm] *= fac

    mcm = mcm[:, :lmax, :lmax]
    
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    mbb_array = np.zeros((5, n_bins, n_bins))
    Bbl_array = np.zeros((5, n_bins, lmax))

    for i in range(5):
        mcm_fortran.bin_mcm((mcm[i, :, :]).T, bin_lo, bin_hi, bin_size, (mbb_array[i, :, :]).T,
                            doDl)
        mcm_fortran.binning_matrix((mcm[i, :, :]).T, bin_lo, bin_hi, bin_size,
                                   (Bbl_array[i, :, :]).T, doDl)

    mbb = get_coupling_dict(mbb_array, fac=-1.0)
    Bbl = get_coupling_dict(Bbl_array, fac=1.0)

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
    mbb_inv = {}
    for s in spin_pairs:
        mbb_inv[s] = np.linalg.inv(mbb[s])
        Bbl[s] = np.dot(mbb_inv[s], Bbl[s])

    if unbin:
        mcm = get_coupling_dict(mcm[:, :lmax - 2, :lmax - 2], fac=-1.0)
        mcm_inv = {}
        for s in spin_pairs:
            mcm_inv[s] = np.linalg.inv(mcm[s])

        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl, spin_pairs=spin_pairs, mcm_inv=mcm_inv)
        return mcm_inv, mbb_inv, Bbl
    else:
        if save_file is not None:
            save_coupling(save_file, mbb_inv, Bbl, spin_pairs=spin_pairs)
        return mbb_inv, Bbl


def coupling_dict_to_array(dict):
    """Take a mcm or Bbl dictionnary with entries:

    -  (spin0xspin0)

    -  (spin0xspin2)

    -  (spin2xspin0)

    -  (spin2xspin2)

    and return a 9xdim1,9xdim2 array.
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


def save_coupling(prefix, mbb_inv, Bbl, spin_pairs=None, mcm_inv=None):
    """Save the inverse of the mode coupling matrix and the binning matrix in npy format

    Parameters
    ----------

    prefix: string
      the prefix for the name of the file
    mbb_inv: 2d array (or dict of 2d array)
      the inverse of the mode coupling matrix, if spin pairs is not none, should be a dictionnary with entries

      -  spin0xspin0

      -  spin0xspin2

      -  spin2xspin0

      -  spin2xspin2

    Bbl: 2d array (or dict of 2d array)
      the binning matrix,if spin pairs is not none, should be a dictionnary with entries

      -  spin0xspin0

      -  spin0xspin2

      -  spin2xspin0

      -  spin2xspin2

      otherwise, it will be a single matrix.
    spin_pairs: list of strings
      needed for spin0 and 2 fields.
    mcm_inv: 2d array (or dict of 2d array)
      the inverse of the unbinned mode coupling matrix, if spin pairs is not none, should be a dictionnary with entries

      -  spin0xspin0

      -  spin0xspin2

      -  spin2xspin0

      -  spin2xspin2
    """

    if spin_pairs is not None:
        for spin in spin_pairs:
            np.save(prefix + "_mbb_inv_%s.npy" % spin, mbb_inv[spin])
            np.save(prefix + "_Bbl_%s.npy" % spin, Bbl[spin])
            if mcm_inv is not None:
                np.save(prefix + "_mcm_inv_%s.npy" % spin, mcm_inv[spin])
    else:
        np.save(prefix + "_mbb_inv.npy", mbb_inv)
        np.save(prefix + "_Bbl.npy", Bbl)
        if mcm_inv is not None:
            np.save(prefix + "_mcm_inv.npy", mcm_inv)


def read_coupling(prefix, spin_pairs=None, unbin=None):
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
        mbb_inv = {}
        mcm_inv = {}
        for spin in spin_pairs:
            if unbin:
                mcm_inv[spin] = np.load(prefix + "_mcm_inv_%s.npy" % spin)
            mbb_inv[spin] = np.load(prefix + "_mbb_inv_%s.npy" % spin)
            Bbl[spin] = np.load(prefix + "_Bbl_%s.npy" % spin)
    else:
        if unbin:
            mcm_inv = np.load(prefix + "_mcm_inv.npy")
        mbb_inv = np.load(prefix + "_mbb_inv.npy")
        Bbl = np.load(prefix + "_Bbl.npy")

    if unbin:
        return mcm_inv, mbb_inv, Bbl
    else:
        return mbb_inv, Bbl
