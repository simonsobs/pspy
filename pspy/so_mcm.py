"""
Routines for mode coupling calculation. For more details on computation of the matrix see
https://pspy.readthedocs.io/en/latest/scientific_doc.pdf.
"""

from copy import deepcopy

import healpy as hp
import numpy as np
from scipy import sparse

from pspy import pspy_utils, so_cov, sph_tools
from ._mcm_fortran import mcm_compute as mcm_fortran

import ducc0
from pixell import curvedsky

def ducc_couplings(spec, lmax, spec_index=(0, 1, 2, 3),
                   mat_index=(0, 1, 2, 3, 4), nthread=0, res=None,
                   dtype=np.float64, coupling=False, pspy_index_convention=True):
    """Return coupling matrices calculated by ducc. This function is a 
    wrapper around ducc0.misc.experimental.coupling_matrix_spin0and2_tri.

    See https://mtr.pages.mpcdf.de/ducc/misc.html#module-ducc0.misc.experimental.

    Parameters
    ----------
    spec : numpy.ndarray ((nspec, 1<=x<=4, lmax_spec+1), dtype=np.float64)
        The window cross-spectra.
    lmax : int
        The lmax of the matrices. If spectra are zero-padded so that they are
        size 2*lmax.
    spec_index : tuple of int, length 4, optional
        See ducc docs, by default (0, 1, 2, 3).
    mat_index : tuple, optional
        See ducc docs, by default (0, 1, 2, 3, 4)
    nthread : int, optional
        The number of threads to use, by default 0. If 0. the output of
        ducc0.misc.thread_pool_size() (e.g., OMP_NUM_THREADS).
    res : numpy.ndarray ((nspec, 1<=x<=5, ((lmax+1)*(lmax+2))/2),
          dtype=np.float32 or np.float64), optional
        Output buffer, by default None. Allocated if None.
    dtype : np.dtype, optional
        Output dtype, by default np.float64. Either np.float64 or np.float32.
    coupling : bool, optional
        If True, return the raw coupling; if False, return the mode-coupling 
        matrix, by default False. The difference is just a factor of 2l+1
        multiplied along the columns.
    pspy_index_convention : bool, optional
        If True, slice out 2:lmax along each axis, by default True.

    Returns
    -------
    numpy.ndarray ((nspec, 1<=x<=5, nl, nl) np.ndarray
        The mode-coupling matrices or coupling matrices.
    """
    if nthread == 0:
        nthread = ducc0.misc.thread_pool_size()
    if np.issubdtype(dtype, np.float64):
        singleprec = False 
    if np.issubdtype(dtype, np.float32):
        singleprec = True

    mcm = ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec, lmax, spec_index, mat_index,
                                                                nthreads=nthread, res=res,
                                                                singleprec=singleprec)
    ainfo_tri = curvedsky.alm_info(lmax=lmax, layout='tri')
    ainfo_rect = curvedsky.alm_info(lmax=lmax, layout='rect')
    mcm = curvedsky.transfer_alm(ainfo_tri, mcm, ainfo_rect).reshape(*mcm.shape[:-1], lmax+1, lmax+1)

    # have C-contiguous array in upper triangle, so use fill_lower instead of fill_upper
    for idx in np.ndindex(mcm.shape[:-2]):
        if singleprec:
            mcm_fortran.fill_lower_single(mcm[idx].T)
        else:
            mcm_fortran.fill_lower(mcm[idx].T)

    if not coupling:
        mcm *= (2 * np.arange(lmax + 1) + 1)

    if pspy_index_convention: # TODO: reconsider this convention
        mcm = mcm[..., 2:lmax, 2:lmax]

    return mcm

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

def invert_mcm(mcm_array):
    """Get the blocks of the inverse of a spinxspin array from its blocks.

    Parameters
    ----------
    mcm_array : (5, nel, nel) np.ndarray
        The blocks of the spinxspin matrix: 0x0, 0x2, 2x0, ++, --

    Returns
    -------
    (5, nel, nel) np.ndarray
        The blocks of the inverse: inv(0x0), inv(0x2), inv(2x0). The inverse
        of the ++, -- blocks are not separable, so the fourth and fifth 
        returned arrays are the ++ and -- blocks of the joint ++-- inverse.
    """
    nmat, nel1, nel2 = mcm_array.shape

    assert nmat == 5, f'expected nmat=5, got {nmat=}'
    assert nel1 == nel2, f'must have square mats, got {nel1=} and {nel2=}'

    inverse_mcm_array = np.zeros_like(mcm_array)
    
    # assume 00, 02, 20, ++, -- ordering
    inverse_mcm_array[:3] = np.linalg.inv(mcm_array[:3])

    # do the 2x2 parts manually to avoid wasted computation and numerical errors
    pp, mm = mcm_array[3:]
    pp_inv = np.linalg.inv(pp)

    X = np.linalg.inv(pp - mm @ pp_inv @ mm)
    Y = -mm @ pp_inv 
    
    inverse_mcm_array[3] = X
    inverse_mcm_array[4] = X @ Y

    return inverse_mcm_array

def get_spec2spec_array_from_spin2spin_array(spin2spin_array, dense=True,
                                             spin0=False):
    """Get a full 9x9 spectrum-to-spectrum matrix from 5 (or 1) blocks of
    a spinxspin array.

    Parameters
    ----------
    spin2spin_array : ({5}, x, y) np.ndarray
        The blocks of the spinxspin matrix: 0x0, 0x2, 2x0, ++, --. If spin0 is
        True, then this is just the (x, y)-shaped 0x0 block.
    dense : bool, optional
        If False, return a scipy.sparse.bsr_array. If True, a dense array
        is returned. By default True
    spin0 : bool, optional
        If True, then spin2spin_array is just the 0x0 block, by default False.
        In that case, it is copied along the block-diagonal for all the 9
        spectra blocks.

    Returns
    -------
    scipy.sparse.bsr_array or np.ndarray
        The full 9x9 block matrix.
    """
    # for each row 0-8, indptr gives the indexes into the indices and data
    # arrays for the placement, and data, of blocks respectively.
    # NOTE: ordering assumes TT, TE, TB, ET, BT, EE, EB, BE, BB order
    if spin0:
        obj = spin2spin_array.squeeze()
        assert obj.ndim == 2, \
            f'If spin0, obj must be squeezable to a 2d array'
        indptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        obj_data = np.tile(obj, (9, 1, 1))
    else:
        obj = spin2spin_array
        assert obj.ndim == 3 and obj.shape[0] == 5, \
            f'If not spin0, obj must be a 3d array whose first axis has size 5'
        indptr = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13] 
        indices = [0, 1, 2, 3, 4, 5, 8, 6, 7, 6, 7, 5, 8]
        obj_data = np.array([obj[0],           # TT (0x0)
                             obj[1],           # TE (0x2)
                             obj[1],           # TB (0x2)
                             obj[2],           # ET (2x0)
                             obj[2],           # BT (2x0)
                             obj[3], obj[4],   # EE2EE BB2EE (pp, mm)
                             obj[3], -obj[4],  # EB2EB BE2EB (pp, -mm)
                             -obj[4], obj[3],  # BE2EB BE2BE (-mm, pp)
                             obj[4], obj[3]])  # EE2BB BB2BB (mm, pp)

    blocksize = obj_data[0].shape
    shape = tuple(9 * x for x in blocksize)
    out = sparse.bsr_array((obj_data, indices, indptr), shape=shape, blocksize=blocksize)
    if dense:
        out = out.toarray()
    return out

def get_coupling_dict(array):
    """Turns an array of spin-dependent mode-coupling matrices into a dict 
    labeled by their spin, i.e., spin0xspin0, spin0xspin2, spin2xspin0, 
    spin2xspin2. For spin2xspin2, it makes the full 4x4 nblock part of the 
    mode-coupling matrix, assuming EE, EB, BE, BB ordering. 

    Parameters
    ----------
    array : (5, dim1, dim2) np.ndarray
        The 0x0, 0x2, 2x0, ++, and -- matrix blocks.

    Returns
    -------
    dict
        A dictionary representation of the input array, with the full 4x4
        PP->PP block realized.
    """
    dim1, dim2 = array.shape[-2:]
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
