"""
Tools for analytical covariance matrix estimation. For more details on computation of the matrix see
https://pspy.readthedocs.io/en/latest/scientific_doc.pdf.
"""
import healpy as hp
import numpy as np
from pixell import colorize, enmap, enplot
from pspy import pspy_utils, so_mcm, sph_tools
from pspy.cov_fortran.cov_fortran import cov_compute as cov_fortran
from pspy.mcm_fortran.mcm_fortran import mcm_compute as mcm_fortran
import matplotlib as mpl

def cov_coupling_spin0(win, lmax, niter=3, save_file=None, l_exact=None, l_band=None, l_toep=None):
    """Compute the coupling kernels corresponding to the T only covariance matrix

    Parameters
    ----------

    win: so_map or dictionnary of so_map
      the window functions, can be a so_map or a dictionnary containing so_map
      if the later, the entry of the dictionnary should be Ta,Tb,Tc,Td.
    lmax: integer
      the maximum multipole to consider
    niter: int
      the number of iteration performed while computing the alm
    save_file: string
      the name of the file in which the coupling kernel will be saved (npy format)
    l_toep: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_band: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_exact: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    """

    coupling_dict = {}
    if l_toep is None: l_toep = lmax
    if l_band is None: l_band = lmax
    if l_exact is None: l_exact = lmax

    if type(win) is not dict:
        sq_win = win.copy()
        sq_win.data *= sq_win.data
        alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
        wcl = hp.alm2cl(alm)
        l = np.arange(len(wcl))
        wcl *= (2 * l + 1) / (4 * np.pi)
        coupling = np.zeros((lmax, lmax))


        mcm_fortran.calc_coupling_spin0(wcl,
                                       l_exact,
                                       l_band,
                                       l_toep,
                                       coupling.T)


        if l_toep < lmax:
            # For toepliz fill the matrix
            coupling = so_mcm.format_toepliz_fortran(coupling, l_toep, lmax)

        mcm_fortran.fill_upper(coupling.T)

        coupling_dict["TaTcTbTd"] = coupling[:lmax - 2, :lmax - 2]
        coupling_dict["TaTdTbTc"] = coupling[:lmax - 2, :lmax - 2]

    else:
        wcl = {}
        for s in ["TaTcTbTd", "TaTdTbTc"]:
            n0, n1, n2, n3 = [s[i * 2:(i + 1) * 2] for i in range(4)]
            sq_win_n0n1 = win[n0].copy()
            sq_win_n0n1.data *= win[n1].data
            sq_win_n2n3 = win[n2].copy()
            sq_win_n2n3.data *= win[n3].data
            alm_n0n1 = sph_tools.map2alm(sq_win_n0n1, niter=niter, lmax=lmax)
            alm_n2n3 = sph_tools.map2alm(sq_win_n2n3, niter=niter, lmax=lmax)
            wcl[n0+n1+n2+n3] = hp.alm2cl(alm_n0n1, alm_n2n3)
            l = np.arange(len(wcl[n0 + n1 + n2 + n3]))
            wcl[n0+n1+n2+n3] *= (2 * l + 1) / (4 * np.pi)

        coupling = np.zeros((2, lmax, lmax))

        cov_fortran.calc_cov_spin0(wcl["TaTcTbTd"], wcl["TaTdTbTc"], l_exact, l_band, l_toep,  coupling.T)

        for id_cov, name in enumerate(["TaTcTbTd", "TaTdTbTc"]):
            if l_toep < lmax:
                coupling[id_cov] = so_mcm.format_toepliz_fortran(coupling[id_cov], l_toep, lmax)
            mcm_fortran.fill_upper(coupling[id_cov].T)

            coupling_dict[name] = coupling[id_cov][:lmax - 2, :lmax - 2]


    if save_file is not None:
        np.save("%s.npy"%save_file, coupling)

    return coupling_dict


def cov_coupling_spin0and2_simple(win, lmax, niter=3, save_file=None, planck=False, l_exact=None, l_band=None, l_toep=None):
    """Compute the coupling kernels corresponding to the T and E covariance matrix

    Parameters
    ----------

    win: so_map or dictionnary of so_map
      the window functions, can be a so_map or a dictionnary containing so_map
      if the later, the entry of the dictionnary should be Ta,Tb,Tc,Td,Pa,Pb,Pc,Pd
    lmax: integer
      the maximum multipole to consider
    niter: int
      the number of iteration performed while computing the alm
    save_file: string
      the name of the file in which the coupling kernel will be saved (npy format)
    l_toep: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_band: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_exact: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)


    """


    win_list = ["TaTcTbTd", "TaTdTbTc", "PaPcPbPd", "PaPdPbPc",
                "TaTcPbPd", "TaPdPbTc", "PaPcTbTd", "PaTdTbPc",
                "TaPcTbPd", "TaPdTbPc", "TaTcTbPd", "TaPdTbTc",
                "TaPcTbTd", "TaTdTbPc", "TaPcPbTd", "TaTdPbPc",
                "TaPcPbPd", "TaPdPbPc", "PaPcTbPd", "PaPdTbPc",
                "TaTcPbTd", "TaTdPbTc", "PaTcTbTd", "PaTdTbTc",
                "PaTcPbTd", "PaTdPbTc", "PaTcTbPd", "PaPdTbTc",
                "PaTcPbPd", "PaPdPbTc", "PaPcPbTd", "PaTdPbPc"]

    coupling_dict = {}

    if l_toep is None: l_toep = lmax
    if l_band is None: l_band = lmax
    if l_exact is None: l_exact = lmax

    if type(win) is not dict:
        sq_win = win.copy()
        sq_win.data *= sq_win.data
        alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
        wcl = hp.alm2cl(alm)
        l = np.arange(len(wcl))
        wcl *= (2 * l + 1) / (4 * np.pi)
        coupling = np.zeros((1, lmax, lmax))


        mcm_fortran.calc_coupling_spin0(wcl,
                                       l_exact,
                                       l_band,
                                       l_toep,
                                       coupling[0].T)


        if l_toep < lmax:
            # For toepliz fill the matrix
            coupling[0] = so_mcm.format_toepliz_fortran(coupling[0], l_toep, lmax)

        indexlist=np.zeros(32, dtype=np.int8)

        for name,index in zip(win_list, indexlist):
            coupling_dict[name] = coupling[index] + coupling[index].T - np.diag(np.diag(coupling[index]))
            coupling_dict[name] = coupling_dict[name][:lmax - 2, :lmax - 2]
    else:
        wcl={}

        for s in win_list:

            n0, n1, n2, n3 = [s[i * 2:(i + 1) * 2] for i in range(4)]

            sq_win_n0n1 = win[n0].copy()
            sq_win_n0n1.data *= win[n1].data
            sq_win_n2n3 = win[n2].copy()
            sq_win_n2n3.data *= win[n3].data

            alm_n0n1 = sph_tools.map2alm(sq_win_n0n1, niter=niter, lmax=lmax)
            alm_n2n3 = sph_tools.map2alm(sq_win_n2n3, niter=niter, lmax=lmax)

            wcl[n0+n1+n2+n3] = hp.alm2cl(alm_n0n1, alm_n2n3)
            l = np.arange(len(wcl[n0+n1+n2+n3]))
            wcl[n0+n1+n2+n3] *= (2 * l + 1) / (4 * np.pi)

        coupling = np.zeros((32, lmax, lmax))

        if planck==True:
            doPlanck = 1
        else:
            doPlanck = 0


        cov_fortran.calc_cov_spin0and2_simple(wcl["TaTcTbTd"], wcl["TaTdTbTc"], wcl["PaPcPbPd"], wcl["PaPdPbPc"],
                                              wcl["TaTcPbPd"], wcl["TaPdPbTc"], wcl["PaPcTbTd"], wcl["PaTdTbPc"],
                                              wcl["TaPcTbPd"], wcl["TaPdTbPc"], wcl["TaTcTbPd"], wcl["TaPdTbTc"],
                                              wcl["TaPcTbTd"], wcl["TaTdTbPc"], wcl["TaPcPbTd"], wcl["TaTdPbPc"],
                                              wcl["TaPcPbPd"], wcl["TaPdPbPc"], wcl["PaPcTbPd"], wcl["PaPdTbPc"],
                                              wcl["TaTcPbTd"], wcl["TaTdPbTc"], wcl["PaTcTbTd"], wcl["PaTdTbTc"],
                                              wcl["PaTcPbTd"], wcl["PaTdPbTc"], wcl["PaTcTbPd"], wcl["PaPdTbTc"],
                                              wcl["PaTcPbPd"], wcl["PaPdPbTc"], wcl["PaPcPbTd"], wcl["PaTdPbPc"],
                                              doPlanck, l_exact, l_band, l_toep, coupling.T)


        indexlist = np.arange(32)
        for name, index in zip(win_list, indexlist):
            if l_toep < lmax:
                coupling[index] = so_mcm.format_toepliz(coupling[index], l_toep, lmax)
            mcm_fortran.fill_upper(coupling[index].T)
            coupling_dict[name] = coupling[index][:lmax - 2, :lmax - 2]

    if save_file is not None:
        np.save("%s.npy"%save_file, coupling)

    return coupling_dict


def fast_cov_coupling_spin0and2(sq_win_alms_dir,
                                id_element,
                                lmax,
                                l_exact=None,
                                l_band=None,
                                l_toep=None):

    """Compute the coupling kernels corresponding to the T and E covariance matrix
    This routine assume that you have already precomputed the alms of the square of the windows
    and that the window in temperature and polarisation are the same.
    Both conditions lead to a very important speed up, that explains the name of the routine

    Parameters
    ----------

    sq_win_alms_dir: string
        the folder with precomputed alms corresponding to the sq of the window function
        this is what enters the analytic computation
    id_element : list
        a list of the form [a,b,c,d] where a = dr6_pa4_090, etc, this identify which pair of power spectrum we want the covariance of
    lmax: integer
      the maximum multipole to consider
    l_toep: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_band: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)
    l_exact: int
      parameter for the toeplitz approximation (see arXiv:2010.14344)

    """

    na, nb, nc, nd = id_element

    if l_toep is None: l_toep = lmax
    if l_band is None: l_band = lmax
    if l_exact is None: l_exact = lmax

    def try_alm_order(dir, a, b):
        try:
            alm = np.load("%s/alms_%sx%s.npy"  % (dir, a, b))
        except:
            alm = np.load("%s/alms_%sx%s.npy"  % (dir, b, a))
        return alm

    alm_TaTc = try_alm_order(sq_win_alms_dir, na, nc)
    alm_TbTd = try_alm_order(sq_win_alms_dir, nb, nd)
    alm_TaTd = try_alm_order(sq_win_alms_dir, na, nd)
    alm_TbTc = try_alm_order(sq_win_alms_dir, nb, nc)

    wcl = {}
    wcl["TaTcTbTd"] = hp.alm2cl(alm_TaTc, alm_TbTd)
    wcl["TaTdTbTc"] = hp.alm2cl(alm_TaTd, alm_TbTc)

    l = np.arange(len(wcl["TaTcTbTd"]))
    wcl["TaTcTbTd"] *= (2 * l + 1) / (4 * np.pi)
    wcl["TaTdTbTc"] *= (2 * l + 1) / (4 * np.pi)

    coupling = np.zeros((2, lmax, lmax))

    cov_fortran.calc_cov_spin0(wcl["TaTcTbTd"], wcl["TaTdTbTc"], l_exact, l_band, l_toep,  coupling.T)

    coupling_dict = {}

    for id_cov, name in enumerate(["TaTcTbTd", "TaTdTbTc"]):
        if l_toep < lmax:
            coupling[id_cov] = so_mcm.format_toepliz_fortran(coupling[id_cov], l_toep, lmax)
        mcm_fortran.fill_upper(coupling[id_cov].T)
        coupling_dict[name] = coupling[id_cov]

    list1 = ["TaTcTbTd", "PaPcPbPd", "TaTcPbPd", "PaPcTbTd",
             "TaPcTbPd", "TaTcTbPd", "TaPcTbTd", "TaPcPbTd",
             "TaPcPbPd", "PaPcTbPd", "TaTcPbTd", "PaTcTbTd",
             "PaTcPbTd", "PaTcTbPd", "PaTcPbPd", "PaPcPbTd"]

    list2 = ["TaTdTbTc", "PaPdPbPc", "TaPdPbTc", "PaTdTbPc",
             "TaPdTbPc", "TaPdTbTc", "TaTdTbPc", "TaTdPbPc",
             "TaPdPbPc", "PaPdTbPc", "TaTdPbTc", "PaTdTbTc",
             "PaTdPbTc", "PaPdTbTc", "PaPdPbTc", "PaTdPbPc"]

    for id1 in list1:
        coupling_dict[id1] = coupling_dict["TaTcTbTd"][:lmax - 2, :lmax - 2]
    for id2 in list2:
        coupling_dict[id2] = coupling_dict["TaTdTbTc"][:lmax - 2, :lmax - 2]

    return coupling_dict



def read_coupling(file, spectra=None):
    """Read a precomputed coupling kernels
    the code use the size of the array to infer what type of survey it corresponds to

    Parameters
    ----------

    file: string
      the name of the npy file
    """

    coupling = np.load("%s.npy"%file)
    coupling_dict = {}
    if spectra is None:
        win_list = ["TaTcTbTd", "TaTdTbTc"]
        if coupling.shape[0] == 2:
            indexlist = [0,1]
        elif coupling.shape[0] == 1:
            indexlist = [0,0]
    else:
        win_list = ["TaTcTbTd", "TaTdTbTc", "PaPcPbPd", "PaPdPbPc",
                    "TaTcPbPd", "TaPdPbTc", "PaPcTbTd", "PaTdTbPc",
                    "TaPcTbPd", "TaPdTbPc", "TaTcTbPd", "TaPdTbTc",
                    "TaPcTbTd", "TaTdTbPc", "TaPcPbTd", "TaTdPbPc",
                    "TaPcPbPd", "TaPdPbPc", "PaPcTbPd", "PaPdTbPc",
                    "TaTcPbTd", "TaTdPbTc", "PaTcTbTd", "PaTdTbTc",
                    "PaTcPbTd", "PaTdPbTc", "PaTcTbPd", "PaPdTbTc",
                    "PaTcPbPd", "PaPdPbTc", "PaPcPbTd", "PaTdPbPc"]

        if coupling.shape[0] == 32:
            indexlist = np.arange(32)
        elif coupling.shape[0] == 1:
            indexlist=np.zeros(32, dtype=np.int8)

    for name, index in zip(win_list, indexlist):
        coupling_dict[name] = coupling[index]

    return coupling_dict


def symmetrize(Clth, mode="arithm"):
    """Take a power spectrum Cl and return a symmetric array C_l1l2=f(Cl)

    Parameters
    ----------
    Clth: 1d array
      the power spectrum to be made symmetric
    mode : string
      geometric or arithmetic mean
      if geo return C_l1l2 = sqrt( |Cl1 Cl2 |)
      if arithm return C_l1l2 = (Cl1 + Cl2)/2
    """

    if mode == "geo":
        return np.sqrt(np.abs(np.outer(Clth, Clth)))
    if mode == "arithm":
        return np.add.outer(Clth, Clth) / 2

def bin_mat(mat, binning_file, lmax, speclist=["TT"]):
    """Take a matrix and bin it Mbb'= Pbl Pb'l' Mll' with  Pbl =1/Nb sum_(l in b)

    Parameters
    ----------
    binning_file: data file
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

    n_bins = len(bin_hi)
    n_spec = len(speclist)
    n_ell = int(mat.shape[0] / n_spec)


    coupling_b = np.zeros((n_spec * n_bins, n_spec * n_bins))

    for i, s1 in enumerate(speclist):
        for j, s2 in enumerate(speclist):

            block = mat[i * n_ell : (i + 1) * n_ell,  j * n_ell : (j + 1) * n_ell]
            binned_block = np.zeros((n_bins, n_bins))
            mcm_fortran.bin_mcm(block.T, bin_lo, bin_hi, bin_size, binned_block.T, 0)
            binned_block /= bin_size
            coupling_b[i * n_bins : (i + 1) * n_bins,  j * n_bins : (j + 1) * n_bins] = binned_block

    return coupling_b

def cov_spin0(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd, binned_mcm=True):
    """From the two point functions and the coupling kernel construct the spin0 analytical covariance matrix of <(C_ab- Clth)(C_cd-Clth)>

    Parameters
    ----------

    Clth_dict: dictionnary
      A dictionnary of theoretical power spectrum (auto and cross) for the different split combinaison ('TaTb' etc)
    coupling_dict: dictionnary
      a dictionnary containing the coupling kernel
    binning_file: data file
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    mbb_inv_ab: 2d array
      the inverse mode coupling matrix for the 'TaTb' power spectrum
    mbb_inv_cd: 2d array
      the inverse mode coupling matrix for the 'TcTd' power spectrum
    binned_mcm: boolean
      specify if the mode coupling matrices are binned or not

    """

    cov = symmetrize(Clth_dict["TaTc"]) * symmetrize(Clth_dict["TbTd"]) * coupling_dict["TaTcTbTd"]
    cov += symmetrize(Clth_dict["TaTd"]) * symmetrize(Clth_dict["TbTc"]) * coupling_dict["TaTdTbTc"]

    if binned_mcm == True:
        analytic_cov = bin_mat(cov, binning_file, lmax)
        analytic_cov = mbb_inv_ab @ analytic_cov @ mbb_inv_cd.T
    else:
        full_analytic_cov = mbb_inv_ab @ cov @ mbb_inv_cd.T
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax)

    return analytic_cov

def cov_spin0and2(Clth_dict,
                  coupling_dict,
                  binning_file,
                  lmax,
                  mbb_inv_ab,
                  mbb_inv_cd,
                  binned_mcm=True,
                  cov_T_E_only=True,
                  dtype=np.float64,
                  old_way=False):

    """From the two point functions and the coupling kernel construct the T and E analytical covariance matrix of <(C_ab- Clth)(C_cd-Clth)>

    Parameters
    ----------

    Clth_dict: dictionnary
        A dictionnary of theoretical power spectrum (auto and cross) for the different split combinaison ('XaYb' etc)
    coupling_dict: dictionnary
        a dictionnary containing the coupling kernel
    binning_file: data file
        a binning file with format bin low, bin high, bin mean
    lmax: int
        the maximum multipole to consider
    mbb_inv_ab: 2d array
        the inverse mode coupling matrix for the 'XaYb' power spectrum
    mbb_inv_cd: 2d array
        the inverse mode coupling matrix for the 'XcYd' power spectrum
    binned_mcm: boolean
      specify if the mode coupling matrices are binned or not
    binned_mcm: boolean
      specify if the mode coupling matrices are binned or not
    cov_T_E_only: boolean
        if true don't do B
    """

    n_ell = Clth_dict["TaTb"].shape[0]

    if cov_T_E_only:
        speclist = ["TT", "TE", "ET", "EE"]
    else:
        speclist =  ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    nspec = len(speclist)
    full_analytic_cov = np.zeros((nspec * n_ell, nspec * n_ell), dtype=dtype)

    for i, (W, X) in enumerate(speclist):
        for j, (Y, Z) in enumerate(speclist):
            if i > j : continue
            id0 = W + "a" + Y + "c"
            id1 = X + "b" + Z + "d"
            id2 = W + "a" + Z + "d"
            id3 = X + "b" + Y + "c"

            Cl0Cl1 = symmetrize(Clth_dict[id0]) * symmetrize(Clth_dict[id1])
            Cl2Cl3 = symmetrize(Clth_dict[id2]) * symmetrize(Clth_dict[id3])

            for field in ["E", "B"]:
                id0 = id0.replace(field, "P")
                id1 = id1.replace(field, "P")
                id2 = id2.replace(field, "P")
                id3 = id3.replace(field, "P")

            M = Cl0Cl1 * coupling_dict[id0 + id1]
            M += Cl2Cl3 * coupling_dict[id2 + id3]

            full_analytic_cov[i * n_ell:(i + 1) * n_ell, j * n_ell:(j + 1) * n_ell] = M

    full_analytic_cov = np.triu(full_analytic_cov) + np.tril(full_analytic_cov.T, -1)

    if old_way == True:
        mbb_inv_ab = extract_mbb(mbb_inv_ab, cov_T_E_only=cov_T_E_only, dtype=dtype)
        mbb_inv_cd = extract_mbb(mbb_inv_cd, cov_T_E_only=cov_T_E_only, dtype=dtype)
        if binned_mcm == True:
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
            analytic_cov = mbb_inv_ab @ analytic_cov @ mbb_inv_cd.T
        else:
            full_analytic_cov = mbb_inv_ab @ full_analytic_cov @ mbb_inv_cd.T
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
    else:
        slices, mbb_inv_ab_list = extract_mbb_list(mbb_inv_ab, cov_T_E_only=cov_T_E_only, dtype=dtype, transpose=False)
        slices, mbb_inv_cd_list = extract_mbb_list(mbb_inv_cd, cov_T_E_only=cov_T_E_only, dtype=dtype, transpose=True)
        if binned_mcm == True:
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
            analytic_cov = block_diagonal_mult(slices, mbb_inv_ab_list, mbb_inv_cd_list, analytic_cov)
        else:
            full_analytic_cov = block_diagonal_mult(slices, mbb_inv_ab_list, mbb_inv_cd_list, full_analytic_cov)
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)

    return analytic_cov


def generalized_cov_spin0and2(coupling_dict,
                              id_element,
                              ns,
                              ps_all,
                              nl_all,
                              lmax,
                              binning_file,
                              mbb_inv_ab,
                              mbb_inv_cd,
                              binned_mcm=True,
                              return_full_cov=False,
                              cov_T_E_only=True,
                              dtype=np.float64,
                              old_way=False):

    """
    This routine deserves some explanation
    We want to compute the covariance between two power spectra
    C1 = Wa * Xb, C2 =  Yc * Zd
    Here W, X, Y, Z can be either T or E and a,b,c,d will be an index
    corresponding to the survey and array we consider so for example a = s17&pa5_150 or a = dr6&pa4_090
    The formula for the analytic covariance of C1, C2 is given by
    Cov( Wa * Xb,  Yc * Zd) = < Wa Yc> <Xb Zd>  + < Wa Zd> <Xb Yc> (this is just from the wick theorem)
    In practice we need to include the effect of the mask (so we have to introduce the coupling dict D)
    and we need to take into account that we use as spectra an average of cross power spectra, that is why we use the chi function
    (and what make it different from cov_spin0and2)
    Cov( Wa * Xb,  Yc * Zd) = D(Wa*Yc,Xb Zd) chi(Wa,Yc,Xb Zd) +  D(Wa*Zd,Xb*Yc) chi(Wa,Zd,Xb,Yc)

    Parameters
    ----------
    coupling_dict : dictionnary
        a dictionnary that countains the coupling terms arising from the window functions
    id_element : list
        a list of the form [a,b,c,d] where a = dr6_pa4_090, etc, this identify which pair of power spectrum we want the covariance of
    ns: dict
        this dictionnary contains the number of split we consider for each of the survey
    ps_all: dict
        this dict contains the theoretical best power spectra, convolve with the beam for example
        ps["dr6&pa5_150", "dr6&pa4_150", "TT"] = bl_dr6_pa5_150 * bl_dr6_pa4_150 * (Dl^{CMB}_TT + fg_TT)
    nl_all: dict
        this dict contains the estimated noise power spectra, note that it correspond to the noise power spectrum per split
        e.g nl["dr6&pa5_150", "dr6&pa4_150", "TT"]
    binning_file:
        a binning file with three columns bin low, bin high, bin mean
    mbb_inv_ab and mbb_inv_cd:
        the inverse mode coupling matrices corresponding to the C1 = Wa * Xb and C2 =  Yc * Zd power spectra
    binned_mcm: boolean
        specify if the mode coupling matrices are binned or not
    return_full_cov: boolean
        an option to return the lbyl cov (if binned_mcm=False)
        mostly used for debugging
    cov_T_E_only: boolean
        if true don't do B
    """

    na, nb, nc, nd = id_element

    n_ell = coupling_dict["TaTcTbTd"].shape[0]

    if cov_T_E_only:
        speclist = ["TT", "TE", "ET", "EE"]
    else:
        speclist =  ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    nspec = len(speclist)
    full_analytic_cov = np.zeros((nspec * n_ell, nspec * n_ell), dtype=dtype)

    for i, (W, X) in enumerate(speclist):
        for j, (Y, Z) in enumerate(speclist):

            id0 = W + "a" + Y + "c"
            id1 = X + "b" + Z + "d"
            id2 = W + "a" + Z + "d"
            id3 = X + "b" + Y + "c"

            for field in ["E", "B"]:
                id0 = id0.replace(field, "P")
                id1 = id1.replace(field, "P")
                id2 = id2.replace(field, "P")
                id3 = id3.replace(field, "P")


            M = coupling_dict[id0 + id1] * chi(na, nc, nb, nd, ns, ps_all, nl_all, W + Y + X + Z)
            M += coupling_dict[id2 + id3] * chi(na, nd, nb, nc, ns, ps_all, nl_all, W + Z + X + Y)
            full_analytic_cov[i * n_ell: (i + 1) * n_ell, j * n_ell: (j + 1) * n_ell] = M

    if old_way == True:
        mbb_inv_ab = extract_mbb(mbb_inv_ab, cov_T_E_only=cov_T_E_only, dtype=dtype)
        mbb_inv_cd = extract_mbb(mbb_inv_cd, cov_T_E_only=cov_T_E_only, dtype=dtype)
        if binned_mcm == True:
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
            analytic_cov = mbb_inv_ab @ analytic_cov @ mbb_inv_cd.T
        else:
            full_analytic_cov = mbb_inv_ab @ full_analytic_cov @ mbb_inv_cd.T
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
    else:
        slices, mbb_inv_ab_list = extract_mbb_list(mbb_inv_ab, cov_T_E_only=cov_T_E_only, dtype=dtype, transpose=False)
        slices, mbb_inv_cd_list = extract_mbb_list(mbb_inv_cd, cov_T_E_only=cov_T_E_only, dtype=dtype, transpose=True)
        if binned_mcm == True:
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
            analytic_cov = block_diagonal_mult(slices, mbb_inv_ab_list, mbb_inv_cd_list, analytic_cov)
        else:
            full_analytic_cov = block_diagonal_mult(slices, mbb_inv_ab_list, mbb_inv_cd_list, full_analytic_cov)
            analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)


    return analytic_cov


def covariance_element_beam(id_element,
                            ps_all,
                            norm_beam_cov,
                            binning_file,
                            lmax,
                            cov_T_E_only=True):
    """
    This routine compute the contribution from beam errors to the analytical covariance of the power spectra
    We want to compute the beam covariance between the two spectra
    C1 = Wa * Xb, C2 =  Yc * Zd
    Here W, X, Y, Z can be either T or E and a,b,c,d will be an index
    corresponding to the survey and array we consider so for example a = dr6&pa5_150 or a = dr6&pa4_090
    The formula for the analytic covariance of C1, C2 is given by
    let's denote the normalised beam covariance <BB>_ac = < delta B_a delta B_c >/np.outer(B_a, B_c)

    Cov(Wa * Xb,  Yc * Zd) = Dl^{WaXb} Dl^{YcZd} ( <BB>_ac + <BB>_ad + <BB>_bc + <BB>_bd )

    Parameters
    ----------
    id_element : list
        a list of the form [a,b,c,d] where a = dr6_pa4_090, etc, this identify which pair of power spectrum we want the covariance of
    ps_all: dict
        this dict contains the theoretical best power spectra, convolve with the beam for example
        ps["dr6&pa5_150", "dr6&pa4_150", "TT"] =  (Dl^{CMB}_TT + fg_TT)
    norm_beam_cov: dict
        this dict contains the normalized beam covariance for each survey and array
    binning_file: str
        a binning file with three columns bin low, bin high, bin mean
    lmax: int
        the maximum multipole to consider
    cov_T_E_only: boolean
        if true don't do B

    """
    na, nb, nc, nd = id_element

    sv_alpha, ar_alpha = na.split("&")
    sv_beta, ar_beta = nb.split("&")
    sv_gamma, ar_gamma = nc.split("&")
    sv_eta, ar_eta = nd.split("&")

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    if cov_T_E_only:
        speclist = ["TT", "TE", "ET", "EE"]
    else:
        speclist = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    nspec = len(speclist)
    analytic_cov_from_beam = np.zeros((nspec * nbins, nspec * nbins))

    M =  (delta2(na, nc) + delta2(na, nd)) * norm_beam_cov[sv_alpha, ar_alpha]
    M += (delta2(nb, nc) + delta2(nb, nd)) * norm_beam_cov[sv_beta, ar_beta]

    for i, spec1 in enumerate(speclist):
        for j, spec2 in enumerate(speclist):

            cov = M.copy()
            cov *=  np.outer(ps_all[na, nb, spec1], ps_all[nc, nd, spec2])

            analytic_cov_from_beam[i * nbins: (i + 1) * nbins, j * nbins: (j + 1) * nbins] = bin_mat(cov, binning_file, lmax)

    return analytic_cov_from_beam


def block_diagonal_mult(slices, mbb_inv_ab_list, mbb_inv_cd_list, analytic_cov):
    """Suggestion by adrien to do an operation of the type A M B, where A and B are block diagonal matrix

    Parameters
    ----------
    slices: list of tuple of integer
        give the min and max indices of each block
    mbb_inv_ab_list: list of 2d array
        list of the inverse of the mode coupling matrix
    mbb_inv_cd_list: list of 2d array
        list of the transpose of the inverse of the mode coupling matrix
    analytic_cov: 2d array
        analytic covariance matrix

    """
    nblocks = len(slices)

    for i in range(nblocks):
        for j in range(nblocks):

            min_i, max_i = slices[i]
            min_j, max_j = slices[j]

            analytic_cov[min_i: max_i, min_j: max_j] = mbb_inv_ab_list[i] @ analytic_cov[min_i: max_i, min_j: max_j] @ mbb_inv_cd_list[j]

    return analytic_cov

def extract_mbb_list(mbb_inv,
                     cov_T_E_only=True,
                     dtype=np.float64,
                     transpose=False):

    """extract a list of inverse mode coupling matrix, you can optionnaly choose the dtype, if you want only the TT-TE-ET-EE part,
    and if you want to transpose it

    Parameters
    ----------

    mbb_inv: dict of 2d arrays
      the inverse spin0 and 2 mode coupling matrix
    cov_T_E_only: boolean
        if true don't do B
    dtype: np dtype
        choose the type to save memory
    transpose: boolean
        wether to transpose it or not
    """


    nbins = mbb_inv["spin0xspin0"].shape[0]
    slices = []

    if cov_T_E_only == False:

        nblocks = 6
        for i in range(nblocks - 1):
            slices += [(i * nbins, (i + 1) * nbins)]
        slices += [(5 * nbins, 9 * nbins)] # the EE-EB-BE-BB block

        mbb_list = [mbb_inv["spin0xspin0"],
                    mbb_inv["spin0xspin2"],
                    mbb_inv["spin0xspin2"],
                    mbb_inv["spin2xspin0"],
                    mbb_inv["spin2xspin0"],
                    mbb_inv["spin2xspin2"]]
    else:

        nblocks = 4
        for i in range(nblocks):
            slices += [(i * nbins, (i + 1) * nbins)]

        mbb_list = [mbb_inv["spin0xspin0"],
                    mbb_inv["spin0xspin2"],
                    mbb_inv["spin2xspin0"],
                    mbb_inv["spin2xspin2"][0:nbins, 0:nbins]]

    for i in range(nblocks):
        if mbb_list[i].dtype != dtype:
            mbb_list[i] = mbb_list[i].astype(dtype)
        if transpose == True:
            mbb_list[i] = mbb_list[i].T

    return slices, mbb_list



def extract_mbb(mbb_inv, cov_T_E_only=True, dtype=np.float64):
    """The mode coupling marix is computed for T,E,B but for if cov_T_E_only=True we only construct analytical covariance matrix for T and E

    Parameters
    ----------

    mbb_inv: dict of 2d arrays
      the inverse spin0 and 2 mode coupling matrix
    cov_T_E_only: boolean
        if true don't do B
    """
    nbins = mbb_inv["spin0xspin0"].shape[0]
    if cov_T_E_only:
        mbb_inv_array = np.zeros((4*nbins, 4*nbins), dtype=dtype)
        # TT
        mbb_inv_array[0*nbins:1*nbins, 0*nbins:1*nbins] = mbb_inv["spin0xspin0"]
        # TE
        mbb_inv_array[1*nbins:2*nbins, 1*nbins:2*nbins] = mbb_inv["spin0xspin2"]
        # ET
        mbb_inv_array[2*nbins:3*nbins, 2*nbins:3*nbins] = mbb_inv["spin2xspin0"]
        # EE
        mbb_inv_array[3*nbins:4*nbins, 3*nbins:4*nbins] = mbb_inv["spin2xspin2"][0:nbins, 0:nbins]
    else:
        mbb_inv_array = np.zeros((9*nbins, 9*nbins), dtype=dtype)
        # TT
        mbb_inv_array[0*nbins:1*nbins, 0*nbins:1*nbins] = mbb_inv["spin0xspin0"]
        # TE
        mbb_inv_array[1*nbins:2*nbins, 1*nbins:2*nbins] = mbb_inv["spin0xspin2"]
        # TB
        mbb_inv_array[2*nbins:3*nbins, 2*nbins:3*nbins] = mbb_inv["spin0xspin2"]
        # ET
        mbb_inv_array[3*nbins:4*nbins, 3*nbins:4*nbins] = mbb_inv["spin2xspin0"]
        # BT
        mbb_inv_array[4*nbins:5*nbins, 4*nbins:5*nbins] = mbb_inv["spin2xspin0"]
        # EE-EB-BE-BB
        mbb_inv_array[5*nbins:9*nbins, 5*nbins:9*nbins] = mbb_inv["spin2xspin2"]

    return mbb_inv_array

def extract_EEEBBB_mbb(mbb_inv):
    """this routine extract the E and B part of the mode coupling matrix

    Parameters
    ----------

    mbb_inv: 2d array
        the inverse spin0 and 2 mode coupling matrix
    """

    return mbb_inv["spin2xspin2"]

def corr2cov(corr, var):
    """Go from correlation and variance to covariance matrix

    Parameters
    ----------
    corr: 2d array
      the correlation matrix
    var: 1d array
      vector of variance of the random variables
    """
    return corr * np.sqrt(var[:, None] * var[None, :])


def cov2corr(cov, remove_diag=False):
    """Go from covariance to correlation matrix, also setting the diagonal to zero

    Parameters
    ----------
    cov: 2d array
      the covariance matrix
    """

    d = np.sqrt(cov.diagonal())
    corr = ((cov.T/d).T)/d
    if remove_diag == True:
        corr -= np.identity(cov.shape[0])
    return corr

def selectblock(cov, spectra_order, n_bins, block="TTTT"):
    """Select a block in a spin0 and 2 covariance matrix

    Parameters
    ----------

    cov: 2d array
      the covariance matrix
    spectra_order: list of strings
      the arangement of the different block
    n_bins: int
      the number of bins for each block
    block: string
      the block you want to look at
    """

    if spectra_order == None:
        print ("cov mat of spin 0, no block selection needed")
        return
    else:
        blockindex = {}
        for c1,s1 in enumerate(spectra_order):
            for c2,s2 in enumerate(spectra_order):
                blockindex[s1 + s2] = [c1 * n_bins, c2 * n_bins]
    id1 = blockindex[block][0]
    id2 = blockindex[block][1]
    cov_select = cov[id1:id1 + n_bins, id2:id2 + n_bins]
    return cov_select

def get_sigma(cov, spectra_order, n_bins, spectrum):
    """get the error of the spectrum for the given cov mat

    Parameters
    ----------

    cov: 2d array
      the covariance matrix
    spectra_order: list of strings
      the arangement of the different block
    n_bins: int
      the number of bins for each block
    spectrum: string
      the spectrum we consider
    """
    cov_sub = selectblock(cov, spectra_order, n_bins, block=spectrum+spectrum)
    err = np.sqrt(cov_sub.diagonal())
    return err



def delta2(a, b):
    """Simple delta function
    """

    if a == b:
        return 1
    else:
        return 0

def delta3(a, b, c):
    """Delta function (3 variables)
    """

    if (a == b) & (b == c):
        return 1
    else:
        return 0

def delta4(a, b, c, d):
    """Delta function (4 variables)
    """

    if (a == b) & (b == c) & (c == d):
        return 1
    else:
        return 0

def f(a, b, c, d, ns):
    """f combination factor in the covariance computation
    """

    result = 1. * ns[a] * (ns[c] * ns[d] * delta2(a, b) - ns[c] * delta3(a, b, d) - ns[d] * delta3(a, b, c) + delta4(a, b, c, d))
    result /= (ns[c] * ns[d] * (ns[a] - delta2(a, c)) * (ns[b] - delta2(b, d)))
    return result

def g(a, b, c, d, ns):
    """g combination factor in the covariance computation
    """

    result = 1. * ns[a] * (ns[c] * delta2(a,b) * delta2(c, d) - delta4(a, b, c, d))
    result /= (ns[a] * ns[b] * (ns[c] - delta2(a, c)) * (ns[d] - delta2(b, d)))
    return result


def chi(alpha, gamma, beta, eta, ns, Dl, DNl, id="TTTT"):
    """doc not ready yet
    """

    sv_alpha, ar_alpha = alpha.split("&")
    sv_beta, ar_beta = beta.split("&")
    sv_gamma, ar_gamma = gamma.split("&")
    sv_eta, ar_eta = eta.split("&")

    AB = id[0] + id[1]
    CD = id[2] + id[3]
    chi = Dl[alpha, gamma, AB] * Dl[beta, eta, CD]
    chi += Dl[alpha, gamma, AB] * DNl[beta, eta, CD] * f(sv_beta, sv_eta, sv_alpha, sv_gamma, ns)
    chi += Dl[beta, eta, CD] * DNl[alpha, gamma, AB] * f(sv_alpha, sv_gamma, sv_beta, sv_eta, ns)
    chi += g(sv_alpha, sv_gamma, sv_beta, sv_eta, ns) * DNl[alpha, gamma, AB] * DNl[beta, eta, CD]

    chi= symmetrize(chi, mode="arithm")

    return chi


def plot_cov_matrix(mat, color_range=None, color="pwhite", file_name=None):
    """plot the covariance matrix at full resolution using pixell plotting routines

    Parameters
    ----------

    mat: 2d array
      the covariance matrix
    color_range: float
      the range of the plot
    color: pixell colormap
      the colormap for the plot (have to be pixell compatible)
    file_name: string
      file_name is the name of the png file that will be created, if None the plot
      will be displayed.
    """

    mat_plot = mat.copy()

    try:
        if color not in mpl.colormaps:
            colorize.mpl_setdefault(color)
    except KeyError:
        raise KeyError("Color name must be a pixell color map name {}!".format(
                        list(colorize.schemes.keys())))

    if color_range is None:
        max_range = np.maximum(np.max(mat_plot),np.abs(np.min(mat_plot)))
        color_range = "%s" % (max_range)

    # We need to revert the order of the array to make the plot it similar to matplotlib imshow

    mat_plot = mat_plot[::-1, ::-1]

    wcs = enmap.create_wcs(mat_plot.shape, proj="car")
    mat_plot = enmap.enmap(mat_plot,wcs)
    plots = enplot.get_plots(mat_plot,
                             color=color,
                             range=color_range,
                             colorbar=1,
                             grid=0)
    for plot in plots:
        if file_name is not None:
            enplot.write(file_name + ".png", plot)
        else:
            plot.img.show()


def mc_cov_from_spectra_list(spec_list_a, spec_list_b, spectra=None):
    """Return the montecarlo covariance of two spectra list estimated from simulations
    This routine is inefficient and need some more work

    Parameters
    ----------
    spec_list_a: first list of spectra (or list of dict if spectra is not None)
        a list of ps computed from simulation (each entry of the list is a ps)
    spec_list_b: second list of spectra (or list of dict if spectra is not None)
        a list of ps computed from simulation (each entry of the list is a ps)

    spectra: list of strings
        needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    if spectra is None:
        mc_cov = np.cov(spec_list_a, y=spec_list_b,  rowvar=False)
        mean_a = np.mean(spec_list_a, axis=0)
        mean_b = np.mean(spec_list_b, axis=0)

    else:
        vec_all_a, vec_all_b = [], []
        for spec_a, spec_b in zip(spec_list_a, spec_list_b):
            vec_a, vec_b = [], []
            for spec in spectra:
                vec_a = np.append(vec_a, spec_a[spec])
                vec_b = np.append(vec_b, spec_b[spec])
            vec_all_a +=  [vec_a]
            vec_all_b +=  [vec_b]

        mc_cov = np.cov(vec_all_a, y=vec_all_b, rowvar=False)

        vec_mean_a = np.mean(vec_all_a, axis=0)
        vec_mean_b = np.mean(vec_all_b, axis=0)

        n_bins = int(len(vec_mean_a)/len(spectra))
        mean_a = {}
        mean_b = {}
        for i, spec in enumerate(spectra):
            mean_a[spec] = vec_mean_a[i * n_bins : (i + 1) * n_bins ]
            mean_b[spec] = vec_mean_b[i * n_bins : (i + 1) * n_bins ]

    n_el = int(mc_cov.shape[0] / 2)
    return mean_a, mean_b, mc_cov[:n_el, n_el:]


def measure_white_noise_level(var, mask, dtype=np.float64):
    """take enmaps of map variance and mask, outputs white noise power spectrum amplitude
    Parameters
    ----------

    var: enmap
      variance map
    mask: enmap
      window map
    """
    buffer = mask.flatten().astype(dtype)  # save some memory by reusing one buffer
    buffer *= buffer        # ZL: np.sum not accurate enough, math.fsum used instead
    mask_tot = np.sum(buffer)  # sum(mask^2)
    buffer *= var.ravel()
    buffer *= var.pixsizemap().flatten()
    noise_tot = np.sum(buffer)
    return noise_tot / mask_tot


def masked_variances(survey_name, win, var):
    var_a, var_b, var_c, var_d = (
        win['Ta'].copy(), win['Tb'].copy(), win['Tc'].copy(), win['Td'].copy())
    pixsizes = win['Ta'].data.pixsizemap()  # assuming they're the same footprint
    var_a.data *= np.sqrt(pixsizes * var['Ta'].data)
    var_b.data *= np.sqrt(pixsizes * var['Tb'].data)
    var_c.data *= np.sqrt(pixsizes * var['Tc'].data)
    var_d.data *= np.sqrt(pixsizes * var['Td'].data)
    return dict(zip(survey_name, (var_a, var_b, var_c, var_d)))



def _same_pol(a, b):  # i.e. 'Ta' and 'Tb'
    return a[0] == b[0]


def get_zero_map(template):
    c = template.copy()
    c.data *= 0
    return c

def _product_of_so_map(a, b):
    c = a.copy()
    c.data *= b.data
    return c

def make_weighted_variance_map(variance, window):
    var = window.copy()  # window^2 * variance * pixsize
    var.data *= window.data
    var.data *= window.data.pixsizemap() * variance.data
    return var


def organize_covmat_products(survey_id, survey_names, weighted_variances, windows):
    survey = dict(zip(survey_id, survey_names))
    zero_map = get_zero_map(weighted_variances[survey_id[0]])
    win = lambda a, b : _product_of_so_map(windows[a], windows[b])
    var = lambda a, b : weighted_variances[a] if (
        _same_pol(a, b) and survey[a] == survey[b]) else zero_map
    return win, var


def generate_aniso_couplings_TTTT(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling = lambda m1, m2 : so_mcm.coupling_block("00", win1=m1, win2=m2, 
                                                      lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling(win(i, p), win(j, q)),
        coupling(win(i, q), win(j, p)),
        coupling(win(i, p), var(j, q)),
        coupling(win(j, q), var(i, p)),
        coupling(win(i, q), var(j, p)),
        coupling(win(j, p), var(i, q)),
        coupling(var(i, p), var(j, q)),
        coupling(var(i, q), var(j, p))]


def generate_aniso_couplings_EEEE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling = lambda m1, m2 : so_mcm.coupling_block("++", win1=m1, win2=m2, 
                                                      lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling(win(i, p), win(j, q)),
        coupling(win(i, q), win(j, p)),
        coupling(win(i, p), var(j, q)),
        coupling(win(j, q), var(i, p)),
        coupling(win(i, q), var(j, p)),
        coupling(win(j, p), var(i, q)),
        coupling(var(i, p), var(j, q)),
        coupling(var(i, q), var(j, p))]


# for TTTT, EEEE
def coupled_cov_aniso_same_pol(survey_id, Clth, Rl, couplings):
    i, j, p, q = survey_id
    geom = lambda cl : symmetrize(cl, mode="geo")
    cov =  geom(Clth[i+p]) * geom(Clth[j+q]) * couplings[0]
    cov += geom(Clth[i+q]) * geom(Clth[j+p]) * couplings[1]
    cov += geom(Rl[j]) * geom(Rl[q]) * geom(Clth[i+p]) * couplings[2]
    cov += geom(Rl[i]) * geom(Rl[p]) * geom(Clth[j+q]) * couplings[3]
    cov += geom(Rl[j]) * geom(Rl[p]) * geom(Clth[i+q]) * couplings[4]
    cov += geom(Rl[i]) * geom(Rl[q]) * geom(Clth[j+p]) * couplings[5]
    cov += geom(Rl[i]) * geom(Rl[j]) * geom(Rl[p]) * geom(Rl[q]) * couplings[6]
    cov += geom(Rl[i]) * geom(Rl[j]) * geom(Rl[p]) * geom(Rl[q]) * couplings[7]
    return cov


def generate_aniso_couplings_TETE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling_TT = lambda m1, m2 : so_mcm.coupling_block("00", win1=m1, win2=m2, 
                                                      lmax=lmax, niter=niter)  / (4 * np.pi)
    coupling_TE = lambda m1, m2 : so_mcm.coupling_block("02", win1=m1, win2=m2, 
                                                      lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling_TE(win(i, p), win(j, q)),
        coupling_TT(win(i, q), win(j, p)),
        coupling_TE(win(i, p), var(j, q)),
        coupling_TE(win(j, q), var(i, p)),
        coupling_TE(var(i, p), var(j, q))]


def arith(cl_a, cl_b): 
    A = (np.repeat(cl_a[:,np.newaxis], len(cl_a), 1) * 
        np.repeat(cl_b[np.newaxis,:], len(cl_b), 0))
    return A

# for TETE
def coupled_cov_aniso_TETE(survey_id, Clth, Rl, couplings):
    i, j, p, q = survey_id
    geom = lambda cl : symmetrize(cl, mode="geo")
    cov =  geom(Clth[i+p]) * geom(Clth[j+q]) * couplings[0]
    cov += 0.5 * (np.outer(Clth[i+q], Clth[j+p]) + np.outer(Clth[j+p], Clth[i+q])) * couplings[1]
    cov += geom(Rl[j]) * geom(Rl[q]) * geom(Clth[i+p]) * couplings[2]
    cov += geom(Rl[i]) * geom(Rl[p]) * geom(Clth[j+q]) * couplings[3]
    cov += geom(Rl[i]) * geom(Rl[j]) * geom(Rl[p]) * geom(Rl[q]) * couplings[4]
    return cov


def coupled_cov_aniso_TTTE(survey_id, Clth, Rl, couplings):
    i, j, p, q = survey_id
    geom = lambda cl : symmetrize(cl, mode="geo")
    arit = lambda cl : symmetrize(cl, mode="arithm")
    cov =  geom(Clth[i+p]) * arit(Clth[j+q]) * couplings[0]
    cov += geom(Clth[j+p]) * arit(Clth[i+q]) * couplings[1]
    cov += geom(Rl[i]) * geom(Rl[p]) * arit(Clth[j+q]) * couplings[2]
    cov += geom(Rl[j]) * geom(Rl[p]) * arit(Clth[i+q]) * couplings[3]
    return cov


def generate_aniso_couplings_TTTE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling_TT = lambda m1, m2 : so_mcm.coupling_block("00", win1=m1, win2=m2, 
                                                         lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling_TT(win(i, p), win(j, q)),
        coupling_TT(win(i, q), win(j, p)),
        coupling_TT(win(j, q), var(i, p)),
        coupling_TT(win(i, q), var(j, p))
    ]


def coupled_cov_aniso_TTEE(survey_id, Clth, Rl, couplings):
    i, j, p, q = survey_id
    cov =  0.5 * (np.outer(Clth[i+p], Clth[j+q]) + np.outer(Clth[j+q], Clth[i+p])) * couplings[0]
    cov += 0.5 * (np.outer(Clth[i+q], Clth[j+p]) + np.outer(Clth[j+p], Clth[i+q])) * couplings[1]
    return cov


def generate_aniso_couplings_TTEE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling_TT = lambda m1, m2 : so_mcm.coupling_block("00", win1=m1, win2=m2, 
                                                         lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling_TT(win(i, p), win(j, q)),
        coupling_TT(win(i, q), win(j, p))
    ]


def coupled_cov_aniso_TEEE(survey_id, Clth, Rl, couplings):
    i, j, p, q = survey_id
    geom = lambda cl : symmetrize(cl, mode="geo")
    arit = lambda cl : symmetrize(cl, mode="arithm")
    cov =  geom(Clth[j+q]) * arit(Clth[i+p]) * couplings[0]
    cov += geom(Clth[j+p]) * arit(Clth[i+q]) * couplings[1]
    cov += geom(Rl[j]) * geom(Rl[q]) * arit(Clth[i+p]) * couplings[2]
    cov += geom(Rl[j]) * geom(Rl[p]) * arit(Clth[i+q]) * couplings[3]
    return cov


def generate_aniso_couplings_TEEE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
    i, j, p, q = survey_id
    win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
    coupling_EE = lambda m1, m2 : so_mcm.coupling_block("++", win1=m1, win2=m2, 
                                                         lmax=lmax, niter=niter)  / (4 * np.pi)
    return [
        coupling_EE(win(i, p), win(j, q)),
        coupling_EE(win(i, q), win(j, p)),
        coupling_EE(win(i, p), var(j, q)),
        coupling_EE(win(i, q), var(j, p))
    ]


# todo: need to check arithmetic mean routine


# def generate_aniso_couplings_TTTE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
#     i, j, p, q = survey_id
#     win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
#     coupling = lambda prod1, prod2 : so_mcm.mcm_and_bbl_spin0and2(
#         prod1, "", lmax, niter, "Cl", prod2, return_coupling_only=True)[3,:,:] / (4 * np.pi)
#     return [
#         coupling(win(i, p), win(j, q)),
#         coupling(win(i, q), win(j, p)),
#         coupling(win(j, q), var(i, p)),
#         coupling(win(i, q), var(j, p))]

# def generate_aniso_couplings_TTEE(survey_id, surveys, windows, weighted_var, lmax, niter=0):
#     i, j, p, q = survey_id
#     win, var = organize_covmat_products(survey_id, surveys, weighted_var, windows)
#     coupling = lambda prod1, prod2 : so_mcm.mcm_and_bbl_spin0and2(
#         prod1, "", lmax, niter, "Cl", prod2, return_coupling_only=True)[3,:,:] / (4 * np.pi)
#     return [
#         coupling(win(i, p), win(j, q)),
#         coupling(win(i, q), win(j, p))]

