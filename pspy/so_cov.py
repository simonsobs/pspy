"""
Tools for analytical covariance matrix estimation.
"""

import healpy as hp
import numpy as np

from pspy import pspy_utils, so_mcm, sph_tools
from pspy.cov_fortran import cov_fortran


def cov_coupling_spin0(win, lmax, niter=0, save_file=None):
    """Compute the coupling kernels corresponding to the T only covariance matrix
       see Section IV A of https://www.overleaf.com/read/fvrcvgbzqwrz

   Parameters
    ----------

    win: ``so_map`` or dictionnary of ``so_map``
      the window functions, can be a ``so_map`` or a dictionnary containing ``so_map``
      if the later, the entry of the dictionnary should be Ta, Tb, Tc, Td.
    lmax: integer
      the maximum multipole to consider
    niter: int
      the number of iteration performed while computing the alm
    save_file: string
      the name of the file in which the coupling kernel will be saved (npy format)
    """

    coupling_dict = {}
    if isinstance(win, dict):
        wcl = {}
        for name in ["TaTcTbTd", "TaTdTbTc"]:
            n0, n1, n2, n3 = [name[i, i + 2] for i in range(4)]
            sq_win_n0n1 = win[n0].copy()
            sq_win_n0n1.data *= win[n1].data
            sq_win_n2n3 = win[n2].copy()
            sq_win_n2n3.data *= win[n3].data
            alm_n0n1 = sph_tools.map2alm(sq_win_n0n1, niter=niter, lmax=lmax)
            alm_n2n3 = sph_tools.map2alm(sq_win_n2n3, niter=niter, lmax=lmax)
            wcl[n0 + n1 + n2 + n3] = hp.alm2cl(alm_n0n1, alm_n2n3)
            ell = np.arange(len(wcl[n0 + n1 + n2 + n3]))
            wcl[n0 + n1 + n2 + n3] *= (2 * ell + 1) / (4 * np.pi)

        coupling = np.zeros((2, lmax, lmax))
        cov_fortran.calc_cov_spin0(wcl["TaTcTbTd"], wcl["TaTdTbTc"], coupling.T)
        coupling_dict["TaTcTbTd"] = coupling[0]
        coupling_dict["TaTdTbTc"] = coupling[1]
    else:
        sq_win = win.copy()
        sq_win.data *= sq_win.data
        alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
        wcl = hp.alm2cl(alm)
        ell = np.arange(len(wcl))
        wcl *= (2 * ell + 1) / (4 * np.pi)
        coupling = np.zeros((1, lmax, lmax))
        cov_fortran.calc_cov_spin0_single_win(wcl, coupling.T)
        coupling_dict["TaTcTbTd"] = coupling[0]
        coupling_dict["TaTdTbTc"] = coupling[0]

    if save_file is not None:
        np.save("%s.npy" % save_file, coupling)

    return coupling_dict


def cov_coupling_spin0and2(win, lmax, niter=0, save_file=None):
    """Compute the coupling kernels corresponding to the T and E covariance matrix
       see Section IV B of https://www.overleaf.com/read/fvrcvgbzqwrz

    Parameters
    ----------

    win: ``so_map`` or dictionnary of ``so_map``
      the window functions, can be a ``so_map`` or a dictionnary containing ``so_map``
      if the later, the entry of the dictionnary should be Ta,Tb,Tc,Td,Pa,Pb,Pc,Pd
    lmax: integer
      the maximum multipole to consider
    niter: int
      the number of iteration performed while computing the alm
    save_file: string
      the name of the file in which the coupling kernel will be saved (npy format)
    """

    win_list = [
        "TaTcTbTd", "TaTdTbTc", "PaPcPbPd", "PaPdPbPc", "TaTcPbPd", "TaPdPbTc", "TaTcTbPd",
        "TaPdTbTc", "TaPcTbPd", "TaPdTbPc", "PaTcPbPd", "PaPdPbTc"
    ]

    coupling_dict = {}
    if isinstance(win, dict):
        wcl = {}
        for name in win_list:
            n0, n1, n2, n3 = [name[i * 2:(i + 1) * 2] for i in range(4)]

            sq_win_n0n1 = win[n0].copy()
            sq_win_n0n1.data *= win[n1].data
            sq_win_n2n3 = win[n2].copy()
            sq_win_n2n3.data *= win[n3].data

            alm_n0n1 = sph_tools.map2alm(sq_win_n0n1, niter=niter, lmax=lmax)
            alm_n2n3 = sph_tools.map2alm(sq_win_n2n3, niter=niter, lmax=lmax)

            wcl[n0 + n1 + n2 + n3] = hp.alm2cl(alm_n0n1, alm_n2n3)
            ell = np.arange(len(wcl[n0 + n1 + n2 + n3]))
            wcl[n0 + n1 + n2 + n3] *= (2 * ell + 1) / (4 * np.pi)

        coupling = np.zeros((12, lmax, lmax))
        cov_fortran.calc_cov_spin0and2(wcl["TaTcTbTd"], wcl["TaTdTbTc"], wcl["PaPcPbPd"],
                                       wcl["PaPdPbPc"], wcl["TaTcPbPd"], wcl["TaPdPbTc"],
                                       wcl["TaTcTbPd"], wcl["TaPdTbTc"], wcl["TaPcTbPd"],
                                       wcl["TaPdTbPc"], wcl["PaTcPbPd"], wcl["PaPdPbTc"],
                                       coupling.T)

        indexlist = np.arange(12)
        for name, index in zip(win_list, indexlist):
            coupling_dict[name] = coupling[index]
    else:
        sq_win = win.copy()
        sq_win.data *= sq_win.data
        alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
        wcl = hp.alm2cl(alm)
        ell = np.arange(len(wcl))
        wcl *= (2 * ell + 1) / (4 * np.pi)
        coupling = np.zeros((3, lmax, lmax))
        cov_fortran.calc_cov_spin0and2_single_win(wcl, coupling.T)

        indexlist = [0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 2, 2]
        for name, index in zip(win_list, indexlist):
            coupling_dict[name] = coupling[index]

    if save_file is not None:
        np.save("%s.npy" % save_file, coupling)

    return coupling_dict


def read_coupling(file_name):
    """Read a precomputed coupling kernels
    the code use the size of the array to infer what type of survey it corresponds to

    Parameters
    ----------

    file: string
      the name of the npy file
    """

    coupling = np.load("%s.npy" % file_name)
    coupling_dict = {}
    if coupling.shape[0] == 12:
        win_list = [
            "TaTcTbTd", "TaTdTbTc", "PaPcPbPd", "PaPdPbPc", "TaTcPbPd", "TaPdPbTc", "TaTcTbPd",
            "TaPdTbTc", "TaPcTbPd", "TaPdTbPc", "PaTcPbPd", "PaPdPbTc"
        ]
        indexlist = np.arange(12)
    elif coupling.shape[0] == 3:
        win_list = [
            "TaTcTbTd", "TaTdTbTc", "PaPcPbPd", "PaPdPbPc", "TaTcPbPd", "TaPdPbTc", "TaTcTbPd",
            "TaPdTbTc", "TaPcTbPd", "TaPdTbPc", "PaTcPbPd", "PaPdPbTc"
        ]
        indexlist = [0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 2, 2]
    elif coupling.shape[0] == 2:
        win_list = ["TaTcTbTd", "TaTdTbTc"]
        indexlist = [0, 1]
    elif coupling.shape[0] == 1:
        win_list = ["TaTcTbTd", "TaTdTbTc"]
        indexlist = [0, 0]
    for name, index in zip(win_list, indexlist):
        coupling_dict[name] = coupling[index]

    return coupling_dict


def symmetrize(cl, mode="arithm"):
    """Take a power spectrum Cl and return a symmetric array C_l1l2=f(Cl)

    Parameters
    ----------
    cl: 1d array
      the power spectrum to be made symmetric
    mode : string
      geometric or arithmetic mean
      if geo return C_l1l2= sqrt( |Cl1 Cl2 |)
      if arithm return C_l1l2= (Cl1 + Cl2)/2
    """

    if mode == "geo":
        return np.sqrt(np.abs(np.outer(cl, cl)))
    if mode == "arithm":
        return np.add.outer(cl, cl) / 2

    raise ValueError("Unkown mode '%s'!" % mode)


def bin_mat(mat, binning_file, lmax):
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
    coupling_b = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            coupling_b[i, j] = np.mean(mat[bin_lo[i]:bin_hi[i], bin_lo[j]:bin_hi[j]])
    return coupling_b


def cov_spin0(cl_th_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd):
    """From the two point functions and the coupling kernel construct the spin0 analytical
       covariance matrix of <(C_ab- Clth)(C_cd-Clth)>

    Parameters
    ----------

    cl_th_dict: dictionnary
      A dictionnary of theoretical power spectrum (auto and cross) for the different
      split combinaison ('TaTb' etc)
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
    """

    cov = symmetrize(cl_th_dict["TaTc"]) * symmetrize(
        cl_th_dict["TbTd"]) * coupling_dict["TaTcTbTd"] + symmetrize(
            cl_th_dict["TaTd"]) * symmetrize(cl_th_dict["TbTc"]) * coupling_dict["TaTdTbTc"]
    analytic_cov = bin_mat(cov, binning_file, lmax)
    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    return analytic_cov


def cov_spin0and2(cl_th_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd):
    """From the two point functions and the coupling kernel construct the T and E analytical
       covariance matrix of <(C_ab- Clth)(C_cd-Clth)>

    Parameters
    ----------

    cl_th_dict: dictionnary
      A dictionnary of theoretical power spectrum (auto and cross) for the different
      split combinaison ('XaYb' etc)
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
    """

    TaTc, TbTd, TaTd, TbTc = symmetrize(cl_th_dict["TaTc"]), symmetrize(
        cl_th_dict["TbTd"]), symmetrize(cl_th_dict["TaTd"]), symmetrize(cl_th_dict["TbTc"])
    EaEc, EbEd, EaEd, EbEc = symmetrize(cl_th_dict["EaEc"]), symmetrize(
        cl_th_dict["EbEd"]), symmetrize(cl_th_dict["EaEd"]), symmetrize(cl_th_dict["EbEc"])
    TaEd, TaEc, TbEc, TbEd, EaTc, EbTc = symmetrize(cl_th_dict["TaEd"]), symmetrize(
        cl_th_dict["TaEc"]), symmetrize(cl_th_dict["TbEc"]), symmetrize(
            cl_th_dict["TbEd"]), symmetrize(cl_th_dict["EaTc"]), symmetrize(cl_th_dict["EbTc"])

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    analytic_cov = np.zeros((3 * n_bins, 3 * n_bins))

    analytic_cov[:n_bins, :n_bins] = bin_mat(TaTc * TbTd * coupling_dict["TaTcTbTd"] +
                                             TaTd * TbTc * coupling_dict["TaTdTbTc"], binning_file,
                                             lmax)  #TTTT
    analytic_cov[n_bins:2 * n_bins, n_bins:2 * n_bins] = bin_mat(
        TaTc * EbEd * coupling_dict["TaTcPbPd"] + TaEd * EbTc * coupling_dict["TaPdPbTc"],
        binning_file, lmax)  #TETE
    analytic_cov[2 * n_bins:3 * n_bins, 2 * n_bins:3 * n_bins] = bin_mat(
        EaEc * EbEd * coupling_dict["PaPcPbPd"] + EaEd * EbEc * coupling_dict["PaPdPbPc"],
        binning_file, lmax)  #EEEE
    analytic_cov[n_bins:2 * n_bins, :n_bins] = bin_mat(TaTc * TbEd * coupling_dict["TaTcTbPd"] +
                                                       TaEd * TbTc * coupling_dict["TaPdTbTc"],
                                                       binning_file, lmax)  #TTTE
    analytic_cov[2 * n_bins:3 * n_bins, :n_bins] = bin_mat(
        TaEc * TbEd * coupling_dict["TaPcTbPd"] + TaEd * TbEc * coupling_dict["TaPdTbPc"],
        binning_file, lmax)  #TTEE
    analytic_cov[2 * n_bins:3 * n_bins, n_bins:2 * n_bins] = bin_mat(
        EaTc * EbEd * coupling_dict["PaTcPbPd"] + EaEd * EbTc * coupling_dict["TaPdTbPc"],
        binning_file, lmax)  #TEEE

    analytic_cov = np.tril(analytic_cov) + np.triu(analytic_cov.T, 1)

    mbb_inv_ab = extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd = extract_TTTEEE_mbb(mbb_inv_cd)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)

    return analytic_cov


def extract_TTTEEE_mbb(mbb_inv):
    """The mode coupling marix is computed for T,E,B but for now we only construct analytical
       covariance matrix for T and E The B modes is complex with important E->B leakage,
       this routine extract the T and E part of the mode coupling matrix

    Parameters
    ----------

    mbb_inv: 2d array
      the inverse spin0 and 2 mode coupling matrix

    """

    mbb_inv_array = so_mcm.coupling_dict_to_array(mbb_inv)
    mbb_array = np.linalg.inv(mbb_inv_array)
    n_bins = int(mbb_array.shape[0] / 9)
    mbb_array_select = np.zeros((3 * n_bins, 3 * n_bins))
    mbb_array_select[:n_bins, :n_bins] = mbb_array[:n_bins, :n_bins]
    mbb_array_select[n_bins:2 * n_bins, n_bins:2 * n_bins] = mbb_array[n_bins:2 * n_bins,
                                                                       n_bins:2 * n_bins]
    mbb_array_select[2 * n_bins:3 * n_bins,
                     2 * n_bins:3 * n_bins] = mbb_array[5 * n_bins:6 * n_bins,
                                                        5 * n_bins:6 * n_bins]
    mbb_inv_array = np.linalg.inv(mbb_array_select)
    return mbb_inv_array


def cov2corr(cov):
    """Go from covariance to correlation matrix, also setting the diagonal to zero

    Parameters
    ----------
    cov: 2d array
      the covariance matrix
    """

    d = np.sqrt(cov.diagonal())
    corr = ((cov.T / d).T) / d - np.identity(cov.shape[0])
    return corr


def selectblock(cov, spectra, n_bins, block="TTTT"):
    """Select a block in a spin0 and 2 covariance matrix

    Parameters
    ----------

    cov: 2d array
      the covariance matrix
    spectra: list of strings
      the arangement of the different block
    n_bins: int
      the number of bins for each block
    block: string
      the block you want to look at
    """

    if spectra is None:
        print("cov mat of spin 0, no block selection needed")
        return cov

    blockindex = {}
    for c1, s1 in enumerate(spectra):
        for c2, s2 in enumerate(spectra):
            blockindex[s1 + s2] = [c1 * n_bins, c2 * n_bins]
    id1 = blockindex[block][0]
    id2 = blockindex[block][1]
    cov_select = cov[id1:id1 + n_bins, id2:id2 + n_bins]
    return cov_select


def delta2(a, b):
    """Simple delta function
    """

    return 1 if a == b else 0


def delta3(a, b, c):
    """Delta function (3 variables)
    """

    return 1 if (a == b) & (b == c) else 0


def delta4(a, b, c, d):
    """Delta function (4 variables)
    """

    return 1 if (a == b) & (b == c) & (c == d) else 0


def f(a, b, c, d, ns):
    """f combination factor in the covariance computation
       (https://www.overleaf.com/read/fvrcvgbzqwrz)
    """

    result = 1. * ns[a] * (ns[c] * ns[d] * delta2(a, b) - ns[c] * delta3(a, b, d) -
                           ns[d] * delta3(a, b, c) + delta4(a, b, c, d))
    result /= (ns[c] * ns[d] * (ns[a] - delta2(a, c)) * (ns[b] - delta2(b, d)))
    return result


def g(a, b, c, d, ns):
    """g combination factor in the covariance computation
       (https://www.overleaf.com/read/fvrcvgbzqwrz)
    """

    result = 1. * ns[a] * (ns[c] * delta2(a, b) * delta2(c, d) - delta4(a, b, c, d))
    result /= (ns[a] * ns[b] * (ns[c] - delta2(a, c)) * (ns[d] - delta2(b, d)))
    return result


def chi(alpha, gamma, beta, eta, ns, Dl, DNl, name="TTTT"):
    """doc not ready yet
    """
    exp_alpha = alpha.split("_")[0]
    exp_beta = beta.split("_")[0]
    exp_gamma = gamma.split("_")[0]
    exp_eta = eta.split("_")[0]

    RX = name[0] + name[2]
    SY = name[1] + name[3]
    chi_value = Dl[alpha, gamma, RX] * Dl[beta, eta, SY]
    chi_value += Dl[alpha, gamma, RX] * DNl[beta, eta, SY] * f(
        exp_beta, exp_eta, exp_alpha, exp_gamma, ns) + Dl[beta, eta, SY] * DNl[
            alpha, gamma, RX] * f(exp_alpha, exp_gamma, exp_beta, exp_eta, ns)
    chi_value += g(exp_alpha, exp_gamma, exp_beta, exp_eta, ns) * DNl[alpha, gamma,
                                                                      RX] * DNl[beta, eta, SY]

    print("RX", RX)
    print("SY", SY)
    print("ns", ns)
    print(r"f_{%s %s}^{%s %s}" % (exp_beta, exp_eta, exp_alpha, exp_gamma),
          f(exp_beta, exp_eta, exp_alpha, exp_gamma, ns))
    print(r"f_{%s %s}^{%s %s}" % (exp_alpha, exp_gamma, exp_beta, exp_eta),
          f(exp_alpha, exp_gamma, exp_beta, exp_eta, ns))
    print(r"g_{%s %s %s %s}" % (exp_alpha, exp_gamma, exp_beta, exp_eta),
          g(exp_alpha, exp_gamma, exp_beta, exp_eta, ns))

    chi_value = symmetrize(chi_value, mode="arithm")

    return chi_value


# def calc_cov_lensed(noise_uK_arcmin,
#                     fwhm_arcmin,
#                     lmin,
#                     lmax,
#                     camb_lensed_theory_file,
#                     camb_unlensed_theory_file,
#                     output_dir,
#                     overwrite=False):
#     """ Wrapper around lenscov (https://github.com/JulienPeloton/lenscov). heavily borrowed
#         from covariance.py compute lensing induced non-gaussian part of covariance matrix
#     """

#     try:
#         import lib_covariances, lib_spectra, misc, util
#     except:
#         print(
#             "[ERROR] failed to load lenscov modules. Make sure that lenscov is properly installed")
#         print("[ERROR] Note: lenscov is not yet python3 compatible")
#     print("[WARNING] calc_cov_lensed requires MPI to be abled")
#     from pspy import so_mpi
#     from mpi4py import MPI
#     so_mpi.init(True)

#     rank, size = so_mpi.rank, so_mpi.size
#     ## The available blocks in the code
#     blocks = ["TTTT", "EEEE", "BBBB", "EEBB", "TTEE", "TTBB", "TETE", "TTTE", "EETE", "TEBB"]

#     ## Initialization of spectra
#     cls_unlensed = lib_spectra.get_camb_cls(fname=os.path.abspath(camb_unlensed_theory_file),
#                                             lmax=lmax)
#     cls_lensed = lib_spectra.get_camb_cls(fname=os.path.abspath(camb_lensed_theory_file),
#                                           lmax=lmax)

#     file_manager = util.file_manager("covariances_CMBxCMB",
#                                      "pspy",
#                                      spec="v1",
#                                      lmax=lmax,
#                                      force_recomputation=overwrite,
#                                      folder=output_dir,
#                                      rank=rank)

#     if file_manager.FileExist is True:
#         if rank == 0:
#             print("Already computed in %s/" % output_dir)
#     else:
#         cov_order0_tot, cov_order1_tot, cov_order2_tot, junk = lib_covariances.analytic_covariances_CMBxCMB(
#             cls_unlensed,
#             cls_lensed,
#             lmin=lmin,
#             blocks=blocks,
#             noise_uK_arcmin=noise_uK_arcmin,
#             TTcorr=False,
#             fwhm_arcmin=fwhm_arcmin,
#             MPI=MPI,
#             use_corrfunc=True,
#             exp="pspy",
#             folder_cache=output_dir)
#         array_to_save = [cov_order0_tot, cov_order1_tot, cov_order2_tot, blocks]

#         if file_manager.FileExist is False and rank == 0:
#             file_manager.save_data_on_disk(array_to_save)

# def load_cov_lensed(cov_lensed_file, include_gaussian_part=False):
#     """wrapper around lenscov (https://github.com/JulienPeloton/lenscov).
#     include_gaussian_part: if False, it returns only lensing induced non-gaussin parts
#     """
#     covs = {}

#     input_data = pickle.load(open(cov_lensed_file, "r"))["data"]
#     cov_G = input_data[0]
#     cov_NG1 = input_data[1]
#     cov_NG2 = input_data[2]
#     combs = input_data[3]

#     for ctr, comb in enumerate(combs):
#         covs[comb] = cov_NG1[ctr].data + cov_NG2[ctr].data
#         if include_gaussian_part:
#             covs[comb] = covs[comb] + cov_G[ctr].data

#     return covs
