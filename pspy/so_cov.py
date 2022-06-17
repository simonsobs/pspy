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


def cov_coupling_spin0(win, lmax, niter=3, save_file=None, l_exact=None, l_band=None, l_toep=None):
    """compute the coupling kernels corresponding to the T only covariance matrix
        
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
            coupling[0] = so_mcm.format_toepliz_fortran(coupling[0], l_toep, maxl)

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
        analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    else:
        full_analytic_cov = np.dot(np.dot(mbb_inv_ab, cov), mbb_inv_cd.T)
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax)
        
    return analytic_cov

def cov_spin0and2(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd, binned_mcm=True):
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

    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_ell = Clth_dict["TaTb"].shape[0]
        
    full_analytic_cov = np.zeros((4 * n_ell, 4 * n_ell))
    
    speclist = ["TT", "TE", "ET", "EE"]
    for i, (W, X) in enumerate(speclist):
        for j, (Y, Z) in enumerate(speclist):
            if i > j : continue
            id0 = W + "a" + Y + "c"
            id1 = X + "b" + Z + "d"
            id2 = W + "a" + Z + "d"
            id3 = X + "b" + Y + "c"

            M = symmetrize(Clth_dict[id0]) * symmetrize(Clth_dict[id1]) * coupling_dict[id0.replace("E","P") + id1.replace("E","P")]
            M += symmetrize(Clth_dict[id2]) * symmetrize(Clth_dict[id3]) * coupling_dict[id2.replace("E","P") + id3.replace("E","P")]
            
            full_analytic_cov[i * n_ell:(i + 1) * n_ell, j * n_ell:(j + 1) * n_ell] = M
    
    full_analytic_cov = np.triu(full_analytic_cov) + np.tril(full_analytic_cov.T, -1)
    
    mbb_inv_ab = extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd = extract_TTTEEE_mbb(mbb_inv_cd)

    if binned_mcm == True:
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
        analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    else:
        full_analytic_cov = np.dot(np.dot(mbb_inv_ab, full_analytic_cov), mbb_inv_cd.T)
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)

    return analytic_cov

def generalized_cov_spin0and2(coupling_dict, id_element, ns, ps_all, nl_all, lmax, binning_file, mbb_inv_ab, mbb_inv_cd, binned_mcm=True, also_return_full_cov=False):

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
    also_return_full_cov: boolean
        an option to also return the lbyl cov (if binned_mcm=False)
        mostly used for debugging

    """

    na, nb, nc, nd = id_element

    n_ell = coupling_dict["TaTcTbTd"].shape[0]
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, n_ell)
    full_analytic_cov = np.zeros((4 * n_ell, 4 * n_ell))

    speclist = ["TT", "TE", "ET", "EE"]
    for i, (W, X) in enumerate(speclist):
        for j, (Y, Z) in enumerate(speclist):
    
            id0 = W + "a" + Y + "c"
            id1 = X + "b" + Z + "d"
            id2 = W + "a" + Z + "d"
            id3 = X + "b" + Y + "c"
        
            M = coupling_dict[id0.replace("E","P") + id1.replace("E","P")] * chi(na, nc, nb, nd, ns, ps_all, nl_all, W + Y + X + Z)
            M += coupling_dict[id2.replace("E","P") + id3.replace("E","P")] * chi(na, nd, nb, nc, ns, ps_all, nl_all, W + Z + X + Y)
            full_analytic_cov[i * n_ell: (i + 1) * n_ell, j * n_ell: (j + 1) * n_ell] = M

    mbb_inv_ab = extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd = extract_TTTEEE_mbb(mbb_inv_cd)

    if binned_mcm == True:
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
        analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    else:
        full_analytic_cov = np.dot(np.dot(mbb_inv_ab, full_analytic_cov), mbb_inv_cd.T)
        analytic_cov = bin_mat(full_analytic_cov, binning_file, lmax, speclist=speclist)
        if also_return_full_cov == True:
            return full_analytic_cov, analytic_cov

    return analytic_cov


def covariance_element_beam(id_element, ps_all, norm_beam_cov, binning_file, lmax):
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
    """
    na, nb, nc, nd = id_element

    sv_alpha, ar_alpha = na.split("&")
    sv_beta, ar_beta = nb.split("&")
    sv_gamma, ar_gamma = nc.split("&")
    sv_eta, ar_eta = nd.split("&")

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    speclist = ["TT", "TE", "ET", "EE"]

    nspec = len(speclist)
    analytic_cov_from_beam = np.zeros((nspec * nbins, nspec * nbins))
    for i, spec1 in enumerate(speclist):
        for j, spec2 in enumerate(speclist):

            M =  (delta2(na, nc) + delta2(na, nd)) * norm_beam_cov[sv_alpha, ar_alpha]
            M += (delta2(nb, nc) + delta2(nb, nd)) * norm_beam_cov[sv_beta, ar_beta]
            M *=  np.outer(ps_all[na, nb, spec1], ps_all[nc, nd, spec2])
        
            analytic_cov_from_beam[i * nbins: (i + 1) * nbins, j * nbins: (j + 1) * nbins] = so_cov.bin_mat(M, binning_file, lmax)

    return analytic_cov_from_beam


def extract_TTTEEE_mbb(mbb_inv):
    """The mode coupling marix is computed for T,E,B but for now we only construct analytical covariance matrix for T and E
    The B modes is complex with important E->B leakage, this routine extract the T and E part of the mode coupling matrix

    Parameters
    ----------

    mbb_inv: 2d array
      the inverse spin0 and 2 mode coupling matrix
    """

    mbb_inv_array = so_mcm.coupling_dict_to_array(mbb_inv)
    mbb_array = np.linalg.inv(mbb_inv_array)
    nbins = int(mbb_array.shape[0] / 9)
    
    mbb_array_select = np.zeros((4*nbins, 4*nbins))
    # TT
    mbb_array_select[0*nbins:1*nbins, 0*nbins:1*nbins] = mbb_array[0*nbins:1*nbins, 0*nbins:1*nbins]
    # TE
    mbb_array_select[1*nbins:2*nbins, 1*nbins:2*nbins] = mbb_array[1*nbins:2*nbins, 1*nbins:2*nbins]
    # ET
    mbb_array_select[2*nbins:3*nbins, 2*nbins:3*nbins] = mbb_array[3*nbins:4*nbins, 3*nbins:4*nbins]
    # EE
    mbb_array_select[3*nbins:4*nbins, 3*nbins:4*nbins] = mbb_array[5*nbins:6*nbins, 5*nbins:6*nbins]

    mbb_inv_array = np.linalg.inv(mbb_array_select)
    
    return mbb_inv_array

def extract_EEEBBB_mbb(mbb_inv):
    """this routine extract the E and B part of the mode coupling matrix

    Parameters
    ----------

    mbb_inv: 2d array
        the inverse spin0 and 2 mode coupling matrix
    """

    mbb_inv_array = so_mcm.coupling_dict_to_array(mbb_inv)
    mbb_array = np.linalg.inv(mbb_inv_array)
    nbins = int(mbb_array.shape[0] / 9)

    mbb_array_select = np.zeros((4*nbins, 4*nbins))
    # EE
    mbb_array_select[0*nbins:1*nbins, 0*nbins:1*nbins] = mbb_array[5*nbins:6*nbins, 5*nbins:6*nbins]
    # EB
    mbb_array_select[1*nbins:2*nbins, 1*nbins:2*nbins] = mbb_array[6*nbins:7*nbins, 6*nbins:7*nbins]
    # BE
    mbb_array_select[2*nbins:3*nbins, 2*nbins:3*nbins] = mbb_array[7*nbins:8*nbins, 7*nbins:8*nbins]
    # BB
    mbb_array_select[3*nbins:4*nbins, 3*nbins:4*nbins] = mbb_array[8*nbins:9*nbins, 8*nbins:9*nbins]
    # EE-BB
    mbb_array_select[0*nbins:1*nbins, 3*nbins:4*nbins] = mbb_array[5*nbins:6*nbins, 8*nbins:9*nbins]
    # BB-EE
    mbb_array_select[3*nbins:4*nbins, 0*nbins:1*nbins] = mbb_array[8*nbins:9*nbins, 5*nbins:6*nbins]

    mbb_inv_array = np.linalg.inv(mbb_array_select)

    return mbb_inv_array


def cov2corr(cov, remove_diag=True):
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

    if spectra == None:
        print ("cov mat of spin 0, no block selection needed")
        return
    else:
        blockindex = {}
        for c1,s1 in enumerate(spectra):
            for c2,s2 in enumerate(spectra):
                blockindex[s1 + s2] = [c1 * n_bins, c2 * n_bins]
    id1 = blockindex[block][0]
    id2 = blockindex[block][1]
    cov_select = cov[id1:id1 + n_bins, id2:id2 + n_bins]
    return cov_select

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
    return mean_a, mean_b, mc_cov[n_el:, :n_el]
