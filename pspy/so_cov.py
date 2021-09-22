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

        coupling_dict["TaTcTbTd"] = coupling
        coupling_dict["TaTdTbTc"] = coupling
        
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

            coupling_dict[name] = coupling[id_cov]

        
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
            coupling_dict[name] = coupling[index]

    if save_file is not None:
        np.save("%s.npy"%save_file, coupling)

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
            coupling_b[i,j] = np.mean(mat[bin_lo[i]-2:bin_hi[i]-1, bin_lo[j]-2:bin_hi[j]-1])
    return coupling_b

def cov_spin0(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd):
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
    """

    cov = symmetrize(Clth_dict["TaTc"]) * symmetrize(Clth_dict["TbTd"]) * coupling_dict["TaTcTbTd"]
    cov += symmetrize(Clth_dict["TaTd"]) * symmetrize(Clth_dict["TbTc"]) * coupling_dict["TaTdTbTc"]
    
    analytic_cov = bin_mat(cov, binning_file, lmax)
    
    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)

    return analytic_cov

    
def cov_spin0and2(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv_ab, mbb_inv_cd):
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
    """

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    analytic_cov = np.zeros((4*n_bins, 4*n_bins))
    
    speclist = ["TT", "TE", "ET", "EE"]
    for i, sp1 in enumerate(speclist):
        for j, sp2 in enumerate(speclist):
            if i > j : continue
                
            id0 = sp1[0] + "a" + sp2[0] + "c"
            id1 = sp1[1] + "b" + sp2[1] + "d"
            id2 = sp1[0] + "a" + sp2[1] + "d"
            id3 = sp1[1] + "b" + sp2[0] + "c"

            M = symmetrize(Clth_dict[id0]) * symmetrize(Clth_dict[id1]) * coupling_dict[id0.replace("E","P") + id1.replace("E","P")]
            M += symmetrize(Clth_dict[id2]) * symmetrize(Clth_dict[id3]) * coupling_dict[id2.replace("E","P") + id3.replace("E","P")]

            analytic_cov[i * n_bins:(i + 1) * n_bins, j * n_bins:(j + 1) * n_bins] = bin_mat(M, binning_file, lmax)

    analytic_cov = np.triu(analytic_cov) + np.tril(analytic_cov.T, -1)

    mbb_inv_ab = extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd = extract_TTTEEE_mbb(mbb_inv_cd)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)

    return analytic_cov



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
    exp_alpha, f_alpha = alpha.split("_")
    exp_beta, f_beta = beta.split("_")
    exp_gamma, f_gamma = gamma.split("_")
    exp_eta, f_eta = eta.split("_")
    
    RX = id[0] + id[1]
    SY = id[2] + id[3]
    chi = Dl[alpha, gamma, RX] * Dl[beta, eta, SY]
    chi += Dl[alpha, gamma, RX] * DNl[beta, eta, SY] * f(exp_beta, exp_eta, exp_alpha, exp_gamma, ns)
    chi += Dl[beta, eta, SY] * DNl[alpha, gamma, RX] * f(exp_alpha, exp_gamma, exp_beta, exp_eta, ns)
    chi += g(exp_alpha, exp_gamma, exp_beta, exp_eta, ns) * DNl[alpha, gamma, RX] * DNl[beta, eta, SY]
    
    chi= symmetrize(chi, mode="arithm")
    
    return chi


def chi_planck(alpha, gamma, beta, eta, ns, Dl, DNl, id="TTTT"):
    """doc not ready yet
        """
    exp_alpha, f_alpha = alpha.split("_")
    exp_beta, f_beta = beta.split("_")
    exp_gamma, f_gamma = gamma.split("_")
    exp_eta, f_eta = eta.split("_")
    
    RX = id[0] + id[1]
    SY = id[2] + id[3]
    
    if RX == "TE":
        Dl[alpha, gamma, RX] = symmetrize(Dl[alpha, gamma, RX], mode="arithm")
        DNl[alpha, gamma, RX] = symmetrize(DNl[alpha, gamma, RX], mode="arithm")

    else:
        Dl[alpha, gamma, RX] = symmetrize(Dl[alpha, gamma, RX], mode="geo")
        DNl[alpha, gamma, RX] = symmetrize(DNl[alpha, gamma, RX], mode="geo")

    if SY == "TE":
        Dl[beta, eta, SY] = symmetrize(Dl[beta, eta, SY] , mode="arithm")
        DNl[beta, eta, SY] = symmetrize(DNl[beta, eta, SY] , mode="arithm")
    else:
        Dl[beta, eta, SY] = symmetrize(Dl[beta, eta, SY] , mode="geo")
        DNl[beta, eta, SY] = symmetrize(DNl[beta, eta, SY] , mode="geo")

    
    chi = Dl[alpha, gamma, RX] * Dl[beta, eta, SY]
    chi += Dl[alpha, gamma, RX] * DNl[beta, eta, SY] * f(exp_beta, exp_eta, exp_alpha, exp_gamma, ns)
    chi += Dl[beta, eta, SY] * DNl[alpha, gamma, RX] * f(exp_alpha, exp_gamma, exp_beta, exp_eta, ns)
    chi += g(exp_alpha, exp_gamma, exp_beta, exp_eta, ns) * DNl[alpha, gamma, RX] * DNl[beta, eta, SY]
    
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

                       
                       

#def chi_old(alpha, gamma, beta, eta, ns, ls, Dl, DNl, id="TTTT"):
#    """doc not ready yet
#    """
#    exp_alpha, f_alpha = alpha.split("_")
#    exp_beta, f_beta = beta.split("_")
#    exp_gamma, f_gamma = gamma.split("_")
#    exp_eta, f_eta = eta.split("_")

#    RX = id[0] + id[2]
#    SY = id[1] + id[3]
#    chi = Dl[alpha, gamma, RX] * Dl[beta, eta, SY]
#    chi += Dl[alpha, gamma, RX] * DNl[beta, eta, SY] * f(exp_beta, exp_eta, exp_alpha, exp_gamma, ns)
#    chi += Dl[beta, eta, SY] * DNl[alpha, gamma, RX] *f(exp_alpha, exp_gamma, exp_beta, exp_eta, ns)
#    chi += g(exp_alpha, exp_gamma, exp_beta, exp_eta, ns) * DNl[alpha, gamma, RX] * DNl[beta, eta, SY]

#    print ("RX",RX)
#    print ("SY",SY)
#    print ("ns",ns)
#    print (r"f_{%s %s}^{%s %s}"%(exp_beta,exp_eta,exp_alpha,exp_gamma),f(exp_beta,exp_eta,exp_alpha,exp_gamma,ns))
#    print (r"f_{%s %s}^{%s %s}"%(exp_alpha,exp_gamma,exp_beta,exp_eta),f(exp_alpha,exp_gamma,exp_beta,exp_eta,ns))
#    print (r"g_{%s %s %s %s}"%(exp_alpha,exp_gamma,exp_beta,exp_eta),g(exp_alpha,exp_gamma,exp_beta,exp_eta,ns))

#    chi= symmetrize(chi, mode="arithm")

#    return chi


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
