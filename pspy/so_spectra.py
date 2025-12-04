"""
Routines for power spectra estimation and debiasing.
"""
import healpy as hp, numpy as np
from pspy import pspy_utils,so_mcm
from pixell import curvedsky

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
      
    Return
    ----------

    The function returns the multipole array l and cl a 1d power spectrum array or
    cl_dict (for spin0 and spin2) a dictionnary of cl with entry spectra
    """

    if spectra is None:
        if alm2 is None:
            cls = hp.sphtfunc.alm2cl(alm1)
        else:
            cls = hp.sphtfunc.alm2cl(alm1, alm2)
        l = np.arange(len(cls))
        return l, cls
        
    cls = hp.sphtfunc.alm2cl(alm1, alm2)
    l = np.arange(len(cls[0]))
    """ spectra_healpix=[TT,EE,BB,TE,EB,TB] """
    spectra_healpix = [spectra[0], spectra[5], spectra[8], spectra[1], spectra[6], spectra[2]]
    cl_dict = {spec: cls[i] for i, spec in enumerate(spectra_healpix)}

    if alm2 is None:
        #here we set ET=TE, BE=EB and BT=TB
        cl_dict[spectra[3]] = cl_dict[spectra[1]]
        cl_dict[spectra[7]] = cl_dict[spectra[6]]
        cl_dict[spectra[4]] = cl_dict[spectra[2]]
    else:
        #here we need to recompute cls inverting the order of the alm to get ET,BT and BE
        cls = hp.sphtfunc.alm2cl(alm2, alm1)
        # spectra_healpix=[TT,EE,BB,ET,BE,BT]
        spectra_healpix = [spectra[0], spectra[5], spectra[8], spectra[3], spectra[7], spectra[4]]
        for i, spec in enumerate(spectra_healpix):
            cl_dict[spec] = cls[i]

    return l, cl_dict
    
    
def get_spectra_pixell(alm1, alm2=None, spectra=None, apply_pspy_cut=False):
    """Get the power spectrum of alm1 and alm2, we use pixell.alm2cl (this is faster)
    Parameters
    ----------
    alm1: 1d array
      the spherical harmonic transform of map1
    alm2: 1d array
      the spherical harmonic transform of map2
    spectra: list of strings
    needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    apply_pspy_cut : bool
      If the returned spectra should cut 0,1 and last elements
  
    Return
    ----------
    
    The function returns the multipole array l and cl a 1d power spectrum array or
    cl_dict (for spin0 and spin2) a dictionnary of cl with entry spectra
    """
    
    if spectra is None:
        if alm2 is None:
            out = curvedsky.alm2cl(alm1)
        else:
            out = curvedsky.alm2cl(alm1, alm2)
        
        l = np.arange(len(out))
        if apply_pspy_cut:
            l = l[2:-1]
            out = out[2:-1]   

    else:
        cl = curvedsky.alm2cl(alm1[:, None], alm2[None, :])
        out = {}
        for i, l1 in enumerate(["T","E","B"]):
            for j, l2 in enumerate(["T","E","B"]):
                out[l1+l2] = cl[i, j]

        l = np.arange(cl.shape[-1])
        if apply_pspy_cut:
            l = l[2:-1]
            for k, v in out.items():
                out[k] = v[2:-1]
      
    return l, out


def deconvolve_mode_coupling_matrix(l, ps, inv_mode_coupling_matrix, spectra=None):
    """deconvolve the mode coupling matrix
    Parameters
    ----------
    l: 1d array
        the multipoles or the location of the center of the bins
    cl: 1d array or dict of 1d array
        the power spectra, can be a 1d array (spin0) or a dictionnary (spin0 and spin2)
    inv_mode_coupling_matrix: 2d array (or dict of 2d arrays)
        the inverse of the  mode coupling matrix can be binned or not
    spectra: list of string
        needed for spin0 and spin2 cross correlation, the arrangement of the spectra

    """

    n_element = len(l)

    if spectra is None:
        ps = np.dot(inv_mode_coupling_matrix, ps)
    else:
        ps["TT"] = inv_mode_coupling_matrix["spin0xspin0"] @ ps["TT"]
        for spec in ["TE", "TB"]:
            ps[spec] = inv_mode_coupling_matrix["spin0xspin2"] @ ps[spec]
        for spec in ["ET", "BT"]:
            ps[spec] = inv_mode_coupling_matrix["spin2xspin0"] @ ps[spec]

        vec = []
        for spec in ["EE", "EB", "BE", "BB"]:
            vec = np.append(vec, ps[spec])
        vec = inv_mode_coupling_matrix["spin2xspin2"] @ vec
        for i, spec in enumerate(["EE", "EB", "BE", "BB"]):
            ps[spec] = vec[i * n_element:(i + 1) * n_element]

    return l, ps


def bin_spectra(l, cl, binning_file, lmax, type, spectra=None, mbb_inv=None, binned_mcm=True):

    """Bin the power spectra according to a binning file and optionnaly deconvolve the (binned) mode coupling matrix
    
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
    type: string
      the type of binning, either bin Cl or bin Dl
    spectra: list of string
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    mbb_inv: 2d array
      optionnaly apply the inverse of the  mode coupling matrix to debiais the spectra
    binned_mcm: boolean
      if mbb_inv is not None, specify if it's binned or not
      
    Return
    ----------
      
    The function return the binned multipole array bin_c and a 1d power spectrum
    array (or dictionnary of 1d power spectra if spectra is not None).
    """

    bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    # the alm2cl return cl starting at l = 0, we use spectra from l = 2
    # this is due in particular to the fact that the mcm is computed only for l>=2
    
    l = np.arange(2, lmax)
    if spectra is None: cl = cl[l]
    else: cl = {f: cl[f][l] for f in spectra}
    
    if type == "Dl": fac = (l * (l + 1) / (2 * np.pi))
    elif type == "Cl": fac = l * 0 + 1

    # we have the option to deconvolve the l-by-l mode coupling matrix
    if (mbb_inv is not None) & (binned_mcm == False):
        l, cl = deconvolve_mode_coupling_matrix(l, cl, mbb_inv, spectra)
    
    # Now the binning part
    if spectra is None:
        ps = np.zeros(n_bins)
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            ps[ibin] = (cl[loc] * fac[loc]).mean()
    else:
        vec = []
        for f in spectra:
            binned_power = np.zeros(n_bins)
            for ibin in range(n_bins):
                loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
                binned_power[ibin] = (cl[f][loc] * fac[loc]).mean()
            vec = np.append(vec, binned_power)
        ps = vec2spec_dict(n_bins, vec, spectra)

    # we have the option to deconvolve the binned mode coupling matrix
    if (mbb_inv is not None) & (binned_mcm == True):
        lb, ps = deconvolve_mode_coupling_matrix(lb, ps, mbb_inv, spectra)
    return lb, ps

def get_binning_matrix(bin_lo, bin_hi, lmax, cltype='Dl'):
    """Returns P_bl, the binning matrix that turns C_ell into C_b or D_b."""
    l = np.arange(2, lmax) # assumes 2:lmax ordering
    if cltype == 'Dl':
        fac = (l * (l + 1) / (2 * np.pi))
    elif cltype == 'Cl':
        fac = l * 0 + 1
    nbins = len(bin_lo)
    Pbl = np.zeros((nbins, lmax-2))
    for ibin in range(nbins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))[0]
        Pbl[ibin, loc] = fac[loc] / len(loc)
    return Pbl

def spec_dict2vec(spec_dict, spectra=None):
    """Take a power spectra dictionary of power spectra and return a vector.
    The dictionary should be of the form {spec: val, ...}
    For example {'TT': cl_TT, 'TE': cl_TE, ..., 'BB': cl_BB}
    
    Parameters
    ----------
    spec_dict: dict of 1d array
      a dict containing spectra
    spectra: list of strings, optional
      the desired ordering of the spectra for example:
      ['TT','TE','TB','ET','BT','EE','EB','BE','BB']. If not provided, the keys
      of spec_dict (in their insertion order)
    """
    if spectra is None:
        spectra = spec_dict.keys()
    return np.concatenate([spec_dict[spec] for spec in spectra])

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

def spin2spin_array_matmul_spec_dict(spin2spin_array, spec_dict, spin0=False,
                                     return_as_vec=False):
    """Apply a 5 (or 1) block spinxspin array to a spectrum stored in a
    dictionary by polarization pair.

    Parameters
    ----------
    spin2spin_array : ({5}, x, y) np.ndarray
        The blocks of the spinxspin matrix: 0x0, 0x2, 2x0, ++, --. If spin0 is
        True, then this is just the (x, y)-shaped 0x0 block.
    spec_dict: dict of 1d array
        A dictionary holding keys like 'TT', 'BE' and pointing to 1d arrays.
    spin0 : bool, optional
        If True, then spin2spin_array is just the 0x0 block, by default False.
        In that case, it is copied along the block-diagonal for all the
        spectra blocks.
    return_as_vec : bool, optional
        If True, return the output dictionary of spectra as a concatenated 1d 
        vector array, by default False.

    Returns
    -------
    dict or np.ndarray
        The output spectrum either as a dictionary behind the same polarization
        pair keys as the input, or a concatenated vector thereof.
    """
    out_dict = {}
    if spin0:
        obj = spin2spin_array.squeeze()
        assert obj.ndim == 2, \
            f'If spin0, obj must be squeezable to a 2d array'
        for spec in spec_dict:
            out_dict[spec] = obj @ spec_dict[spec]
    else:
        obj = spin2spin_array
        assert obj.ndim == 3 and obj.shape[0] == 5, \
            f'If not spin0, obj must be a 3d array whose first axis has size 5'
        spin_diag_idxs = {'TT': 0, 'TE': 1, 'TB': 1, 'ET': 2, 'BT': 2}
        spin2_pairs_and_signs = {
            'EE': ('BB', 1),
            'EB': ('BE', -1),
            'BE': ('EB', -1),
            'BB': ('EE', 1)
        }
    
        for spec in spec_dict:
            if spec in spin_diag_idxs:
                idx = spin_diag_idxs[spec]
                out_dict[spec] = obj[idx] @ spec_dict[spec]
            else:
                pair, sign = spin2_pairs_and_signs[spec]
                out_dict[spec] = obj[3] @ spec_dict[spec] + sign * obj[4] @ spec_dict[pair]
    
    if return_as_vec:
        out = spec_dict2vec(out_dict)
    else:
        out = out_dict

    return out

def write_ps(file_name, l, ps, type, spectra=None):
    """Write down the power spectra to disk.
    
    Parameters
    ----------
    file_name: str
      the name of the file to write the spectra
    l: 1d array
      the multipoles (or binned multipoles)
    ps: 1d array or dict of 1d array
      the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    type:  string
      'Cl' or 'Dl'
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    if spectra is None:
        ps_list = [ps]
        ps_list[0:0] = [l]
        str = "l"
        str += " %s"%type
        ps_list = np.array(ps_list)
        np.savetxt(file_name, np.transpose(ps_list), header=str)
    else:
        ps_list = [ps[f] for f in spectra]
        ps_list[0:0] = [l]
        str = "l"
        for l1 in spectra:
            str += " %s_%s"%(type, l1)
        ps_list = np.array(ps_list)
        np.savetxt(file_name, np.transpose(ps_list), header=str)

def write_ps_matrix(file_name, l, ps_matrix, type, spectra=None):
    """Write down the power spectra to disk, except the input file
    can be a (3, 3, nl) np.ndarray, not a dict.
    
    Parameters
    ----------
    file_name: str
      the name of the file to write the spectra
    l: 1d array
      the multipoles (or binned multipoles)
    ps: (nl,) or (3, 3, nl) np.ndarray
      the power spectrum in 'TEB x TEB' ordering (if 3x3)
    type:  string
      'Cl' or 'Dl'
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """
    if spectra is None:
        ps = ps_matrix
    else:
        ps = {}
        for pi, poli in enumerate('TEB'):
            for pj, polj in enumerate('TEB'):
                ps[poli + polj] = ps_matrix[pi, pj]
    
    write_ps(file_name, l, ps, type, spectra=spectra)

def read_ps(file_name, spectra=None, return_type=None, return_dtype=None):
    """Read the power spectra.
        
    Parameters
    ----------
    file_name: str
      the name of the file to read the spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the desired spectra. NOTE:
      the spectra are extracted by their header so the order does not need to
      match
    return_type : str, optional
      If not None, return loaded data as either 'Dl' or 'Cl'. The type-on-disk
      if inferred from the header, and the data converted between types if 
      necessary. If None, return as-is.
    return_dtype : np.dtype, optional
      Return as this dtype. If None, return as the default np.loadtxt dtype.
      
    Return
    ----------
      
    The function return the multipole l (or binned multipoles) and a 1d power spectrum
    array or a dictionnary of power spectra (if spectra is not None).
    """
    with open(file_name) as f:
      header = f.readline()
      assert header[:2] == '# '
      assert header[-1] == '\n'
      header = header[2:-1].split(' ')

    type = header[1][:2]
    spec_list = ['l']
    for h in header[1:]: # skip l column
      _type, _spec = h.split('_') 
      assert _type == type, f'Inconsistent type in columns, expected {type}'
      spec_list.append(_spec) # holds order of spectra in file

    if return_dtype is None:
      data = np.loadtxt(file_name)
    else:
      data = np.loadtxt(file_name, dtype=return_dtype)
    l = data[:, 0]

    if return_type is None:
      if spectra is None:
        ps = data[:, 1]
      else:
        ps = {spec: data[:, spec_list.index(spec)] for spec in spectra} # gets spec by order
    else:
      if type == return_type and return_type in ('Cl', 'Dl'):
          lfac = 1
      elif type == 'Cl' and return_type == 'Dl':
          lfac = l*(l + 1) / (2*np.pi)
      elif type == 'Dl' and return_type == 'Cl':
          lfac = 2*np.pi / (l*(l + 1))
      
      if spectra is None:
          ps = data[:, 1] * lfac
      else:
          ps = {spec: data[:, spec_list.index(spec)] * lfac for spec in spectra} # gets spec by order
    
    return l, ps
    
def read_ps_matrix(file_name, spectra=None, return_type=None, return_dtype=None):
    """Read the power spectra, except into a (3, 3, nl) np.ndarray, not a dict.
        
    Parameters
    ----------
    file_name: str
      the name of the file to read the spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the desired spectra. NOTE:
      the spectra are extracted by their header so the order does not need to
      match. If a pol pair is missing, it is replaced with zeros.
    return_type : str, optional
      If not None, return loaded data as either 'Dl' or 'Cl'. The type-on-disk
      if inferred from the header, and the data converted between types if 
      necessary. If None, return as-is.
    return_dtype : np.dtype, optional
      Return as this dtype. If None, return as the default np.loadtxt dtype.
      
    Return
    ----------
      
    The function return the multipole l (or binned multipoles) and a 1d power spectrum
    array or a (3, 3) array of power spectra in 'TEB x TEB' ordering
    (if spectra is not None).
    """
    l, ps = read_ps(file_name, spectra=spectra, return_type=return_type,
                    return_dtype=return_dtype)

    if spectra is None:
        ps_matrix = ps
    else:
        ps_matrix = np.zeros((3, 3, l.size), dtype=l.dtype)
        for pi, poli in enumerate('TEB'):
            for pj, polj in enumerate('TEB'):
                spec = poli + polj
                if spec in ps:
                    ps_matrix[pi, pj] = ps[spec]
        ps_matrix = np.array(ps_matrix).reshape(3, 3, -1)
    
    return l, ps_matrix

def write_ps_hdf5(file, spec_name, l, ps, spectra=None):
    """Write down the power spectra in a hdf5 file.
        
    Parameters
    ----------
    file: hdf5
      the name of the hdf5 file
    spec_name: string
      the name of the group in the hdf5 file
    l: 1d array
      the multipoles (or binned multipoles)
    ps: 1d array or dict of 1d array
      the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    def array_from_dict(l,ps,spectra=None):
        array = []
        array += [l]
        if spectra == None:
            array += [ps]
        else:
            for spec in spectra:
                array += [ps[spec]]
        return array
    
    group = file.create_group(spec_name)
    array = array_from_dict(l, ps, spectra=spectra)
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

    Return
    ----------

    The function returns the multipole l (or binned multipoles) and a 1d power spectrum
    array (or a dictionnary of power spectra if spectra is not None).

    """
    
    spec = file[spec_name]
    data = np.array(spec["data"]).T
    
    l = data[:,0]
    if spectra is None:
        ps = data[:, 1]
    else:
        ps = {spec: data[:, i + 1] for i, spec in enumerate(spectra)}

    return l, ps


