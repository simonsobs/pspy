import numpy as np, pylab as plt
from scipy.integrate import quad
from pspy import flat_tools, pspy_utils
from pixell import enmap, utils
from itertools import product

def build_std_filter(shape, wcs, vk_mask, hk_mask, dtype=np.float64,
                     shape_y=None, wcs_y=None):
    """Construct a "standard," binary-cross style filter.

    Parameters
    ----------
    shape : (ny, nx) tuple
        Shape of map and therefore shape of DFT.
    wcs : astropy.wcs.WCS
        wcs describing the map, used in enmap.laxes
    vk_mask : (kx0, kx1) tuple
        Modes between kx0 and kx1 (exclusive) are set to 0, otherwise 1.
    hk_mask : (ky0, ky1) tuple
        Modes between ky0 and ky1 (exclusive) are set to 0, otherwise 1.
    dtype : np.dtype, optional
        Type of output filter array, by default np.float64.
    shape_y : (ny, nx) tuple, optional
        If provided, together with wcs_y, used to override the y-spacing and
        shape in Fourier space in enmap.laxes, by default None.
    wcs_y : astropy.wcs.WCS, optional
        If provided, together with shape_y, used to override the y-spacing and
        shape in Fourier space in enmap.laxes, by default None.

    Returns
    -------
    (ny, nx) enmap.ndmap
        The binary-cross filter.
    """
    ly, lx  = enmap.laxes(shape, wcs, method="auto")
    if shape_y is not None and wcs_y is not None:
        ly, _  = enmap.laxes(shape_y, wcs_y, method="auto")
    ly = ly.astype(dtype)
    lx = lx.astype(dtype)
    filter = enmap.ones((ly.size, lx.size), wcs, dtype)
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
        filter[:, id_vk] = 0
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))
        filter[id_hk, :] = 0
    return filter

def apply_std_filter(imap, filter):
    filtered_map = imap.copy()
    if imap.ncomp == 1:
        filtered_map.data = enmap.ifft(filter*enmap.fft(imap.data)).real
    else:
        for i in range(imap.ncomp):
            filtered_map.data[i] = enmap.ifft(filter*enmap.fft(imap.data[i])).real

    return filtered_map

def build_sigurd_filter(shape, wcs, lbounds, dtype=np.float64,
                        shape_y=None, wcs_y=None):
    ly, lx  = enmap.laxes(shape, wcs, method="auto")
    if shape_y is not None and wcs_y is not None:
        ly, _  = enmap.laxes(shape_y, wcs_y, method="auto")
    ly = ly.astype(dtype)
    lx = lx.astype(dtype)
    lbounds = np.asarray(lbounds)
    if lbounds.ndim < 2:
        lbounds = np.broadcast_to(lbounds, (1,2))
    if lbounds.ndim > 2 or lbounds.shape[-1] != 2:
        raise ValueError("lbounds must be [:,{ly,lx}]")
    filter = enmap.ones((ly.size, lx.size), wcs, dtype)
    # Apply the filters
    for i , (ycut, xcut) in enumerate(lbounds):
        filter *= 1-(np.exp(-0.5*(ly/ycut)**2)[:,None]*np.exp(-0.5*(lx/xcut)**2)[None,:])
    return filter

def apply_sigurd_filter(imap, ivar, filter, tol=1e-4, ref=0.9):
    """Filter enmap imap with the given 2d fourier filter while
    weithing spatially with ivar"""
    
    filtered_map = imap.copy()
    rhs    = enmap.ifft((1-filter)*enmap.fft(imap.data*ivar)).real
    div    = enmap.ifft((1-filter)*enmap.fft(ivar)).real
    # Avoid division by very low values
    div    = np.maximum(div, np.percentile(ivar[::10,::10],ref*100)*tol)
    filtered_map.data = imap.data - rhs/div
    return filtered_map

def analytical_tf(map, filter, binning_file, lmax):
    lmap  = map.data.modlmap() 
    twod_index = lmap.copy()
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_lo)
    tf = np.zeros(nbins)
    for ii in range(nbins):
        id = np.where((lmap >= bin_lo[ii]) & (lmap <= bin_hi[ii]))
        twod_index *= 0
        twod_index[id] = 1
        bin_area = np.sum(twod_index)
        masked_bin_area= np.sum(twod_index * filter)
        tf[ii] = masked_bin_area / bin_area

    return bin_c, tf

def analytical_tf_vkhk(vk_mask, hk_mask, ells, dtype=np.float64):
    if vk_mask is None:
        vk_mask = [0, 0]
    if hk_mask is None:
        hk_mask = [0, 0]
    assert len(vk_mask) == 2 and len(hk_mask) == 2, \
        f'{len(vk_mask)=} and {len(hk_mask)=} should both be 2'
    
    out = np.zeros(ells.shape, dtype=dtype)
    for vk, hk in product(vk_mask, hk_mask):
        rk = np.sqrt(vk**2 + hk**2)
        out[ells > rk] += .25 - (np.abs(np.arcsin(vk/ells[ells > rk])) + np.abs(np.arcsin(hk/ells[ells > rk]))) / (2*np.pi)
    return out

def analytical_tf_yxint(yxfilter_func, ells, x_fac=1, y_fac=1, dtype=np.float64):
    def x(r, t):
        return r*x_fac*np.cos(t)
    def y(r, t):
        return r*y_fac*np.sin(t)
    
    if x_fac == y_fac:
        def v(r, t):
            return r*x_fac
    else:
        def v(r, t):
            return r*np.sqrt(x_fac**2 * np.sin(t)**2 + y_fac**2 * np.cos(t)**2)

    assert ells.ndim == 1, 'expect 1d ells'
    assert np.all(ells >= 0), 'expect non-negative ells'
    out = np.zeros(ells.shape, dtype=dtype)
    for i, ell in enumerate(ells):
        if i % 100 == 0:
            print(ell)
            
        def vell(t):
            return v(ell, t)
        def rphifilter_func(t):
            return yxfilter_func(y(ell, t), x(ell, t)) * vell(t)
        
        if x_fac == y_fac:
            out[i] = quad(rphifilter_func, 0, 2*np.pi)[0] / (2*np.pi*ell*x_fac)
        else:
            out[i] = quad(rphifilter_func, 0, 2*np.pi)[0] / quad(vell, 0, 2*np.pi)[0]
    
    return out

#def analytical_tf_old(map, binning_file, lmax, vk_mask=None, hk_mask=None):
#    import time
#    t = time.time()
#    ft = flat_tools.fft_from_so_map(map)
#    ft.create_kspace_mask(vertical_stripe=vk_mask, horizontal_stripe=hk_mask)
#    lmap = ft.lmap
#    kmask = ft.kmask
#    twod_index = lmap.copy()

#    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
#    nbins = len(bin_lo)
#    print(lmap.shape)
#    tf = np.zeros(nbins)
#    print("init tf", time.time()-t)
#    for ii in range(nbins):
#        id = np.where( (lmap >= bin_lo[ii]) & (lmap <= bin_hi[ii]))
#        twod_index *= 0
#        twod_index[id] = 1
#        bin_area = np.sum(twod_index)
#        masked_bin_area= np.sum(twod_index * kmask)
#        tf[ii] = masked_bin_area / bin_area
#    print("loop", time.time()-t)

#    return bin_c, tf


#def kspace_filter_old(map, vk_mask=None, hk_mask=None, window=None, normalize=False):
#    filtered_map = map.copy()

 #   ft = flat_tools.fft_from_so_map(map, normalize=normalize)
 #   if window is not None:
 #       ft = flat_tools.get_ffts(map, window, normalize=normalize)

 #   if vk_mask is not None:
 #       id_vk = np.where((ft.lx > vk_mask[0]) & (ft.lx < vk_mask[1]))
 #   if hk_mask is not None:
 #       id_hk = np.where((ft.ly > hk_mask[0]) & (ft.ly < hk_mask[1]))

 #   if map.ncomp == 1:
 #       if vk_mask is not None:
 #           ft.kmap[: , id_vk] = 0.
 #       if hk_mask is not None:
 #           ft.kmap[id_hk , :] = 0.
    
 #   if map.ncomp == 3:
 #      for i in range(3):
 #           if vk_mask is not None:
 #               ft.kmap[i, : , id_vk] = 0.
 #           if hk_mask is not None:
 #               ft.kmap[i, id_hk , :] = 0.

 #   filtered_map.data[:] = ft.map_from_fft(normalize=normalize)
 #   return filtered_map
