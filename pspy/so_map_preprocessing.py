import numpy as np, pylab as plt
from pspy import flat_tools, pspy_utils
from pixell import enmap, utils

def build_std_filter(shape, wcs, vk_mask, hk_mask, dtype=np.float64):
    ly, lx  = enmap.laxes(shape, wcs, method="auto")
    ly = ly.astype(dtype)
    lx = lx.astype(dtype)
    filter = enmap.ones(shape[-2:], wcs, dtype)
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))
    filter[:, id_vk] = 0
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

def build_sigurd_filter(shape, wcs, lbounds, dtype=np.float64):
    ly, lx  = enmap.laxes(shape, wcs, method="auto")
    ly = ly.astype(dtype)
    lx = lx.astype(dtype)
    lbounds = np.asarray(lbounds)
    if lbounds.ndim < 2:
        lbounds = np.broadcast_to(lbounds, (1,2))
    if lbounds.ndim > 2 or lbounds.shape[-1] != 2:
        raise ValueError("lbounds must be [:,{ly,lx}]")
    filter = enmap.ones(shape[-2:], wcs, dtype)
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
    lmap  = enmap.modlmap(map.data.shape, map.data.wcs, method="auto")
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
