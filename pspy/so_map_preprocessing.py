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

def analytical_std_tf(lmax, vk_masks=None, hk_masks=None, geometries=None,
                      geometries_to_filt=None, dtype=np.float64,
                      binning_file=None):
    """The 'analytic' per-ell kspace filter for the standard binary cross filter.
    Can also be binned. Better than the old analytic function in several ways:
    1. Is substantially faster.
    2. Can provide per-ell transfer functions.
    3. Appears to better match simulations.
    4. Better handles multiple patches with different geometries.

    Parameters
    ----------
    lmax : int
        Maximum ell of the filter, also see binning_file.
    vk_masks : iterable of (-n, n) tuple of ints
        The vertical filter stripe edge in 2d Fourier space for legs of the 
        spectrum that are filtered. If None (the default), assumed no filter(s)
        are applied in this direction. If geometries are provided, then the i'th
        vk_mask corresponds to the i'th entry of geometries_to_filt. The largest
        vk_mask, as projected on the overlap of the geometries, sets the filter
        edge. If geometries are not provided, then an "exact" flat-sky
        approximation is assumed, in which case the largest vk_mask, as
        provided, in vk_masks sets the filter edge. Each vk_mask be symmetric
        when provided. A vk_mask can also be None, inside of the vk_masks list.
    hk_mask : iterable of (-n, n) tuple of ints
        Like vk_masks, but for the horizontal stripe in 2d Fourier space.
    geometries : iterable of (shape, wcs), optional
        Geometries of the maps in the spectrum. These geometries are used to 
        compute the "overlap" geometry that is the intersection of them. This
        "overlap" region defines where the information in the spectrum actually
        comes from, and therefore the proper flat-sky-to-ell conversion. Also,
        discretization of the overlap geometry is considered (i.e., that the
        actual filter edges are not exactly the requested ones). If None, then
        the transfer function of an "exact" flat-sky approximation is returned.
    geometries_to_filt : iterable of ints, optional
        Indexes into geometries that correspond to the geometries that are 
        actually filtered. This is important for a cross-spectrum between a 
        filtered and unfiltered map, especially when the two have different
        declination ranges. The filtered geometries need to project their filter
        onto the Fourier-space of the overlap region to be accurate. 
        The i'th geometry index in geometries_to_filt also uses the i'th 
        vk_mask in vk_masks and hk_mask in hk_masks. If not provided, assumed no
        filters are applied. Not used at all if geometries is None.
    dtype : np.dtype, optional
        dtype of output array, by default np.float64.
    binning_file : path-like, optional
        If supplied, apply this binning with 2l+1 weights, by default None.

    Returns
    -------
    (x, tf)
        The ell (or binned-ell) points, and the value of the function at those 
        points.

    Notes
    -----
    If geometries is provided, then for example if geometries_to_filt is [1, 3],
    then the 0'th and 2'th geometry are not filtered but the 1'th and 3'th are.
    The filtered geometries uses the 0'th and 1'th entries in vk_masks and
    hk_masks. The 'overlap' geometry uses all 4 geometries.

    This function just performs the 2d integral in Fourier space, so it's not 
    intrinsically accurate. Any such function will fail when the declination
    range of a map is too wide. Will also fail if the data in a wide declination
    range map is actually confined to a small declination range.
    """
    lx_edge, ly_edge = 0, 0

    if geometries is not None:
        shape_overlap, wcs_overlap = geometries[0]
        for i in range(1, len(geometries)):
            shape_overlap, wcs_overlap = enmap.overlap(shape_overlap, wcs_overlap, *geometries[i])
        if shape_overlap[0] <= 0 or shape_overlap[1] <= 0:
            raise ValueError('Overlap of all input geometries is zero')
        
        ly_overlap, lx_overlap  = enmap.laxes(shape_overlap, wcs_overlap, method="auto")

        if geometries_to_filt is None:
            geometries_to_filt = []

        if vk_masks is not None:
            # get the lx_edge for each geometry, as projected in the overlap geometry.
            # because of the binary mask, we want the largest of these lx_edges
            for mask_idx, geom_idx in enumerate(geometries_to_filt):
                vk_mask = vk_masks[mask_idx]
                if vk_mask is not None:
                    assert vk_mask[0] == -vk_mask[1], 'vk_mask must be symmetric'

                    # we want the l of "boundary" between filt and no filt, which is halfway
                    # between the highest filt pixel and lowest unfilt pixel
                    _shape, _wcs = geometries[geom_idx]
                    _, _lx  = enmap.laxes(_shape, _wcs, method="auto")
                    _lx_edge = (np.max(_lx[_lx < vk_mask[1]]) + np.min(_lx[_lx >= vk_mask[1]]))/2

                    # get the dlx overlap if it were sampled with the resolution of this geometry
                    _dlx_overlap = abs(lx_overlap[1]) * shape_overlap[-1] / _shape[-1] 

                    # rescale the l boundary from this geometry to the resampled overlap geometry
                    _lx_edge *= (_dlx_overlap / abs(_lx[1]))

                    if _lx_edge > lx_edge:
                        lx_edge = _lx_edge

        if hk_masks is not None:
            # get the ly_edge for each geometry, as projected in the overlap geometry.
            # because of the binary mask, we want the largest of these ly_edges
            for mask_idx, geom_idx in enumerate(geometries_to_filt):
                hk_mask = hk_masks[mask_idx]
                if hk_mask is not None:
                    assert hk_mask[0] == -hk_mask[1], 'hk_mask must be symmetric'

                    # we want the l of "boundary" between filt and no filt, which is halfway
                    # between the highest filt pixel and lowest unfilt pixel
                    _shape, _wcs = geometries[geom_idx]
                    _ly, _  = enmap.laxes(_shape, _wcs, method="auto")
                    _ly_edge = (np.max(_ly[_ly < hk_mask[1]]) + np.min(_ly[_ly >= hk_mask[1]]))/2

                    # get the dly overlap if it were sampled with the resolution of this geometry
                    _dly_overlap = abs(ly_overlap[1]) * shape_overlap[-2] / _shape[-2]

                    # rescale the l boundary from this geometry to the resampled overlap geometry
                    _ly_edge *= (_dly_overlap / abs(_ly[1]))

                    if _ly_edge > ly_edge:
                        ly_edge = _ly_edge
        else:
            ly_edge = 0
        
    else:
        if vk_masks is not None:
            for vk_mask in vk_masks:
                assert vk_mask[0] == -vk_mask[1], 'vk_mask must be symmetric'
                _lx_edge = vk_mask[1]
                if _lx_edge > lx_edge:
                    lx_edge = _lx_edge
        
        if hk_masks is not None:
            for hk_mask in hk_masks:
                assert hk_mask[0] == -hk_mask[1], 'hk_mask must be symmetric'
                _ly_edge = hk_mask[1]
                if _ly_edge > ly_edge:
                    ly_edge = _ly_edge

    l = np.arange(lmax+1)

    if lx_edge == 0 and ly_edge == 0:
        tf = np.ones(l.size)
    else:
        # if l <= l_edge, then tf will be <= 0 or undefined
        tf = 1 - 2/np.pi*(np.arcsin(np.divide(lx_edge, l, out=np.ones(l.size), where=l>lx_edge)) + np.arcsin(np.divide(ly_edge, l, out=np.ones(l.size), where=l>ly_edge)))
        tf[tf < 0] = 0

    if binning_file is not None:
        lb, num = pspy_utils.naive_binning(l, tf * (2*l+1), binning_file, lmax)
        _, den = pspy_utils.naive_binning(l, (2*l+1), binning_file, lmax)
        tfb =  (num / den).astype(dtype)
        return lb, tfb
    else:
        tf = tf.astype(dtype)
        return l, tf

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
