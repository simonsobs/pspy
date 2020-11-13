import numpy as np, pylab as plt
from pspy import flat_tools, pspy_utils

def kspace_filter(map, vk_mask=None, hk_mask=None, window=None):

    filtered_map = map.copy()

    ft = flat_tools.fft_from_so_map(map, normalize=False)
    if window is not None:
        ft = flat_tools.get_ffts(map, window, normalize=False)

    if vk_mask is not None:
        id_vk = np.where((ft.lx > vk_mask[0]) & (ft.lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ft.ly > hk_mask[0]) & (ft.ly < hk_mask[1]))

    if map.ncomp == 1:
        if vk_mask is not None:
            ft.kmap[: , id_vk] = 0.
        if hk_mask is not None:
            ft.kmap[id_hk , :] = 0.
        
    if map.ncomp == 3:
        for i in range(3):
            if vk_mask is not None:
                ft.kmap[i, : , id_vk] = 0.
            if hk_mask is not None:
                ft.kmap[i, id_hk , :] = 0.

    filtered_map.data[:] = ft.map_from_fft(normalize=False)
    return filtered_map


def analytical_tf(map, binning_file, lmax, vk_mask=None, hk_mask=None):
    import time
    t = time.time()
    ft = flat_tools.fft_from_so_map(map)
    ft.create_kspace_mask(vertical_stripe=vk_mask, horizontal_stripe=hk_mask)
    lmap = ft.lmap
    kmask = ft.kmask
    twod_index = lmap.copy()

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_lo)
    print(lmap.shape)
    tf = np.zeros(nbins)
    print("init tf", time.time()-t)
    for ii in range(nbins):
        id = np.where( (lmap >= bin_lo[ii]) & (lmap <= bin_hi[ii]))
        twod_index *= 0
        twod_index[id] = 1
        bin_area = np.sum(twod_index)
        masked_bin_area= np.sum(twod_index * kmask)
        tf[ii] = masked_bin_area / bin_area
    print("loop", time.time()-t)

    return bin_c, tf
