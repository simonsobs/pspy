import numpy as np, pylab as plt
from pspy import flat_tools

def kspace_filter(map, vk_mask=None, hk_mask=None, window=None):

    filtered_map = map.copy()

    ft = flat_tools.fft_from_so_map(map)
    if window is not None:
        ft = flat_tools.get_ffts(map,window)

    if vk_mask is not None:
        id_vk = np.where((ft.lx > vk_mask[0]) & (ft.lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ft.lx > hk_mask[0]) & (ft.lx < hk_mask[1]))

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

    filtered_map.data[:] = ft.map_from_fft()
    return filtered_map




