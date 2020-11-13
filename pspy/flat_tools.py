"""
Routines for handling flat maps (inspired by flipper from sudeep das)
"""
import numpy as np
from pixell import enmap
from pspy import so_map, pspy_utils
import pylab as plt
from copy import deepcopy

class fft2D:
    """
    class describing the two dimensional FFTs of a so_map
    """
    def __init__(self):
        pass

    def copy(self):
        return copy.deepcopy(self)
        
    def trim_fft(self, lmax):
    
        ft = fft2D()
        ft.ncomp = self.ncomp

        ly, lx = self.ly, self.lx
        idy = np.where((ly < lmax) & (ly > -lmax))[0]
        idx = np.where((lx < lmax) & (lx > -lmax))[0]
        npix_y, npix_x = len(idy), len(idx)
    
        if self.ncomp == 1:
            trimmed_fft = np.zeros((npix_y, npix_x), dtype=complex)
            trim = self.kmap[idy, :]
            trimmed_fft = trim[:, idx]
        else:
            trimmed_fft = np.zeros((3, npix_y, npix_x), dtype=complex)
            for i in range(3):
                trim = self.kmap[i, idy, :]
                trimmed_fft[i] = trim[:, idx]

        ft.kmap = trimmed_fft
        
        lxmap = self.lxmap[idy,: ]
        ft.lxmap = lxmap[:,idx]
        
        lymap = self.lymap[idy,:]
        ft.lymap = lymap[:,idx]
        
        ft.ly, ft.lx = ft.lymap[:,0], ft.lxmap[0,:]
        
        ft.lmap = np.sqrt(ft.lxmap**2 + ft.lymap**2)
        ft.thetamap = np.arctan2(ft.lymap,ft.lxmap) * 180/np.pi

        return ft
        
        
    def create_kspace_mask(self, vertical_stripe=None, horizontal_stripe=None):
        mask = np.real(self.kmap.copy())
        if mask.ndim == 3:
            mask = mask[0]
        mask[:,:] = 1.
        if vertical_stripe is not None:
            idx = np.where((self.lx < vertical_stripe[1]) & (self.lx > vertical_stripe[0]))
            mask[:,idx] = 0.
        if vertical_stripe is not None:
            idy = np.where((self.ly < horizontal_stripe[1]) & (self.ly > horizontal_stripe[0]))
            mask[idy,:] = 0.
        self.kmask = mask
        
        
    def map_from_fft(self, normalize="phys"):
        return enmap.ifft(self.kmap, normalize=normalize)

        
class power2D:
    """
    a class describing the 2-D power spectra of a so_map
    """
    def __init__(self):
        pass
        
    def copy(self):
        return deepcopy(self)

    def create_kspace_mask(self, vertical_stripe=None, horizontal_stripe=None):
        mask = self.powermap["II"].copy()
        mask[:,:] = 1.
        if vertical_stripe is not None:
            idx = np.where((self.lx < vertical_stripe[1]) & (self.lx > vertical_stripe[0]))
            mask[:,idx] = 0.
        if vertical_stripe is not None:
            idy = np.where((self.ly < horizontal_stripe[1]) & (self.ly > horizontal_stripe[0]))
            mask[idy,:] = 0.
        self.kmask = mask

    def plot(self,
             log=False,
             title="",
             power_of_ell=0,
             show_bins_from_files=None,
             draw_circles_at_ell=None,
             draw_vertical_lines_at_ell=None,
             value_range=None,
             show= True,
             png_file=None):
             


        for spec in self.spectra:
            p =  self.powermap[spec].copy()
     
            p[:] *= (self.lmap + 1.)**power_of_ell
            p = np.fft.fftshift(p)
     
            if show_bins_from_files is not None:
                bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
                theta = np.arange(0, 2. * np.pi + 0.05, 0.05)
         
                for i in xrange(len(bin_lo)):
                    x, y = bin_hi[i] * np.cos(theta), bin_hi[i] * np.sin(theta)
                    plt.plot(x,y,'k')

            if draw_circles_at_ell is not None:
                for ell in draw_circles_at_ell:
                    theta = np.arange(0, 2. * np.pi + 0.05, 0.05)
                    x, y = ell * np.cos(theta), ell * np.sin(theta)
                    plt.plot(x, y, "k")
                    if len(draw_circles_at_ell) < 5:
                        plt.text(ell * np.cos(np.pi / 4.),
                                   ell * np.sin(np.pi / 4.),
                                   "%d" % np.int(ell),
                                   rotation=-45,
                                   horizontalalignment="center",
                                   verticalalignment="bottom",
                                   fontsize=8)
                 
            if draw_vertical_lines_at_ell is not None:
                for ell in draw_vertical_lines_at_ell:
                    plt.axvline(ell)
             
            if log:
                p = np.log10(np.abs(p))

            vmin = p.min()
            vmax = p.max()

            if value_range is not None:
                vmin = value_range[0]
                vmax = value_range[1]
            im = plt.imshow(p,
                            origin="down",
                            extent=[np.min(self.lx), np.max(self.lx), np.min(self.ly), np.max(self.ly)],
                            aspect="equal",
                            vmin=vmin,
                            vmax=vmax)
            plt.title(title+ "%s"%spec, fontsize=13)
            plt.colorbar()
            plt.xlabel(r'$\ell_x$',fontsize=15)
            plt.ylabel(r'$\ell_y$',fontsize=15)
     
            if png_file is not None:
                plt.savefig("%s_%s.png" % (png_file, spec))
                plt.clf()
                plt.close()
            else:
                plt.show()



def fft_from_so_map(so_map, normalize="phys"):
    """
    Creates an fft2D object out of a so_map
    """
    ft = fft2D()

    ft.lymap, ft.lxmap = so_map.data.lmap()
    ft.ly, ft.lx = ft.lymap[:,0], ft.lxmap[0,:]
    ft.lmap = np.sqrt(ft.lxmap**2 + ft.lymap**2)
    ft.thetamap = np.arctan2(ft.lymap, ft.lxmap) * 180 / np.pi
    ft.ncomp = so_map.ncomp
    ft.kmap = enmap.fft(so_map.data, normalize=normalize)
    
    return ft
    
def power_from_fft(ft, ft2=None, type = "Cl"):
    """
    Creates an 2-D power spectra object out of fft2D objects
    """

    p2d = power2D()
    p2d.lymap, p2d.lxmap = ft.lymap, ft.lxmap
    p2d.ly, p2d.lx = ft.ly, ft.lx
    p2d.lmap = ft.lmap
    p2d.thetamap = ft.thetamap
    p2d.powermap = {}
    
    fac = 1
    if type == "Dl":
        fac = p2d.lmap**2/(2*np.pi)
    
    if ft.ncomp == 1:
        p2d.spectra = ["II"]

        if ft2 is None:
            p2d.powermap["II"] = (ft.kmap*np.conj(ft.kmap)).real * fac
        else:
            p2d.powermap["II"]= (ft.kmap*np.conj(ft2.kmap)).real * fac
                  
    elif ft.ncomp == 3:
        spectra_list = []
        for i, m1 in enumerate(["I","Q","U"]):
            for j, m2 in enumerate(["I","Q","U"]):
                if ft2 is None:
                    p2d.powermap[m1+m2]= (ft.kmap[i]*np.conj(ft.kmap[j])).real * fac
                else:
                    p2d.powermap[m1+m2]= (ft.kmap[i]*np.conj(ft2.kmap[j])).real * fac
                spectra_list += [m1+m2]
                
        p2d.spectra = spectra_list
    
        
    return p2d.lmap, p2d


def get_ffts(so_map, window, lmax=None, normalize="phys"):

    windowed_map = so_map.copy()
    if so_map.ncomp == 3:
        windowed_map.data[0] = so_map.data[0]*window[0].data
        windowed_map.data[1] = so_map.data[1]*window[1].data
        windowed_map.data[2] = so_map.data[2]*window[1].data
    if so_map.ncomp == 1:
        windowed_map.data = so_map.data * window.data
    ffts = fft_from_so_map(windowed_map, normalize=normalize)
    if lmax is not None:
        ffts=ffts.trim_fft(lmax)
    return ffts
