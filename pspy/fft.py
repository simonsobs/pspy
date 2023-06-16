""" stolen from Zach atkins mnms, this implement ducc real fft and ifft for enmap object  """
import numpy as np
from pixell import enmap


def rfft(emap, kmap=None, nthread=0, normalize='ortho', adjoint_ifft=False):
    """Perform a 'real'-FFT: an FFT over a real-valued function, such
    that only half the usual frequency modes are required to recover
    the full information.

    Parameters
    ----------
    emap : (..., ny, nx) ndmap
        Map to transform.
    kmap : ndmap, optional
        Output buffer into which result is written, by default None.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    normalize : bool, optional
        The FFT normalization, by default 'ortho'. If 'backward', no
        normalization is applied. If 'ortho', 1/sqrt(npix) normalization
        is applied. If 'forward', 1/npix normalization is applied. If in
        ['phy', 'phys', 'physical'], normalize by both 'ortho' and sky area.
    adjoint_ifft : bool, optional
        Whether to perform the adjoint FFT, by default False.

    Returns
    -------
    (..., ny, nx//2+1) ndmap
        Half of the full FFT, sufficient to recover a real-valued
        function.
    """
    
    try:
        import ducc0
    except ModuleNotFoundError:
        raise ModuleNotFoundError("you need to install ducc to use this function")

    if adjoint_ifft:
        raise NotImplementedError()

    # store wcs if imap is ndmap
    if hasattr(emap, 'wcs'):
        is_enmap = True
        wcs = emap.wcs
    else:
        is_enmap = False

    # need to remove wcs for ducc0 for some reason
    if kmap is not None:
        kmap = np.asarray(kmap)
    emap = np.asarray(emap)

    if normalize in ['phy', 'phys', 'physical']:
        inorm = 1
    else:
        inorm = ['backward', 'ortho', 'forward'].index(normalize)

    res = ducc0.fft.r2c(
        emap, out=kmap, axes=[-2, -1], nthreads=nthread, inorm=inorm, forward=True,
        )
    
    if is_enmap:
        res = enmap.ndmap(res, wcs)
        
    # phys norms
    if normalize in ['phy', 'phys', 'physical']:
        if adjoint_ifft:
            res /= res.pixsize()**0.5
        else:
            res *= res.pixsize()**0.5

    return res

# normalizations adapted from pixell.enmap
def irfft(emap, omap=None, n=None, nthread=0, normalize='ortho', adjoint_fft=False):
    """Perform a 'real'-iFFT: an iFFT to recover a real-valued function,
    over only half the usual frequency modes.

    Parameters
    ----------
    emap : (..., nky, nkx) ndmap
        FFT to inverse transform.
    omap : ndmap, optional
        Output buffer into which result is written, by default None.
    n : int, optional
        Number of pixels in real-space x-direction, by default None.
        If none, assumed to be 2(nkx-1), ie that real-space nx was
        even.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    normalize : bool, optional
        The FFT normalization, by default 'ortho'. If 'forward', no
        normalization is applied. If 'ortho', 1/sqrt(npix) normalization
        is applied. If 'backward', 1/npix normalization is applied. If in
        ['phy', 'phys', 'physical'], normalize by both 'ortho' and sky area.
    adjoint_fft : bool, optional
        Whether to perform the adjoint iFFT, by default False.

    Returns
    -------
    (..., ny, nx) ndmap
        A real-valued real-space map.
    """
    
    try:
        import ducc0
    except ModuleNotFoundError:
        raise ModuleNotFoundError("you need to install ducc to use this function")

    if adjoint_fft:
        raise NotImplementedError()

    # store wcs if imap is ndmap
    if hasattr(emap, 'wcs'):
        is_enmap = True
        wcs = emap.wcs
    else:
        is_enmap = False

    # need to remove wcs for ducc0 for some reason
    if omap is not None:
        omap = np.asarray(omap)
    emap = np.asarray(emap)

    if normalize in ['phy', 'phys', 'physical']:
        inorm = 1
    else:
        inorm = ['forward', 'ortho', 'backward'].index(normalize)

    if n is None:
        n = 2*(emap.shape[-1] - 1)

    res = ducc0.fft.c2r(
        emap, out=omap, axes=[-2, -1], nthreads=nthread, inorm=inorm, forward=False,
        lastsize=n
        )
    
    if is_enmap:
        res = enmap.ndmap(res, wcs)
    
    # phys norms
    if normalize in ['phy', 'phys', 'physical']:
        if adjoint_fft:
            res *= res.pixsize()**0.5
        else:
            res /= res.pixsize()**0.5

    return res


