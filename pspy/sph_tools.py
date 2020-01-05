"""
Routines for generalized map2alm and alm2map (healpix and CAR).
"""

import healpy as hp
import numpy as np
from pixell import curvedsky, enmap

from pspy import so_window


def map2alm(so_map, niter, lmax, theta_range=None):
    """Map2alm transform (for HEALPIX or CAR).

    Parameters
    ----------
    so_map: ``so_map``
      the map from which to compute the alm
    niter: integer
      the number of iteration performed while computing the alm
      not that for CAR niter=0 should be enough
    lmax: integer
      the maximum multipole of the transform
    theta_range: list of 2 elements
      [theta_min, theta_max] in radian.
      for HEALPIX pixellisation all pixel outside this range will be assumed to be zero.
    """
    if so_map.pixel == "HEALPIX":
        if theta_range is None:
            alm = hp.map2alm(so_map.data, lmax=lmax, iter=niter)

        else:
            alm = curvedsky.map2alm_healpix(so_map.data,
                                            lmax=lmax,
                                            theta_min=theta_range[0],
                                            theta_max=theta_range[1])
            if niter != 0:
                map_copy = so_map.copy()
                for _ in range(niter):
                    alm += curvedsky.map2alm_healpix(so_map.data -
                                                     curvedsky.alm2map_healpix(alm, map_copy.data),
                                                     lmax=lmax,
                                                     theta_min=theta_range[0],
                                                     theta_max=theta_range[1])
    elif so_map.pixel == "CAR":
        alm = curvedsky.map2alm(so_map.data, lmax=lmax)
        if niter != 0:
            map_copy = so_map.copy()
            for _ in range(niter):
                alm += curvedsky.map2alm(so_map.data - curvedsky.alm2map(alm, map_copy.data),
                                         lmax=lmax)
    else:
        raise ValueError("Map is neither a CAR nor a HEALPIX")

    alm = alm.astype(np.complex128)
    return alm


def alm2map(alms, so_map):
    """alm2map transform (for healpix and CAR).

    Parameters
    ----------
    alms: array
      a set of alms, the shape of alms should correspond to map.ncomp
    so_map: ``so_map``
      the map template that will contain the results of the harmonic transform
    """
    if so_map.ncomp == 1:
        spin = 0
    else:
        spin = [0, 2]
    if so_map.pixel == "HEALPIX":
        so_map.data = curvedsky.alm2map_healpix(alms, so_map.data, spin)
    elif so_map.pixel == "CAR":
        so_map.data = curvedsky.alm2map(alms, so_map.data, spin)
    else:
        raise ValueError("Map is neither a CAR nor a HEALPIX")
    return so_map


def get_alms(so_map, window, niter, lmax, theta_range=None):
    """Get a map, multiply by a window and return alms
    This is basically map2alm but with application of the
    window functions.

    Parameters
    ----------

    so_map: ``so_map``
      the data we want alms from
    window: so_map or tuple of so_map
      a so map with the window function, if the so map has 3 components
      (for spin0 and 2 fields) expect a tuple (window, window_pol)
    theta range: list of 2 elements
      for healpix pixellisation you can specify
      a range [theta_min,theta_max] in radian. All pixel outside this range
      will be assumed to be zero.
    """
    windowed_map = so_map.copy()
    if so_map.ncomp == 3:
        windowed_map.data[0] *= window[0].data
        windowed_map.data[1] *= window[1].data
        windowed_map.data[2] *= window[1].data
    if so_map.ncomp == 1:
        windowed_map.data *= window.data
    alms = map2alm(windowed_map, niter, lmax, theta_range=theta_range)
    return alms


def get_pure_alms(so_map, window, niter, lmax):
    """Compute pure alms from maps and window function

    Parameters
    ----------

    so_map: ``so_map``
      the data we want alms from
    window: ``so_map`` or tuple of ``so_map``
      a so map with the window function, if the so map has 3 components
      (for spin0 and 2 fields) expect a tuple (window, window_pol)
    niter: integer
      the number of iteration performed while computing the alm
      not that for CAR niter=0 should be enough
    lmax: integer
      the maximum multipole of the transform

    """

    s1_a, s1_b, s2_a, s2_b = so_window.get_spinned_windows(window[1], lmax, niter=niter)
    p2 = np.array([window[1].data * so_map.data[1], window[1].data * so_map.data[2]])
    p1 = np.array([(s1_a.data * so_map.data[1] + s1_b.data * so_map.data[2]),
                   (s1_a.data * so_map.data[2] - s1_b.data * so_map.data[1])])
    p0 = np.array([(s2_a.data * so_map.data[1] + s2_b.data * so_map.data[2]),
                   (s2_a.data * so_map.data[2] - s2_b.data * so_map.data[1])])

    if so_map.pixel == "CAR":
        p0 = enmap.samewcs(p0, so_map.data)
        p1 = enmap.samewcs(p1, so_map.data)
        p2 = enmap.samewcs(p2, so_map.data)

        alm = curvedsky.map2alm(so_map.data[0] * window[0].data, lmax=lmax)
        s2eblm = curvedsky.map2alm(p2, spin=2, lmax=lmax)
        s1eblm = curvedsky.map2alm(p1, spin=1, lmax=lmax)
        s0eblm = s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm(p0[0], spin=0, lmax=lmax)
        s0eblm[1] = curvedsky.map2alm(p0[1], spin=0, lmax=lmax)

    if so_map.pixel == "HEALPIX":
        alm = hp.map2alm(so_map.data[0] * window[0].data, lmax=lmax, iter=niter)
        #curvedsky.map2alm_healpix(so_map.data[0]*window[0].data, lmax=lmax)
        s2eblm = curvedsky.map2alm_healpix(p2, spin=2, lmax=lmax)
        s1eblm = curvedsky.map2alm_healpix(p1, spin=1, lmax=lmax)
        s0eblm = s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm_healpix(p0[0], spin=0, lmax=lmax)
        s0eblm[1] = curvedsky.map2alm_healpix(p0[1], spin=0, lmax=lmax)

    ell = np.arange(lmax)
    filter_1 = np.zeros(lmax)
    filter_2 = np.zeros(lmax)
    filter_3 = np.zeros(lmax)

    filter_1[2:] = 2 * np.sqrt(1.0 / ((ell[2:] + 2.) * (ell[2:] - 1.)))
    filter_2[2:] = np.sqrt(1.0 / ((ell[2:] + 2.) * (ell[2:] + 1.) * ell[2:] * (ell[2:] - 1.)))
    filter_3[2:] = ell[2:] * 0 + 1
    for k in range(2):
        s1eblm[k] = hp.almxfl(s1eblm[k], filter_1)
        s0eblm[k] = hp.almxfl(s0eblm[k], filter_2)
        s2eblm[k] = hp.almxfl(s2eblm[k], filter_3)

    elm_p = s2eblm[0] + s1eblm[0] - s0eblm[0]
    blm_b = s2eblm[1] + s1eblm[1] - s0eblm[1]

    return np.array([alm, elm_p, blm_b])
