"""
Routines for window function generation
"""

import healpy as hp
import numpy as np
from pixell import curvedsky, enmap
from scipy.ndimage import distance_transform_edt

from pspy import sph_tools


def get_distance(binary):
    """Get the distance to the closest masked pixels for CAR and HEALPIX ``so_map`` binary.

    Parameters
    ----------
    binary: ``so_map``
      a ``so_map`` with binary data (1 is observed, 0 is masked)
    """

    dist = binary.copy()
    if binary.pixel == "HEALPIX":
        dist.data = enmap.distance_transform_healpix(binary.data, method="heap")
        dist.data *= 180 / np.pi
    if binary.pixel == "CAR":
        pix_size_arcmin = np.sqrt(binary.data.pixsize() * (60 * 180 / np.pi)**2)
        dist.data[:] = distance_transform_edt(binary.data)
        dist.data[:] *= pix_size_arcmin / 60

    return dist


def create_apodization(binary, apo_type, apo_radius_degree):
    """Create a apodized window function from a binary mask.

    Parameters
    ----------
    binary: ``so_map``
      a ``so_map`` with binary data (1 is observed, 0 is masked)
    apo_type: string
      the type of apodisation you want to use ("C1","C2" or "Rectangle")
    apo_radius: float
      the apodisation radius in degrees
    """

    if apo_type == "C1":
        window = apod_C1(binary, apo_radius_degree)
    if apo_type == "C2":
        window = apod_C2(binary, apo_radius_degree)
    if apo_type == "Rectangle":
        if binary.pixel == "HEALPIX":
            raise ValueError("No rectangle apodization for HEALPIX map")
        if binary.pixel == "CAR":
            window = apod_rectangle(binary, apo_radius_degree)

    return window


def apod_C2(binary, radius):
    """Create a C2 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf

    Parameters
    ----------
    binary: ``so_map``
        a ``so_map`` with binary data (1 is observed, 0 is masked)
    apo_radius: float
        the apodisation radius in degrees

    """

    if radius == 0:
        return binary

    dist = get_distance(binary)
    idx = np.where(dist.data > radius)
    win = binary.copy()
    win.data = dist.data / radius - np.sin(2 * np.pi * dist.data / radius) / (2 * np.pi)
    win.data[idx] = 1

    return win


def apod_C1(binary, radius):
    """Create a C1 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf

    Parameters
    ----------
    binary: ``so_map``
      a ``so_map`` with binary data (1 is observed, 0 is masked)
    apo_radius: float
      the apodisation radius in degrees

    """

    if radius == 0:
        return binary

    dist = get_distance(binary)
    idx = np.where(dist.data > radius)
    win = binary.copy()
    win.data = 1. / 2 - 1. / 2 * np.cos(-np.pi * dist.data / radius)
    win.data[idx] = 1

    return win


def apod_rectangle(binary, radius):
    """Create an apodisation able for rectangle window (in CAR) (smoother at the corner)

    Parameters
    ----------
    binary: ``so_map``
      a ``so_map`` with binary data (1 is observed, 0 is masked)
    apo_radius: float
      the apodisation radius in degrees

    """

    if radius == 0:
        return binary

    wcs = binary.data.wcs
    win = binary.copy()
    win.data = win.data * 0 + 1
    win_x = win.copy()
    win_y = win.copy()
    shape = binary.data.shape
    pix_scale_y, pix_scale_x = enmap.pixshape(shape, wcs)
    deg_to_pix_x = np.pi / 180 / pix_scale_x
    deg_to_pix_y = np.pi / 180 / pix_scale_y
    len_apod_x = int(radius * deg_to_pix_x)
    len_apod_y = int(radius * deg_to_pix_y)

    ones = np.ones(shape)
    ny, nx = shape

    for i in range(len_apod_x):
        r = float(i)
        win_x.data[:, i] = 1. / 2 * (ones[:, i] - np.cos(-np.pi * r / len_apod_x))
        win_x.data[:, nx - i - 1] = win_x.data[:, i]
    for i in range(len_apod_y):
        r = float(i)
        win_y.data[i, :] = 1. / 2 * (ones[i, :] - np.cos(-np.pi * r / len_apod_y))
        win_y.data[ny - i - 1, :] = win_y.data[i, :]

    win.data = win_x.data * win_y.data

    return win


def get_spinned_windows(window, lmax, niter):
    """Compute the spinned window functions (for pure B modes method)

    Parameters
    ----------
    window: ``so_map``
      map of the window function
    lmax: integer
      maximum value of the multipole for the harmonic transform
    niter: integer
      number of iteration for the harmonic transform

    """

    template = np.array([window.data.copy(), window.data.copy()])
    s1_a, s1_b, s2_a, s2_b = window.copy(), window.copy(), window.copy(), window.copy()

    if window.pixel == "CAR":
        template = enmap.samewcs(template, window.data)

    wlm = sph_tools.map2alm(window, lmax=lmax, niter=niter)
    ell = np.arange(lmax)
    filter_1 = -np.sqrt((ell + 1) * ell)
    filter_2 = -np.sqrt((ell + 2) * (ell + 1) * ell * (ell - 1))

    filter_1[:1] = 0
    filter_2[:2] = 0
    wlm1_e = hp.almxfl(wlm, filter_1)
    wlm2_e = hp.almxfl(wlm, filter_2)
    wlm1_b = np.zeros_like(wlm1_e)
    wlm2_b = np.zeros_like(wlm2_e)

    window1 = template.copy()
    window2 = template.copy()

    if window.pixel == "HEALPIX":
        curvedsky.alm2map_healpix(np.array([wlm1_e, wlm1_b]), window1, spin=1)
        curvedsky.alm2map_healpix(np.array([wlm2_e, wlm2_b]), window2, spin=2)
    if window.pixel == "CAR":
        curvedsky.alm2map(np.array([wlm1_e, wlm1_b]), window1, spin=1)
        curvedsky.alm2map(np.array([wlm2_e, wlm2_b]), window2, spin=2)

    s1_a.data[window.data != 0] = window1[0][window.data != 0]
    s1_b.data[window.data != 0] = window1[1][window.data != 0]
    s2_a.data[window.data != 0] = window2[0][window.data != 0]
    s2_b.data[window.data != 0] = window2[1][window.data != 0]

    return s1_a, s1_b, s2_a, s2_b
