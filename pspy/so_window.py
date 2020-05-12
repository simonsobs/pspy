"""
routines for window function generation
"""

import healpy as hp
import numpy as np

from pixell import curvedsky, enmap
from pspy import sph_tools


def get_distance(binary, rmax=None):
    """Get the distance to the closest masked pixels for CAR and healpix so_map binary.

    Parameters
    ----------
    binary: ``so_map``
    a ``so_map`` with binary data (1 is observed, 0 is masked)
    """

    dist = binary.copy()
    # Make sure distances are floating point number
    dist.data = dist.data.astype(float)

    if binary.pixel == "HEALPIX":
        dist.data[:] = enmap.distance_transform_healpix(binary.data, method="heap", rmax=rmax)
        dist.data[:] *= 180 / np.pi
    if binary.pixel == "CAR":
        dist.data[:] = enmap.distance_transform(binary.data, rmax=rmax)
        dist.data[:] *= 180 / np.pi

    return dist


def create_apodization(binary, apo_type, apo_radius_degree, use_rmax=False):
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

    if use_rmax:
        rmax = (apo_radius_degree * 1.1) * np.pi / 180
    else:
        rmax = None

    if apo_type == "C1":
        window = apod_C1(binary, apo_radius_degree, rmax)
    if apo_type == "C2":
        window = apod_C2(binary, apo_radius_degree, rmax)
    if apo_type == "Rectangle":
        if binary.pixel == "HEALPIX":
            raise ValueError("No rectangle apodization for HEALPIX map")
        if binary.pixel == "CAR":
            window = apod_rectangle(binary, apo_radius_degree)

    return window


def apod_C2(binary, radius, rmax=None):
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
    else:
        dist = get_distance(binary, rmax)
        win = binary.copy()
        idx = np.where(dist.data > radius)
        win.data = dist.data / radius - np.sin(2 * np.pi * dist.data / radius) / (2 * np.pi)
        win.data[idx] = 1

    return win


def apod_C1(binary, radius, rmax=None):
    """Create a C1 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf

    Parameters
    ----------
    binary: ``so_map``
      a so_map with binary data (1 is observed, 0 is masked)
    apo_radius: float
      the apodisation radius in degrees

    """

    if radius == 0:
        return binary
    else:
        dist = get_distance(binary, rmax)
        win = binary.copy()
        idx = np.where(dist.data > radius)
        win.data = 1.0 / 2 - 1.0 / 2 * np.cos(-np.pi * dist.data / radius)
        win.data[idx] = 1

    return win


def apod_rectangle(binary, radius):
    """Create an apodisation for rectangle window (in CAR) (smoother at the corner)

    Parameters
    ----------
    binary: ``so_map``
      a so_map with binary data (1 is observed, 0 is masked)
    apo_radius: float
      the apodisation radius in degrees

    """

    # TODO: clean this one

    if radius == 0:
        return binary
    else:
        shape = binary.data.shape
        wcs = binary.data.wcs
        Ny, Nx = shape
        pix_scale_y, pix_scale_x = enmap.pixshape(shape, wcs)
        win = binary.copy()
        win.data = win.data * 0 + 1
        win_x = win.copy()
        win_y = win.copy()
        ones = np.ones((Ny, Nx))
        deg_to_pix_x = np.pi / 180 / pix_scale_x
        deg_to_pix_y = np.pi / 180 / pix_scale_y
        lenApod_x = int(radius * deg_to_pix_x)
        lenApod_y = int(radius * deg_to_pix_y)

        for i in range(lenApod_x):
            r = float(i)
            win_x.data[:, i] = 1.0 / 2 * (ones[:, i] - np.cos(-np.pi * r / lenApod_x))
            win_x.data[:, Nx - i - 1] = win_x.data[:, i]
        for j in range(lenApod_y):
            r = float(j)
            win_y.data[j, :] = 1.0 / 2 * (ones[j, :] - np.cos(-np.pi * r / lenApod_y))
            win_y.data[Ny - j - 1, :] = win_y.data[j, :]

        win.data = win_x.data * win_y.data

        return win


def get_spinned_windows(w, lmax, niter):
    """Compute the spinned window functions (for pure B modes method)

    Parameters
    ----------
    w: ``so_map``
      map of the window function
    lmax: integer
      maximum value of the multipole for the harmonic transform
    niter: integer
      number of iteration for the harmonic transform

    """

    template = np.array([w.data.copy(), w.data.copy()])
    w1_plus, w1_minus, w2_plus, w2_minus = w.copy(), w.copy(), w.copy(), w.copy()

    if w.pixel == "CAR":
        template = enmap.samewcs(template, w.data)

    wlm = sph_tools.map2alm(w, lmax=lmax, niter=niter)
    ell = np.arange(lmax + 1)
    filter_1 = -np.sqrt((ell + 1) * ell)
    filter_2 = np.sqrt((ell + 2) * (ell + 1) * ell * (ell - 1))

    filter_2[:1] = 0

    wlm1_e = hp.almxfl(wlm, filter_1)
    wlm2_e = hp.almxfl(wlm, filter_2)
    wlm1_b = np.zeros_like(wlm1_e)
    wlm2_b = np.zeros_like(wlm2_e)

    w1 = template.copy()
    w2 = template.copy()

    if w.pixel == "HEALPIX":
        curvedsky.alm2map_healpix(np.array([wlm1_e, wlm1_b]), w1, spin=1)
        curvedsky.alm2map_healpix(np.array([wlm2_e, wlm2_b]), w2, spin=2)
    if w.pixel == "CAR":
        curvedsky.alm2map(np.array([wlm1_e, wlm1_b]), w1, spin=1)
        curvedsky.alm2map(np.array([wlm2_e, wlm2_b]), w2, spin=2)

    w1_plus.data = w1[0]
    w1_minus.data = w1[1]
    w2_plus.data = w2[0]
    w2_minus.data = w2[1]

    return w1_plus, w1_minus, w2_plus, w2_minus
