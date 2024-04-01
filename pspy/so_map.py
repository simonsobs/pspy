"""
SO map class for handling healpix and CAR maps.
This is a wrapper around healpix and enlib (pixell).
"""

import os
from copy import deepcopy

import astropy.io.fits as pyfits
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pixell import colorize, curvedsky, enmap, enplot, powspec, reproject

from pspy.pspy_utils import ps_lensed_theory_to_dict
from pspy.sph_tools import alm2map, map2alm
from pspy.fft import rfft, irfft

class so_map:
    """Class defining a ``so_map`` object."""

    def __init__(self):
        self.pixel = None
        self.nside = None
        self.ncomp = None
        self.data = None
        self.coordinate = None

    def copy(self):
        """Create a copy of the ``so_map`` object."""

        return deepcopy(self)

    def info(self):
        """Print information about the ``so_map`` object."""

        print("pixellisation:", self.pixel)
        print("number of components:", self.ncomp)
        print("number of pixels:", self.data.shape[:] if self.ncomp == 1 else self.data.shape[1:])
        print("nside:", self.nside)
        print("coordinates:", self.coordinate)

    def write_map(self, file_name):
        """Write the ``so_map`` to disk.

        Parameters
        ----------
        filename : string
          the name of the fits file
        """

        if self.pixel == "HEALPIX":
            hp.fitsfunc.write_map(file_name, self.data, overwrite=True)
        if self.pixel == "CAR":
            enmap.write_map(file_name, self.data)

    def calibrate(self, cal=1.0, pol_eff=1.0):
        """Calibrate the ``so_map``, also optionnaly apply a pol efficiency
        Parameters
        ----------
        cal : float
            the calibration factor to apply
        pol_eff : float
            the polarisation efficiency
        """

        if self.ncomp == 1:
            self.data *= cal
        else:
            self.data[0] *= cal
            self.data[1] *= cal / pol_eff
            self.data[2] *= cal / pol_eff

        return self

    def upgrade(self, factor):
        """Upgrade the ``so_map``.

        Parameters
        ----------
        factor : integer
          factor of increased pixel resolution (should be a factor of 2)

        """

        assert factor % 2 == 0, "factor should be a factor of 2"

        upgrade = self.copy()
        if self.pixel == "HEALPIX":
            nside_out = int(self.nside * factor)
            upgrade.data = hp.pixelfunc.ud_grade(self.data, nside_out=nside_out)
            upgrade.nside = nside_out
        if self.pixel == "CAR":
            upgrade.data = enmap.upgrade(self.data, factor)
        return upgrade

    def downgrade(self, factor):
        """Downgrade the ``so_map``.

        Parameters
        ----------
        factor : integer
          factor of decreased pixel resolution (should be a factor of 2)

        """

        assert factor % 2 == 0, "factor should be a factor of 2"

        downgrade = self.copy()
        if self.pixel == "HEALPIX":
            nside_out = int(self.nside / factor)
            downgrade.data = hp.pixelfunc.ud_grade(self.data, nside_out=nside_out)
            downgrade.nside = nside_out
        if self.pixel == "CAR":
            downgrade.data = enmap.downgrade(self.data, factor)
        return downgrade

    def synfast(self, clfile):
        """Fill a ``so_map`` with a cmb gaussian simulation.

        Parameters
        ----------
        clfile : CAMB data file
          lensed power spectra file from CAMB

        """

        synfast = self.copy()
        if self.pixel == "HEALPIX":
            l, ps = ps_lensed_theory_to_dict(clfile, output_type="Cl", start_at_zero=True)
            if self.ncomp == 1:
                synfast.data = hp.sphtfunc.synfast(ps["TT"], self.nside, new=True)
            else:
                synfast.data = hp.sphtfunc.synfast(
                    (ps["TT"], ps["EE"], ps["BB"], ps["TE"]), self.nside, new=True
                )

        if self.pixel == "CAR":
            ps = powspec.read_spectrum(clfile)[: self.ncomp, : self.ncomp]
            synfast.data = curvedsky.rand_map(self.data.shape, self.data.wcs, ps)

        return synfast

    def get_lmax_limit(self):
        """Return the maximum lmax corresponding to the ``so_map`` pixellisation"""

        if self.pixel == "HEALPIX":
            l_max_limit = 3 * self.nside - 1
        elif self.pixel == "CAR":
            cdelt = self.data.wcs.wcs.cdelt[1]
            l_max_limit = 360 / cdelt / 4
        return l_max_limit

    def get_pixwin(self, order=0, dtype=np.float64):
        """compute the pixel window function corresponding to the map pixellisation
           order stands for the map making pointing matrix
           order=0 is Neareast Neighbour while order=1 is bilinear interpolation
        """
        if self.pixel == "HEALPIX":
            pixwin = hp.pixwin(self.nside)
        if self.pixel == "CAR":
            wy, wx = enmap.calc_window(self.data.shape, order=order)
            pixwin = (wy[:,None] * wx[None,:])
        return pixwin.astype(dtype)

    def convolve_with_pixwin(self, niter=3, window=None, pixwin=None, order=0, use_ducc_rfft=False):
        """Convolve a ``so_map`` object with a pixel window function
        The convolution is done in harmonics space, for CAR maps
        the pixwin is anisotropic (the pixel varies in size across the maps)
        and the convolution is done in Fourier space.
        We optionaly apply a window before doing the operation to remove pathological pixels, note
        that this operation is dangerous since we do harmonic transform of a masked map.

        Parameters
        ----------
        niter: integer
          the number of iteration performed while computing the alm
          not that for CAR niter=0 should be enough
        window: so_map
            a window that remove pathological pixel before doing the harmonic
            space operation
        pixwin: 1d array for healpix, 2d array for CAR
            this allow you to pass a precomputed pixel window function
        """

        lmax = self.get_lmax_limit()

        if window is not None: self.data *= window.data
        if pixwin is None: pixwin = self.get_pixwin(order=order)

        if self.pixel == "HEALPIX":
            alms = map2alm(self, niter, lmax)
            alms = curvedsky.almxfl(alms, pixwin)
            self = alm2map(alms, self)
        if self.pixel == "CAR":
            self = fourier_convolution(self, pixwin, use_ducc_rfft=use_ducc_rfft)

        return self

    def subtract_mean(self, mask=None):
        """Subtract mean from a ``so_map`` object, optionnaly within a mask

        Parameters
        ----------
        mask: either a single so_map (for ncomp = 1) or a tuple of SO map e.g (mask_T, mask_P)
        """

        if mask is None:
            self.data -= np.mean(self.data)
        else:
            if self.ncomp == 1:
                self.data -= np.mean(self.data * mask.data)
            else:
                self.data[0] -= np.mean(self.data * mask[0].data)
                self.data[1] -= np.mean(self.data * mask[1].data)
                self.data[2] -= np.mean(self.data * mask[1].data)

        return self

    def subtract_mono_dipole(self, mask=None, values=None, bunch=24):
        """Subtract monopole and dipole from a ``so_map`` object.

        Parameters
        ----------
        mask: either a single so_map (for ncomp = 1) or a tuple of SO map e.g (mask_T, mask_P)
        values: the value of the monopole and dipole to subtract
        bunch: int
            the bunch size (default: 24)
        """

        if self.ncomp == 1:
            self.data = subtract_mono_dipole(
                emap=self.data,
                mask=None if mask is None else mask.data,
                healpix=self.pixel == "HEALPIX",
                bunch=bunch,
                values=values,
            )
        else:
            for i in range(self.ncomp):
                self.data[i] = subtract_mono_dipole(
                    emap=self.data[i],
                    mask=None if mask is None else mask[i if i < 2 else 1].data,
                    healpix=self.pixel == "HEALPIX",
                    bunch=bunch,
                    values=values,
                )

    def plot(
        self,
        color="planck",
        color_range=None,
        file_name=None,
        ticks_spacing_car=1,
        title="",
        cbar=True,
        hp_gnomv=None,
    ):
        """Plot a ``so_map``.

        Parameters
        ----------
        color: cmap
          a matplotlib colormap (or 'planck')
        color_range: scalar for single component or len(3) list for T,Q,U.
          the range of the colorscale
        file_name: string
          file_name is the name of the png file that will be created, if None the plot
          will be displayed.
        title: string
          the title of the plot.
        cbar: boolean
          set to True to display the colorbar.
        ticks_spacing_CAR: float
          for CAR plot, choose the spacing of the ticks.
        hp_gnomv: tuple
          gnomview projection for HEALPIX plotting, expected (lon_c,lat_c,xsize,reso).

        """
        try:
            if color not in mpl.colormaps:
                colorize.mpl_setdefault(color)
        except KeyError:
            if self.pixel == "CAR":
                raise KeyError(
                    "Color name must be a pixell color map name {}!".format(
                        list(colorize.schemes.keys())
                    )
                )

        if self.pixel == "HEALPIX":
            cmap = plt.get_cmap(color)
            cmap.set_bad("white")
            cmap.set_under("white")
            if self.ncomp == 1:

                min_range = -color_range if color_range is not None else None
                max_range = +color_range if color_range is not None else None

                if hp_gnomv is not None:
                    lon, lat, xsize, reso = hp_gnomv
                    hp.gnomview(
                        self.data,
                        min=min_range,
                        max=max_range,
                        cmap=cmap,
                        notext=True,
                        title=title,
                        cbar=cbar,
                        rot=(lon, lat, 0),
                        xsize=xsize,
                        reso=reso,
                    )
                else:
                    hp.mollview(
                        self.data,
                        min=min_range,
                        max=max_range,
                        cmap=cmap,
                        notext=True,
                        title=title,
                        cbar=cbar,
                    )
                if file_name is not None:
                    plt.savefig(file_name + ".png", bbox_inches="tight")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()
            else:
                fields = ["T", "Q", "U"]
                min_ranges = {field: None for field in fields}
                max_ranges = {field: None for field in fields}
                if color_range is not None:
                    for i, field in enumerate(fields):
                        min_ranges[field] = -color_range[i]
                        max_ranges[field] = +color_range[i]

                for data, field in zip(self.data, fields):
                    if hp_gnomv is not None:
                        lon, lat, xsize, reso = hp_gnomv
                        hp.gnomview(
                            data,
                            min=min_ranges[field],
                            max=max_ranges[field],
                            cmap=cmap,
                            notext=True,
                            title=field + "" + title,
                            cbar=cbar,
                            rot=(lon, lat, 0),
                            xsize=xsize,
                            reso=reso,
                        )
                    else:
                        hp.mollview(
                            data,
                            min=min_ranges[field],
                            max=max_ranges[field],
                            cmap=cmap,
                            notext=True,
                            title=field + "" + title,
                            cbar=cbar,
                        )
                    if file_name is not None:
                        plt.savefig(file_name + "_%s" % field + ".png", bbox_inches="tight")
                        plt.clf()
                        plt.close()
                    else:
                        plt.show()

        if self.pixel == "CAR":
            if self.ncomp == 1:
                if color_range is not None:
                    max_range = "%s" % (color_range)
                else:
                    max_range = "%s" % (np.max(self.data))

                plots = enplot.get_plots(
                    self.data, color=color, range=max_range, colorbar=1, ticks=ticks_spacing_car
                )

                for plot in plots:
                    if file_name is not None:
                        enplot.write(file_name + ".png", plot)
                    else:
                        plot.img.show()

            if self.ncomp == 3:
                fields = ["T", "Q", "U"]

                if color_range is not None:
                    max_range = "%s:%s:%s" % (color_range[0], color_range[1], color_range[2])
                else:
                    max_range = "%s:%s:%s" % (
                        np.max(self.data[0]),
                        np.max(self.data[1]),
                        np.max(self.data[2]),
                    )

                plots = enplot.get_plots(
                    self.data, color=color, range=max_range, colorbar=1, ticks=ticks_spacing_car
                )

                for (plot, field) in zip(plots, fields):
                    if file_name is not None:
                        enplot.write(file_name + "_%s" % field + ".png", plot)
                    else:
                        # enplot.show(plot,method="ipython")
                        plot.img.show()




def read_map(file, coordinate=None, fields_healpix=None, car_box=None, geometry=None):
    """Create a ``so_map`` object from a fits file.

    Parameters
    ----------
    file: fits file
      name of the fits file
    coordinate: string
      coordinate system of the map
    fields_healpix: integer
      if fields_healpix is not None, load the specified field
    car_box:  2x2 array
        [[dec0,ra0],[dec1,ra1]] in degree

    """

    new_map = so_map()
    hdulist = pyfits.open(file)
    try:
        header = hdulist[1].header
        new_map.pixel = "HEALPIX"
        if fields_healpix is None:
            new_map.ncomp = header["TFIELDS"]
            new_map.data = hp.fitsfunc.read_map(file, field=np.arange(new_map.ncomp))
        else:
            try:
                new_map.ncomp = len(fields_healpix)
            except:
                new_map.ncomp = 1
            new_map.data = hp.fitsfunc.read_map(file, field=fields_healpix)

        new_map.nside = hp.pixelfunc.get_nside(new_map.data)
        try:
            new_map.coordinate = header["SKYCOORD"]
        except:
            new_map.coordinate = None

    except:
        header = hdulist[0].header
        new_map.pixel = header["CTYPE1"][-3:]
        try:
            new_map.ncomp = header["NAXIS3"]
        except:
            new_map.ncomp = 1

        if car_box is not None:
            car_box = np.array(car_box) * np.pi / 180
            new_map.data = enmap.read_map(file, box=car_box)
        elif geometry is not None:
            new_map.data = enmap.read_map(file, geometry=geometry)
        else:
            new_map.data = enmap.read_map(file)

        new_map.nside = None
        new_map.coordinate = header["RADESYS"]
        if new_map.coordinate == "ICRS":
            new_map.coordinate = "equ"

    hdulist.close()

    if coordinate is not None:
        new_map.coordinate = coordinate

    return new_map


def from_components(T, Q, U):
    """Create a (T,Q,U) ``so_map`` object from three fits files.

    Parameters
    ----------
    T : fits file
      name of the T fits file
    Q : fits file
      name of the Q fits file
    U : fits file
      name of the U fits file
    """

    ncomp = 3
    T = enmap.read_map(T)
    Q = enmap.read_map(Q)
    U = enmap.read_map(U)
    shape, wcs = T.geometry
    shape = (ncomp,) + shape
    new_map = so_map()
    new_map.data = enmap.zeros(shape, wcs=wcs, dtype=T.dtype)
    new_map.data[0] = T
    new_map.data[1] = Q
    new_map.data[2] = U
    new_map.pixel = "CAR"
    new_map.nside = None
    new_map.ncomp = ncomp
    new_map.coordinate = "equ"

    return new_map

def get_submap_car(map_car, box, mode="round"):
    """Cut a CAR submap (using pixell).

    Parameters
    ----------
    map : CAR map in ``so_map`` format
      the map to be cut
    box : array_like
      The [[fromy,fromx],[toy,tox]] bounding box to select.
      The resulting map will have a bounding box as close
      as possible to this, but will differ slightly due to
      the finite pixel size.

    mode : str
      How to handle partially selected pixels:
      "round": round bounds using standard rules
      "floor": both upper and lower bounds will be rounded down
      "ceil":  both upper and lower bounds will be rounded up
      "inclusive": lower bounds are rounded down, and upper bounds up
      "exclusive": lower bounds are rounded up, and upper bounds down"""

    submap = map_car.copy()
    submap.data = map_car.data.submap(box, mode=mode)

    return submap


def get_box(ra0, ra1, dec0, dec1):
    """Create box in equatorial coordinates.

    Parameters
    ----------
    ra0, dec0, ra1, dec1 : floats
      coordinates of the box in degrees
    """

    box = np.array([[dec0, ra1], [dec1, ra0]]) * np.pi / 180

    return box


def bounding_box_from_map(map_car):
    """Get a coordinate box from a map.

    Parameters
    ----------
    map_car : ``so_map`` in CAR coordinates
      the map used to define the box
    """

    shape, wcs = map_car.data.geometry

    return enmap.box(shape, wcs)


def from_enmap(emap):
    """Get a ``so_map`` from an enmap (pixell format).

    Parameters
    ----------
    emap : a ndmap object
      the enmap we want to use to define the ``so_map``
    """

    new_map = so_map()
    hdulist = emap.wcs.to_fits()
    header = hdulist[0].header
    new_map.pixel = header["CTYPE1"][-3:]
    shape, wcs = emap.geometry
    new_map.ncomp = shape[0]
    new_map.data = emap.copy()
    new_map.nside = None
    new_map.coordinate = header["RADESYS"]
    if new_map.coordinate == "ICRS":
        new_map.coordinate = "equ"

    return new_map


def healpix2car(healpix_map, template, lmax=None):
    """Project a HEALPIX ``so_map`` into a CAR ``so_map``.

    The projection will be done in harmonic space, you can specify a lmax
    to choose a range of multipoles considered in the projection.
    If the coordinate of the map and the template differ, a rotation will be performed.

    Parameters
    ----------
    healpix_map : ``so_map`` in healpix pixellisation
      the map to be projected
    template: ``so_map`` in CAR pixellisation
      the template that will be projected onto
    lmax: integer
      the maximum multipole in the HEALPIX map to project
    """

    project = template.copy()

    if healpix_map.coordinate is None or template.coordinate is None:
        rot = None
    elif healpix_map.coordinate == template.coordinate:
        rot = None
    else:
        print(
            "will rotate from %s to %s coordinate system"
            % (healpix_map.coordinate, template.coordinate)
        )
        rot = "%s,%s" % (healpix_map.coordinate, template.coordinate)
    if lmax is None:
        lmax = 3 * healpix_map.nside - 1
    if lmax > 3 * healpix_map.nside - 1:
        print("WARNING: your lmax is too large, setting it to 3*nside-1 now")
        lmax = 3 * healpix_map.nside - 1

    spin = [0, 2] if healpix_map.ncomp == 3 else [0]
    project.data = reproject.healpix2map(
        healpix_map.data,
        template.data.shape,
        template.data.wcs,
        lmax=lmax,
        rot=rot,
        spin=spin
    )

    project.ncomp == healpix_map.ncomp
    if project.ncomp == 1:
        project.data = project.data[0]

    return project


def car2car(map_car, template):
    """Project a CAR map into another CAR map with different pixellisation

    Parameters
    ----------
    map : ``so_map`` in CAR pixellisation
      the map to be projected
    template: ``so_map`` in CAR pixellisation
        the template that will be projected onto
    """

    project = template.copy()
    project.data = enmap.project(map_car.data, template.data.shape, template.data.wcs)
    return project


def healpix_template(ncomp, nside, coordinate=None):
    """Create a ``so_map`` template with healpix pixellisation.

    Parameters
    ----------
    ncomp: integer
      the number of components of the map can be 1 or 3 (for T,Q,U)
    nside: integer
      the nside of the healpix map
    coordinate: string
      coordinate system of the map
    """

    temp = so_map()

    if ncomp == 3:
        temp.data = np.zeros((3, 12 * nside ** 2))
    else:
        temp.data = np.zeros((12 * nside ** 2))

    temp.pixel = "HEALPIX"
    temp.ncomp = ncomp
    temp.nside = nside
    temp.coordinate = coordinate
    return temp


def car_template(ncomp, ra0, ra1, dec0, dec1, res, dtype=np.float64):
    """Create a ``so_map`` template with CAR pixellisation in equ coordinates.

    Parameters
    ----------
    ncomp: integer
      the number of components of the map can be 1 or 3 (for T,Q,U)
    ra0,dec0,ra1,dec1: floats
      coordinates of the box in degrees
    res: float
      resolution in arcminute
    """

    if ncomp == 3:
        pre = (3,)
    else:
        pre = ()

    box = get_box(ra0, ra1, dec0, dec1)
    res = res * np.pi / (180 * 60)
    shape, wcs = enmap.geometry(box, res=res, pre=pre)
    return car_template_from_shape_wcs(ncomp, shape, wcs, dtype=dtype)


def full_sky_car_template(ncomp, res, dtype=np.float64):
    """Create a ``so_map`` full sky template with CAR pixellisation in equ coordinates.

    Parameters
    ----------
    ncomp: integer
        the number of components of the map can be 1 or 3 (for T,Q,U)
    res: float
        resolution in arcminute
    """

    if ncomp == 3:
        pre = (3,)
    else:
        pre = ()

    res = res * np.pi / (180 * 60)
    shape, wcs = enmap.fullsky_geometry(res=res, dims=pre)
    return car_template_from_shape_wcs(ncomp, shape, wcs, dtype=dtype)

def car_template_from_shape_wcs(ncomp_out, shape, wcs, dtype=np.float64):
    """
    Create a template from shape and wcs args with
    a number of components `ncomp_out`

    Parameters
    ----------
    ncomp_out: int
      number of components needed (e.g. 3 to create a
      template for (I, Q, U))
    shape: tuple
    wcs: wcs object
    """
    shape_out = shape[-2:]
    if ncomp_out > 1:
        shape_out = (ncomp_out,) + shape_out

    template = so_map()
    template.data = enmap.zeros(shape_out, wcs=wcs, dtype=dtype)
    template.pixel = "CAR"
    template.nside = None
    template.ncomp = ncomp_out
    template.coordinate = "equ"

    return template


def draw_random_location_car(map_car, n_pts):
    """
    Draw n_pts uniformly on a car template and return their pixel indices, taking into accout the fact that the car pixels have different area
    according to their declination.
    also returns a corresponding hit_map also accounting for the fact that a pixel can be drawn
    more than once.
    
    Parameters
    ----------
    map_car : ``so_map`` in CAR coordinates
      the map used to define the box
    n_pts : integer
        the number of point to draw on the map
    """
            
    cumarea = np.cumsum(map_car.data.pixsizemap())
    # Normalize from 0 to 1. We now have something like a CDF
    cumarea /= cumarea[-1]
    # Generate random numbers from 0 to 1:
    r = np.random.uniform(0, 1, n_pts)
    
    # Find where in cumarea each should be
    mpix = np.searchsorted(cumarea, r)
    
    # Expand masked pixel index to full pixel index
    # You can use this directly on the flattened map, but if you want it as a 2d index you can use
    pix2d = np.unravel_index(mpix, map_car.data.shape)
    
    # also produce a map describing how many time a given pixel has been hit
    hitmap = np.bincount(mpix, minlength=map_car.data.npix).reshape(map_car.data.shape[:])

    return pix2d, hitmap




def white_noise(template, rms_uKarcmin_T, rms_uKarcmin_pol=None):
    """Generate a white noise realisation corresponding to the template pixellisation

    Parameters
    ----------
    template: ``so_map`` template
      the template for the white noise generalisation
    rms_uKarcmin_T: float
      the white noise temperature rms in uK.arcmin
    rms_uKarcmin_pol: float
      the white noise polarisation rms in uK.arcmin
      if None set it to sqrt(2)*rms_uKarcmin_T

    """

    noise = template.copy()
    rad_to_arcmin = 60 * 180 / np.pi
    if noise.pixel == "HEALPIX":
        nside = noise.nside
        pixArea = hp.pixelfunc.nside2pixarea(nside) * rad_to_arcmin ** 2
    if noise.pixel == "CAR":
        pixArea = noise.data.pixsizemap() * rad_to_arcmin ** 2
    if noise.ncomp == 1:
        if noise.pixel == "HEALPIX":
            size = len(noise.data)
            noise.data = np.random.randn(size) * rms_uKarcmin_T / np.sqrt(pixArea)
        if noise.pixel == "CAR":
            size = noise.data.shape
            noise.data = np.random.randn(size[0], size[1]) * rms_uKarcmin_T / np.sqrt(pixArea)
    if noise.ncomp == 3:
        if rms_uKarcmin_pol is None:
            rms_uKarcmin_pol = rms_uKarcmin_T * np.sqrt(2)
        if noise.pixel == "HEALPIX":
            size = len(noise.data[0])
            noise.data[0] = np.random.randn(size) * rms_uKarcmin_T / np.sqrt(pixArea)
            noise.data[1] = np.random.randn(size) * rms_uKarcmin_pol / np.sqrt(pixArea)
            noise.data[2] = np.random.randn(size) * rms_uKarcmin_pol / np.sqrt(pixArea)
        if noise.pixel == "CAR":
            size = noise.data[0].shape
            noise.data[0] = np.random.randn(size[0], size[1]) * rms_uKarcmin_T / np.sqrt(pixArea)
            noise.data[1] = np.random.randn(size[0], size[1]) * rms_uKarcmin_pol / np.sqrt(pixArea)
            noise.data[2] = np.random.randn(size[0], size[1]) * rms_uKarcmin_pol / np.sqrt(pixArea)

    return noise


def simulate_source_mask(binary, n_holes, hole_radius_arcmin):
    """Simulate a point source mask in a binary template

    Parameters
    ----------
    binary:  ``so_map`` binary template
      the binay map in which we generate the source mask
    n_holes: integer
      the number of masked point sources
    hole_radius_arcmin: float
      the radius of the holes
    """

    mask = binary.copy()
    if binary.pixel == "HEALPIX":
        idx = np.where(binary.data == 1)
        for i in range(n_holes):
            random_index1 = np.random.choice(idx[0])
            vec = hp.pixelfunc.pix2vec(binary.nside, random_index1)
            disc = hp.query_disc(binary.nside, vec, hole_radius_arcmin / (60.0 * 180) * np.pi)
            mask.data[disc] = 0

    if binary.pixel == "CAR":
        random_index1 = np.random.randint(0, binary.data.shape[0], size=n_holes)
        random_index2 = np.random.randint(0, binary.data.shape[1], size=n_holes)
        mask.data[random_index1, random_index2] = 0
        dist = enmap.distance_transform(mask.data)
        mask.data[dist * 60 * 180 / np.pi < hole_radius_arcmin] = 0

    return mask


def generate_source_mask(binary, coordinates, point_source_radius_arcmin):
    """Generate a point source mask in a binary template

    Parameters
    ----------
    binary:  ``so_map`` binary template
      the binay map in which we generate the source mask
    coordinates: list
      the list of point sources in DEC, RA coordinates (radians)
    point_source_radius_arcmin: float
      the radius of the point sources
    """
    mask = binary.copy()
    mask.data[:] = 1

    if mask.pixel == "HEALPIX":
        vectors = hp.ang2vec(np.pi / 2.0 - coordinates[0], 2 * np.pi - coordinates[1])
        for vec in vectors:
            disc = hp.query_disc(mask.nside, vec, point_source_radius_arcmin / (60.0 * 180) * np.pi)
            mask.data[disc] = 0

    if mask.pixel == "CAR":
        xpix, ypix = mask.data.sky2pix(coordinates).astype(int)
        mask.data[xpix, ypix] = 0
        dist = enmap.distance_transform(
            mask.data, rmax=2 * point_source_radius_arcmin * np.pi / (60.0 * 180)
        )
        mask.data[dist * 60 * 180 / np.pi < point_source_radius_arcmin] = 0

    return mask


def subtract_mono_dipole(emap, mask=None, healpix=True, values=None, bunch=24, return_values=False):
    """Subtract monopole and dipole from a ``enmap`` object.

    Parameters
    ----------
    emap: ``enmap`` or numpy array
      the map from which to compute mono/dipole values
    mask: ``enmap`` or numpy array
      a mask to put on top of the map
    healpix: bool
      flag for using HEALPIX (default) or CAR pixellisation
    bunch: int
      the bunch size (default: 24)
    values
    return_values: bool
      Return mono/dipole values with the subtracted map (default: False)
    """
    map_cleaned = emap.copy()
    if healpix:
        map_masked = hp.ma(emap)
        if mask is not None:
            map_masked.mask = mask < 1
        if values is None:
            mono, dipole = hp.fit_dipole(map_masked)
        else:
            mono, dipole = values
            
        npix = len(emap)
        nside = hp.npix2nside(npix)
        bunchsize = npix // bunch
        for ibunch in range(npix // bunchsize):
            ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
            ipix = ipix[(np.isfinite(emap.flat[ipix]))]
            x, y, z = hp.pix2vec(nside, ipix, False)
            map_cleaned.flat[ipix] -= dipole[0] * x
            map_cleaned.flat[ipix] -= dipole[1] * y
            map_cleaned.flat[ipix] -= dipole[2] * z
            map_cleaned.flat[ipix] -= mono

    else:
    
        if values is None:
            def _get_xyz(dec, ra):
                x = np.cos(dec) * np.cos(ra)
                y = np.cos(dec) * np.sin(ra)
                z = np.sin(dec)
                return x, y, z

            dec, ra = emap.posmap()
            dec = dec.flatten()
            ra = ra.flatten()
            weights = emap.pixsizemap()
            weights = weights.flatten() / (4 * np.pi)

            npix = dec.size
            bunchsize = npix // bunch

            aa = np.zeros((4, 4))
            v = np.zeros(4)
            for ibunch in range(npix // bunchsize):
                ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
                if mask is not None:
                    ipix = ipix[mask.flat[ipix] > 0]

                x, y, z = _get_xyz(dec[ipix], ra[ipix])
                w = weights[ipix]

            # aa[0, 0] += ipix.size
                aa[0, 0] += np.sum(w)
                aa[1, 0] += np.sum(x * w)
                aa[2, 0] += np.sum(y * w)
                aa[3, 0] += np.sum(z * w)
                aa[1, 1] += np.sum(x ** 2 * w)
                aa[2, 1] += np.sum(x * y * w)
                aa[3, 1] += np.sum(x * z * w)
                aa[2, 2] += np.sum(y ** 2 * w)
                aa[3, 2] += np.sum(y * z * w)
                aa[3, 3] += np.sum(z ** 2 * w)
                v[0] += np.sum(emap.flat[ipix] * w)
                v[1] += np.sum(emap.flat[ipix] * x * w)
                v[2] += np.sum(emap.flat[ipix] * y * w)
                v[3] += np.sum(emap.flat[ipix] * z * w)

            aa[0, 1] = aa[1, 0]
            aa[0, 2] = aa[2, 0]
            aa[0, 3] = aa[3, 0]
            aa[1, 2] = aa[2, 1]
            aa[1, 3] = aa[3, 1]
            aa[2, 3] = aa[3, 2]
            res = np.dot(np.linalg.inv(aa), v)
            mono = res[0]
            dipole = res[1:4]
        else:
            mono, dipole = values
            
        for ibunch in range(npix // bunchsize):
            ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)

            x, y, z = _get_xyz(dec[ipix], ra[ipix])
            map_cleaned.flat[ipix] -= dipole[0] * x
            map_cleaned.flat[ipix] -= dipole[1] * y
            map_cleaned.flat[ipix] -= dipole[2] * z
            map_cleaned.flat[ipix] -= mono

    if return_values:
        return map_cleaned, mono, dipole
    return map_cleaned

def fourier_convolution(map_car, fourier_kernel, window=None, use_ducc_rfft=False):

    """do a convolution in fourier space with a fourier_kernel,
    you can optionnaly use a window to remove pathological pixels

    Parameters
    ---------
    map_car : ``so_map`` in CAR pixellisation
        the map to be convolved
    fourier_kernel: 2d array
        the convolution kernel in Fourier space
    window:  ``so_map``
        a window removing pathological pixels
    use_ducc_rfft: boolean
        wether to use ducc real fft instead of enmap complex fft

    """
    if window is not None:
        map_car.data *= window.data

    if use_ducc_rfft == True:
        ft = rfft(map_car.data[:])
        ft  *= fourier_kernel[: ft.shape[-2], : ft.shape[-1]]
        map_car.data[:] = irfft(ft)
    else:
        ft = enmap.fft(map_car.data[:], normalize=True)
        ft  *= fourier_kernel
        map_car.data[:] = enmap.ifft(ft, normalize=True).real

    return map_car
