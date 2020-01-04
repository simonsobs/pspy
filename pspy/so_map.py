"""
SO map class for handling healpix and CAR maps.
This is a wrapper around healpix and enlib (pixell).
"""

import copy
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from pixell import curvedsky, enmap, enplot, powspec, reproject


class so_map:
    """Class defining a ``so_map`` object.
    """
    def __init__(self):
        pass

    def copy(self):
        """ Create a copy of the ``so_map`` object.
        """

        return copy.deepcopy(self)

    def info(self):
        """ Print information about the ``so_map`` object.
        """

        print("pixellisation:", self.pixel)
        print("number of components:", self.ncomp)
        print("number of pixels:", self.data.shape[:] if self.ncomp == 1 else self.data.shape[1:])
        print("nside:", self.nside)
        print("geometry:", self.geometry)
        print("coordinates:", self.coordinate)

    def write_map(self, file_name):
        """Write the ``so_map`` to disk.

        Parameters
        ----------
        file_name : string
          the name of the fits file
        """

        if self.pixel == "HEALPIX":
            hp.write_map(file_name, self.data, overwrite=True)
        if self.pixel == "CAR":
            enmap.write_map(file_name, self.data)

    def upgrade(self, factor):
        """Upgrade the ``so_map``.

        Parameters
        ----------
        factor : integer
          factor of increased pixel resolution (should be a factor of 2)

        """

        assert (factor % 2 == 0), "factor should be a factor of 2"

        upgrade = self.copy()
        if self.pixel == "HEALPIX":
            nside_out = self.nside * factor
            upgrade.data = hp.ud_grade(self.data, nside_out=nside_out)
            upgrade.nside = nside_out
        if self.pixel == "CAR":
            upgrade.data = enmap.upgrade(self.data, factor)
            upgrade.geometry = upgrade.data.geometry[1:]
        return upgrade

    def downgrade(self, factor):
        """Downgrade the ``so_map``.

        Parameters
        ----------
        factor : integer
          factor of decreased pixel resolution (should be a factor of 2)

        """

        assert (factor % 2 == 0), "factor should be a factor of 2"

        downgrade = self.copy()
        if self.pixel == "HEALPIX":
            nside_out = nside / factor
            downgrade.data = hp.ud_grade(self.data, nside_out=nside_out)
            downgrade.nside = nside_out
        if self.pixel == "CAR":
            downgrade.data = enmap.downgrade(self.data, factor)
            downgrade.geometry = downgrade.data.geometry[1:]
        return downgrade

    def synfast(self, clfile):
        """Fill a ``so_map`` with a cmb gaussian simulation.

        Parameters
        ----------
        clfile : ``CAMB`` data file
          lensed power spectra file from ``CAMB``

        """

        if self.pixel == "HEALPIX":
            from pspy.pspy_utils import ps_lensed_theory_to_dict
            l, ps = ps_lensed_theory_to_dict(clfile, output_type="Cl", startAtZero=True)
            if self.ncomp == 1:
                self.data = hp.synfast(ps["TT"], self.nside, new=True, verbose=False)
            else:
                self.data = hp.synfast((ps["TT"], ps["EE"], ps["BB"], ps["TE"]),
                                       self.nside,
                                       new=True,
                                       verbose=False)

        if self.pixel == "CAR":
            ps = powspec.read_spectrum(clfile)[:self.ncomp, :self.ncomp]
            self.data = curvedsky.rand_map(self.data.shape, self.data.wcs, ps)

        return self

    def plot(self,
             color="planck",
             color_range=None,
             file_name=None,
             ticks_spacing_car=1,
             title="",
             cbar=True,
             hp_gnomv=None):
        """Plot a ``so_map``.

        Parameters
        ----------
        color: cmap
          a ``matplotlib`` colormap (or "planck")
        color_range: scalar for single component or len(3) list for T, Q, U.
          the range of the colorscale
        file_name: string
          file_name is the name of the png file that will be created, if ``None`` the plot will be displayed.
        title: string
          the title of the plot.
        cbar: boolean
          set to ``True`` to display the colorbar.
        ticks_spacing_CAR: float
          for CAR plot, choose the spacing of the ticks.
        hp_gnomv: boolean
          gnomview projection for HEALPIX plotting, expected (lon_c, lat_c, xsize, reso).

        """

        if self.pixel == "HEALPIX":
            if color == "planck":
                from matplotlib.colors import ListedColormap
                from pspy.so_config import DEFAULT_DATA_DIR
                planck_rgb_file = os.path.join(DEFAULT_DATA_DIR, "Planck_Parchment_RGB.txt")
                colombi1_cmap = ListedColormap(np.loadtxt(planck_rgb_file) / 255.)
                colombi1_cmap.set_bad("white")
                colombi1_cmap.set_under("white")
                cmap = colombi1_cmap
            else:
                cmap = plt.get_cmap(color)
                cmap.set_bad("white")
                cmap.set_under("white")

            if self.ncomp == 1:

                min_range, max_range = None, None
                if color_range is not None:
                    min_range = -color_range
                    max_range = +color_range

                if hp_gnomv is not None:
                    lon, lat, xsize, reso = hp_gnomv
                    hp.gnomview(self.data,
                                min=min_range,
                                max=max_range,
                                cmap=cmap,
                                notext=True,
                                title=title,
                                cbar=cbar,
                                rot=(lon, lat, 0),
                                xsize=xsize,
                                reso=reso)
                else:
                    hp.mollview(self.data,
                                min=min_range,
                                max=max_range,
                                cmap=cmap,
                                notext=True,
                                title=title,
                                cbar=cbar)
                if file_name is not None:
                    plt.savefig(file_name + ".png", bbox_inches="tight")
                    plt.clf()
                    plt.close
                else:
                    plt.show()
            else:
                fields = ["T", "Q", "U"]
                min_ranges = {l1: None for l1 in fields}
                max_ranges = {l1: None for l1 in fields}
                if color_range is not None:
                    for i, l1 in enumerate(fields):
                        min_ranges[l1] = -color_range[i]
                        max_ranges[l1] = +color_range[i]
                for data, l1 in zip(self.data, fields):

                    if hp_gnomv is not None:
                        lon, lat, xsize, reso = hp_gnomv
                        hp.gnomview(data,
                                    min=min_ranges[l1],
                                    max=max_ranges[l1],
                                    cmap=cmap,
                                    notext=True,
                                    title=title,
                                    cbar=cbar,
                                    rot=(lon, lat, 0),
                                    xsize=xsize,
                                    reso=reso)
                    else:
                        hp.mollview(data,
                                    min=min_ranges[l1],
                                    max=max_ranges[l1],
                                    cmap=cmap,
                                    notext=True,
                                    title=l1 + "" + title,
                                    cbar=cbar)
                    if file_name is not None:
                        plt.savefig(file_name + "_%s" % l1 + ".png", bbox_inches="tight")
                        plt.clf()
                        plt.close
                    else:
                        plt.show()

        if self.pixel == "CAR":
            if self.ncomp == 1:
                if color_range is not None:
                    max_range = "%s" % (color_range)
                else:
                    max_range = "%s" % (np.max(self.data))

                plots = enplot.get_plots(self.data,
                                         color=color,
                                         range=max_range,
                                         colorbar=1,
                                         ticks=ticks_spacing_car)

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
                    max_range = "%s:%s:%s" % (np.max(self.data[0]), np.max(
                        self.data[1]), np.max(self.data[2]))

                plots = enplot.get_plots(self.data,
                                         color=color,
                                         range=max_range,
                                         colorbar=1,
                                         ticks=ticks_spacing_car)

                for (plot, l1) in zip(plots, fields):
                    if file_name is not None:
                        enplot.write(file_name + "_%s" % l1 + ".png", plot)
                    else:
                        #enplot.show(plot,method="ipython")
                        plot.img.show()


def read_map(file_name, coordinate=None, fields_healpix=None):
    """Create a ``so_map`` object from a fits file.

    Parameters
    ----------
    file_name: fits file
      name of the fits file
    coordinate: string
      coordinate system of the map
    fields_healpix: integer
      if fields_healpix is not None, load the specified field

    """

    new_map = so_map()
    from astropy.io import fits
    hdulist = fits.open(file_name)
    try:
        header = hdulist[1].header
        new_map.pixel = "HEALPIX"
        if fields_healpix is None:
            new_map.ncomp = header["TFIELDS"]
            new_map.data = hp.read_map(file_name, field=np.arange(new_map.ncomp), verbose=False)
        else:
            try:
                new_map.ncomp = len(fields_healpix)
            except:
                new_map.ncomp = 1
            new_map.data = hp.read_map(file_name, verbose=False, field=fields_healpix)

        new_map.nside = hp.get_nside(new_map.data)
        new_map.geometry = "healpix geometry"
        try:
            new_map.coordinate = header["SKYCOORD"]
        except:
            new_map.coordinate = None

    except:
        header = hdulist[0].header
        new_map.pixel = (header["CTYPE1"][-3:])
        try:
            new_map.ncomp = header["NAXIS3"]
        except:
            new_map.ncomp = 1
        new_map.data = enmap.read_map(file_name)
        new_map.nside = None
        new_map.geometry = new_map.data.geometry[1:]
        new_map.coordinate = header["RADESYS"]
        if new_map.coordinate == "ICRS":
            new_map.coordinate = "equ"

    if coordinate is not None:
        new_map.coordinate = coordinate

    return new_map


def from_components(T, Q, U):
    """Create a (T, Q, U) ``so_map`` object from three fits files.

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
    shape = ((ncomp, ) + shape)
    new_map = so_map()
    new_map.data = enmap.zeros(shape, wcs=wcs, dtype=None)
    new_map.data[0] = T
    new_map.data[1] = Q
    new_map.data[2] = U
    new_map.pixel = "CAR"
    new_map.nside = None
    new_map.ncomp = ncomp
    new_map.geometry = T.geometry[1:]
    new_map.coordinate = "equ"

    return new_map


def get_submap_car(map_car, box, mode):
    """Cut a CAR submap (using pixell).

    Parameters
    ----------
    map_car : CAR map in so_map format
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
    submap.geometry = map_car.data.submap(box, mode=mode).geometry[1:]

    return submap


def get_box(ra0, ra1, dec0, dec1):
    """Create box in equatorial coordinates.

    Parameters
    ----------
    ra0, dec0, ra1, dec1 : floats
      coordinates of the box in degrees
    """

    box = np.deg2rad(np.array([[dec0, ra1], [dec1, ra0]]))

    return box


def bounding_box_from_map(map_car):
    """Get a coordinate box from a map.

    Parameters
    ----------
    map_car : so_map in CAR coordinates
      the map used to define the box
    """

    shape, wcs = map_car.data.geometry

    return enmap.box(shape, wcs)


def from_enmap(emap):
    """Get a ``so_map`` from an enmap (pixell format).

    Parameters
    ----------
    emap : a ndmap object
      the enmap we want to use to define the so_map
    """

    new_map = so_map()
    hdulist = emap.wcs.to_fits()
    header = hdulist[0].header
    new_map.pixel = (header["CTYPE1"][-3:])
    try:
        new_map.ncomp = header["NAXIS3"]
    except:
        new_map.ncomp = 1
    new_map.data = emap.copy()
    new_map.nside = None
    new_map.geometry = new_map.data.geometry[1:]
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
        print("will rotate from %s to %s coordinate system" %
              (healpix_map.coordinate, template.coordinate))
        rot = "%s,%s" % (healpix_map.coordinate, template.coordinate)
    if lmax > 3 * healpix_map.nside - 1:
        print("WARNING: your lmax is too large, setting it to 3*nside-1 now")
        lmax = 3 * healpix_map.nside - 1
    if lmax is None:
        lmax = 3 * healpix_map.nside - 1
    project.data = reproject.enmap_from_healpix(healpix_map.data,
                                                template.data.shape,
                                                template.data.wcs,
                                                ncomp=healpix_map.ncomp,
                                                unit=1,
                                                lmax=lmax,
                                                rot=rot,
                                                first=0)

    return project


def car2car(map_car, template):
    """Project a CAR map into another CAR map with different pixellisation

    Parameters
    ----------
    car_map : ``so_map`` in CAR pixellisation
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
        temp.data = np.zeros((3, 12 * nside**2))
    else:
        temp.data = np.zeros((12 * nside**2))

    temp.pixel = "HEALPIX"
    temp.ncomp = ncomp
    temp.nside = nside
    temp.geometry = "healpix geometry"
    temp.coordinate = coordinate
    return temp


def car_template(ncomp, ra0, ra1, dec0, dec1, res):
    """Create a ``so_map`` template with CAR pixellisation in equatorial coordinates.

    Parameters
    ----------
    ncomp: integer
      the number of components of the map can be 1 or 3 (for T,Q,U)
    ra0, dec0, ra1, dec1: floats
      coordinates of the box in degrees
    res: float
      resolution in arcminute
    """

    pre = (3, ) if ncomp == 3 else ()
    box = get_box(ra0, ra1, dec0, dec1)
    res = np.deg2rad(res) / 60
    temp = so_map()
    shape, wcs = enmap.geometry(box, res=res, pre=pre)
    temp.data = enmap.zeros(shape, wcs=wcs, dtype=None)
    temp.pixel = "CAR"
    temp.nside = None
    temp.ncomp = ncomp
    temp.geometry = temp.data.geometry[1:]
    temp.coordinate = "equ"
    return temp


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
    rad_to_arcmin = np.rad2deg(60)
    if noise.pixel == "HEALPIX":
        nside = noise.nside
        pixArea = hp.nside2pixarea(nside) * rad_to_arcmin**2
    if noise.pixel == "CAR":
        pixArea = noise.data.pixsizemap() * rad_to_arcmin**2
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
    binary:  so_map binary template
      the binay map in which we generate the source mask
    n_holes: integer
      the number of masked point sources
    hole_radius_arcmin: float
      the radius of the holes
    """

    mask = binary.copy()
    rad_to_arcmin = np.rad2deg(60)
    if binary.pixel == "HEALPIX":
        idx = np.where(binary.data == 1)
        for i in range(n_holes):
            random_index1 = np.random.choice(idx[0])
            vec = hp.pix2vec(binary.nside, random_index1)
            disc = hp.query_disc(binary.nside, vec, hole_radius_arcmin / rad_to_arcmin)
            mask.data[disc] = 0

    if binary.pixel == "CAR":
        pixSize_arcmin = np.sqrt(binary.data.pixsize() * rad_to_arcmin**2)
        random_index1 = np.random.randint(0, binary.data.shape[0], size=n_holes)
        random_index2 = np.random.randint(0, binary.data.shape[1], size=n_holes)
        mask.data[random_index1, random_index2] = 0
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask.data)
        mask.data[dist * pixSize_arcmin < hole_radius_arcmin] = 0

    return mask
