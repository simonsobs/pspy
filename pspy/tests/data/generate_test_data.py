import os
import pickle
import tempfile
import time
from itertools import combinations_with_replacement as cwr

import healpy as hp
import numpy as np
import yaml
from pspy import pspy_utils, so_map, so_mcm, so_spectra, so_window, sph_tools


class TimeLogger:
    def __init__(self, verbose):
        self.t0 = None
        self.verbose = verbose

    def start(self, msg):
        if self.verbose:
            self.t0 = time.time()
            print(msg, end="... ", flush=True)

    def stop(self, msg="done"):
        if self.verbose:
            print(f"{msg} in {time.time() - self.t0:.2f} sec")


data_dir = os.path.dirname(__file__)
with open(os.path.join(data_dir, "parameters_test_data.yml"), "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


def do_simulation(car, window_data=None, verbose=True):
    time_log = TimeLogger(verbose)
    tmpl_config = config.get("car" if car else "healpix")

    if verbose:
        print("{} template: {}".format("CAR" if car else "HEALPIX", tmpl_config))

    if car:
        template = so_map.car_template(ncomp=config.get("ncomp"), **tmpl_config)
        binary = so_map.car_template(ncomp=1, **tmpl_config)
        # binary.data[:] = 0
        binary.data[1:-1, 1:-1] = 1
    else:
        nside = tmpl_config.get("nside")
        template = so_map.healpix_template(ncomp=config.get("ncomp"), nside=nside)
        binary = so_map.healpix_template(ncomp=1, nside=nside)
        vec = hp.ang2vec(tmpl_config.get("lon"), tmpl_config.get("lat"), lonlat=True)
        disc = hp.query_disc(nside, vec, radius=tmpl_config.get("radius") * np.pi / 180)
        binary.data[disc] = 1

    time_log.start("Generate binning file")
    binning_file = os.path.join(tempfile.gettempdir(), "binning.dat")
    pspy_utils.create_binning_file(bin_size=40, n_bins=100, file_name=binning_file)
    time_log.stop()

    time_log.start("Generate CMB")
    np.random.seed(config.get("seed"))
    cmb = template.synfast(os.path.join(data_dir, config.get("cl_file")))
    nsplits = config.get("nsplits", 2)
    rms_uKarcmin_T = config.get("rms_uKarcmin_T", 20)
    splits = [cmb.copy() for i in range(nsplits)]
    for i in range(nsplits):
        noise = so_map.white_noise(
            cmb, rms_uKarcmin_T=rms_uKarcmin_T, rms_uKarcmin_pol=np.sqrt(2) * rms_uKarcmin_T
        )
        splits[i].data += noise.data
    time_log.stop()

    if window_data is not None:
        window = binary.copy()
        if car:
            from pixell.enmap import ndmap

            window.data = ndmap(window_data, binary.data.wcs)
        else:
            window.data = window_data
    else:
        time_log.start("Generate mask")
        window = so_window.create_apodization(
            binary,
            apo_type="Rectangle" if car else "C1",
            apo_radius_degree=config.get("apo_radius_degree_survey"),
        )
        mask = so_map.simulate_source_mask(
            binary,
            n_holes=config.get("source_mask_nholes"),
            hole_radius_arcmin=config.get("source_mask_radius"),
        )
        mask = so_window.create_apodization(
            mask, apo_type="C1", apo_radius_degree=config.get("apo_radius_degree_mask")
        )
        window.data *= mask.data
        time_log.stop()

    lmax = config.get("lmax")
    niter = config.get("niter")
    time_log.start(f"Compute MCM and Bbl for lmax={lmax} and niter={niter}")
    windows = (window, window)
    mbb_inv, bbl = so_mcm.mcm_and_bbl_spin0and2(windows, binning_file, lmax=lmax, niter=niter)
    alms = [sph_tools.get_alms(split, windows, lmax=lmax, niter=niter) for split in splits]
    time_log.stop()

    time_log.start("Compute power spectra")
    spectra = config.get("spectra")
    Db = {
        f"split{i1}xsplit{i2}": so_spectra.bin_spectra(
            *so_spectra.get_spectra(alm1, alm2, spectra=spectra),
            binning_file,
            lmax,
            type="Dl",
            mbb_inv=mbb_inv,
            spectra=spectra,
        )
        for (i1, alm1), (i2, alm2) in cwr(enumerate(alms), 2)
    }
    time_log.stop()

    # Reduce size of map by only keeping first 1000 values
    if car:
        cmb.data = cmb.data[:, :100, :100]
    else:
        cmb.data = cmb.data[:, :1000]
    return {
        "cmb": np.asarray(cmb.data),
        "window": np.asarray(window.data),
        "mbb_inv": mbb_inv,
        "bbl": bbl,
        "spectra": Db,
    }


def store_test_data():
    test_data_dict = {"config": config}
    test_data_dict["car"] = do_simulation(car=True)
    test_data_dict["healpix"] = do_simulation(car=False)

    test_data_file = config.get("test_data_file", os.path.join(data_dir, "test_data.pkl"))
    with open(test_data_file, "wb") as f:
        pickle.dump(test_data_dict, f)


if __name__ == "__main__":
    store_test_data()
