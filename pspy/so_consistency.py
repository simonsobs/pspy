from itertools import combinations_with_replacement as cwr
from pspy import so_spectra, so_cov
import matplotlib.pyplot as plt
import numpy as np

def append_spectra_and_cov(ps_dict, cov_dict, spec_list):
    """
    Get the power spectra vector containing
    the spectra in `spec_list` with the
    associated covariance matrix.

    Parameters
    ----------
    ps_dict: dict
      dict containing the different power spectra
    cov_list: dict
      dict containing the covariances
    spec_list: list
      list containing the name of the spectra

    """
    n_bins = len(ps_dict[spec_list[0]])
    n_spec = len(spec_list)

    spectra_vec = np.zeros(n_bins * n_spec)
    full_cov = np.zeros((n_bins * n_spec,
                         n_bins * n_spec))

    for i, key1 in enumerate(spec_list):
        spectra_vec[i*n_bins:(i+1)*n_bins] = ps_dict[key1]
        for j, key2 in enumerate(spec_list):
            if j < i: continue
            full_cov[i*n_bins:(i+1)*n_bins,
                     j*n_bins:(j+1)*n_bins] = cov_dict[key1, key2]

    full_cov = np.triu(full_cov)
    transpose_cov = full_cov.T
    full_cov += transpose_cov - np.diag(full_cov.diagonal())

    return spectra_vec, full_cov


def get_projector(n_bins, proj_pattern):
    """
    Get the projection operator
    to apply to a spectra vector to
    get the residual power spectra

    Parameters
    ----------
    n_bins: int
      number of bins
    proj_pattern: 1D array
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    """
    N = len(proj_pattern)
    identity = np.identity(n_bins)

    projector = (np.tile(identity, (1, N)) *
                 np.tile(np.repeat(proj_pattern, n_bins), (n_bins, 1)))
    return projector


def project_spectra_vec_and_cov(spectra_vec, full_cov, proj_pattern, calib_vec = None):
    """
    Get the residual spectrum and the associated
    covariance matrix from the spectra vector and
    the full covmat.

    Parameters
    ----------
    spectra_vec: 1D array
      spectra vector
    full_cov: 2D array
      full covariance matrix associated with spectra_vec
    proj_pattern: 1D array
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    calib_vec: 1D array
      calibration amplitudes to apply to the
      different power spectra
    """
    n_bins = len(spectra_vec) // len(proj_pattern)
    projector_uncal = get_projector(n_bins, proj_pattern)
    if calib_vec is None:
        calib_vec = np.ones(len(proj_pattern))
    projector_cal = get_projector(n_bins, proj_pattern * calib_vec)

    res_spectrum = projector_cal @ spectra_vec
    res_cov = projector_uncal @ full_cov @ projector_uncal.T

    return res_spectrum, res_cov

def get_chi2(spectra_vec, full_cov, proj_pattern, calib_vec = None, lrange = None):
    """
    Compute the chi2 of the residual
    power spectrum

    Parameters
    ----------
    spectra_vec: 1D array
      spectra vector
    full_cov: 2D array
      full covariance matrix associated with spectra_vec
    proj_pattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    lrange: 1D array
    calib_vec: 1D array
      calibration amplitudes to apply to the
      different power spectra
    """
    res_spec, res_cov = project_spectra_vec_and_cov(spectra_vec, full_cov,
                                                    proj_pattern, calib_vec)
    if lrange is not None:
        return res_spec[lrange] @ np.linalg.inv(res_cov[np.ix_(lrange, lrange)]) @ res_spec[lrange]
    else:
        return res_spec @ np.linalg.inv(res_cov) @ res_spec

def plot_residual(lb, res_spec, res_cov, mode, title, file_name):
    """
    Plot the residual power spectrum and
    save it at a png file

    Parameters
    ----------
    lb: 1D array
    res_spec: 1D array
      Residual power spectrum
    res_cov: 2D array
      Residual covariance matrix
    mode: string
    title: string
    fileName: string
    """

    chi2 = res_spec @ np.linalg.inv(res_cov) @ res_spec

    plt.figure(figsize = (8, 6))
    plt.axhline(0, color = "k", ls = "--")
    plt.errorbar(lb, res_spec, yerr = np.sqrt(res_cov.diagonal()),
                 ls = "None", marker = ".",
                 label = f"Chi2 : {chi2:.1f}/{len(lb)}")
    plt.title(title)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\Delta D_\ell^\mathrm{%s}$" % mode)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{file_name}.png", dpi = 300)

def get_calibration_amplitudes(spectra_vec, full_cov, proj_pattern, mode, lrange, chain_name):
    """
    Get the calibration amplitude and the
    associated error

    Parameters
    ----------
    spectra_vec: 1D array
    full_cov: 2D array
    proj_pattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; and
      proj_pattern is [A, B, C], applying the
      projector to a spectra vector will give :
        A * psA + B * psB + C * psC
    mode: string
    lrange: 1D array
    chain_name: string
    """
    try:
        from cobaya.run import run
        from getdist.mcsamples import loadMCSamples
    except ModuleNotFoundError:
        raise ModuleNotFoundError("You need to install Cobaya to use this function")

    cal_vec = {# Global calibration :
               #  multiplicative factor to apply to
               # the different cross spectra to obtain
               # a calibration amplitude. Here the cross
               # spectra are [AxA, AxR, RxR] with A the
               # array you want to calibrate, and R the
               # reference array
              "TT": lambda c: np.array([c**2, c, 1]),
              # Pol. Eff.
              "EE": lambda e: np.array([e**2, e, 1]),
              "TE": lambda e: np.array([e, 1, 1]),
              "ET": lambda e: np.array([e, e, 1])}

    def logL(cal):
        if (proj_pattern == np.array([1, -1, 0])).all():
            if mode == "TT" or mode == "EE":
                # If we want to fit for the calib (resp. pol.eff.)
                # using the combination (c**2)AxA - c AxR,
                # we set the multiplicative factor to be
                # [c, 1, 1] such that we are minimizing
                # the residual c AxA - AxR
                calib_vec = np.array([cal, 1, 1])
        else:
            calib_vec = cal_vec[mode](cal)

        chi2 = get_chi2(spectra_vec, full_cov, proj_pattern, calib_vec, lrange)
        return -0.5 * chi2

    info = {
        "likelihood": {"my_like": logL},
        "params": {
            "cal": {
                "prior": {
                    "min": 0.5,
                    "max": 1.5
                         },
                "latex": "c"
                 }
                  },
        "sampler": {
            "mcmc": {
                "max_tries": 1e4,
                "Rminus1_stop": 0.001,
                "Rminus1_cl_stop": 0.03,
                    }
                   },
        "output": chain_name,
        "force": True
           }

    updated_info, sampler = run(info)
    samples = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})
    cal_mean = samples.mean("cal")
    cal_std = np.sqrt(samples.cov(["cal"])[0, 0])

    return cal_mean, cal_std
