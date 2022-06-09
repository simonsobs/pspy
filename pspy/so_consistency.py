from itertools import combinations_with_replacement as cwr
from pspy import so_spectra, so_cov
import matplotlib.pyplot as plt
import numpy as np

def get_spectraVec_and_fullCov(specFile, covFile, arA, arB, mode, nBins, tfA = 1, tfB = 1):
    """
    Get the power spectra vector containing auto
    and cross for arA and arB with the associated
    covariance matrix

    Parameters
    ----------
    specFile: string
      name template for the power spectra (using %s as formatter)
    covFile: string
      name template for the covariance matrices
    arA, arB: strings
      name of the two array
    mode: string
      mode to consider (TT, TE, EE, ...)
    nBins: int
      number of bins
    tfA, tfB: 1D arrays
      map maker TF to deconvolve. Default is set to 1
    """

    spectraVec = np.zeros(3*nBins)
    fullCov = np.zeros((3*nBins, 3*nBins))

    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    modes = ["TT", "TE", "ET", "EE"]

    tf = {"arA": tfA, "arB": tfB}
    tfDict = {"TT": [tfA*tfA, tfA*tfB, tfB*tfB],
              "TE": [tfA, tfA, tfB],
              "ET": [tfA, tfB, tfB],
              "EE": [1, 1, 1]}

    for i, xar1 in enumerate(cwr([arA, arB], 2)):

        specName = specFile % xar1
        lb, ps = so_spectra.read_ps(specName, spectra = spectra)
        ps[mode] /= tfDict[mode][i]

        spectraVec[i*nBins:(i+1)*nBins] = ps[mode]

        for j, xar2 in enumerate(cwr([arA, arB], 2)):

            if i <= j: covName = covFile % (xar1 + xar2)
            else: covName = covFile % (xar2 + xar1)

            cov = np.load(covName)
            cov = so_cov.selectblock(cov, modes, n_bins = nBins, block = mode + mode)

            if i <= j:
                fullCov[i*nBins:(i+1)*nBins,
                        j*nBins:(j+1)*nBins] = cov
            else:
                fullCov[i*nBins:(i+1)*nBins,
                        j*nBins:(j+1)*nBins] = cov.T

    # Symmetrize covmat
    fullCov = np.block(fullCov)
    fullCov = np.triu(fullCov)
    transposeCov = fullCov.T
    fullCov += transposeCov - np.diag(fullCov.diagonal())

    return lb, spectraVec, fullCov


def get_projector(nBins, projPattern):
    """
    Get the projection operator
    to apply to a spectra vector to
    get the residual power spectra

    Parameters
    ----------
    nBins: int
      number of bins
    projPattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; applying the
      projector will give : A * psA + B * psB + C * psC
    """

    identity = np.identity(nBins)
    projector = np.hstack((
        identity * projPattern[0],
        identity * projPattern[1],
        identity * projPattern[2]
                          ))
    return projector

def get_residual_spectra_and_cov(spectraVec, fullCov, projPattern, calibVec = np.array([1, 1, 1])):
    """
    Get the residual spectrum and the associated
    covariance matrix from the spectra vector and
    the full covmat.

    Parameters
    ----------
    spectraVec: 1D array
      spectra vector containing the auto and cross spectra
    fullCov: 2D array
      full covariance matrix associated with spectraVec
    nBins: int
      number of bins
    projPattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; applying the
      projector will give : A * psA + B * psB + C * psC
    calibVec: 1D array (len = 3)
      calibration amplitudes to apply to the
      different power spectra
    """
    nBins = len(spectraVec) // 3
    projectorNoCal = get_projector(nBins, projPattern)
    projectorCal = get_projector(nBins, projPattern * calibVec)

    resSpectrum = projectorCal @ spectraVec
    resCov = projectorNoCal @ fullCov @ projectorNoCal.T

    return resSpectrum, resCov

def plot_residual(lb, resPs, resCov, mode, title, fileName):
    """
    Plot the residual power spectrum and
    save it at a png file

    Parameters
    ----------
    lb: 1D array
    resPs: 1D array
      Residual power spectrum
    resCov: 2D array
      Residual covariance matrix
    mode: string
    title: string
    fileName: string
    """

    chi2 = resPs @ np.linalg.inv(resCov) @ resPs

    plt.figure(figsize = (8, 6))
    plt.axhline(0, color = "k", ls = "--")
    plt.errorbar(lb, resPs, yerr = np.sqrt(resCov.diagonal()),
                 ls = "None", marker = ".",
                 label = f"Chi2 : {chi2:.1f}/{len(lb)}")
    plt.title(title)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\Delta D_\ell^\mathrm{%s}$" % mode)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{fileName}.png", dpi = 300)

def get_chi2(spectraVec, fullCov, projPattern, lrange, calibVec):
    """
    Compute the chi2 of the residual
    power spectrum

    Parameters
    ----------
    spectraVec: 1D array
    fullCov: 2D array
    projPattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; applying the
      projector will give : A * psA + B * psB + C * psC
    lrange: 1D array
    calibVec: 1D array (len = 3)
      calibration amplitudes to apply to the
      different power spectra
    """
    resSpec, resCov = get_residual_spectra_and_cov(spectraVec, fullCov,
                                                   projPattern, calibVec = calibVec)
    return resSpec[lrange] @ np.linalg.inv(resCov[np.ix_(lrange[0], lrange[0])]) @ resSpec[lrange]

def get_calibration_amplitudes(spectraVec, fullCov, projPattern, mode, lrange, chainName):
    """
    Get the calibration amplitude and the
    associated error

    Parameters
    ----------
    spectraVec: 1D array
    fullCov: 2D array
    projPattern: 1D array (len = 3)
      If the spectra vector is a concatenation
      of three spectra psA, psB, psC; applying the
      projector will give : A * psA + B * psB + C * psC
    mode: string
    lrange: 1D array
    chainName: string
    """
    from cobaya.run import run
    from getdist.mcsamples import loadMCSamples
    calVec = {"TT": lambda c: np.array([c**2, c, 1]),
              "EE": lambda e: np.array([e**2, e, 1]),
              "TE": lambda e: np.array([e, 1, 1]),
              "ET": lambda e: np.array([e, e, 1])}

    def logL(cal):
        if (projPattern == np.array([1, -1, 0])).all():
            if mode == "TT" or mode == "EE":
                calibVec = np.array([cal, 1, 1])
        else:
            calibVec = calVec[mode](cal)

        chi2 = get_chi2(spectraVec, fullCov, projPattern, lrange, calibVec)
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
        "output": chainName,
        "force": True
           }

    updated_info, sampler = run(info)
    samples = loadMCSamples(chainName, settings = {"ignore_rows": 0.5})
    calMean = samples.mean("cal")
    calStd = np.sqrt(samples.cov(["cal"])[0, 0])

    return calMean, calStd
