import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from setuptools import find_packages

import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

compile_opts = {
    "extra_f90_compile_args": [
        "-fopenmp",
        "-ffree-line-length-none",
        "-fdiagnostics-color=always",
        "-Wno-tabs",
    ],
    "f2py_options": ["skip:", "map_border", "calc_weights", ":"],
    "extra_link_args": ["-fopenmp"],
}

mcm = Extension(
    name="pspy.mcm_fortran.mcm_fortran",
    sources=["pspy/mcm_fortran/mcm_fortran.f90", "pspy/wigner3j/wigner3j_sub.f"],
    **compile_opts
)
cov = Extension(
    name="pspy.cov_fortran.cov_fortran",
    sources=["pspy/cov_fortran/cov_fortran.f90", "pspy/wigner3j/wigner3j_sub.f"],
    **compile_opts
)

setup(
    name="pspy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simons Observatory Collaboration Power Spectrum Task Force",
    url="https://github.com/simonsobs/pspy",
    description="Python power spectrum code",
    long_description=readme,
    long_description_content_type="text/x-rst",
    license="BSD license",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={},
    ext_modules=[mcm, cov],
    install_requires=[
        "numpy>=1.20",
        "healpy",
        "pyFFTW",
        "pillow",  # this one should be installed by pixell
        "pixell>=0.7.0",
    ],
    packages=find_packages(),
    package_data={"pspy": ["pspy/tests/data/*.pkl"]},
    include_package_data=True,
    scripts=["scripts/test-pspy"],
)
