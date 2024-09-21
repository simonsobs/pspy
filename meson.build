# Main build file for pspy, helping with building the fortran
project(
  'pspy',
  ['c', 'fortran']
)

py = import('python').find_installation(pure: false)

# Dependencies
py_dep = py.dependency()
omp_dep = dependency('openmp')

# Libraries
cc = meson.get_compiler('c')
c_m_dep = cc.find_library('m', required: true)

fc = meson.get_compiler('fortran')
fortran_m_dep = fc.find_library('m', required: true)

# Directories
library_install_dir = py.get_install_dir() / 'pspy'

# Includes

# Need to massage these into relative paths to keep meson happy.
# It does not allow for absolute paths.
incdir_numpy = run_command(
    py,
    ['-c', 'import numpy; import os; print(os.path.relpath(numpy.get_include()))'],
    check: true
).stdout().strip()

incdir_f2py = run_command(
    py,
    ['-c', 'import numpy.f2py; import os; print(os.path.relpath(numpy.f2py.get_include()))'],
    check: true
).stdout().strip()

# Build fortran extensions
fortran_include = include_directories(incdir_numpy, incdir_f2py)

fortran_sources = {
    'pspy/mcm_fortran/mcm_fortran.f90': '_mcm_fortran',
    'pspy/cov_fortran/cov_fortran.f90': '_cov_fortran',
}

foreach source_name, module_name : fortran_sources
    f2py_output = custom_target(
        input: source_name,
        output: [module_name + '-f2pywrappers2.f90', module_name + 'module.c'],
        command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', module_name, '--lower'],
    )

    py.extension_module(
        module_name,
        [source_name, f2py_output],
        'pspy/wigner3j/wigner3j_sub.f',
        incdir_f2py / 'fortranobject.c',
        include_directories: fortran_include,
        dependencies: [py_dep, omp_dep, c_m_dep, fortran_m_dep],
        install: true,
        subdir: 'pspy'
    )
endforeach

subdir('pspy')
subdir('scripts')
