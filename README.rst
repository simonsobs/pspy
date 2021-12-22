====
pspy
====
.. inclusion-marker-do-not-remove

``pspy`` is a cosmology code for calculating CMB power spectra and covariance matrices. See the
python example notebooks for an introductory set of examples on how to use the package.

.. image:: https://img.shields.io/pypi/v/pspy.svg?style=flat
   :target: https://pypi.python.org/pypi/pspy/
.. image:: https://img.shields.io/badge/license-BSD-yellow
   :target: https://github.com/simonsobs/pspy/blob/master/LICENSE
.. image:: https://img.shields.io/github/workflow/status/simonsobs/pspy/Testing/feature-github-actions
   :target: https://github.com/simonsobs/pspy/actions?query=workflow%3ATesting
.. image:: https://readthedocs.org/projects/pspy/badge/?version=latest
   :target: https://pspy.readthedocs.io/en/latest/?badge=latest
.. image:: https://codecov.io/gh/simonsobs/pspy/branch/master/graph/badge.svg?token=HHAJ7NQ5CE
   :target: https://codecov.io/gh/simonsobs/pspy
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/simonsobs/pspy/master?filepath=notebooks/%2Findex.ipynb

* Free software: BSD license
* ``pspy`` documentation: https://pspy.readthedocs.io.
* Scientific documentation: https://pspy.readthedocs.io/en/latest/scientific_doc.pdf


Installing
----------

.. code:: shell

    $ pip install pspy [--user]

You can test your installation by running

.. code:: shell

    $ test-pspy

If everything goes fine, no errors will occur. Otherwise, you can report your problem on the `Issues tracker <https://github.com/simonsobs/pspy/issues>`_.

If you plan to develop ``pspy``, it is better to checkout the latest version by doing

.. code:: shell

    $ git clone https://github.com/simonsobs/pspy.git /where/to/clone

Then you can install the ``pspy`` library and its dependencies *via*

.. code:: shell

    $ pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``pspy`` directory without having
to reinstall at every changes.


Ipython notebooks
-----------------

* `Reading, writing and plotting SO maps  <https://pspy.readthedocs.org/en/latest/tutorial_io.html>`_
* `Generate spin0 and spin2 spectra for CAR  <https://pspy.readthedocs.org/en/latest/tutorial_spectra_car_spin0and2.html>`_
* `Generate spin0 and spin2 spectra for HEALPIX  <https://pspy.readthedocs.org/en/latest/tutorial_spectra_healpix_spin0and2.html>`_
* `Projecting HEALPIX to CAR  <https://pspy.readthedocs.org/en/latest/tutorial_projection.html>`_
* `Compute spectra for standard and pure B modes  <https://pspy.readthedocs.org/en/latest/tutorial_purebb.html>`_

Others tutorials can be found under the ``tutorials`` directory.

Dependencies
------------

* Python >= 3.7
* ``pyFFTW`` https://pyfftw.readthedocs.io
* ``healpy`` https://healpy.readthedocs.io
* ``pixell`` >= 0.7.0 https://pixell.readthedocs.io


Authors
------------
* `Thibaut Louis <https://thibautlouis.github.io>`_
* Steve Choi
* DW Han
* Xavier Garrido
* Sigurd Naess

The code is part of `PSpipe <https://github.com/simonsobs/PSpipe>`_ the Simons Observatory power spectrum pipeline.
