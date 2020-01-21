
Installation on MAC OS
----------------------

On MAC OS, one difficulty for installing ``pspy`` is the dependency on the ``pixell`` library.
``pixell`` includes ``sharp``, a C library allowing to do spherical harmonic transforms and we will need to compile it.

First you should go to the Apple Store and get or update ``Xcode``, then install ``macport`` corresponding to your OS version : https://www.macports.org/.

We will need to install a c/fortran compiler and some configure tools

.. code:: shell

    sudo port install gcc9
    sudo port select --set gcc mp-gcc9
    sudo port install automake autoconf libtool

you can check that your ``gcc`` is now the macport one.

.. code:: shell

    which gcc

it should print ``/opt/local/bin/gcc``, if it doesn't, close your terminal, open a new one and try again.

Now the difficult part is done, and the rest is installing python softwares, a complete installation for ``python3.7`` goes as follow (you can skip the part you have already done)

.. code:: shell

    sudo port install python37
    sudo port select --set python python37
    sudo port install py37-ipython
    sudo port select --set ipython py37-ipython
    sudo port install py37-numpy py37-matplotlib
    sudo port select --set cython cython37
    sudo port install py37-pip
    sudo port select --set pip pip37
    sudo pip install scipy==1.3.2
    sudo pip install pyyaml
    sudo pip install Pillow
    sudo pip install astropy==3.1.0
    sudo pip install healpy

Note that we use ``scipy < 1.4`` for pyfftw compatibility and ``astropy < 4`` because of changes in fits convention happening in version 4.
You should now have all the dependencies to install ``pixell``, go to a location where you want to install ``pixell`` and run

.. code:: shell

    git clone https://github.com/simonsobs/pixell.git
    cd pixell
    sudo CC=gcc python setup.py build_ext -i
    sudo CC=gcc python setup.py install
    sudo python setup.py test

the last command will run a bunch of tests that should all be succesfuls.
Then go to a folder where you want to install ``pspy`` and run

.. code:: shell

    git clone https://github.com/simonsobs/pspy.git
    cd pspy
    sudo python setup.py install

to check that the installation is working, go to the tutorials folder of ``pspy`` and  run the scripts !
.. code:: shell
    cd tutorials
    python tuto_spectra_spin0.py
