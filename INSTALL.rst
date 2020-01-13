
Install at NERSC
----------------

First, make sure you load a nersc python module

.. code:: shell

    module load python/3.7-anaconda-2019.07

You can put this line in your .bashrc.ext in order to load it automatically.
Then, clone pixell and install as follow

.. code:: shell

    git clone https://github.com/simonsobs/pixell.git
    cd pixell
    python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
    pip install -e . --user

Next, clone pspy and install as follow

.. code:: shell

    git clone https://github.com/simonsobs/pspy.git
    cd pspy
    python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
    pip install -e . --user

That is it, you should be done !
note that the pixell library does not run on the login node, so to test go to an interactive one !
