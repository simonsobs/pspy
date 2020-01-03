====
pspy
====

.. image:: https://travis-ci.com/thibautlouis/pspy.svg?branch=master
   :target: https://travis-ci.com/thibautlouis/pspy
.. image:: https://readthedocs.org/projects/pspy/badge/?version=latest
   :target: https://pspy.readthedocs.io/en/latest/?badge=latest
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/thibautlouis/pspy/master?filepath=notebooks/%2Findex.ipynb

Python power spectrum library

https://pspy.readthedocs.io/en/latest/


Installing the code
-------------------

.. code:: shell

    $ git clone https://github.com/thibautlouis/pspy.git /where/to/clone

Then you can install the ``pspy`` library and its dependencies *via*

.. code:: shell

    $ pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``pspy`` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.
