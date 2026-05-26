Installation
============================================================================

There is no single `orbax` package, but rather a separate package for each
functionality provided by the Orbax namespace.

Checkpointing
-------------

The latest release of `orbax-checkpoint` can be installed from
`PyPI <https://pypi.org/project/orbax-checkpoint/>`_ using:

.. code-block:: bash

   pip install orbax-checkpoint

You may also install directly from GitHub, using the following command. This
can be used to obtain the most recent version of Orbax:

.. code-block:: bash

   pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'

.. note::
   Certain edge cases of `orbax-checkpoint` may not work on Windows. Support is not currently planned.

Exporting
---------

Similarly, `orbax-export` can be installed from
`PyPI <https://pypi.org/project/orbax-export/>`_ using:

.. code-block:: bash

   pip install orbax-export

Install from GitHub using the following:

.. code-block:: bash

   pip install 'git+https://github.com/google/orbax/#subdirectory=export'

Installing from a Pull Request
------------------------------

To install and test changes from an open GitHub Pull Request, you can use `pip` to install directly from the PR's head reference.

Replace ``<PR_NUMBER>`` with the actual pull request ID.

.. code-block:: bash

   pip install 'git+https://github.com/google/orbax.git@refs/pull/<PR_NUMBER>/head#subdirectory=checkpoint'

