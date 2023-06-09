:github_url: https://github.com/google/orbax

Orbax (Note: Under Construction)
-----


Orbax is a namespace providing common utility libraries for JAX users.
Currently, Orbax encompasses the following functionalities:
*   Checkpointing
*   Exporting

Installation
------------

There is no single `orbax` package, but rather a separate package for each
functionality provided by the Orbax namespace.

The latest release of `orbax-checkpoint` can be installed from
`PyPI <https://pypi.org/project/orbax-checkpoint/>`_ using

``pip install orbax-checkpoint``

You may also install directly from GitHub, using the following command. This
can be used to obtain the most recent version of Optax.

``pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'`

Similarly, `orbax-export` can be installed from
`PyPI <https://pypi.org/project/orbax-export/>`_ using

``pip install orbax-export``

Install from GitHub using the following.

``pip install 'git+https://github.com/google/orbax/#subdirectory=export'` 


.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   orbax_checkpoint_101


.. toctree::
   :caption: Examples
   :maxdepth: 1


.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api

The Team
--------

TODO(cpgaffney)

Support
-------

TODO(cpgaffney)


License
-------

TODO(cpgaffney)


Indices and Tables
==================

* :ref:`genindex`
