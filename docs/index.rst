:github_url: https://github.com/google/orbax

Orbax
------------------------------------------------------


Orbax is an umbrella namespace providing common training utilities for JAX
users. It includes multiple distinct but interrelated libraries.

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Checkpointing
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            A flexible and customizable API for managing checkpoints consisting
            of various user-defined objects in multi-host, multi-device settings.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Exporting
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            A library for exporting JAX models to Tensorflow SavedModel format.


Installation
---------------

There is no single `orbax` package, but rather a separate package for each
functionality provided by the Orbax namespace.

The latest release of `orbax-checkpoint` can be installed from
`PyPI <https://pypi.org/project/orbax-checkpoint/>`_ using

``pip install orbax-checkpoint``

You may also install directly from GitHub, using the following command. This
can be used to obtain the most recent version of Optax.

``pip install 'git+https://github.com/google/orbax/#subdirectory=checkpoint'``

Similarly, `orbax-export` can be installed from
`PyPI <https://pypi.org/project/orbax-export/>`_ using

``pip install orbax-export``

Install from GitHub using the following.

``pip install 'git+https://github.com/google/orbax/#subdirectory=export'``


.. For TOC
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Checkpointing

   orbax_checkpoint_101
   orbax_checkpoint_api_overview
   optimized_checkpointing
   custom_handlers
   transformations
   preemption_checkpointing
   async_checkpointing
   api_reference/checkpoint

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Exporting

   orbax_export_101
   api_reference/export


Checkpointing
---------------

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Getting Started
         :class-card: sd-text-black sd-bg-light
         :link: orbax_checkpoint_101.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: API Overview
         :class-card: sd-text-black sd-bg-light
         :link: orbax_checkpoint_api_overview.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Optimized Checkpointing
         :class-card: sd-text-black sd-bg-light
         :link: optimized_checkpointing.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Custom Handlers
         :class-card: sd-text-black sd-bg-light
         :link: custom_handlers.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Transformations
         :class-card: sd-text-black sd-bg-light
         :link: transformations.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Preemption Tolerance
         :class-card: sd-text-black sd-bg-light
         :link: preemption_checkpointing.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Async Checkpointing
         :class-card: sd-text-black sd-bg-light
         :link: async_checkpointing.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/checkpoint.html


Exporting
----------------
.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: Getting Started
         :class-card: sd-text-black sd-bg-light
         :link: orbax_export_101.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/export.html


.. For TOC
.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Additional Information

   contributors


Support
------------

Please report any issues or request support using our `issue tracker <https://github.com/google/orbax/issues>`_.
