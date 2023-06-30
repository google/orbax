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

For more information how to install orbax, see the project README.


.. For TOC
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Checkpointing

   orbax_checkpoint_101
   api_reference/checkpoint

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Exporting

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

      .. card:: API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/checkpoint.html


Exporting
----------------
.. grid::

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
