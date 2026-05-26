:github_url: https://github.com/google/orbax

Orbax
------------------------------------------------------


Orbax is a modular and customizable JAX checkpointing library built for high performance at scale,
allowing for distributed array storage and checkpoint lifecycle management.

We are focused on providing a JAX-native approach to model persistence and recovery, with the goals of providing an
API that is **easy to use**, **highly performant**, and **maximimally compatible** across the JAX ecosystem.

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Performance
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Checkpointing with Orbax is fast and memory-efficient, allowing for quick start-up and minimal training impact.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Distributed
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Orbax abstracts the details of persisting disributed arrays and provides a unified API for single- and multi-process checkpointing.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Management
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Orbax facilitates checkpointing in a training loop, e.g. through metadata management, garbage collection, and saving policies. 

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Flexibility
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Orbax features out-of-the-box support for advanced workflows like topology-agnostic loading (resharding), partial loading, incremental saving, and more.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Extensibility
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Orbax provides extensibility for user-defined types and logic through customizable handler interfaces.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Exporting
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Orbax provides an associated library, `orbax-export`, for exporting JAX models to Tensorflow SavedModel format.



.. Table of Contents

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation
   guides/checkpoint/v1/orbax_checkpoint_101
   guides/checkpoint/v1/checkpointables
   guides/checkpoint/v1/checkpointing_pytrees
   guides/checkpoint/v1/training

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Performance

   guides/checkpoint/v1/async_checkpointing
   guides/checkpoint/v1/checkpoint_format

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Usage

   guides/checkpoint/v1/checkpoint_format
   guides/checkpoint/v1/customization
   guides/checkpoint/v1/partial_saving
   guides/checkpoint/v1/model_surgery

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api_reference/checkpoint.v1

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Migrating to Orbax V1

   guides/checkpoint/v1/orbax_v0_to_v1_migration
   index.v0

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Exporting

   guides/export/orbax_export_101
   guides/checkpoint/v1/checkpointing_and_exporting_jax_models
   api_reference/export

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Additional Information

   contributors


.. Make sure the grid is not ragged.
Quick Links
---------------
.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`download;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: installation.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Getting Started
         :class-card: sd-text-black sd-bg-light
         :link: guides/checkpoint/v1/orbax_checkpoint_101.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` MNIST Example
         :class-card: sd-text-black sd-bg-light
         :link: guides/checkpoint/v1/training.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`transform;2em` V1 API Migration
         :class-card: sd-text-black sd-bg-light
         :link: guides/checkpoint/v1/orbax_v0_to_v1_migration.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`unarchive;2em` Exporting
         :class-card: sd-text-black sd-bg-light
         :link: guides/export/orbax_export_101.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/checkpoint.v1.html


Support
------------

Please report any issues or request support using our `issue tracker <https://github.com/google/orbax/issues>`_.
