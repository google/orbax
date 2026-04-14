orbax.checkpoint V1 API Reference
============================================================================

.. currentmodule:: orbax.checkpoint.v1

Submodules
----------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   checkpoint.v1.arrays
   checkpoint.v1.errors
   checkpoint.v1.handlers
   checkpoint.v1.handler_registration
   checkpoint.v1.options
   checkpoint.v1.partial
   checkpoint.v1.path
   checkpoint.v1.serialization
   checkpoint.v1.training
   checkpoint.v1.tree
   checkpoint.v1.multihost

Top-level Symbols
-----------------

Loading
~~~~~~~
.. autofunction:: load_pytree
.. autofunction:: load_pytree_async
.. autofunction:: load_checkpointables
.. autofunction:: load_checkpointables_async

Saving
~~~~~~
.. autofunction:: save_pytree
.. autofunction:: save_pytree_async
.. autofunction:: save_checkpointables
.. autofunction:: save_checkpointables_async

Metadata
~~~~~~~~
.. autofunction:: pytree_metadata
.. autofunction:: checkpointables_metadata
.. autoclass:: PyTreeMetadata
.. autoclass:: CheckpointMetadata

Path Utilities
~~~~~~~~~~~~~~
.. autofunction:: is_orbax_checkpoint

Constants
~~~~~~~~~
.. autodata:: PLACEHOLDER

Synchronization
~~~~~~~~~~~~~~~
.. autoclass:: AsyncResponse

Checkpointable Handlers
~~~~~~~~~~
.. autoclass:: StatefulCheckpointable
.. autoclass:: CheckpointableHandler

Context
~~~~~~~
.. autoclass:: Context
