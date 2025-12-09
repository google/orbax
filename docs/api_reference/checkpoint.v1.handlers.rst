Checkpointable Handlers
============================================================================

.. currentmodule:: orbax.checkpoint.experimental.v1.handlers

.. automodule:: orbax.checkpoint.experimental.v1.handlers
  :members:

Types
------------------------------------------------------------
.. autoclass:: CheckpointableHandler
.. autoclass:: StatefulCheckpointable

Handlers
------------------------------------------------------------
.. autoclass:: PyTreeHandler
.. autoclass:: ProtoHandler
.. autoclass:: JsonHandler

Registration
------------------------------------------------------------
.. autoclass:: CheckpointableHandlerRegistry
.. autofunction:: global_registry
.. autofunction:: local_registry
.. autofunction:: register_handler