TypeHandlers
============================================================================

.. currentmodule:: orbax.checkpoint.type_handlers

.. automodule:: orbax.checkpoint.type_handlers

Arguments for PyTreeCheckpointHandler
--------------------------------------------------------------------------
.. autoclass:: SaveArgs
  :members:

.. autoclass:: RestoreArgs
  :members:

.. autoclass:: ArrayRestoreArgs
  :members:

TypeHandler
------------------------------------------------
.. autoclass:: TypeHandler
  :members:

NumpyHandler
------------------------------------------------
.. autoclass:: NumpyHandler
  :members:

ScalarHandler
------------------------------------------------
.. autoclass:: ScalarHandler
  :members:

ArrayHandler
------------------------------------------------
.. autoclass:: ArrayHandler
  :members:

SingleReplicaArrayHandler
------------------------------------------------
.. autoclass:: SingleReplicaArrayHandler
  :members:

StringHandler
------------------------------------------------
.. autoclass:: StringHandler
  :members:

Tensorstore functions
------------------------------------------------
.. autofunction:: is_ocdbt_checkpoint
.. autofunction:: merge_ocdbt_per_process_files
.. autofunction:: get_json_tspec_write
.. autofunction:: get_json_tspec_read
.. autofunction:: get_ts_context
.. autofunction:: get_cast_tspec_serialize
.. autofunction:: get_cast_tspec_deserialize

TypeHandler registry
------------------------------------------------
.. autoclass:: TypeHandlerRegistry
.. autofunction:: create_type_handler_registry
.. autofunction:: register_type_handler
.. autofunction:: get_type_handler
.. autofunction:: has_type_handler
.. autofunction:: register_standard_handlers_with_options
