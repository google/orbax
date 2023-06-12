TypeHandlers
======================================

.. currentmodule:: orbax.checkpoint.type_handlers

.. automodule:: orbax.checkpoint.type_handlers

Arguments for PyTreeCheckpointHandler
-------------------------------------
.. autoclass:: SaveArgs
  :members:

.. autoclass:: RestoreArgs
  :members:

.. autoclass:: ArrayRestoreArgs
  :members:

TypeHandler
------------------------
.. autoclass:: TypeHandler
  :members:

NumpyHandler
------------------------
.. autoclass:: NumpyHandler
  :members:

ScalarHandler
------------------------
.. autoclass:: ScalarHandler
  :members:

ArrayHandler
------------------------
.. autoclass:: ArrayHandler
  :members:

StringHandler
------------------------
.. autoclass:: StringHandler
  :members:

OCDBT functions
------------------------
.. autofunction:: create_coordinator_server_and_context
.. autofunction:: is_ocdbt_checkpoint

TypeHandler registry
------------------------
.. autofunction:: register_type_handler
.. autofunction:: get_type_handler
.. autofunction:: has_type_handler
.. autofunction:: register_standard_handlers_with_options