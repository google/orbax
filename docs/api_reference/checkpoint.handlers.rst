CheckpointHandlers
============================================================================

.. automodule:: orbax.checkpoint.handlers

.. currentmodule:: orbax.checkpoint

CheckpointHandler
------------------------------------------------------------
.. autoclass:: CheckpointHandler
  :members:

AsyncCheckpointHandler
------------------------------------------------------------------
.. autoclass:: AsyncCheckpointHandler
  :members:

StandardCheckpointHandler
--------------------------------------------------------------------
.. autoclass:: StandardCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: StandardSaveArgs
  :members:

.. autoclass:: StandardRestoreArgs
  :members:

PyTreeCheckpointHandler
----------------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: PyTreeCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: PyTreeSaveArgs
  :members:

.. autoclass:: PyTreeRestoreArgs
  :members:

CompositeCheckpointHandler
----------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: CompositeCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: CompositeArgs
  :members:

.. autoclass:: CompositeResults
  :members:

JsonCheckpointHandler
----------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: JsonCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: JsonSaveArgs
  :members:

.. autoclass:: JsonRestoreArgs
  :members:

ArrayCheckpointHandler
----------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: ArrayCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: ArraySaveArgs
  :members:

.. autoclass:: ArrayRestoreArgs
  :members:


ProtoCheckpointHandler
----------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: ProtoCheckpointHandler
  :members:

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: ProtoSaveArgs
  :members:

.. autoclass:: ProtoRestoreArgs
  :members:


JaxRandomKeyCheckpointHandler
----------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: JaxRandomKeyCheckpointHandler
  :members: save, async_save, restore

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: JaxRandomKeySaveArgs
  :members:

.. autoclass:: JaxRandomKeyRestoreArgs
  :members:

NumpyRandomKeyCheckpointHandler
--------------------------------------------------------------
.. currentmodule:: orbax.checkpoint

.. autoclass:: NumpyRandomKeyCheckpointHandler
  :members: save, async_save, restore

.. currentmodule:: orbax.checkpoint.handlers

.. autoclass:: NumpyRandomKeySaveArgs
  :members:

.. autoclass:: NumpyRandomKeyRestoreArgs
  :members:
