CheckpointHandlers
======================================

.. currentmodule:: orbax.checkpoint.handlers

.. automodule:: orbax.checkpoint.handlers


CheckpointHandler
------------------------------
.. autoclass:: CheckpointHandler
  :members:

AsyncCheckpointHandler
---------------------------------
.. autoclass:: AsyncCheckpointHandler
  :members:

StandardCheckpointHandler
----------------------------------
.. autoclass:: StandardCheckpointHandler
  :members:

.. autoclass:: StandardSaveArgs
  :members:

.. autoclass:: StandardRestoreArgs
  :members:

PyTreeCheckpointHandler
--------------------------------
.. autoclass:: PyTreeCheckpointHandler
  :members:

.. autoclass:: PyTreeSaveArgs
  :members:

.. autoclass:: PyTreeRestoreArgs
  :members:

CompositeCheckpointHandler
-----------------------------
.. autoclass:: CompositeCheckpointHandler
  :members:

.. autoclass:: CompositeArgs
  :members:

.. autoclass:: CompositeResults
  :members:

JsonCheckpointHandler
-----------------------------
.. autoclass:: JsonCheckpointHandler
  :members:

.. autoclass:: JsonSaveArgs
  :members:

.. autoclass:: JsonRestoreArgs
  :members:


ArrayCheckpointHandler
-----------------------------
.. autoclass:: ArrayCheckpointHandler
  :members:

.. autoclass:: ArraySaveArgs
  :members:

.. autoclass:: ArrayRestoreArgs
  :members:


ProtoCheckpointHandler
-----------------------------
.. autoclass:: ProtoCheckpointHandler
  :members:

.. autoclass:: ProtoSaveArgs
  :members:

.. autoclass:: ProtoRestoreArgs
  :members:


JaxRandomKeyCheckpointHandler
-----------------------------
.. autoclass:: JaxRandomKeyCheckpointHandler
  :members: save, async_save, restore

.. autoclass:: JaxRandomKeySaveArgs
  :members:

.. autoclass:: JaxRandomKeyRestoreArgs
  :members:

NumpyRandomKeyCheckpointHandler
-------------------------------
.. autoclass:: NumpyRandomKeyCheckpointHandler
  :members: save, async_save, restore

.. autoclass:: NumpyRandomKeySaveArgs
  :members:

.. autoclass:: NumpyRandomKeyRestoreArgs
  :members:
