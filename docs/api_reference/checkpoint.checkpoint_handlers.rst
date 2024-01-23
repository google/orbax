CheckpointHandlers
======================================

.. currentmodule:: orbax.checkpoint

.. automodule:: orbax.checkpoint


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

PyTreeCheckpointHandler
--------------------------------
.. autoclass:: PyTreeCheckpointHandler
  :members:

CompositeCheckpointHandler
-----------------------------
.. autoclass:: CompositeCheckpointHandler
  :members:

JsonCheckpointHandler
-----------------------------
.. autoclass:: JsonCheckpointHandler
  :members:


ArrayCheckpointHandler
-----------------------------
.. autoclass:: ArrayCheckpointHandler
  :members:


ProtoCheckpointHandler
-----------------------------
.. autoclass:: ProtoCheckpointHandler
  :members:


JaxRandomKeyCheckpointHandler
-----------------------------
.. autoclass:: JaxRandomKeyCheckpointHandler
  :members: save, async_save, restore

NumpyRandomKeyCheckpointHandler
-----------------------------
.. autoclass:: NumpyRandomKeyCheckpointHandler
  :members: save, async_save, restore