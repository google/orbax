Metadata Utilities
======================================

.. currentmodule:: orbax.checkpoint.metadata

.. automodule:: orbax.checkpoint.metadata


Tree Metadata
------------------------

.. autoclass:: Metadata
  :members:

.. autoclass:: ArrayMetadata
  :members:

.. autoclass:: ScalarMetadata
  :members:

.. autoclass:: StringMetadata
  :members:

Sharding Metadata
------------------------

.. autoclass:: ShardingMetadata
  :members:

.. autoclass:: NamedShardingMetadata
  :members:

.. autoclass:: SingleDeviceShardingMetadata
  :members:

.. autoclass:: GSPMDShardingMetadata
  :members:

.. autoclass:: PositionalShardingMetadata
  :members:

.. autoclass:: ShardingTypes
  :members:

.. autofunction:: from_jax_sharding
.. autofunction:: from_serialized_string
.. autofunction:: get_sharding_or_none


Internal Metadata
------------------------

.. autoclass:: TreeMetadata
  :members:
