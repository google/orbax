``ocp.v1.tree`` module
============================================================================

.. automodule:: orbax.checkpoint.experimental.v1.tree

.. currentmodule:: orbax.checkpoint.v1.tree

Types
------------------------------------------------------------
.. autoclass:: PyTree
.. autoclass:: PyTreeOf
.. autodata:: Leaf
   :annotation: = jax.Array | np.ndarray | int | float | np.number | bytes | bool | str
.. autodata:: AbstractLeaf
   :annotation: = ocp.v1.arrays.AbstractArray | ocp.v1.arrays.AbstractShardedArray | int | float | np.number | bytes | bool | str
.. autodata:: PyTreeKey
   :annotation: = jax.tree_util.SequenceKey | jax.tree_util.DictKey | jax.tree_util.GetAttrKey | jax.tree_util.FlattenedIndexKey
.. autodata:: PyTreeKeyPath
   :annotation: = tuple[PyTreeKey, ...]
.. autodata:: JsonType
   :annotation: = list[JsonValue] | dict[str, JsonValue]

Structure Utils
------------------------------------------------------------
.. autofunction:: merge
