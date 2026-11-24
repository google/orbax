``ocp.v1.arrays`` module
============================================================================

.. automodule:: orbax.checkpoint.experimental.v1.arrays

.. currentmodule:: orbax.checkpoint.v1.arrays

Types
------------------------------------------------------------
.. autoclass:: AbstractArray
.. autoclass:: AbstractShardedArray
.. autodata:: ArrayLike
   :annotation: = AbstractArray | AbstractShardedArray
.. autodata:: Scalar
   :annotation: = int | float | np.number | bytes | bool
.. autodata:: AbstractScalar
   :annotation: = Scalar
.. autodata:: Shape
   :annotation: = tuple[int, ...]
.. autodata:: DType
   :annotation: = jax.numpy.dtype | np.dtype

Utilities
------------------------------------------------------------
.. autofunction:: to_shape_dtype_struct
