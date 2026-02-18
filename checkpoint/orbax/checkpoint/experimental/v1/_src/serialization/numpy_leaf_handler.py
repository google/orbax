# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""":py:class:`.NumpyLeafHandler` that implements the :py:class:`~.v1.serialization.LeafHandler` Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for NumPy arrays.
"""

import asyncio
import copy
import dataclasses
import typing
from typing import Any, Awaitable, Sequence

from absl import logging
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
import tensorstore as ts


NumpySerializationParam = types.SerializationParam[np.ndarray]
NumpyDeserializationParam = types.DeserializationParam[
    types.AbstractArray
]
Shape = arrays_types.Shape
AbstractArray = types.AbstractArray


@dataclasses.dataclass
class NumpyShapeDtype(AbstractArray):
  """To implement the :py:class:`.AbstractArray` protocol."""

  shape: Shape | None
  dtype: np.dtype | None


@dataclasses.dataclass
class NumpyMetadata(AbstractArray):
  """NumPy Metadata for the :py:class:`.NumpyLeafHandler`.

  shape:
    A tuple of integers describing the array shape.
  dtype:
    The `dtype` of array elements.
  storage:
    Optional metadata describing how the array is stored in a checkpoint.
  """

  shape: Shape | None
  dtype: np.dtype | None
  storage_metadata: value_metadata.StorageMetadata | None


async def _open_and_write(
    value: np.ndarray, tspec: dict[str, Any], *, ts_context: ts.Context
):
  """Opens and writes using Tensorstore."""
  t = await ts.open(ts.Spec(tspec), create=True, open=True, context=ts_context)
  await t.write(value, can_reference_source_data_indefinitely=True)  # pytype: disable=attribute-error


async def _maybe_check_param_dir_existence(
    params: Sequence[NumpyDeserializationParam],
    deserialization_context: types.DeserializationContext,
):
  """Checks if the parameter directory exists."""
  if deserialization_context.ocdbt_checkpoint:
    return
  ops = [
      ts_utils.assert_parameter_files_exist(
          deserialization_context.parent_dir / param.name,
          None,
          use_zarr3=deserialization_context.zarr3_checkpoint,
      )
      for param in params
  ]
  await asyncio.gather(*ops)


def _dtype_for_deserialization(
    param: NumpyDeserializationParam,
) -> np.dtype | None:
  """Deserializes the dtype of a numpy array."""
  value = param.value
  if value is None or isinstance(value, type):
    return None
  else:
    value = typing.cast(AbstractArray, value)
    return value.dtype


def _check_array_values(
    params: Sequence[NumpySerializationParam],
):
  """Checks array values for zero size."""
  for param in params:
    if param.value.shape is None or np.prod(param.value.shape) == 0:
      raise ValueError(f'Cannot save an array with zero size: {param.name}')


class NumpyLeafHandler(types.LeafHandler[np.ndarray, AbstractArray]):
  """:py:class:`.NumpyLeafHandler` that implements the :py:class:`~.v1.serialization.LeafHandler` Protocol."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
  ):
    self._context = context_lib.get_context(context)
    self._override_ocdbt_process_id = None
    if multihost.is_pathways_backend():
      self._override_ocdbt_process_id = 'pwcontroller'
    logging.vlog(1, 'NumpyLeafHandler created.')

  async def _background_serialize(
      self,
      params: Sequence[NumpySerializationParam],
      serialization_context: types.SerializationContext,
  ):
    """Serializes numpy arrays in a background thread."""
    _check_array_values(params)
    parent_dir = await serialization_context.parent_dir.await_creation()
    write_coros = []
    for param in params:
      storage_options = self._context.array_options.saving.storage_options
      # Individualized settings in PyTreeOptions take precedence.
      if self._context.pytree_options.saving.create_array_storage_options_fn:
        storage_options = (
            self._context.pytree_options.saving.create_array_storage_options_fn(
                param.keypath, param.value
            )
        )
      array_write_spec = ts_utils.ArrayWriteSpec(
          parent_dir.as_posix(),
          relative_array_filename=param.name,
          global_shape=param.value.shape,
          write_shape=param.value.shape,
          dtype=jnp.dtype(param.value.dtype),
          target_dtype=jnp.dtype(storage_options.dtype)
          if storage_options.dtype is not None
          else None,
          chunk_byte_size=storage_options.chunk_byte_size,
          shard_axes=storage_options.shard_axes,
          use_compression=self._context.array_options.saving.use_compression,
          use_zarr3=self._context.array_options.saving.use_zarr3,
          use_ocdbt=self._context.array_options.saving.use_ocdbt,
          process_id=ocdbt_utils.get_process_index_for_subdir(
              use_ocdbt=self._context.array_options.saving.use_ocdbt,
              override_ocdbt_process_id=self._override_ocdbt_process_id,
          ),
          replica_separate_folder=self._context.array_options.saving.enable_replica_parallel_separate_folder,
          ocdbt_target_data_file_size=self._context.array_options.saving.ocdbt_target_data_file_size,
      )
      tspec = array_write_spec.json
      if multihost.is_primary_host(
          self._context.multiprocessing_options.primary_host
      ):
        write_coros.append(
            _open_and_write(
                param.value, tspec, ts_context=serialization_context.ts_context
            )
        )
    await asyncio.gather(*write_coros)

  async def serialize(
      self,
      params: Sequence[NumpySerializationParam],
      serialization_context: types.SerializationContext,
  ) -> Awaitable[None]:
    """Serializes `np.ndarrays` as a checkpointable to a storage location.

    Args:
      params: a sequence of NumpySerializationParam per leaf.
      serialization_context: SerializationContext for the NumpyLeafHandler.

    Returns:
      Sequence of commit futures which can be awaited to complete the save
      operation.
    """
    copied_params = [
        dataclasses.replace(p, value=copy.deepcopy(p.value)) for p in params
    ]
    return self._background_serialize(copied_params, serialization_context)

  async def _background_deserialize(
      self,
      params: Sequence[NumpyDeserializationParam],
      deserialization_context: types.DeserializationContext,
  ):
    await _maybe_check_param_dir_existence(params, deserialization_context)
    open_futures = []
    for param in params:
      array_read_spec = ts_utils.ArrayReadSpec(
          directory=deserialization_context.parent_dir.as_posix(),
          relative_array_filename=param.name,
          use_zarr3=deserialization_context.zarr3_checkpoint,
          use_ocdbt=deserialization_context.ocdbt_checkpoint,
          raise_array_data_missing_error=self._context.array_options.loading.raise_array_data_missing_error,
          target_dtype=_dtype_for_deserialization(param),
      )
      open_futures += [
          ts.open(
              ts.Spec(array_read_spec.json),
              open=True,
              context=deserialization_context.ts_context,
          )
      ]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [t.read() for t in tensorstores]
    return await asyncio.gather(*read_ops)

  async def deserialize(
      self,
      params: Sequence[NumpyDeserializationParam],
      deserialization_context: types.DeserializationContext,
  ) -> Awaitable[Sequence[np.ndarray]]:
    """Returns a sequence of `np.ndarrays` from a stored checkpointable location.

    Args:
      params: sequence of NumpyDeserializationParam per leaf.
      deserialization_context: NumpyDeserializationContext for the leaf handler.

    Returns:
      The deserialized sequence of nd.ndarays as leaves.
    """
    # validate all parameters
    return self._background_deserialize(params, deserialization_context)

  async def metadata(
      self,
      params: Sequence[types.DeserializationParam[None]],
      deserialization_context: types.DeserializationContext,
  ) -> Sequence[AbstractArray]:
    """Returns a squence of NumpyMetadata from a stored checkpointable location.

    Args:
      params: sequence of NumpyDeserializationParam per Numpy array leaf.
      deserialization_context: DeserializationContext for the array leaf
        handler.

    Returns:
      Sequence of NumpyMetadata for each provided DeserializationParam.
    """
    open_ops = []
    for param in params:
      array_read_spec = ts_utils.ArrayReadSpec(
          directory=deserialization_context.parent_dir.as_posix(),
          relative_array_filename=param.name,
          use_zarr3=deserialization_context.zarr3_checkpoint,
          use_ocdbt=deserialization_context.ocdbt_checkpoint,
          raise_array_data_missing_error=self._context.array_options.loading.raise_array_data_missing_error,
      )
      open_ops.append(
          ts.open(
              ts.Spec(array_read_spec.json),
              open=True,
              context=deserialization_context.ts_context,
          )
      )

    tensorstores = await asyncio.gather(*open_ops)
    return [
        NumpyMetadata(
            shape=t.shape,
            dtype=jnp.dtype(t.dtype.name),
            storage_metadata=value_metadata.StorageMetadata(
                chunk_shape=t.chunk_layout.read_chunk_template.shape,
                # TODO(b/407609827): In jax.Array this comes from array metadata
                # store, and is kind of extraneous for numpy arrays since the
                # write shape is the same as the array shape. However, we should
                # ideally just set it for completeness. In V0, it is already
                # not being set. Several unit tests must be updated.
                write_shape=None,
            ),
        )
        for t in tensorstores
    ]
