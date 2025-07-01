# Copyright 2025 The Orbax Authors.
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

"""ArrayLeafHandler that implements the types.LeafHandler Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for jax.Arrays.
"""

import asyncio
import dataclasses
from typing import Awaitable, Protocol, Sequence, cast

from absl import logging
import jax
import jax.experimental.layout as jax_layout
import jax.numpy as jnp
from orbax.checkpoint._src.arrays import types as arrays_types_v0
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import type_handlers as type_handlers_v0
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import types


ArraySerializationParam = types.SerializationParam[jax.Array]
ArrayDeserializationParam = types.DeserializationParam["AbstractArray"]
Shape = arrays_types_v0.Shape

if jax.__version_info__ >= (0, 6, 2):
  Format = jax_layout.Format
else:
  Format = jax_layout.Layout


class AbstractArray(Protocol):
  """Abstract representation of an array.

  This is a protocol for an abstract array that can be used to represent various
  metadata types such as jax.ShapeDtypeStruct and ArrayMetadata.

  #TODO(dnlng): All attributes are made optional to support the case where
  # the ArrayMetadata is passed into the metadata() call to pass only the
  # `write_shape`.  Optional attributes are not needed once write_shape is
  # refactored.


  shape:
    Tuple of integers describing the array shape.
  dtype:
    Dtype of array elements.
  Sharding:
    Sharding to indicate how the array is sharded. This can be jax's Sharding or
    Layout or None.
  """

  shape: Shape | None
  dtype: jax.numpy.dtype | None
  sharding: jax.sharding.Sharding | Format | None  # pytype: disable=unsupported-operands


@dataclasses.dataclass
class ArrayMetadata:
  """Array Metadata for the ArrayLeafHandler.

  shape:
    Tuple of integers describing the array shape.
  sharding_metadata:
    ShardingMetadata to indicate how the array is sharded. ShardingMetadata is
    an orbax representation of `jax.sharding.Sharding` which stores the same
    properties but not require accessing real devices.
  dtype:
    Dtype of array elements.
  storage:
    Optional metadata describing how the array is stored in a checkpoint.
  """

  shape: Shape | None
  dtype: jax.numpy.dtype | None
  sharding_metadata: sharding_metadata.ShardingMetadata | None
  storage_metadata: value_metadata.StorageMetadata | None

  @property
  def sharding(self) -> jax.sharding.Sharding | None:
    """Returns the jax.sharding.Sharding from the sharding_metadata if possible.

    Exception will be thrown if the hardware topology has changed and the
    sharding cannot be restored from stored metadata.
    """
    if self.sharding_metadata is None:
      return None
    return self.sharding_metadata.to_jax_sharding()


def _create_v0_array_handler(
    context: context_lib.Context,
) -> type_handlers_v0.ArrayHandler:
  """Creates a V0 array handler from a V1 context."""

  saving_options = context.array_options.saving
  primary_host = context.multiprocessing_options.primary_host
  array_handler = type_handlers_v0.ArrayHandler(
      primary_host=primary_host,
      replica_id=None if primary_host is None else 0,
      use_replica_parallel=saving_options.use_replica_parallel,
      enable_write_sharding_file=saving_options.enable_write_sharding_file,
      array_metadata_store=saving_options.array_metadata_store,
  )


  return array_handler


def _create_v0_saving_paraminfo(
    param: ArraySerializationParam,
    context: context_lib.Context,
    serialization_context: types.SerializationContext,
) -> type_handlers_v0.ParamInfo:
  """Creates a V0 ParamInfo from V1 params andn contexts for saving."""

  saving_options = context.array_options.saving

  return type_handlers_v0.ParamInfo(
      name=param.name,
      path=serialization_context.parent_dir.path / param.name,
      parent_dir=serialization_context.parent_dir.path,
      byte_limiter=serialization_context.byte_limiter,
      is_ocdbt_checkpoint=saving_options.use_ocdbt,
      use_zarr3=saving_options.use_zarr3,
      ocdbt_target_data_file_size=saving_options.ocdbt_target_data_file_size,
      ts_context=serialization_context.ts_context,
      value_typestr=None,  # TODO(dnlng): Add value typestr.
      enable_pinned_host_transfer=saving_options.enable_pinned_host_transfer,
  )


def _create_v0_savearg(
    param: ArraySerializationParam,
    context: context_lib.Context,
) -> type_handlers_v0.SaveArgs:
  """Creates a V0 SaveArgs from V1 params and context for saving."""

  fn = context.pytree_options.saving.create_array_storage_options_fn

  if fn:
    storage_options = fn(param.keypath, param.value)
    savearg = type_handlers_v0.SaveArgs(
        dtype=jnp.dtype(storage_options.dtype)
        if storage_options.dtype
        else None,
        chunk_byte_size=storage_options.chunk_byte_size,
        shard_axes=storage_options.shard_axes,
    )
  else:
    savearg = type_handlers_v0.SaveArgs()

  return savearg


def _create_v0_restore_paraminfo(
    param: (
        types.DeserializationParam[None]
        | types.DeserializationParam[AbstractArray]
    ),
    context: context_lib.Context,
    deserialization_context: types.DeserializationContext,
) -> type_handlers_v0.ParamInfo:
  """Creates a V0 ParamInfo from V1 params and contexts for loading."""

  loading_options = context.array_options.Loading

  if isinstance(param.value, ArrayMetadata):
    # the write_shape is populated for metadata() calls.
    v = cast(ArrayMetadata, param.value)
    write_shape = v.storage_metadata.write_shape
  else:
    write_shape = None

  return type_handlers_v0.ParamInfo(
      name=param.name,
      path=deserialization_context.parent_dir / param.name,
      parent_dir=deserialization_context.parent_dir,
      skip_deserialize=False,
      byte_limiter=deserialization_context.byte_limiter,
      is_ocdbt_checkpoint=deserialization_context.ocdbt_checkpoint,
      ts_context=deserialization_context.ts_context,
      raise_array_data_missing_error=loading_options.raise_array_data_missing_error,
      use_zarr3=deserialization_context.zarr3_checkpoint,
      write_shape=write_shape,
  )


def _create_v0_restorearg(
    param: ArrayDeserializationParam,
    context: context_lib.Context,
) -> type_handlers_v0.ArrayRestoreArgs:
  """Creates a V0 ArrayRestoreArgs from V1 params."""

  if param.value is None:
    return type_handlers_v0.ArrayRestoreArgs(restore_type=jax.Array)
  else:
    v = param.value
    if not isinstance(v, (jax.Array, jax.ShapeDtypeStruct, ArrayMetadata)):
      raise ValueError(
          "ArrayDeserializationParam.value is an unsupported type:"
          f" {type(v)} for param.name: {param.name}"
      )
    return type_handlers_v0.ArrayRestoreArgs(
        restore_type=jax.Array,
        dtype=v.dtype,
        sharding=v.sharding,
        shape=v.shape,
        strict=not context.array_options.loading.enable_padding_and_truncation,
    )


async def _async_futures(commit_futures: Sequence[future.Future]):
  await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])


class ArrayLeafHandler(types.LeafHandler[jax.Array, AbstractArray]):
  """ArrayLeafHandler that implements the types.LeafHandler Protocol."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
  ):
    self._context = context_lib.get_context(context)
    self._handler_impl = _create_v0_array_handler(
        self._context,
    )

    logging.vlog(1, "ArrayLeafHandler created.")

  async def serialize(
      self,
      params: Sequence[ArraySerializationParam],
      serialization_context: types.SerializationContext,
  ) -> Awaitable[None]:
    """Serializes jax.Arrays as a checkpointable to a storage location.

    Args:
      params: a sequence of ArraySerializationParam per leaf.
      serialization_context: SerializationContext for the array leaf handler.

    Returns:
      Sequence of commit futures which can be awaited to complete the save
      operation.
    """
    values = [p.value for p in params]
    paraminfos = [
        _create_v0_saving_paraminfo(p, self._context, serialization_context)
        for p in params
    ]
    saveargs = [_create_v0_savearg(p, self._context) for p in params]

    commit_futures = await self._handler_impl.serialize(
        values, paraminfos, saveargs
    )
    assert commit_futures

    return _async_futures(commit_futures)

  async def deserialize(
      self,
      params: Sequence[ArrayDeserializationParam],
      deserialization_context: types.DeserializationContext,
  ) -> Awaitable[Sequence[jax.Array]]:
    """Returns sequence of jax.Arrays from a stored checkpointable location.

    Args:
      params: sequence of ArrayDeserializationParam per leaf.
      deserialization_context: ArrayDeserializationContext for the leaf handler.

    Returns:
      The deserialized sequence of jax.Arrays as leaves.
    """

    # validate all parameters
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]
    restoreargs = [_create_v0_restorearg(p, self._context) for p in params]

    return asyncio.create_task(
        self._handler_impl.deserialize(paraminfos, restoreargs)
    )

  async def metadata(
      self,
      params: Sequence[types.DeserializationParam[None | AbstractArray]],
      deserialization_context: types.DeserializationContext,
  ) -> Sequence[AbstractArray]:
    """Returns a squence of ArrayMetadata from a stored checkpointable location.

    Args:
      params: sequence of ArrayDeserializationParam per jax.Array leaf.
      deserialization_context: DeserializationContext for the array leaf
        handler.

    Returns:
      Sequence of ArrayMetadata for each provided ArrayDeserializationParam.
    """
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]

    async def _convert_to_array_metadata() -> Sequence[ArrayMetadata]:
      v0_metadatas = await self._handler_impl.metadata(paraminfos)

      ret = []
      for meta in v0_metadatas:
        array_metadata = ArrayMetadata(
            shape=meta.shape,
            dtype=meta.dtype,
            sharding_metadata=meta.sharding,
            storage_metadata=meta.storage,
        )
        ret.append(array_metadata)

        logging.vlog(1, "array_metadata: %r", array_metadata)

      return ret

    return await _convert_to_array_metadata()
