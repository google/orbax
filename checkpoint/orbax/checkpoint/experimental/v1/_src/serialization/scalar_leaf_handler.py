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

"""ScalarLeafHandler that implements the types.LeafHandler Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for scalar values.
"""

import asyncio
from typing import Awaitable, Sequence, Type

from absl import logging
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.serialization import type_handlers as type_handlers_v0
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import types


Scalar = int | float | np.number
ScalarSerializationParam = types.SerializationParam[Scalar]
ScalarDeserializationParam = types.DeserializationParam["AbstractScalar"]


# Optional type hint for a scalar leaf handler. If provided, the restored scalar
# will be cast to this type.  Only casting to int or float is supported.
AbstractScalar = Type[Scalar] | Scalar


def _create_v0_scalar_handler() -> type_handlers_v0.ScalarHandler:
  """Creates a V0 ScalarHandler."""
  scalar_handler = type_handlers_v0.ScalarHandler()
  return scalar_handler


def _create_v0_saving_paraminfo(
    param: ScalarSerializationParam,
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
      value_typestr="scalar",
  )


def _create_v0_savearg(
    param: ScalarSerializationParam,
    context: context_lib.Context,
) -> type_handlers_v0.SaveArgs:
  """Creates a V0 SaveArgs from V1 params and context for saving."""

  fn = context.pytree_options.saving.create_array_storage_options_fn

  if fn:
    storage_options = fn(param.keypath, param.value)
    savearg = type_handlers_v0.SaveArgs(
        dtype=storage_options.dtype,
        chunk_byte_size=storage_options.chunk_byte_size,
        shard_axes=storage_options.shard_axes,
    )
  else:
    savearg = type_handlers_v0.SaveArgs()

  return savearg


def _create_v0_restore_paraminfo(
    param: types.DeserializationParam[None | AbstractScalar],
    context: context_lib.Context,
    deserialization_context: types.DeserializationContext,
) -> type_handlers_v0.ParamInfo:
  """Creates a V0 ParamInfo from V1 params and contexts for loading."""

  loading_options = context.array_options.Loading

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
  )


def _create_v0_restorearg(
    param: ScalarDeserializationParam,
) -> type_handlers_v0.RestoreArgs:
  """Creates a V0 RestoreArgs from V1 params."""
  if isinstance(param.value, Scalar):
    # users pass values direclty
    restore_type = type(param.value)
  else:
    restore_type = param.value

  logging.vlog(1, "setting restore_type: %r", restore_type)
  return type_handlers_v0.RestoreArgs(
      restore_type=restore_type,
  )


async def _async_futures(commit_futures: Sequence[future.Future]):
  await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])


class ScalarLeafHandler(types.LeafHandler[Scalar, AbstractScalar]):
  """ScalarLeafHandler that implements the types.LeafHandler Protocol."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
  ):
    self._context = context_lib.get_context(context)
    self._handler_impl = _create_v0_scalar_handler()

    logging.vlog(1, "ScalarLeafHandler created.")

  async def serialize(
      self,
      params: Sequence[ScalarSerializationParam],
      serialization_context: types.SerializationContext,
  ) -> Awaitable[None]:
    """Serializes scalar values as a checkpointable to a storage location.

    Args:
      params: a sequence of ScalarSerializationParam per leaf.
      serialization_context: SerializationContext for the scalar leaf handler.

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
      params: Sequence[types.DeserializationParam[AbstractScalar]],
      deserialization_context: types.DeserializationContext,
  ) -> Awaitable[Sequence[Scalar]]:
    """Returns sequence of Scalar values from a stored checkpointable location.

    Args:
      params: sequence of ScalarDeserializationParam per leaf.
      deserialization_context: ScalarDeserializationContext for the leaf
        handler.

    Returns:
      The deserialized sequence of scalar values as leaves.
    """

    # validate all parameters
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]
    restoreargs = [_create_v0_restorearg(p) for p in params]

    return asyncio.create_task(
        self._handler_impl.deserialize(paraminfos, restoreargs)
    )

  async def metadata(
      self,
      params: Sequence[types.DeserializationParam[None]],
      deserialization_context: types.DeserializationContext,
  ) -> Sequence[AbstractScalar]:
    """Returns a squence of AbstractScalar from a stored checkpointable location.

    Args:
      params: sequence of ScalarDeserializationParam per scalar value leaf.
      deserialization_context: DeserializationContext for the scalar leaf
        handler.

    Returns:
      Sequence of ScalarMetadata for each provided DeserializationParam.
    """
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]

    async def _convert_to_scalar_metadata() -> Sequence[AbstractScalar]:
      v0_metadatas = await self._handler_impl.metadata(paraminfos)

      def _get_type(meta: type_handlers_v0.ScalarMetadata):
        if meta.dtype is None:
          # this shouldn't happen even though ScalarMetadata.dtype is Optional,
          # but each scalar should have a dtype.
          raise ValueError("dtype is None")

        if isinstance(meta.dtype, (np.dtype | jnp.dtype)):
          return meta.dtype.type
        else:
          return meta.dtype

      ret = [_get_type(meta) for meta in v0_metadatas]

      logging.vlog(1, "scalar_metadata: %r", ret)

      return ret

    return await _convert_to_scalar_metadata()
