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

"""StringLeafHandler that implements the types.LeafHandler Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for strings.
"""

import asyncio
from typing import Awaitable, Sequence, Type

from absl import logging
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.serialization import type_handlers as type_handlers_v0
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import types


AbstractString = Type[str]
StringSerializationParam = types.SerializationParam[str]
StringDeserializationParam = types.DeserializationParam[AbstractString]


def _create_v0_saving_paraminfo(
    param: StringSerializationParam,
    context: context_lib.Context,
    serialization_context: types.SerializationContext,
) -> type_handlers_v0.ParamInfo:
  """Creates a V0 ParamInfo from V1 params and contexts for saving."""

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
      value_typestr="string",
  )


def _create_v0_restore_paraminfo(
    param: types.DeserializationParam[None | AbstractString],
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


async def _async_futures(commit_futures: Sequence[future.Future]):
  await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])


class StringLeafHandler(types.LeafHandler[str, AbstractString]):
  """StringLeafHandler that implements the types.LeafHandler Protocol."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
  ):
    """Initializes the StringLeafHandler.

    This handler underneath uses the V0 StringHandler.

    Args:
      context: Context that will be used for this leaf handler.
    """
    self._context = context_lib.get_context(context)
    self._handler_impl = type_handlers_v0.StringHandler()

    logging.vlog(1, "StringLeafHandler created.")

  async def serialize(
      self,
      params: Sequence[StringSerializationParam],
      serialization_context: types.SerializationContext,
  ) -> Awaitable[None]:
    """Serializes scalar values as a checkpointable to a storage location.

    Args:
      params: a sequence of StringSerializationParam per leaf.
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

    # `args` is not used by StringHandler.serialize, so it's not passed in.
    commit_futures = await self._handler_impl.serialize(values, paraminfos)
    if not commit_futures:
      raise ValueError("No commit futures returned by StringHandler.serialize.")

    return _async_futures(commit_futures)

  async def deserialize(
      self,
      params: Sequence[types.DeserializationParam[AbstractString]],
      deserialization_context: types.DeserializationContext,
  ) -> Awaitable[Sequence[str]]:
    """Returns sequence of String values from a stored checkpointable location.

    Args:
      params: sequence of StringDeserializationParam per leaf.
      deserialization_context: StringDeserializationContext for the leaf
        handler.

    Returns:
      The deserialized sequence of scalar values as leaves.
    """

    # validate all parameters
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]

    async def _background_deserialize() -> Sequence[str]:
      # This is needed because StringHandler.deserialize could return None
      # values.  However, it should be very rare.  This is to make sure
      # everything is string.
      return [p or "" for p in await self._handler_impl.deserialize(paraminfos)]

    return asyncio.create_task(_background_deserialize())

  async def metadata(
      self,
      params: Sequence[types.DeserializationParam[None]],
      deserialization_context: types.DeserializationContext,
  ) -> Sequence[AbstractString]:
    """Returns a squence of str from a stored checkpointable location.

    Args:
      params: sequence of StringDeserializationParam per scalar value leaf.
      deserialization_context: DeserializationContext for the scalar leaf
        handler.

    Returns:
      Sequence of StringMetadata for each provided DeserializationParam.
    """
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]

    async def _get_metadata() -> Sequence[AbstractString]:
      v0_metadatas = await self._handler_impl.metadata(paraminfos)
      return [str] * len(v0_metadatas)

    return await _get_metadata()
