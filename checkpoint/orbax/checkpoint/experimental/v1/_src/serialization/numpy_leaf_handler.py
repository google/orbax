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

"""NumpyLeafHandler that implements the types.LeafHandler Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for numpy arrays
"""

import asyncio
import dataclasses
from typing import Awaitable, Protocol, Sequence

from absl import logging
import numpy as np
from orbax.checkpoint._src.arrays import types as arrays_types_v0
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import type_handlers as type_handlers_v0
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import types


NumpySerializationParam = types.SerializationParam[np.ndarray]
NumpyDeserializationParam = types.DeserializationParam["AbstractNumpy"]
Shape = arrays_types_v0.Shape


class AbstractNumpy(Protocol):
  """Abstract representation of a numpy array.

  This is a protocol for an abstract numpy array that can be used to represent
  the metadata belonging to a numpy array.

  shape:
    Tuple of integers describing the array shape.
  dtype:
    Dtype of array elements.
  """

  shape: Shape | None
  dtype: np.dtype


@dataclasses.dataclass
class NumpyShapeDtype:
  """To implement the AbstractNumpy protocol."""

  shape: Shape | None
  dtype: np.dtype


@dataclasses.dataclass
class NumpyMetadata:
  """Numpy Metadata for the NumpyLeafHandler.

  shape:
    Tuple of integers describing the array shape.
  dtype:
    Dtype of array elements.
  storage:
    Optional metadata describing how the array is stored in a checkpoint.
  """

  shape: Shape | None
  dtype: np.dtype
  storage_metadata: value_metadata.StorageMetadata | None


def _create_v0_numpy_handler() -> type_handlers_v0.NumpyHandler:
  """Creates a V0 NumpyHandler."""
  numpy_handler = type_handlers_v0.NumpyHandler()
  return numpy_handler


def _create_v0_saving_paraminfo(
    param: NumpySerializationParam,
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
      value_typestr="np.ndarray",
  )


def _create_v0_savearg(
    param: NumpySerializationParam,
    context: context_lib.Context,
) -> type_handlers_v0.SaveArgs:
  """Creates a V0 SaveArgs from V1 params and context for saving."""

  fn = context.pytree_options.saving.create_array_storage_options_fn

  if fn:
    storage_options = fn(param.keypath, param.value)
    savearg = type_handlers_v0.SaveArgs(
        dtype=np.dtype(storage_options.dtype)
        if storage_options.dtype
        else None,
        chunk_byte_size=storage_options.chunk_byte_size,
        shard_axes=storage_options.shard_axes,
    )
  else:
    savearg = type_handlers_v0.SaveArgs()

  return savearg


def _create_v0_restore_paraminfo(
    param: types.DeserializationParam[None | AbstractNumpy],
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
    param: NumpyDeserializationParam,
) -> type_handlers_v0.RestoreArgs:
  """Creates a V0 RestoreArgs from V1 params."""

  if param.value is None:
    return type_handlers_v0.RestoreArgs(restore_type=np.ndarray)
  else:
    v = param.value
    if not isinstance(
        v,
        (
            np.ndarray,
            NumpyShapeDtype,
            NumpyMetadata,
        ),
    ):
      raise ValueError(
          f"NumpyDeserializationParam.value is an unsupported type: {type(v)}"
      )

    logging.vlog(1, "name: %s, v.dtype: %s", param.name, v.dtype)
    return type_handlers_v0.RestoreArgs(
        restore_type=np.ndarray,
        dtype=v.dtype,
    )


async def _async_futures(commit_futures: Sequence[future.Future]):
  await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])


class NumpyLeafHandler(types.LeafHandler[np.ndarray, AbstractNumpy]):
  """NumpyLeafHandler that implements the types.LeafHandler Protocol."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
  ):
    self._context = context_lib.get_context(context)
    self._handler_impl = _create_v0_numpy_handler()

    logging.vlog(1, "NumpyLeafHandler created.")

  async def serialize(
      self,
      params: Sequence[NumpySerializationParam],
      serialization_context: types.SerializationContext,
  ) -> Awaitable[None]:
    """Serializes np.ndarrays as a checkpointable to a storage location.

    Args:
      params: a sequence of NumpySerializationParam per leaf.
      serialization_context: SerializationContext for the NumpyLeafHandler.

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
      params: Sequence[types.DeserializationParam[AbstractNumpy]],
      deserialization_context: types.DeserializationContext,
  ) -> Awaitable[Sequence[np.ndarray]]:
    """Returns sequence of np.ndarrays from a stored checkpointable location.

    Args:
      params: sequence of NumpyDeserializationParam per leaf.
      deserialization_context: NumpyDeserializationContext for the leaf handler.

    Returns:
      The deserialized sequence of nd.ndarays as leaves.
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
  ) -> Sequence[AbstractNumpy]:
    """Returns a squence of NumpyMetadata from a stored checkpointable location.

    Args:
      params: sequence of NumpyDeserializationParam per Numpy array leaf.
      deserialization_context: DeserializationContext for the array leaf
        handler.

    Returns:
      Sequence of NumpyMetadata for each provided DeserializationParam.
    """
    paraminfos = [
        _create_v0_restore_paraminfo(p, self._context, deserialization_context)
        for p in params
    ]

    async def _convert_to_numpy_metadata() -> Sequence[NumpyMetadata]:
      v0_metadatas = await self._handler_impl.metadata(paraminfos)

      ret = []
      for meta in v0_metadatas:
        numpy_metadata = NumpyMetadata(
            shape=meta.shape,
            dtype=meta.dtype,
            storage_metadata=meta.storage,
        )
        ret.append(numpy_metadata)

        logging.vlog(1, "numpy_metadata: %r", numpy_metadata)

      return ret

    return await _convert_to_numpy_metadata()
