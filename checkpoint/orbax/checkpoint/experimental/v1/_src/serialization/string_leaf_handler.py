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

""":py:class:`.StringLeafHandler` that implements the :py:class:`~.v1.serialization.LeafHandler` Protocol.

The primary purpose of this handler is to provide serialization and
deserialization for strings.
"""

import asyncio
from typing import Any, Awaitable, Sequence

from absl import logging
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
import tensorstore as ts


AbstractString = types.AbstractString
StringSerializationParam = types.SerializationParam[str]
StringDeserializationParam = types.DeserializationParam[
    AbstractString
]


class StringLeafHandler(types.LeafHandler[str, AbstractString]):
  """:py:class:`.StringLeafHandler` that implements the :py:class:`~.v1.serialization.LeafHandler` Protocol."""

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
    self._filename = '_strings.json'
    logging.vlog(1, 'StringLeafHandler created.')

  def _get_json_tspec(
      self,
      param_name: str,
      parent_dir: path_types.Path,
  ) -> dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    directory = (parent_dir / self._filename).as_posix()
    kvstore_tspec = ts_utils.build_kvstore_tspec(directory, use_ocdbt=False)
    tspec = {
        'driver': 'json',
        'kvstore': kvstore_tspec,
        'json_pointer': '/' + param_name,
    }
    return tspec

  async def _background_serialize(
      self,
      params: Sequence[StringSerializationParam],
      serialization_context: types.SerializationContext,
  ):
    """Writes strings using Tensorstore in the background thread."""
    parent_dir = await serialization_context.parent_dir.await_creation()
    write_coros = []
    txn = ts.Transaction()
    for param in params:
      tspec = self._get_json_tspec(param.name, parent_dir)
      if multihost.is_primary_host(
          self._context.multiprocessing_options.primary_host
      ):
        t = await ts.open(
            tspec,
            open=True,
            context=serialization_context.ts_context,
        )
        write_coros.append(t.with_transaction(txn).write(param.value))  # pytype: disable=attribute-error
    await asyncio.gather(*write_coros)
    await txn.commit_async()

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
    return self._background_serialize(params, serialization_context)

  async def _background_deserialize(
      self,
      params: Sequence[StringDeserializationParam],
      deserialization_context: types.DeserializationContext,
  ) -> Sequence[str]:
    """Deserializes strings using Tensorstore in the background thread."""

    async def _convert_to_string(tensorstore):
      result = await tensorstore.read()
      return str(result)

    open_futures = []
    for param in params:
      tspec = self._get_json_tspec(
          param.name, deserialization_context.parent_dir
      )
      open_future = ts.open(
          tspec,
          open=True,
          read=True,
          context=deserialization_context.ts_context,
      )
      open_futures += [open_future]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [_convert_to_string(t) for t in tensorstores]
    return await asyncio.gather(*read_ops)

  async def deserialize(
      self,
      params: Sequence[StringDeserializationParam],
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
    return self._background_deserialize(params, deserialization_context)

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
    return ['string'] * len(params)

