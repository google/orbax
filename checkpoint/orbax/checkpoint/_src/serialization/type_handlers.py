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

"""Provides utils for PytreeCheckpointHandler."""

from __future__ import annotations

import asyncio
import copy
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, TypeAlias, Union

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types
import tensorstore as ts

ParamInfo: TypeAlias = types.ParamInfo
SaveArgs: TypeAlias = types.SaveArgs
RestoreArgs: TypeAlias = types.RestoreArgs
TypeHandler: TypeAlias = types.TypeHandler
TypeHandlerRegistry: TypeAlias = types.TypeHandlerRegistry
PLACEHOLDER = ...
PLACEHOLDER_TYPESTR = 'placeholder'
JAX_ARRAY_TYPE_STR = jax_array_handlers.JAX_ARRAY_TYPE_STR

Scalar = Union[int, float, np.number]
ScalarMetadata = value_metadata.ScalarMetadata
StringMetadata = value_metadata.StringMetadata
is_ocdbt_checkpoint = format_utils.is_ocdbt_checkpoint
check_array_values = jax_array_handlers.check_array_values
represents_jax_array = jax_array_handlers.represents_jax_array
any_jax_array_param_info = jax_array_handlers.any_jax_array_param_info
ArrayRestoreArgs = jax_array_handlers.ArrayRestoreArgs
ArrayHandler = jax_array_handlers.ArrayHandler
SingleReplicaArrayRestoreArgs = jax_array_handlers.SingleReplicaArrayRestoreArgs
SingleReplicaArrayHandler = jax_array_handlers.SingleReplicaArrayHandler
InvalidShardingError = jax_array_handlers.InvalidShardingError


class NumpyHandler(types.TypeHandler):
  """Provides an implementation of TypeHandler for replicated numpy arrays."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
      ocdbt_process_id: str | None = None,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
      ocdbt_process_id: name of the process id to be used by single controller
        systems to write in OCDBT format. The checkpoints are written in a
        subdir with this name to avoid collisions with the subdir names used by
        other host processes managed by this controller.
    """
    self._metadata_key = metadata_key
    self._override_ocdbt_process_id = ocdbt_process_id

  def typestr(self) -> str:
    return 'np.ndarray'

  async def metadata(
      self, infos: Sequence[types.ParamInfo]
  ) -> Sequence[value_metadata.ArrayMetadata]:
    open_ops = []
    for info in infos:
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      array_read_spec = ts_utils.build_array_read_spec(
          info,
          use_ocdbt=use_ocdbt,
          metadata_key=self._metadata_key,
          raise_array_data_missing_error=info.raise_array_data_missing_error,
      )
      tspec = array_read_spec.json
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=info.ts_context)
      )

    tensorstores = await asyncio.gather(*open_ops)
    return [
        ts_utils.array_metadata_from_tensorstore(t, info, sharding=None)
        for t, info in zip(tensorstores, infos)
    ]

  async def _open_and_write(
      self, value: np.ndarray, tspec: Dict[str, Any], ts_context: ts.Context
  ):
    """Opens and writes using Tensorstore."""
    t = await ts.open(
        ts.Spec(tspec), create=True, open=True, context=ts_context
    )
    await t.write(value, can_reference_source_data_indefinitely=True)  # pytype: disable=attribute-error

  async def _background_serialize(
      self,
      values: Sequence[np.ndarray],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ):
    """Serializes numpy arrays in a background thread."""
    write_coros = []
    for value, info, arg in zip(values, infos, args):
      array_write_spec = ts_utils.build_array_write_spec(
          info=info,
          arg=arg,
          global_shape=value.shape,
          local_shape=value.shape,
          dtype=value.dtype,
          use_ocdbt=info.is_ocdbt_checkpoint,
          process_index=ocdbt_utils.get_process_index_for_subdir(
              use_ocdbt=info.is_ocdbt_checkpoint,
              override_ocdbt_process_id=self._override_ocdbt_process_id,
          ),
          metadata_key=self._metadata_key,
      )
      tspec = array_write_spec.json
      if logging.vlog_is_on(1):
        logging.vlog(1, 'tspec = %s', tspec)
        logging.vlog(1, 'infos = %s', info)
        logging.vlog(1, 'args = %s', arg)
      if multihost.process_index() == 0:
        ts_context = info.ts_context
        write_coros.append(self._open_and_write(value, tspec, ts_context))
    await asyncio.gather(*write_coros)

  async def serialize(
      self,
      values: Sequence[np.ndarray],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ) -> Sequence[future.Future]:
    """Uses Tensorstore to serialize a numpy array."""
    args = args or [types.SaveArgs()] * len(values)
    types.check_input_arguments(values, infos, args)
    check_array_values(values, infos)
    if logging.vlog_is_on(1):
      ts_utils.print_ts_debug_data(self._metadata_key, infos)
    copied_values = [copy.deepcopy(v) for v in values]
    return [
        future.CommitFutureAwaitingContractedSignals(
            self._background_serialize(copied_values, infos, args),
            name='np_type_handler',
        )
    ]

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[np.ndarray]:
    """Deserializes the array using Tensorstore."""
    args = args or [RestoreArgs()] * len(infos)
    types.check_input_arguments(infos, args)
    open_futures = []
    for info, arg in zip(infos, args):
      if not info.is_ocdbt_checkpoint:
        await ts_utils.assert_parameter_files_exist(
            info.parent_dir / info.name, self._metadata_key, info.use_zarr3
        )
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      array_read_spec = ts_utils.build_array_read_spec(
          info,
          use_ocdbt=use_ocdbt,
          metadata_key=self._metadata_key,
          raise_array_data_missing_error=info.raise_array_data_missing_error,
      )
      tspec = array_read_spec.json
      tspec = ts_utils.get_cast_tspec_deserialize(tspec, arg)

      if logging.vlog_is_on(1):
        logging.vlog(1, 'tspec = %s', tspec)
        logging.vlog(1, 'infos = %s', infos)
        logging.vlog(1, 'args = %s', args)
      open_futures += [
          ts.open(ts.Spec(tspec), open=True, context=info.ts_context)
      ]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [t.read() for t in tensorstores]
    ret = await asyncio.gather(*read_ops)

    if logging.vlog_is_on(1):
      for a in ret:
        logging.vlog(
            1, 'restored ndarray.shape = %s, array.dtype = %s', a.shape, a.dtype
        )
        ts_utils.print_ts_debug_data(self._metadata_key, infos)

    return ret

  def memory_size(
      self, values: Sequence[np.ndarray]
  ) -> Sequence[Tuple[int, int]]:
    actual_sizes = [v.size * v.dtype.itemsize for v in values]
    if multihost.process_index() == 0:
      write_sizes = actual_sizes
    else:
      write_sizes = [0 for _ in values]
    read_sizes = actual_sizes
    return list(zip(write_sizes, read_sizes))


class ScalarHandler(NumpyHandler):
  """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.)."""

  def typestr(self) -> str:
    return 'scalar'

  async def metadata(
      self, infos: Sequence[types.ParamInfo]
  ) -> Sequence[ScalarMetadata]:
    metadatas = await super().metadata(infos)
    return [
        ScalarMetadata(name=m.name, directory=m.directory, dtype=m.dtype)
        for m in metadatas
    ]

  async def serialize(
      self,
      values: Sequence[Scalar],  # pytype: disable=signature-mismatch
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ) -> Sequence[future.Future]:
    """See superclass documentation."""
    values = [np.asarray(v) for v in values]
    return await super().serialize(values, infos, args)

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[Scalar]:  # pytype: disable=signature-mismatch
    """See superclass documentation."""
    results = await super().deserialize(infos, args)
    for r in results:
      if r.ndim != 0:
        raise ValueError('Restored result is not a scalar.')
    results = [r.item() for r in results]
    if args:
      # Cast to the intended `restore_type` if it is provided.
      return [
          a.restore_type(r) if a.restore_type else r
          for a, r in zip(args, results)
      ]
    return results

  def memory_size(self, values: Sequence[Scalar]) -> Sequence[Tuple[int, int]]:  # pytype: disable=signature-mismatch
    actual_sizes = [sys.getsizeof(v) for v in values]
    if multihost.process_index() == 0:
      write_sizes = actual_sizes
    else:
      write_sizes = [0 for _ in values]
    read_sizes = actual_sizes
    return list(zip(write_sizes, read_sizes))


class StringHandler(types.TypeHandler):
  """TypeHandler for strings."""

  def __init__(
      self,
      filename: Optional[str] = None,
  ):
    self._filename = filename or '_strings.json'
    self._ts_context = ts_utils.get_ts_context(use_ocdbt=False)

  def _get_json_tspec(
      self,
      info: types.ParamInfo,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.parent_dir is None:
      raise ValueError('Must provide info.parent_dir.')
    directory = (info.parent_dir / self._filename).as_posix()
    kvstore_tspec = ts_utils.build_kvstore_tspec(directory, use_ocdbt=False)
    tspec = {
        'driver': 'json',
        'kvstore': kvstore_tspec,
        'json_pointer': '/' + info.name,
    }
    return tspec

  def typestr(self) -> str:
    return 'string'

  async def metadata(
      self, infos: Sequence[types.ParamInfo]
  ) -> Sequence[StringMetadata]:
    return [
        StringMetadata(name=info.name, directory=info.parent_dir)
        for info in infos
    ]

  async def _convert_to_string(self, tensorstore):
    result = await tensorstore.read()
    return str(result)

  async def _background_serialize(
      self,
      values: Sequence[str],
      infos: Sequence[types.ParamInfo],
  ):
    """Writes strings using Tensorstore in the background thread."""
    types.check_input_arguments(values, infos)
    write_coros = []
    txn = ts.Transaction()
    for (
        info,
        value,
    ) in zip(infos, values):
      tspec = self._get_json_tspec(info)
      if multihost.process_index() == 0:
        t = await ts.open(
            tspec,
            open=True,
            context=self._ts_context,
        )
        write_coros.append(t.with_transaction(txn).write(value))  # pytype: disable=attribute-error
    await asyncio.gather(*write_coros)
    await txn.commit_async()

  async def serialize(
      self,
      values: Sequence[str],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ) -> Sequence[future.Future]:
    """See superclass documentation."""
    del args
    # Copy is not needed since strings are passed by value.
    return [
        future.CommitFutureAwaitingContractedSignals(
            self._background_serialize(values, infos),
            name='string_type_handler',
        )
    ]

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[Optional[str]]:
    """See superclass documentation."""
    del args
    types.check_input_arguments(infos)
    open_futures = []

    for info in infos:
      tspec = self._get_json_tspec(info)
      open_future = ts.open(
          tspec, open=True, read=True, context=self._ts_context
      )
      open_futures += [open_future]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [self._convert_to_string(t) for t in tensorstores]
    return await asyncio.gather(*read_ops)

  def memory_size(self, values: Sequence[str]) -> Sequence[Tuple[int, int]]:
    actual_sizes = [len(v.encode('utf-8')) for v in values]
    if multihost.process_index() == 0:
      write_sizes = actual_sizes
    else:
      write_sizes = [0 for _ in values]
    read_sizes = actual_sizes
    return list(zip(write_sizes, read_sizes))


def is_placeholder(value: Any) -> bool:
  return value is PLACEHOLDER


class PlaceholderHandler(types.TypeHandler):
  """TypeHandler for placeholders."""

  def typestr(self) -> str:
    return PLACEHOLDER_TYPESTR

  async def metadata(self, infos: Sequence[types.ParamInfo]) -> Sequence[Any]:
    raise NotImplementedError('Placeholders do not have metadata.')

  async def serialize(
      self,
      values: Sequence[str],
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.SaveArgs] | None = None,
  ) -> Sequence[future.Future]:
    """See superclass documentation."""
    raise NotImplementedError('Placeholders cannot be serialized.')

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Sequence[RestoreArgs] | None = None,
  ) -> Sequence[Any]:
    """See superclass documentation."""
    del args
    return [PLACEHOLDER] * len(infos)


# TODO(b/253238305) Deprecate when all checkpoints have saved types.
def default_restore_type(args: RestoreArgs) -> Any:
  if isinstance(args, ArrayRestoreArgs):
    return jax.Array
  elif isinstance(args, RestoreArgs):
    return np.ndarray
  else:
    raise ValueError(f'Unsupported restore_args type: {type(args)}')
