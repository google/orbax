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
import dataclasses
import functools
import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeAlias, Union, cast
import warnings

from absl import logging
from etils import epath
import jax
from jax.experimental import layout
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types
import tensorstore as ts

ParamInfo: TypeAlias = types.ParamInfo
SaveArgs: TypeAlias = types.SaveArgs
RestoreArgs: TypeAlias = types.RestoreArgs
TypeHandler: TypeAlias = types.TypeHandler
TypeHandlerRegistry: TypeAlias = types.TypeHandlerRegistry
Pytree = Any
PLACEHOLDER = ...
PLACEHOLDER_TYPESTR = 'placeholder'

if jax.__version_info__ >= (0, 6, 2):
  Format = layout.Format
else:
  Format = layout.Layout
Shape = arrays_types.Shape
Scalar = Union[int, float, np.number]
NamedSharding = jax.sharding.NamedSharding
SingleDeviceSharding = jax.sharding.SingleDeviceSharding
ScalarMetadata = value_metadata.ScalarMetadata
ArrayMetadata = value_metadata.ArrayMetadata
StringMetadata = value_metadata.StringMetadata
ShardingMetadata = sharding_metadata.ShardingMetadata
LimitInFlightBytes = limits.LimitInFlightBytes
is_ocdbt_checkpoint = format_utils.is_ocdbt_checkpoint
is_supported_empty_value = empty_values.is_supported_empty_value
is_supported_type = types.is_supported_type


_SHARDING = '_sharding'


def check_input_arguments(*args):
  l = None
  for arg in args:
    if l == 0:
      raise ValueError('Cannot pass TypeHandler input of length 0.')
    if l is None:
      l = len(arg)
    elif len(arg) != l:
      raise ValueError('Found input args with mismatched lengths.')


def check_array_values(
    values: Sequence[Union[jax.Array, np.ndarray]],
    infos: Sequence[types.ParamInfo],
):
  for v, info in zip(values, infos):
    if v.size == 0:
      raise ValueError(
          f'Cannot save arrays with zero size: ParamInfo: [name={info.name},'
          f'value_typestr={info.value_typestr}]'
      )


def _array_metadata_from_tensorstore(
    t: Any,
    info: ParamInfo,
    sharding: Optional[sharding_metadata.ShardingMetadata] = None,
) -> ArrayMetadata:
  return ArrayMetadata(
      name=info.name,
      directory=info.parent_dir,
      shape=t.shape,
      dtype=jnp.dtype(t.dtype.name),
      sharding=sharding,
      storage=value_metadata.StorageMetadata(
          chunk_shape=t.chunk_layout.read_chunk_template.shape,
          write_shape=info.write_shape,
      ),
  )


def _print_ts_debug_data(key, infos):
  """Log Tensorstore related metrics."""
  ts_metrics = ts.experimental_collect_matching_metrics('/tensorstore')
  ts_metrics += ts.experimental_collect_matching_metrics('/mallocz')
  ts_metrics += ts.experimental_collect_matching_metrics('/tcmalloc/')
  ts_metrics += [
      {'key': key},
      {'infos': [f'{info.name}' for info in infos]},
  ]

  for metrics in ts_metrics:
    logging.vlog(1, 'ts_metric: %s', metrics)

  return json.dumps(ts_metrics)


class NumpyHandler(types.TypeHandler):
  """Provides an implementation of TypeHandler for replicated numpy arrays."""

  def __init__(self, metadata_key: Optional[str] = None):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
    """
    self._metadata_key = metadata_key
    # Name of the process id to be used by single controller systems to write
    # in OCDBT format. The checkpoints are written in a subdir with this name
    # to avoid collisions with the subdir names used by other host processes
    # managed by this controller.
    self._override_ocdbt_process_id: Optional[str] = None

  def _get_array_write_spec(
      self,
      info: types.ParamInfo,
      value: np.ndarray,
      use_ocdbt: bool,
      process_index: Optional[Union[int, str]] = None,
      arg: Optional[types.SaveArgs] = None,
  ) -> ts_utils.ArrayWriteSpec:
    """Gets ArrayWriteSpec for writing."""
    return ts_utils.build_array_write_spec(
        info=info,
        arg=arg,
        global_shape=value.shape,
        local_shape=value.shape,
        dtype=value.dtype,
        use_ocdbt=use_ocdbt,
        process_index=process_index,
        metadata_key=self._metadata_key,
    )

  def _get_json_tspec_read(
      self,
      info: types.ParamInfo,
      use_ocdbt: bool,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for reading."""
    return ts_utils.get_json_tspec_read(
        info,
        use_ocdbt=use_ocdbt,
        metadata_key=self._metadata_key,
        raise_array_data_missing_error=info.raise_array_data_missing_error,
    )

  def typestr(self) -> str:
    return 'np.ndarray'

  async def metadata(
      self, infos: Sequence[types.ParamInfo]
  ) -> Sequence[ArrayMetadata]:
    open_ops = []
    for info in infos:
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=info.ts_context)
      )

    tensorstores = await asyncio.gather(*open_ops)
    return [
        _array_metadata_from_tensorstore(t, info, sharding=None)
        for t, info in zip(tensorstores, infos)
    ]

  async def _open_and_write(
      self, value: np.ndarray, tspec: Dict[str, Any], ts_context: ts.Context
  ):
    """Opens and writes using Tensorstore."""
    t = await ts.open(
        ts.Spec(tspec), create=True, open=True, context=ts_context
    )
    await t.write(value, can_reference_source_data_indefinitely=True)

  async def _background_serialize(
      self,
      values: Sequence[np.ndarray],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ):
    """Serializes numpy arrays in a background thread."""
    write_coros = []
    for value, info, arg in zip(values, infos, args):
      array_write_spec = self._get_array_write_spec(
          info,
          value,
          use_ocdbt=info.is_ocdbt_checkpoint,
          process_index=ocdbt_utils.get_process_index_for_subdir(
              use_ocdbt=info.is_ocdbt_checkpoint,
              override_ocdbt_process_id=self._override_ocdbt_process_id,
          ),
          arg=arg,
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
    check_input_arguments(values, infos, args)
    check_array_values(values, infos)
    if logging.vlog_is_on(1):
      _print_ts_debug_data(self._metadata_key, infos)
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
    check_input_arguments(infos, args)
    open_futures = []
    for info, arg in zip(infos, args):
      if not info.is_ocdbt_checkpoint:
        await ts_utils.assert_parameter_files_exist(
            info.path, self._metadata_key, info.use_zarr3
        )
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
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
        _print_ts_debug_data(self._metadata_key, infos)

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


@dataclasses.dataclass
class ArrayRestoreArgs(RestoreArgs):
  """Arguments used when restoring with ArrayHandler.

  restore_type:
    See parent class.
  mesh:
    The device mesh that the array should be restored as. Cannot be None.
  mesh_axes:
    The mesh_axes that the array should be restored as. Cannot be None.
  sharding:
   `jax.sharding.Sharding`, `ShardingMetadata`, or `Layout` object which takes
   precedence over mesh and mesh_axes if provided. Otherwise, mesh and mesh_axes
   will be used to construct a NamedSharding object OR `ShardingMetadata` which
   is an orbax representation of `jax.sharding.Sharding` that stores the same
   properties but does not require accessing real devices.
  global_shape: The global shape that the array should be restored into. If not
    provided, the shape will be restored as written. Presently, arbitrary shape
    transformations are not supported (for example, reshaping to different
    dimensions). Padding and truncating are supported. When the global_shape is
    greater than that of the saved array, 0's will be appended. If the
    global_shape is shorter than that of the saved array, excess elements will
    be dropped from the end of the array.
  shape: Interchangeable with global_shape.
  strict:
    True by default. If True, enforces that the target global shape and the
    origin global shape (as recorded by the saved array) are the same. If False,
    the returned array will be silently truncated or padded to fit the target
    global shape as necessary.
  """

  restore_type: Optional[Any] = jax.Array
  mesh: Optional[jax.sharding.Mesh] = None
  mesh_axes: Optional[jax.sharding.PartitionSpec] = None
  # pyformat: disable
  sharding: Optional[Union[jax.sharding.Sharding, ShardingMetadata, Format]] = (  # type: ignore[invalid-annotation]
      None
  )
  # pyformat: enable
  global_shape: Optional[Tuple[int, ...]] = None
  shape: Optional[Tuple[int, ...]] = None
  strict: bool = True

  def __post_init__(self):
    if self.shape is not None and self.global_shape is not None:
      if self.shape != self.global_shape:
        raise ValueError(
            'If `shape` and `global_shape` are both provided, they must match.'
        )
    elif self.shape is None:
      self.shape = self.global_shape
    elif self.global_shape is None:
      self.global_shape = self.shape


@dataclasses.dataclass
class SingleReplicaArrayRestoreArgs(ArrayRestoreArgs):
  """Arguments used when restoring with SingleReplicaArrayHandler.

  In case when training at scale loading checkpoint to all host may be
  very slow especially when checkpoint file is large. To mitigate this
  issue `SingleReplicaArrayHandler` suggests to read the checkpoint only
  on one replica hosts and do broadcasting which should significantly
  improve the training start time at scale.

  single_replica_sharding:
    jax.sharding.NamedSharding object which describes the single replica
    sharding to which current host belongs to.
  """

  single_replica_sharding: Optional[jax.sharding.NamedSharding] = None


JAX_ARRAY_TYPE_STR = 'jax.Array'


def represents_jax_array(param_info: types.ParamInfo) -> bool:
  """Returns True if the param_info represents a jax.Array."""
  assert (
      param_info.value_typestr is not None
  ), f'ParamInfo.value_typestr cannot be None: {param_info}'
  return param_info.value_typestr == JAX_ARRAY_TYPE_STR


def any_jax_array_param_info(param_infos: Pytree) -> types.ParamInfo | None:
  """Returns any jax.Array param_info in the PyTree, or None."""
  return jax.tree_util.tree_reduce(
      lambda found_jax_array, param_info: (
          found_jax_array
          or (param_info if represents_jax_array(param_info) else None)
      ),
      tree=param_infos,
      initializer=None,
  )


@functools.lru_cache(maxsize=4096)
def _is_replicated_sharding(sharding: jax.sharding.Sharding) -> bool:
  """Returns True if the sharding is replicated.

  This is to provide a quick check to decide whether to the sharding would
  produce replicated data. For namedsharding, if any axis is not specified in
  the PartitionSpec, it is considered as replicated.  This function doesn't take
  in the array shape into account as the shape isn't know at the point of
  deserialization.

  We can cache results because we typically expect `save` to be called
  repeatedly on the same model (with changing array values).

  Args:
    sharding: The sharding to check.

  Returns:
    True if the sharding is replicated.
  """
  if isinstance(sharding, NamedSharding):
    pspec = sharding.spec
    pspec_len = len(pspec)
    mesh_len = len(sharding.mesh.axis_names)
    if mesh_len > pspec_len or not pspec or any((i is None for i in pspec)):
      # replica
      return True
    else:
      return False
  elif isinstance(sharding, SingleDeviceSharding):
    return True
  else:
    logging.warning(
        'Unsupported sharding type, assuming not replicated: %s', sharding
    )
    return False


class ArrayHandler(types.TypeHandler):
  """An implementation of TypeHandler for jax.Array."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
      primary_host: Optional[int] = 0,
      replica_id: Optional[int] = 0,
      use_replica_parallel: bool = True,
      min_slice_bytes_for_replica_parallel: Optional[int] = None,
      max_replicas_for_replica_parallel: Optional[int] = None,
      enable_write_sharding_file: bool = True,
      array_metadata_store: array_metadata_store_lib.Store | None = None,
      enable_replica_parallel_separate_folder: bool = False,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
      primary_host: the host id of the primary host.  Default to 0.  If it's set
        to None, then all hosts will be considered as primary.  It's useful in
        the case that all hosts are only working with local storage.
      replica_id: the replica id to be used for saving.  Default to 0.  If it's
        set to None, each shards will pick first replica_id to be used.  It's
        useful in the case that all hosts are only working with local storage.
      use_replica_parallel: Whether to parallelize saving across replicas.
      min_slice_bytes_for_replica_parallel: Minimum number of bytes per replica
        slice. Only uses replica-parallel when the amount of data written per
        replica is greater than or equal to this number.
      max_replicas_for_replica_parallel: Maximum number of replicas over which
        saving will be parallelized if use_replica_parallel is True.
      enable_write_sharding_file: whether to write sharding file, defaults to
        True.
      array_metadata_store: Store to manage per host ArrayMetadata. To disable
        ArrayMetadata persistence, set it to None.
      enable_replica_parallel_separate_folder: If True, save replica and sharded
        arrays in separate folders when use_replica_parallel is active.
    """
    self._metadata_key = metadata_key
    self._primary_host = primary_host
    self._replica_id = replica_id
    self._enable_write_sharding_file = enable_write_sharding_file
    self._use_replica_parallel = use_replica_parallel
    self._min_slice_bytes_for_replica_parallel = (
        min_slice_bytes_for_replica_parallel
    )
    self._max_replicas_for_replica_parallel = max_replicas_for_replica_parallel
    self._array_metadata_store = array_metadata_store
    self._enable_replica_parallel_separate_folder = (
        enable_replica_parallel_separate_folder
    )
    self._ext_metadata = dict()

    logging.vlog(
        1,
        'Created `%s` with primary_host=%s, replica_id=%s,'
        ' use_replica_parallel=%s, array_metadata_store=%s',
        self.__class__.__qualname__,
        self._primary_host,
        self._replica_id,
        self._use_replica_parallel,
        self._array_metadata_store,
    )

    if self._primary_host is None and jax.__version_info__ <= (0, 4, 25):  # pylint:disable=unreachable
      raise ValueError(
          'Setting `primary_host` to None requires JAX version > 0.4.25.'
      )

  def _get_array_write_spec(
      self,
      info: types.ParamInfo,
      value: replica_slices.ReplicaSlices,
      *,
      use_ocdbt: bool,
      process_index: Optional[Union[int, str]] = None,
      replica_separate_folder: bool = False,
      arg: Optional[types.SaveArgs] = None,
  ) -> ts_utils.ArrayWriteSpec:
    """Gets ArrayWriteSpec for writing."""
    return ts_utils.build_array_write_spec(
        info=info,
        arg=arg,
        global_shape=value.global_shape,
        local_shape=value.local_shape,
        dtype=value.dtype,
        use_ocdbt=use_ocdbt,
        process_index=process_index,
        replica_separate_folder=replica_separate_folder,
        metadata_key=self._metadata_key,
        ext_metadata=self._ext_metadata.get(info.name),
    )

  def _get_json_tspec_read(
      self,
      info: types.ParamInfo,
      use_ocdbt: bool,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for reading."""
    return ts_utils.get_json_tspec_read(
        info,
        use_ocdbt=use_ocdbt,
        metadata_key=self._metadata_key,
        raise_array_data_missing_error=info.raise_array_data_missing_error,
    )

  def typestr(self) -> str:
    return JAX_ARRAY_TYPE_STR

  async def metadata(
      self, infos: Sequence[types.ParamInfo]
  ) -> Sequence[ArrayMetadata]:
    open_ops = []
    sharding_open_ops = []
    shardings = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING
    sharding_file_exists = await async_path.exists(sharding_file_path)
    for info in infos:
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=info.ts_context)
      )

      assert info.parent_dir is not None
      sharding_op = None
      if info.name:
        tspec_sharding = ts_utils.get_sharding_tensorstore_spec(
            info.parent_dir.as_posix(), info.name
        )
        if sharding_file_exists:
          sharding_op = ts.open(
              tspec_sharding,
              open=True,
              read=True,
              # OCDBT is not used for sharding metadata.
              context=info.ts_context,
          )
      sharding_open_ops.append(sharding_op)

    tensorstores = await asyncio.gather(*open_ops)

    if sharding_file_exists:
      sharding_tensorstores = await asyncio.gather(*sharding_open_ops)
      for sharding_tensorstore in sharding_tensorstores:
        if sharding_tensorstore:
          sharding_string = await sharding_tensorstore.read()
          if not sharding_string.item():
            shardings.append(None)
            continue
          deserialized = sharding_metadata.from_serialized_string(
              sharding_string.item()
          )
          shardings.append(deserialized)
        else:
          shardings.append(None)
    else:
      shardings = [None] * len(tensorstores)
    return [
        _array_metadata_from_tensorstore(t, info, sharding)
        for (t, info, sharding) in zip(tensorstores, infos, shardings)
    ]

  async def _serialize_sharding(
      self,
      sharding: jax.sharding.Sharding,
      info: types.ParamInfo,
      sharding_metadata_txn: ts.Transaction,
  ):
    """Serializes sharding metadata."""
    if info.parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    tspec_sharding = ts_utils.get_sharding_tensorstore_spec(
        info.parent_dir.as_posix(), info.name
    )
    if multihost.is_primary_host(self._primary_host):
      # OCDBT is not used for sharding metadata.
      sharding_ts_context = info.ts_context
      t = await ts.open(
          tspec_sharding,
          open=True,
          context=sharding_ts_context,
      )
      serialized_sharding = None
      sharding_metadata_value = sharding_metadata.from_jax_sharding(sharding)
      if sharding_metadata_value is not None:
        serialized_sharding = sharding_metadata_value.to_serialized_string()
      if serialized_sharding is not None:
        await t.with_transaction(sharding_metadata_txn).write(
            serialized_sharding
        )

  async def _background_serialize(
      self,
      values: Sequence[replica_slices.ReplicaSlices],
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.SaveArgs],
  ):
    """Runs serialization in a background thread."""
    write_coros = []
    sharding_metadata_txn = ts.Transaction()
    ocdbt_transaction: Optional[ts.Transaction] = None
    array_metadatas = []
    for value, info, arg in zip(values, infos, args):
      # The byte_limiter can't be used with a transaction, because awaiting the
      # `write` only waits until the in-memory transaction state reflects the
      # write, but the memory will remain in use until the transaction is
      # committed.
      if info.is_ocdbt_checkpoint and info.byte_limiter is None:
        if ocdbt_transaction is None:
          ocdbt_transaction = ts.Transaction(atomic=True)

      replica_separate_folder = False
      if (
          self._use_replica_parallel
          and self._enable_replica_parallel_separate_folder
      ):
        if info.is_ocdbt_checkpoint:
          replica_separate_folder = _is_replicated_sharding(value.sharding)
        else:
          logging.log_first_n(
              logging.WARNING,
              'Replica_separate_folder is disabled as OCDBT is not enabled.',
              1,
          )
      array_write_spec = self._get_array_write_spec(
          info,
          value,
          use_ocdbt=info.is_ocdbt_checkpoint,
          process_index=ocdbt_utils.get_process_index_for_subdir(
              info.is_ocdbt_checkpoint
          ),
          replica_separate_folder=replica_separate_folder,
          arg=arg,
      )
      tspec = array_write_spec.json
      ts_context = info.ts_context

      if logging.vlog_is_on(1):
        logging.vlog(1, 'info: %s', info)
        logging.vlog(1, 'arg: %s', arg)
        logging.vlog(
            1,
            'value.global_shape: %s, value.sharding: %s',
            value.global_shape,
            value.sharding,
        )
        logging.vlog(1, 'tspec: %s', tspec)

      write_coros.append(
          serialization.async_serialize_from_host(
              value,
              tspec,
              primary_host=self._primary_host,
              context=ts_context,
              transaction=ocdbt_transaction,
              byte_limiter=info.byte_limiter,
          )
      )
      if self._enable_write_sharding_file and value.sharding is not None:
        write_coros.append(
            self._serialize_sharding(
                value.sharding, info, sharding_metadata_txn
            )
        )
      array_metadatas.append(array_write_spec.metadata)
    if self._array_metadata_store is not None:
      write_coros.append(
          self._array_metadata_store.write(
              checkpoint_dir=infos[0].parent_dir,
              array_metadatas=array_metadatas,
              process_index=multihost.process_index(),
          )
      )

    await asyncio.gather(*write_coros)
    await sharding_metadata_txn.commit_async()
    if ocdbt_transaction is not None:
      await ocdbt_transaction.commit_async()

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ) -> Sequence[future.Future]:
    """See superclass documentation."""
    args = args or [types.SaveArgs()] * len(values)
    check_input_arguments(values, infos, args)
    check_array_values(values, infos)

    self._ext_metadata = dict()
    arrays = []
    for v, info in zip(values, infos):
      if (
          isinstance(v, jax.Array)
          and jax.process_count() > 1
          and v.is_fully_addressable
      ):
        debug_param_info = (
            f'ParamInfo=[name={info.name},value_typestr={info.value_typestr}]'
        )
        debug_array = (
            f'jax.Array=[value={v},shape={v.shape},dtype={v.dtype},'
            f'sharding={v.sharding},device={v.device}]'
        )
        raise ValueError(
            f'Cannot serialize host local jax.Array ({debug_param_info},'
            f' {debug_array}) in multi-host setting. Arrays like this are'
            ' typically obtained using pmap. Consider using'
            ' fully_replicated_host_local_array_to_global_array in'
            ' orbax/checkpoint/utils.py to convert your arrays into'
            f' serializable objects. Array.sharding: {v.sharding}'
        )

      if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key):
        # a JAX random key
        arrays.append(jax.random.key_data(v))
        self._ext_metadata[info.name] = {
            array_metadata_lib.RANDOM_KEY_IMPL: str(jax.random.key_impl(v))
        }
      else:
        # regular array
        arrays.append(v)

    assert all([info.enable_pinned_host_transfer for info in infos]) or all(
        [not info.enable_pinned_host_transfer for info in infos]
    )

    # Complete D2H transfer in parallel for each array.
    values_on_host = replica_slices.transfer_arrays_to_host(
        arrays,
        self._replica_id,
        self._use_replica_parallel,
        enable_pinned_host_transfer=infos[0].enable_pinned_host_transfer,
        min_slice_bytes_for_replica_parallel=self._min_slice_bytes_for_replica_parallel,
        max_replicas_for_replica_parallel=self._max_replicas_for_replica_parallel,
    )

    return [
        future.CommitFutureAwaitingContractedSignals(
            self._background_serialize(values_on_host, infos, args),
            name='array_type_handler',
        )
    ]

  def _parse_array_metadatas(self, array_metadatas, infos, deserialized_arrays):
    """Parse array_metadatas and adjust deserialized_arrays accordingly."""

    logging.vlog(1, 'array_metadatas = %s', array_metadatas)
    if not isinstance(array_metadatas, Dict):
      raise ValueError(
          'Expecting array_metadatas to be a "Dict" but got'
          f' {type(array_metadatas)}.'
      )

    # use the first available array_metadata
    array_metadatas_cache = {
        array_metadata.param_name: array_metadata
        for array_metadata in next(iter(array_metadatas.values()))
    }

    for i, (info, v) in enumerate(zip(infos, deserialized_arrays)):
      if meta := array_metadatas_cache.get(info.name):
        assert isinstance(
            meta, array_metadata_lib.SerializedArrayMetadata
        ), f'Expecting SerializedArrayMetadata but got {type(meta)}.'
        if meta.ext_metadata is None or not isinstance(meta.ext_metadata, dict):
          continue

        if impl := meta.ext_metadata.get(array_metadata_lib.RANDOM_KEY_IMPL):  # pytype: disable=attribute-error
          deserialized_arrays[i] = jax.random.wrap_key_data(v, impl=impl)
          logging.vlog(
              1,
              '%s: recreated as a random key: %s',
              info.name,
              deserialized_arrays[i],
          )

    return deserialized_arrays

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[jax.Array]:
    """See superclass documentation.

    Args:
      infos: ParamInfo.
      args: must be of type `ArrayRestoreArgs`.

    Returns:
      The deserialized parameter.

    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.sharding` is not provided or `args.mesh` and
      `args.mesh_axes` are not provided.
    """
    if args is None:
      raise ValueError('Must provide ArrayRestoreArgs to restore as jax.Array.')
    check_input_arguments(infos, args)
    deserialize_ops = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING
    sharding_file_exists = await async_path.exists(sharding_file_path)
    for info, arg in zip(infos, args):
      sharding = None
      arg = cast(ArrayRestoreArgs, arg)
      if (
          isinstance(arg, ArrayRestoreArgs)
          and arg.mesh is not None
          and arg.mesh_axes is not None
      ):
        sharding = NamedSharding(arg.mesh, arg.mesh_axes)
      elif isinstance(arg, ArrayRestoreArgs) and arg.sharding is not None:
        if isinstance(arg.sharding, ShardingMetadata):
          sharding = arg.sharding.to_jax_sharding()
        else:
          sharding = arg.sharding
      elif sharding_file_exists:
        warnings.warn(
            'Sharding info not provided when restoring. Populating sharding'
            ' info from sharding file. Please note restoration time will be'
            ' slightly increased due to reading from file. Note also that this'
            ' option is unsafe when restoring on a different topology than the'
            ' checkpoint was saved with.'
        )
        assert info.parent_dir is not None
        if info.name:
          tspec_sharding = ts_utils.get_sharding_tensorstore_spec(
              info.parent_dir.as_posix(), info.name
          )
          t = await ts.open(
              tspec_sharding,
              # OCDBT is not used for sharding metadata.
              context=info.ts_context,
              open=True,
              read=True,
          )
          serialized_string = await t.read()
          if serialized_string:
            sharding = sharding_metadata.get_sharding_or_none(serialized_string)
        else:
          raise ValueError('Unable to deserialize sharding.')
      else:
        raise ValueError(
            'Sharding of jax.Array cannot be None. Provide `mesh`'
            ' and `mesh_axes` OR `sharding`'
        )
      if not info.is_ocdbt_checkpoint:
        await ts_utils.assert_parameter_files_exist(
            info.path,
            self._metadata_key,
            info.use_zarr3,
        )
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = ts_utils.get_cast_tspec_deserialize(tspec, arg)
      if logging.vlog_is_on(1):
        logging.vlog(1, 'tspec = %s', tspec)
        logging.vlog(1, 'info = %s', info)
        logging.vlog(1, 'arg = %s', arg)
        logging.vlog(1, 'sharding = %s', sharding)
      deserialize_ops += [
          serialization.async_deserialize(
              sharding,
              tspec,
              global_shape=arg.global_shape
              if hasattr(arg, 'global_shape')
              else None,
              byte_limiter=info.byte_limiter,
              context=info.ts_context,
              strict=arg.strict if hasattr(arg, 'strict') else True,
          )
      ]

    if self._array_metadata_store is not None:
      deserialize_ops.append(
          self._array_metadata_store.read(
              checkpoint_dir=infos[0].parent_dir,
          )
      )
      *ret, array_metadatas = await asyncio.gather(*deserialize_ops)

      if array_metadatas:
        ret = self._parse_array_metadatas(array_metadatas, infos, ret)
    else:
      ret = await asyncio.gather(*deserialize_ops)

    if logging.vlog_is_on(1):
      for a in ret:
        logging.vlog(
            1,
            'restored jax.Array.shape = %s, jax.array.dtype = %s,'
            ' jax.array.format = %s',
            getattr(a, 'shape', None),
            getattr(a, 'dtype', None),
            getattr(a, 'format', None),
        )
      _print_ts_debug_data(self._metadata_key, infos)

    return ret  # pytype: disable=bad-return-type

  def memory_size(
      self, values: Sequence[jax.Array]
  ) -> Sequence[Tuple[int, int]]:
    write_sizes = []
    read_sizes = []
    shard_size = lambda shard: shard.data.size * shard.data.dtype.itemsize
    for v in values:
      write_sizes.append(
          replica_slices.get_replica_slices(
              v,
              replica_id=self._replica_id,
              use_replica_parallel=self._use_replica_parallel,
              min_slice_bytes_for_replica_parallel=self._min_slice_bytes_for_replica_parallel,
              max_replicas_for_replica_parallel=self._max_replicas_for_replica_parallel,
          ).nbytes
      )
      read_sizes.append(
          sum(shard_size(shard) for shard in v.addressable_shards)
      )
    return list(zip(write_sizes, read_sizes))


def _is_host_for_primary_replica(primary_replica_ids: set[int]) -> bool:
  return multihost.process_index() in primary_replica_ids


class InvalidShardingError(ValueError):
  """Error raised when sharding is not valid."""


class SingleReplicaArrayHandler(ArrayHandler):
  """An implementation TypeHandler for jax.

  ArrayHandler that optimizes checkpoint read performance during multi-pod or
  multihost training. Normally each host reads relevant data from the
  checkpoint, even if other hosts are reading the exact same data. This can be
  very inefficient with large number of pods/hosts and large checkpoint size.
  With SingleReplicaArrayhandler the data is read only on hosts that are in
  primary replica. Then these hosts broadcast the data to other hosts. It is
  assumed that all hosts have ALL their devices either inside the primary
  replica or outside.
  Consider, for example, the following sharding on v4-128 which has 16 hosts and
  64 devices::

    shape = (32, 2)
    mesh = jax.sharding.Mesh(jax.devices().reshape(shape), ('x', 'y'))
    pspec = jax.sharding.PartitionSpec(None, 'y')
    sharding=jax.sharding.NamedSharding(mesh, pspec)

  This sharding will not work since the primary replica has only two devices,
  and hence there is a host which has 2 devices in the primary replica, and 2
  devices outside of primary replica. However, changing shape, for example, to
  (4, 16) will result in a valid sharding.

  This TypeHandler can be registered by running::

    ocp.type_handlers.register_type_handler(
        jax.Array,
        type_handlers.SingleReplicaArrayHandler(),
        override=True)

  Example usage can be found in MaxText (TO BE MERGED).
  https://github.com/google/maxtext/blob/main/MaxText/checkpointing.py
  """

  def __init__(
      self,
      replica_axis_index: Optional[int] = 0,
      primary_replica_id: Optional[int] = 0,
      broadcast_memory_limit_bytes: Optional[int] = None,
      broadcast_memory_scaling_factor: Optional[float] = 0.75,
      use_replica_parallel: bool = True,
      enable_write_sharding_file: bool = True,
      array_metadata_store: array_metadata_store_lib.Store | None = None,
  ):
    """Constructor.

    Args:
      replica_axis_index: Defines the axis of the global mesh along which
        replicas are defined. E.g. all devices in
        global_mesh.devices[replica_axis_index] are part of the same replica.
      primary_replica_id: The id of the replica hosts that is used to load and
        broadcast the checkpoint.
      broadcast_memory_limit_bytes: Specifies the memory size (in bytes) used
        for broadcasting data.
      broadcast_memory_scaling_factor: Specifies the fraction of available
        memory to use for broadcasting data.
      use_replica_parallel: Whether to parallelize saving across replicas.
      enable_write_sharding_file: whether to write sharding file, defaults to
        True.
      array_metadata_store: Store to manage per host ArrayMetadata. To disable
        ArrayMetadata persistence, set it to None.
    """

    super(SingleReplicaArrayHandler, self).__init__(
        use_replica_parallel=use_replica_parallel,
        enable_write_sharding_file=enable_write_sharding_file,
        array_metadata_store=array_metadata_store,
    )
    self.replica_axis_index = replica_axis_index
    self.primary_replica_id = primary_replica_id
    self.broadcast_memory_limit_bytes = broadcast_memory_limit_bytes
    self.broadcast_memory_scaling_factor = broadcast_memory_scaling_factor

  def _validate_sharding_and_get_primary_replica_processes(
      self,
      sharding: jax.sharding.Sharding,
  ) -> Set[int]:
    """Validates sharding for restoration."""
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise InvalidShardingError(
          'The provided sharding is not a NamedSharding. Please use'
          ' NamedSharding instead.'
      )
    primary_replica_device_ids, primary_replica_pids = (
        multislice.get_primary_replica_ids_and_pids(
            replica_axis_idx=self.replica_axis_index,
            mesh=sharding.mesh,  # pytype: disable=attribute-error
            primary_replica_id=self.primary_replica_id,
        )
    )
    if len(primary_replica_device_ids) == len(jax.devices()):
      raise InvalidShardingError(
          'All devices are in the primary replica. There are no non-primary'
          ' replicas to broadcast to.'
      )

    expected_primary_replica_device_ids = {
        d.id
        for d in jax.devices()
        if multihost.process_index_from_device(d) in primary_replica_pids
    }
    if not primary_replica_device_ids.issubset(
        expected_primary_replica_device_ids
    ):
      raise InvalidShardingError(
          'The provided sharding is not valid. The primary replica has the'
          f' following devices: {primary_replica_device_ids}, which is not a'
          ' subset of the expected devices:'
          f' {expected_primary_replica_device_ids}. for the primary processes:'
          f' {primary_replica_pids}.'
      )

    return primary_replica_pids

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Optional[
          Sequence[SingleReplicaArrayRestoreArgs]
      ] = None,  # pytype: disable=signature-mismatch
  ) -> Sequence[jax.Array]:
    """Deserializing in case of single replica broadcasting.

    Args:
      infos: ParamInfo.
      args: must be of type `SingleReplicaArrayRestoreArgs`.

    Returns:
      Deserialized parameters.
    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.sharding` is not provided or `args.mesh` and
      `args.mesh_axes` or `single_replica_pids` or `single_replica_ids` are
      not provided.
    """
    if args is None:
      raise ValueError(
          'Must provide SingleReplicaArrayRestoreArgs to restore as jax.Array.'
      )
    check_input_arguments(infos, args)
    deserialize_ops = []
    shardings = []
    primary_replica_pids = set()
    single_replica_shardings = []
    for info, arg in zip(infos, args):
      arg = cast(SingleReplicaArrayRestoreArgs, arg)
      if not info.is_ocdbt_checkpoint:
        await ts_utils.assert_parameter_files_exist(  # pylint: disable=protected-access
            info.path, self._metadata_key
        )
      if not isinstance(arg, SingleReplicaArrayRestoreArgs):
        raise ValueError(
            'Must provide `SingleReplicaArrayRestoreArgs`, but got'
            f' {type(arg)}.'
        )
      if arg.sharding is None:
        raise ValueError('Must provide `sharding`.')
      sharding = arg.sharding
      shardings.append(sharding)
      primary_replica_pids = (
          self._validate_sharding_and_get_primary_replica_processes(sharding)
      )
      if arg.single_replica_sharding is None:
        raise ValueError('Must provide `single_replica_sharding`.')
      single_replica_sharding = arg.single_replica_sharding
      single_replica_shardings.append(single_replica_sharding)

      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = ts_utils.get_cast_tspec_deserialize(tspec, arg)  # pylint: disable=protected-access

      if _is_host_for_primary_replica(primary_replica_pids):
        deserialize_ops += [
            serialization.async_deserialize(
                single_replica_sharding,
                tspec,
                global_shape=arg.global_shape
                if hasattr(arg, 'global_shape')
                else None,
                byte_limiter=info.byte_limiter,
                context=info.ts_context,
            )
        ]

    @functools.partial(
        jax.jit, static_argnums=0, out_shardings=tuple(single_replica_shardings)
    )
    def create_zeros(shape_dtype_tup):
      return jax.tree.map(
          lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
      )

    if _is_host_for_primary_replica(primary_replica_pids):
      start_deserialization = time.time()
      deserialized = await asyncio.gather(*deserialize_ops)
      deserialization_elapsed_s = time.time() - start_deserialization
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/read/primary_replica_deserialization_duration_secs',
          deserialization_elapsed_s,
      )
      logging.info(
          'Finished primary replica deserialization in %.2f',
          deserialization_elapsed_s,
      )

    else:
      shape_dtype = [
          jax.ShapeDtypeStruct(arg.global_shape, arg.dtype) for arg in args
      ]
      deserialized = create_zeros(tuple(shape_dtype))

    deserialized = tuple(deserialized)

    start_broadcast = time.time()
    global_mesh = cast(jax.sharding.NamedSharding, shardings[0]).mesh
    shared_state, _ = multislice.broadcast_one_replica_to_all(
        deserialized,
        global_mesh,
        self.replica_axis_index,
        _is_host_for_primary_replica(primary_replica_pids),
        memory_limit_bytes=self.broadcast_memory_limit_bytes,
        memory_scaling_factor=self.broadcast_memory_scaling_factor,
    )
    broadcast_elapsed_s = time.time() - start_broadcast
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/read/broadcast_duration_secs', broadcast_elapsed_s
    )
    logging.info('Finished broadcasting in %.2f', broadcast_elapsed_s)
    return shared_state

  # TODO(b/370396118): Calculation overestimates bytes read.
  def memory_size(  # pylint: disable=useless-parent-delegation
      self, values: Sequence[jax.Array]
  ) -> Sequence[Tuple[int, int]]:
    return super().memory_size(values)


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
    if info.path is None:
      raise ValueError('Must construct serialization path.')
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
    check_input_arguments(values, infos)
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
        write_coros.append(t.with_transaction(txn).write(value))
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
    check_input_arguments(infos)
    directory = epath.Path(infos[0].path).parent
    open_futures = []

    for info in infos:
      info.path = epath.Path(directory / self._filename)
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


class _TypeHandlerRegistryImpl(types.TypeHandlerRegistry):
  """The implementation for TypeHandlerRegistry."""

  def __init__(self, *handlers: Tuple[Any, types.TypeHandler]):
    """Create a type registry.

    Args:
      *handlers: an optional list of handlers to initialize with.
    """
    self._type_registry: List[
        Tuple[Callable[[Any], bool], types.TypeHandler]
    ] = []
    self._typestr_registry: Dict[str, types.TypeHandler] = {}
    if handlers:
      for ty, h in handlers:
        self.add(ty, h, override=True, ignore_warnings=True)

  def add(
      self,
      ty: Any,
      handler: types.TypeHandler,
      func: Optional[Callable[[Any], bool]] = None,
      override: bool = False,
      ignore_warnings: bool = False,
  ):
    if func is None:
      func = lambda t: issubclass(t, ty)

    existing_handler_idx = None
    for i, (f, _) in enumerate(self._type_registry):
      if f(ty):
        existing_handler_idx = i
        # Ignore the possibility for subsequent matches, as these will not be
        # used anyway.
        break

    if existing_handler_idx is None:
      if handler.typestr() in self._typestr_registry:
        if override:
          if not ignore_warnings:
            logging.warning(
                'Type handler registry overriding type "%s" collision on %s',
                ty,
                handler.typestr(),
            )
        else:
          raise ValueError(
              f'Type "{ty}" has a `typestr` ("{handler.typestr()}") which'
              ' collides with that of an existing TypeHandler.'
          )
      self._type_registry.append((func, handler))
      self._typestr_registry[handler.typestr()] = handler
    elif override:
      if not ignore_warnings:
        logging.warning(
            'Type handler registry type "%s" overriding %s',
            ty,
            handler.typestr(),
        )
      self._type_registry[existing_handler_idx] = (func, handler)
      self._typestr_registry[handler.typestr()] = handler
    else:
      raise ValueError(f'A TypeHandler for "{ty}" is already registered.')

  def get(self, ty: Any) -> types.TypeHandler:
    if isinstance(ty, str):
      if ty in self._typestr_registry:
        return self._typestr_registry[ty]
    else:
      for func, handler in self._type_registry:
        if func(ty):
          return handler
    raise ValueError(f'Unknown type: "{ty}". Must register a TypeHandler.')

  def has(self, ty: Any) -> bool:
    try:
      self.get(ty)
      return True
    except ValueError:
      return False


def create_type_handler_registry(
    *handlers: Tuple[Any, types.TypeHandler]
) -> types.TypeHandlerRegistry:
  """Create a type registry.

  Args:
    *handlers: optional pairs of `(<type>, <handler>)` to initialize the
      registry with.

  Returns:
    A TypeHandlerRegistry instance with only the specified handlers.
  """
  return _TypeHandlerRegistryImpl(*handlers)


_DEFAULT_TYPE_HANDLERS = tuple([
    (int, ScalarHandler()),
    (float, ScalarHandler()),
    (bytes, ScalarHandler()),
    (np.number, ScalarHandler()),
    (np.ndarray, NumpyHandler()),
    (
        jax.Array,
        ArrayHandler(array_metadata_store=array_metadata_store_lib.Store()),
    ),
    (str, StringHandler()),
    (type(PLACEHOLDER), PlaceholderHandler()),
])


GLOBAL_TYPE_HANDLER_REGISTRY = create_type_handler_registry(
    *_DEFAULT_TYPE_HANDLERS
)


def register_type_handler(
    ty: Any,
    handler: types.TypeHandler,
    func: Optional[Callable[[Any], bool]] = None,
    override: bool = False,
):
  """Registers a type for serialization/deserialization with a given handler.

  Note that it is possible for a type to match multiple different entries in
  the registry, each with a different handler. In this case, only the first
  match is used.

  Args:
    ty: A type to register.
    handler: a TypeHandler capable of reading and writing parameters of type
      `ty`.
    func: A function that accepts a type and returns True if the type should be
      handled by the provided TypeHandler. If this parameter is not specified,
      defaults to `lambda t: issubclass(t, ty)`.
    override: if True, will override an existing mapping of type to handler.

  Raises:
    ValueError if a type is already registered and override is False.
  """
  GLOBAL_TYPE_HANDLER_REGISTRY.add(ty, handler, func, override)


def get_type_handler(ty: Any) -> types.TypeHandler:
  """Returns the handler registered for a given type, if available."""
  return GLOBAL_TYPE_HANDLER_REGISTRY.get(ty)


def has_type_handler(ty: Any) -> bool:
  """Returns if there is a handler registered for a given type."""
  return GLOBAL_TYPE_HANDLER_REGISTRY.has(ty)


def supported_types() -> list[Any]:
  """Returns the default list of supported types."""
  return [ty for ty, _ in _DEFAULT_TYPE_HANDLERS]


def register_standard_handlers_with_options(**kwargs):
  """Re-registers a select set of handlers with the given options.

  This is intended to override options en masse for the standard numeric
  TypeHandlers and their corresponding types (scalars, numpy arrays and
  jax.Arrays).

  Args:
    **kwargs: keyword arguments to pass to each of the standard handlers.
  """
  # TODO(b/314258967): clean those up.
  del kwargs['use_ocdbt'], kwargs['ts_context']
  GLOBAL_TYPE_HANDLER_REGISTRY.add(int, ScalarHandler(**kwargs), override=True)
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      float, ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      bytes, ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      np.number, ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      np.ndarray, NumpyHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      jax.Array, ArrayHandler(**kwargs), override=True
  )


# TODO(b/253238305) Deprecate when all checkpoints have saved types.
def default_restore_type(args: RestoreArgs) -> Any:
  if isinstance(args, ArrayRestoreArgs):
    return jax.Array
  elif isinstance(args, RestoreArgs):
    return np.ndarray
  else:
    raise ValueError(f'Unsupported restore_args type: {type(args)}')
