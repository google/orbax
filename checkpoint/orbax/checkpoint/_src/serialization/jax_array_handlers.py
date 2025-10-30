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

"""Provides handlers for Jax Arrays."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import time
from typing import Any, Dict, Optional, Sequence, Set, Tuple, TypeAlias, Union, cast
import warnings

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.serialization import jax_array_restore_args
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types
import tensorstore as ts


Pytree: TypeAlias = Any
ArrayRestoreArgs = jax_array_restore_args.ArrayRestoreArgs
SingleReplicaArrayRestoreArgs = (
    jax_array_restore_args.SingleReplicaArrayRestoreArgs
)


_SHARDING_FILE_NAME = '_sharding'


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
  if isinstance(sharding, jax.sharding.NamedSharding):
    pspec = sharding.spec
    pspec_len = len(pspec)
    mesh_len = len(sharding.mesh.axis_names)
    if mesh_len > pspec_len or not pspec or any((i is None for i in pspec)):
      # replica
      return True
    else:
      return False
  elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
    return True
  else:
    logging.warning(
        'Unsupported sharding type, assuming not replicated: %s', sharding
    )
    return False


async def _async_serialize_shardings(
    shardings: Sequence[jax.sharding.Sharding | None],
    infos: Sequence[types.ParamInfo],
    *,
    primary_host: int | None,
):
  """Serializes sharding metadata."""
  sharding_metadata_txn = ts.Transaction()

  for sharding, info in zip(shardings, infos):
    if sharding is None:
      continue
    if info.parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    tspec_sharding = ts_utils.get_sharding_tensorstore_spec(
        info.parent_dir.as_posix(), info.name
    )
    if multihost.is_primary_host(primary_host):
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
        await t.with_transaction(sharding_metadata_txn).write(  # pytype: disable=attribute-error
            serialized_sharding
        )

  await sharding_metadata_txn.commit_async()


def _get_replica_slices(
    arrays: Sequence[jax.Array],
    replica_id: int,
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: int | None,
    max_replicas_for_replica_parallel: int | None,
) -> Sequence[replica_slices.ReplicaSlices]:
  """Returns ReplicaSlices for arrays."""
  rslices_per_array = [
      replica_slices.get_replica_slices(
          arr,
          replica_id,
          use_replica_parallel,
          min_slice_bytes_for_replica_parallel,
          max_replicas_for_replica_parallel,
      )
      for arr in arrays
  ]
  # D2H copy is performed automatically as part of dispatcher call, but
  # we must set properties correctly to pass later consistency checks.
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice,
                  unsliced_data=np.asarray(rslice.data()),
                  slice_args=None,
              )
              for rslice in rslices.replica_slices
          ],
      )
      for rslices in rslices_per_array
  ]


def _worker_serialize_arrays(
    arrays: Sequence[jax.Array],
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.SaveArgs],
    replica_id: int,
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: int | None,
    max_replicas_for_replica_parallel: int | None,
    primary_host: int | None,
    metadata_key: str | None,
    array_metadata_store: array_metadata_store_lib.Store | None,
    enable_replica_parallel_separate_folder: bool,
    ext_metadata: Dict[str, Any],
):
  """Worker function to serialize arrays."""
  rslices_per_array = _get_replica_slices(
      arrays,
      replica_id,
      use_replica_parallel,
      min_slice_bytes_for_replica_parallel,
      max_replicas_for_replica_parallel,
  )

  asyncio_utils.run_sync(
      _async_serialize_replica_slices(
          rslices_per_array,
          infos,
          args,
          primary_host=primary_host,
          metadata_key=metadata_key,
          array_metadata_store=array_metadata_store,
          enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
          use_replica_parallel=use_replica_parallel,
          ext_metadata=ext_metadata,
      )
  )


def _serialize_arrays(
    arrays: Sequence[jax.Array],
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.SaveArgs],
    dispatcher: dispatchers.Dispatcher | None,
    replica_id: int,
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: int | None,
    max_replicas_for_replica_parallel: int | None,
    primary_host: int | None,
    metadata_key: str | None,
    array_metadata_store: array_metadata_store_lib.Store | None,
    enable_replica_parallel_separate_folder: bool,
    ext_metadata: Dict[str, Any],
) -> future.Future:
  """D2H transfer and serialize arrays using dispatcher if provided."""
  if dispatcher is None:
    # Complete D2H transfer in parallel for each array.
    values_on_host = replica_slices.transfer_arrays_to_host(
        arrays,
        replica_id,
        use_replica_parallel,
        enable_pinned_host_transfer=infos[0].enable_pinned_host_transfer,
        min_slice_bytes_for_replica_parallel=min_slice_bytes_for_replica_parallel,
        max_replicas_for_replica_parallel=max_replicas_for_replica_parallel,
    )
    return future.CommitFutureAwaitingContractedSignals(
        _async_serialize_replica_slices(
            values_on_host,
            infos,
            args,
            primary_host=primary_host,
            metadata_key=metadata_key,
            array_metadata_store=array_metadata_store,
            enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
            use_replica_parallel=use_replica_parallel,
            ext_metadata=ext_metadata,
        ),
        name='array_type_handler',
    )
  else:

    async def _serialize():
      ret = dispatcher.dispatch(
          _worker_serialize_arrays,
          input_arrays=arrays,
          func_kwargs={
              'infos': infos,
              'args': args,
              'replica_id': replica_id,
              'use_replica_parallel': use_replica_parallel,
              'min_slice_bytes_for_replica_parallel': (
                  min_slice_bytes_for_replica_parallel
              ),
              'max_replicas_for_replica_parallel': (
                  max_replicas_for_replica_parallel
              ),
              'primary_host': primary_host,
              'metadata_key': metadata_key,
              'array_metadata_store': array_metadata_store,
              'enable_replica_parallel_separate_folder': (
                  enable_replica_parallel_separate_folder
              ),
              'ext_metadata': ext_metadata,
          },
      )
      jax.block_until_ready(ret)

    return future.CommitFutureAwaitingContractedSignals(
        _serialize(),
        name='array_type_handler',
    )


async def _async_serialize_replica_slices(
    values: Sequence[replica_slices.ReplicaSlices],
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.SaveArgs],
    *,
    primary_host: int | None,
    metadata_key: str | None,
    array_metadata_store: array_metadata_store_lib.Store | None,
    enable_replica_parallel_separate_folder: bool,
    use_replica_parallel: bool,
    ext_metadata: Dict[str, Any],
) -> None:
  """This function contains the logic from ArrayHandler._background_serialize."""
  write_coros = []
  ocdbt_transaction: Optional[ts.Transaction] = None
  array_metadatas = []
  for value, info, arg in zip(values, infos, args):
    if info.is_ocdbt_checkpoint and info.byte_limiter is None:
      if ocdbt_transaction is None:
        ocdbt_transaction = ts.Transaction(atomic=True)
    replica_separate_folder = False
    if use_replica_parallel and enable_replica_parallel_separate_folder:
      if info.is_ocdbt_checkpoint:
        replica_separate_folder = _is_replicated_sharding(value.sharding)
      else:
        logging.log_first_n(
            logging.WARNING,
            'Replica_separate_folder is disabled as OCDBT is not enabled.',
            1,
        )
    array_write_spec = ts_utils.build_array_write_spec(
        info=info,
        arg=arg,
        global_shape=value.global_shape,
        local_shape=value.local_shape,
        dtype=value.dtype,
        use_ocdbt=info.is_ocdbt_checkpoint,
        process_index=ocdbt_utils.get_process_index_for_subdir(
            info.is_ocdbt_checkpoint
        ),
        replica_separate_folder=replica_separate_folder,
        metadata_key=metadata_key,
        ext_metadata=ext_metadata.get(info.name),
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
            primary_host=primary_host,
            context=ts_context,
            transaction=ocdbt_transaction,
            byte_limiter=info.byte_limiter,
        )
    )
    array_metadatas.append(array_write_spec.metadata)
  if array_metadata_store is not None:
    write_coros.append(
        array_metadata_store.write(
            checkpoint_dir=infos[0].parent_dir,
            array_metadatas=array_metadatas,
            process_index=multihost.process_index(),
        )
    )

  await asyncio.gather(*write_coros)
  if ocdbt_transaction is not None:
    await ocdbt_transaction.commit_async()


def _wrap_random_key_data(
    array_metadatas: Any,
    infos: Sequence[types.ParamInfo],
    deserialized_arrays: list[jax.Array],
) -> Sequence[jax.Array]:
  """Parse array_metadatas and wrap deserialized_arrays as random keys."""

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


def _validate_ocdbt_settings(infos: Sequence[types.ParamInfo]) -> bool:
  """Checks that all parameters have matching OCDBT flags set."""
  assert infos
  use_ocdbt = infos[0].is_ocdbt_checkpoint
  for info in infos:
    if info.is_ocdbt_checkpoint != use_ocdbt:
      raise ValueError(
          f"OCDBT settings for parameter {info.name} don't match those for the"
          f' rest of parameters: got ({info.is_ocdbt_checkpoint=}, expected'
          f' {use_ocdbt=}.'
      )
  if use_ocdbt is None:
    raise ValueError('Setting of `use_ocdbt` may not be None.')
  return use_ocdbt


async def _validate_non_ocdbt_files(
    infos: Sequence[types.ParamInfo], metadata_key: str
):
  await asyncio.gather(*[
      ts_utils.assert_parameter_files_exist(  # pylint: disable=protected-access
          info.path, metadata_key, info.use_zarr3
      )
      for info in infos
  ])


async def _deserialize_shardings(
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.RestoreArgs],
    sharding_file_exists: bool,
) -> Sequence[Any]:
  """Deserializes shardings from file or infers from args."""
  shardings = []
  for info, arg in zip(infos, args):
    sharding = None
    if (
        isinstance(arg, ArrayRestoreArgs)
        and arg.mesh is not None
        and arg.mesh_axes is not None
    ):
      sharding = jax.sharding.NamedSharding(arg.mesh, arg.mesh_axes)
    elif isinstance(arg, ArrayRestoreArgs) and arg.sharding is not None:
      if isinstance(arg.sharding, sharding_metadata.ShardingMetadata):
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
        serialized_string = await t.read()  # pytype: disable=attribute-error
        if serialized_string:
          sharding = sharding_metadata.get_sharding_or_none(serialized_string)
      else:
        raise ValueError('Unable to deserialize sharding.')
    else:
      raise ValueError(
          'Sharding of jax.Array cannot be None. Provide `mesh`'
          ' and `mesh_axes` OR `sharding`'
      )
    shardings.append(sharding)
  return shardings


async def _deserialize_arrays(
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.RestoreArgs],
    shardings: Sequence[jax.sharding.Sharding],
    metadata_key: str | None,
    array_metadata_store: array_metadata_store_lib.Store | None,
) -> Sequence[jax.Array]:
  """Deserializes arrays and applies array_metadata if available."""

  async def _async_deserialize(
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.RestoreArgs],
      shardings: Sequence[jax.sharding.Sharding],
      *,
      metadata_key: str | None,
  ) -> list[jax.Array]:
    """This function contains the core TensorStore read logic from ArrayHandler.deserialize."""
    use_ocdbt = _validate_ocdbt_settings(infos)
    if not use_ocdbt:
      await _validate_non_ocdbt_files(infos, metadata_key)
    deserialize_ops = []
    for info, arg, sharding in zip(infos, args, shardings):
      tspec = ts_utils.get_json_tspec_read(
          info,
          use_ocdbt=use_ocdbt,
          metadata_key=metadata_key,
          raise_array_data_missing_error=info.raise_array_data_missing_error,
      )
      tspec = ts_utils.get_cast_tspec_deserialize(tspec, arg)
      if logging.vlog_is_on(1):
        logging.vlog(1, 'tspec = %s', tspec)
        logging.vlog(1, 'info = %s', info)
        logging.vlog(1, 'arg = %s', arg)
        logging.vlog(1, 'sharding = %s', sharding)
      deserialize_ops.append(
          serialization.async_deserialize(
              sharding,
              tspec,
              global_shape=arg.global_shape
              if hasattr(arg, 'global_shape')
              else None,
              dtype=arg.dtype,
              byte_limiter=info.byte_limiter,
              context=info.ts_context,
              strict=arg.strict if hasattr(arg, 'strict') else True,
          )
      )
    return await asyncio.gather(*deserialize_ops)

  if array_metadata_store is not None:
    ret, array_metadatas = await asyncio.gather(
        _async_deserialize(
            infos,
            args,
            shardings,
            metadata_key=metadata_key,
        ),
        array_metadata_store.read(
            checkpoint_dir=infos[0].parent_dir,
        ),
    )
    if array_metadatas:
      ret = _wrap_random_key_data(array_metadatas, infos, ret)
  else:
    ret = await _async_deserialize(
        infos,
        args,
        shardings,
        metadata_key=metadata_key,
    )
  return ret


def _sync_deserialize_arrays(
    infos: Sequence[types.ParamInfo],
    args: Sequence[types.RestoreArgs],
    shardings: Sequence[jax.sharding.Sharding],
    metadata_key: str | None,
    array_metadata_store: array_metadata_store_lib.Store | None,
) -> Sequence[jax.Array]:
  """Deserializes arrays and applies array_metadata if available."""
  return asyncio_utils.run_sync(
      _deserialize_arrays(
          infos,
          args,
          shardings,
          metadata_key,
          array_metadata_store,
      )
  )


def _get_abstract_arrays(
    args: Sequence[types.RestoreArgs],
    shardings: Sequence[jax.sharding.Sharding],
) -> Sequence[jax.ShapeDtypeStruct]:
  """Returns result specs for the given restore args."""
  abstract_arrays = []
  for arg, sharding in zip(args, shardings):
    assert isinstance(arg, ArrayRestoreArgs)
    assert arg.global_shape is not None
    assert arg.dtype is not None
    abstract_arrays.append(
        jax.ShapeDtypeStruct(
            shape=arg.global_shape, dtype=arg.dtype, sharding=sharding
        )
    )
  return abstract_arrays


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
      dispatcher: dispatchers.Dispatcher | None = None,
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
      dispatcher: The dispatcher to use for executing operations on workers if
        provided.
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
    self._dispatcher = dispatcher

    logging.vlog(
        1,
        'Created `%s` with primary_host=%s, replica_id=%s,'
        ' use_replica_parallel=%s, array_metadata_store=%s, dispatcher=%s',
        self.__class__.__qualname__,
        self._primary_host,
        self._replica_id,
        self._use_replica_parallel,
        self._array_metadata_store,
        self._dispatcher,
    )

    if self._primary_host is None and jax.__version_info__ <= (0, 4, 25):  # pylint:disable=unreachable
      raise ValueError(
          'Setting `primary_host` to None requires JAX version > 0.4.25.'
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
  ) -> Sequence[value_metadata.ArrayMetadata]:
    open_ops = []
    sharding_open_ops = []
    shardings = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING_FILE_NAME
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
          if not sharding_string.item():  # pytype: disable=attribute-error
            shardings.append(None)
            continue
          deserialized = sharding_metadata.from_serialized_string(
              sharding_string.item()  # pytype: disable=attribute-error
          )
          shardings.append(deserialized)
        else:
          shardings.append(None)
    else:
      shardings = [None] * len(tensorstores)
    return [
        ts_utils.array_metadata_from_tensorstore(t, info, sharding)
        for (t, info, sharding) in zip(tensorstores, infos, shardings)
    ]

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[types.ParamInfo],
      args: Optional[Sequence[types.SaveArgs]] = None,
  ) -> Sequence[future.Future]:
    """See superclass documentation."""
    args = args or [types.SaveArgs()] * len(values)
    types.check_input_arguments(values, infos, args)
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

    future_list = []
    if self._enable_write_sharding_file:
      future_list.append(
          future.CommitFutureAwaitingContractedSignals(
              _async_serialize_shardings(
                  shardings=[arr.sharding for arr in arrays],
                  infos=infos,
                  primary_host=self._primary_host,
              ),
              name='serialize_shardings',
          )
      )
    future_list.append(
        _serialize_arrays(
            arrays=arrays,
            infos=infos,
            args=args,
            dispatcher=self._dispatcher,
            primary_host=self._primary_host,
            replica_id=self._replica_id,
            use_replica_parallel=self._use_replica_parallel,
            min_slice_bytes_for_replica_parallel=self._min_slice_bytes_for_replica_parallel,
            max_replicas_for_replica_parallel=self._max_replicas_for_replica_parallel,
            enable_replica_parallel_separate_folder=self._enable_replica_parallel_separate_folder,
            metadata_key=self._metadata_key,
            ext_metadata=self._ext_metadata,
            array_metadata_store=self._array_metadata_store,
        )
    )

    return future_list

  async def deserialize(
      self,
      infos: Sequence[types.ParamInfo],
      args: Sequence[types.RestoreArgs] | None = None,
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
    types.check_input_arguments(infos, args)
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING_FILE_NAME
    sharding_file_exists = await async_path.exists(sharding_file_path)
    shardings = await _deserialize_shardings(infos, args, sharding_file_exists)

    if self._dispatcher is None:
      ret = await _deserialize_arrays(
          infos, args, shardings, self._metadata_key, self._array_metadata_store
      )
    else:
      ret = self._dispatcher.dispatch(
          _sync_deserialize_arrays,
          result_specs=_get_abstract_arrays(args, shardings),
          func_kwargs={
              'infos': infos,
              'args': args,
              'shardings': shardings,
              'metadata_key': self._metadata_key,
              'array_metadata_store': self._array_metadata_store,
          },
      )
      jax.block_until_ready(ret)
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
      ts_utils.print_ts_debug_data(self._metadata_key, infos)

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


def _validate_sharding_and_get_primary_replica_processes(
    replica_axis_index: int,
    primary_replica_id: int,
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
          replica_axis_idx=replica_axis_index,
          mesh=sharding.mesh,
          primary_replica_id=primary_replica_id,
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


async def _single_replica_deserialize_and_broadcast(
    infos: Sequence[types.ParamInfo],
    args: Sequence[SingleReplicaArrayRestoreArgs],
    shardings: Sequence[jax.sharding.Sharding],
    single_replica_shardings: Sequence[jax.sharding.Sharding],
    replica_axis_index: int,
    primary_replica_id: int,
    metadata_key: str | None,
    broadcast_memory_limit_bytes: int | None,
    broadcast_memory_scaling_factor: float | None,
) -> Sequence[jax.Array]:
  """Deserializes and broadcasts a single replica."""
  primary_replica_pids = _validate_sharding_and_get_primary_replica_processes(
      replica_axis_index=replica_axis_index,
      primary_replica_id=primary_replica_id,
      sharding=shardings[0],
  )
  if _is_host_for_primary_replica(primary_replica_pids):
    start_deserialization = time.time()
    deserialized = await _deserialize_arrays(
        infos,
        args,
        single_replica_shardings,
        metadata_key,
        None,
    )
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

    @functools.partial(
        jax.jit, static_argnums=0, out_shardings=tuple(single_replica_shardings)
    )
    def create_zeros(shape_dtype_tup):
      return jax.tree.map(
          lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
      )

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
      replica_axis_index,
      _is_host_for_primary_replica(primary_replica_pids),
      memory_limit_bytes=broadcast_memory_limit_bytes,
      memory_scaling_factor=broadcast_memory_scaling_factor,
  )
  broadcast_elapsed_s = time.time() - start_broadcast
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/read/broadcast_duration_secs', broadcast_elapsed_s
  )
  logging.info('Finished broadcasting in %.2f', broadcast_elapsed_s)

  return shared_state


def _single_replica_deserialize_on_worker(
    _,
    infos: Sequence[types.ParamInfo],
    args: Sequence[SingleReplicaArrayRestoreArgs],
    single_replica_shardings: Sequence[jax.sharding.Sharding],
    metadata_key: str | None,
):
  """Deserializes a single replica on a worker."""
  return asyncio_utils.run_sync(
      _deserialize_arrays(
          infos,
          args,
          single_replica_shardings,
          metadata_key,
          None,
      )
  )


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
      dispatcher: dispatchers.Dispatcher | None = None,
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
      dispatcher: The dispatcher to use for executing operations on workers if
        provided.
    """

    super(SingleReplicaArrayHandler, self).__init__(
        use_replica_parallel=use_replica_parallel,
        enable_write_sharding_file=enable_write_sharding_file,
        array_metadata_store=array_metadata_store,
        dispatcher=dispatcher,
    )
    self.replica_axis_index = replica_axis_index
    self.primary_replica_id = primary_replica_id
    self.broadcast_memory_limit_bytes = broadcast_memory_limit_bytes
    self.broadcast_memory_scaling_factor = broadcast_memory_scaling_factor

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
    types.check_input_arguments(infos, args)
    for arg in args:
      if not isinstance(arg, SingleReplicaArrayRestoreArgs):
        raise ValueError(
            'Must provide `SingleReplicaArrayRestoreArgs`, but got'
            f' {type(arg)}.'
        )
      if arg.sharding is None:
        raise ValueError('Must provide `sharding`.')
      if arg.single_replica_sharding is None:
        raise ValueError('Must provide `single_replica_sharding`.')

    single_replica_shardings = [arg.single_replica_sharding for arg in args]
    shardings = [arg.sharding for arg in args]

    if self._dispatcher is None:
      ret = await _single_replica_deserialize_and_broadcast(
          infos,
          args,
          shardings,
          single_replica_shardings,
          self.replica_axis_index,
          self.primary_replica_id,
          self._metadata_key,
          self.broadcast_memory_limit_bytes,
          self.broadcast_memory_scaling_factor,
      )
    else:
      primary_replica_devices = multislice.replica_devices(
          shardings[0].mesh,
          replica_id=self.primary_replica_id,
          replica_axis_index=self.replica_axis_index,
      ).flatten()
      dummy_input_array = dispatchers.get_dummy_input_array(
          primary_replica_devices
      )
      # Step 1: Deserialize arrays on a single replica.
      ret = self._dispatcher.dispatch(
          _single_replica_deserialize_on_worker,
          input_arrays=dummy_input_array,
          result_specs=_get_abstract_arrays(args, single_replica_shardings),
          func_kwargs={
              'infos': infos,
              'args': args,
              'single_replica_shardings': single_replica_shardings,
              'metadata_key': self._metadata_key,
          },
      )
      # Step 2: Use `jax.device_put` to broadcast/reshard the data to all
      # devices according to the final desired sharding. This is the equivalent
      # of multislice.broadcast_one_replica_to_all in non-dispatcher based
      # implementation.
      ret = jax.tree.map(jax.device_put, ret, shardings)
      jax.block_until_ready(ret)

    return ret

  # TODO(b/370396118): Calculation overestimates bytes read.
  def memory_size(  # pylint: disable=useless-parent-delegation
      self, values: Sequence[jax.Array]
  ) -> Sequence[Tuple[int, int]]:
    return super().memory_size(values)
