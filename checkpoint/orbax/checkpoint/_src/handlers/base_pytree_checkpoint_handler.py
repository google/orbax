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

"""BasePyTreeCheckpointHandler class.

Implementation of `CheckpointHandler` interface dealing with JAX PyTrees. Much
of the underlying reading/writing logic for individual leaf types can be
customized, and is delegated to the `TypeHandler` class.
"""

from __future__ import annotations

import asyncio
from collections.abc import Set
import dataclasses
import functools
import json
import threading
import time
from typing import Any, List, Mapping, Optional, Tuple, Union

from absl import logging
from etils import epath
import humanize
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import partial_save_utils
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import metadata_manager as metadata_manager_lib
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.path import types as path_types
from orbax.checkpoint._src.serialization import async_io_engine
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import memory_regulator
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handler_registry as type_handler_registry_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint._src.tree import types as tree_types
from orbax.checkpoint._src.tree import utils as tree_utils
import tensorstore as ts



PyTree = Any
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
SaveArgs = type_handlers.SaveArgs
ParamInfo = types.ParamInfo
TypeHandlerRegistry = types.TypeHandlerRegistry
BatchRequest = async_io_engine.BatchRequest
BatchRequests = async_io_engine.BatchRequests
CommitFutures = async_io_engine.CommitFutures

# TODO(b/298487158) Clean up protected access.
CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler
get_param_names = tree_utils.get_param_names
PYTREE_METADATA_FILE = format_utils.PYTREE_METADATA_FILE
PLACEHOLDER = type_handlers.PLACEHOLDER
TMP_DIR_SUFFIX = atomicity_types.TMP_DIR_SUFFIX
PartialSaveReplacementError = partial_save_utils.PartialSaveReplacementError




def batched_serialization_requests(
    tree: PyTree,
    param_infos: PyTree,
    args: PyTree,
    registry: TypeHandlerRegistry,
) -> BatchRequests:
  """Gets a list of batched serialization or deserialization requests."""
  grouped = {}

  def _group_value(
      keypath: Tuple[Any, ...],
      info: ParamInfo,
      value: Union[Any, tree_metadata.ValueMetadataEntry],
      arg: Union[SaveArgs, RestoreArgs],
  ):
    nonlocal grouped
    tuple_key = tree_utils.tuple_path_from_keypath(keypath)
    if info.skip_deserialize:
      return

    if not isinstance(arg, (SaveArgs, RestoreArgs)):
      if tree_utils.is_empty_node(arg):
        return

    if isinstance(arg, RestoreArgs):
      assert isinstance(value, tree_metadata.ValueMetadataEntry), type(value)
      metadata_restore_type = value.value_type
      requested_restore_type = arg.restore_type or metadata_restore_type
      # TODO(cpgaffney): Add a warning message if the requested_restore_type
      # is not the same as the metadata_restore_type.
      if empty_values.is_empty_typestr(requested_restore_type):
        # Skip deserialization of empty node using TypeHandler.
        return
      type_for_registry_lookup = requested_restore_type
    elif isinstance(arg, SaveArgs):
      # Skip serialization of empty node using TypeHandler.
      if tree_utils.is_empty_node(value):
        return
      type_for_registry_lookup = type(value)
    else:
      raise AssertionError(
          f'Expected `RestoreArgs` or `SaveArgs`. Got {type(arg)}.'
      )

    try:
      handler = registry.get(type_for_registry_lookup)
    except ValueError as e:
      raise ValueError(
          f'TypeHandler lookup failed for: type={type_for_registry_lookup},'
          f' keypath={keypath}, ParamInfo={info}, RestoreArgs={arg},'
          f' value={value}'
      ) from e

    if handler not in grouped:
      grouped[handler] = BatchRequest(handler, [], [], [], [])
    request = grouped[handler]
    grouped[handler] = dataclasses.replace(
        request,
        keys=request.keys + [tuple_key],
        values=request.values + [value],
        infos=request.infos + [info],
        args=request.args + [arg],
    )

  jax.tree_util.tree_map_with_path(
      _group_value,
      param_infos,
      tree,
      args,
  )
  return list(grouped.values())


def _update_array_restore_args(
    v: Any, leaf_args: ArrayRestoreArgs
) -> ArrayRestoreArgs:
  """Updates ArrayRestoreArgs with global shape and dtype."""
  if isinstance(v, type):
    return leaf_args
  is_array = getattr(v, 'shape', False) and getattr(v, 'dtype', False)
  is_prng_key = jax.dtypes.issubdtype(
      getattr(v, 'dtype', None), jax.dtypes.prng_key
  )
  if is_array and not is_prng_key:
    updates = {}
    if leaf_args.strict:
      if leaf_args.global_shape is None and leaf_args.shape is None:
        updates['global_shape'] = getattr(v, 'shape', None)
      if getattr(leaf_args, 'dtype', None) is None:
        updates['dtype'] = getattr(v, 'dtype', None)
    if updates:
      return dataclasses.replace(leaf_args, **updates)
  return leaf_args


def _fill_missing_save_or_restore_args(
    item: PyTree, args: Optional[PyTree], *, mode: str
) -> PyTree:
  """Fills in missing values in the tree of SaveArgs or RestoreArgs.

  Values may be "missing" because of empty nodes in `item`. After returning, all
  keys in `item`, with empty nodes or not, will have a corresponding value
  in the result.

  Args:
    item: tree to save or target to restore.
    args: tree of SaveArgs or RestoreArgs. May be None, if the user did not
      provide it.
    mode: 'save' or 'restore'.

  Returns:
    A tree of SaveArgs or RestoreArgs with missing values filled in.
  """

  # Because of empty states, the user-provided args may not contain
  # all necessary arguments. These should be filled in with default args.
  def _maybe_set_default_save_restore_args(v, leaf_args):
    if mode == 'save':
      return leaf_args if isinstance(leaf_args, SaveArgs) else SaveArgs()
    if mode == 'restore':
      if isinstance(leaf_args, ArrayRestoreArgs):
        return _update_array_restore_args(v, leaf_args)
      return leaf_args if isinstance(leaf_args, RestoreArgs) else RestoreArgs()
    raise ValueError(f'Unknown mode: {mode}.')

  return jax.tree_util.tree_map(
      _maybe_set_default_save_restore_args,
      item,
      item if args is None else args,
      is_leaf=utils.is_empty_or_leaf,
  )




def _format_bytes(bytes_value: Optional[int]) -> str:
  return (
      'None'
      if bytes_value is None
      else f'{bytes_value} ({humanize.naturalsize(bytes_value, binary=True)})'
  )


def _is_prefix(t1: Tuple[Any, ...], t2: Tuple[Any, ...]) -> bool:
  """Checks if tuple t1 is a prefix of tuple t2."""
  return len(t1) < len(t2) and t2[: len(t1)] == t1


async def _get_partial_save_additions(
    merged_metadata: list[tree_metadata.InternalTreeMetadataEntry],
    flat_item: Mapping[Any, Any],
) -> set[Any]:
  """Returns the set of additions for partial saving."""
  merged_tuples_set = {
      tree_utils.tuple_path_from_keypath(entry.jax_keypath())
      for entry in merged_metadata
  }

  def _validate_key(key, merged_tuples_set=merged_tuples_set):
    is_exact_match = key in merged_tuples_set
    has_prefix_conflict = isinstance(key, tuple) and any(
        isinstance(mt, tuple) and (_is_prefix(key, mt) or _is_prefix(mt, key))
        for mt in merged_tuples_set
    )
    if is_exact_match or has_prefix_conflict:
      raise PartialSaveReplacementError(
          f'Key "{key!r}" was found in a previous partial save in this session.'
          ' Partial saving currently does not support REPLACEMENT.'
      )
    return key

  additions = {_validate_key(key) for key in flat_item}
  return additions


def _filter_batch_requests(
    batch_requests: BatchRequests,
    additions: Set[Any],
) -> BatchRequests:
  """Filters batch requests to include only items matching the additions."""
  filtered_requests = []
  for request in batch_requests:
    filtered_items = []
    for key, value, info, arg in zip(
        request.keys, request.values, request.infos, request.args
    ):
      for add in additions:
        # Additions may be a prefix/parent of the key.
        if add == key[: len(add)]:
          filtered_items.append((key, value, info, arg))
    if filtered_items:
      keys, values, infos, args = zip(*filtered_items)
      filtered_requests.append(
          dataclasses.replace(
              request,
              keys=list(keys),
              values=list(values),
              infos=list(infos),
              args=list(args),
          )
      )
  return filtered_requests


class BasePyTreeCheckpointHandler(
    async_checkpoint_handler.DeferredPathAsyncCheckpointHandler
):
  """A CheckpointHandler implementation for any PyTree structure.

  Largely serves as the implementation for `PyTreeCheckpointHandler`. Users are
  advised not to use this class directly.
  """

  def __init__(
      self,
      *,
      save_concurrent_bytes: Optional[int] = None,
      restore_concurrent_bytes: Optional[int] = None,
      save_device_host_concurrent_bytes: int | str | None = None,
      memory_limit_options: options_lib.MemoryLimitOptions | None = None,
      use_ocdbt: bool = True,
      use_zarr3: bool = False,
      use_compression: bool = True,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      type_handler_registry: TypeHandlerRegistry = type_handler_registry_lib.GLOBAL_TYPE_HANDLER_REGISTRY,
      enable_post_merge_validation: bool = True,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
          tree_metadata.PYTREE_METADATA_OPTIONS
      ),
      array_metadata_validator: array_metadata_store_lib.Validator = (
          array_metadata_store_lib.Validator()
      ),
      enable_pinned_host_transfer: Optional[bool] = None,
      is_prioritized_key_fn: Optional[types.IsPrioritizedKeyFn] = None,
      metadata_manager: Optional[metadata_manager_lib.MetadataManager] = None,
  ):
    """Creates BasePyTreeCheckpointHandler.

    Args:
      save_concurrent_bytes: max concurrent bytes that are allowed to be
        written. Can help to reduce the possibility of OOM's when large
        checkpoints are saved. Note that this also applies when arrays are
        tranferred to host memory, and so can result in a slowdown of async
        saves.
      restore_concurrent_bytes: max concurrent bytes that are allowed to be
        restored. Can help to reduce the possibility of OOM's when large
        checkpoints are restored.
      save_device_host_concurrent_bytes: max concurrent bytes allowed to be
        transferred from device to host memory at once when saving. When the
        limit is reached, arrays must be finished writing to the checkpoint
        before a new array can start being transferred. Can be "auto".
      memory_limit_options: Options for configuring memory limits for save. Can
        help to reduce the possibility of OOM's when checkpoints are saved.
      use_ocdbt: Whether to use OCDBT format for saving.
      use_zarr3: If True, use Zarr ver3 otherwise Zarr ver2.
      use_compression: If True, use zstd compression.
      multiprocessing_options: See orbax.checkpoint.options.
      type_handler_registry: a type_handlers.TypeHandlerRegistry. If not
        specified, the global type handler registry will be used.
      enable_post_merge_validation: If True, enables validation of the
        parameters after the finalize step.
      pytree_metadata_options: `PyTreeMetadataOptions` to manage metadata.
      array_metadata_validator: Validator for ArrayMetadata.
      enable_pinned_host_transfer: Whether to use pinned_host memory for the
        transfer from device to host memory. Passing None will enable
        pinned_host memory depending on the platform used (currently only
        enables it for the GPU backend).
      is_prioritized_key_fn: A function that accepts a PyTree keypath (obtained
        using jax.tree.map_with_path) that should be scheduled for D2H transfer
        before other keys. The transfer is scheduled before returning to the
        caller, so the values will never be corrupted by a concurrent update.
        Keys that are not prioritized will not be scheduled for transfer until
        all prioritized keys have been fully written to the checkpoint. This
        means that these values may be altered if the values are updated
        concurrently. Callers should take care to call `wait_until_finished`
        before updating array values (e.g. `apply_gradients`) if some keys are
        not prioritized. Note that any "prioritized" keys are assumed to be
        lightweight, and `save_device_host_concurrent_gb` will be ignored for
        them.
      metadata_manager: Optional `MetadataManager` instance to manage
        persistence.
    """
    self._save_concurrent_bytes = save_concurrent_bytes
    self._restore_concurrent_bytes = restore_concurrent_bytes
    self._save_device_host_concurrent_bytes = save_device_host_concurrent_bytes
    self._max_save_device_host_concurrent_bytes = None
    if memory_limit_options is not None:
      if memory_limit_options.max_transfer_concurrent_gb is not None:
        self._max_save_device_host_concurrent_bytes = (
            memory_limit_options.max_transfer_concurrent_gb * 10**9
        )

    if self._save_device_host_concurrent_bytes == 'auto':
      if self._max_save_device_host_concurrent_bytes is None:
        raise ValueError(
            'max_save_device_host_concurrent_bytes must be provided if'
            ' save_device_host_concurrent_bytes is "auto"'
        )
      max_memory_limit_gib = self._max_save_device_host_concurrent_bytes / (
          1024**3
      )
      self._memory_regulator = memory_regulator.MemoryRegulator(
          max_memory_limit_gib=max_memory_limit_gib,
      )
      self._current_device_host_limit_bytes = int(
          self._memory_regulator.min_memory_limit_gib * 1024**3
      )
    self._use_ocdbt = use_ocdbt
    self._use_zarr3 = use_zarr3
    self._use_compression = use_compression
    self._primary_host = multiprocessing_options.primary_host
    self._type_handler_registry = type_handler_registry
    self._enable_post_merge_validation = enable_post_merge_validation
    self._pytree_metadata_options = pytree_metadata_options
    # Get ArrayMetadata Store from TypeHandler for jax.Array.
    # ArrayMetadata persistence is only supported for jax.Array.
    self._array_metadata_store = (
        array_metadata_store_lib.resolve_array_metadata_store(
            type_handler_registry
        )
    )
    if self._array_metadata_store:
      self._array_metadata_store.set_primary_host(self._primary_host)
    self._array_metadata_validator = array_metadata_validator
    self._metadata_manager = (
        metadata_manager
        if metadata_manager is not None
        else metadata_manager_lib.MetadataManager()
    )

    if enable_pinned_host_transfer is None:
      enable_pinned_host_transfer = jax.default_backend() == 'gpu'
    self._enable_pinned_host_transfer = enable_pinned_host_transfer
    self._is_prioritized_key_fn = is_prioritized_key_fn
    if self._is_prioritized_key_fn:
      jax.monitoring.record_event(
          '/jax/orbax/pytree_checkpoint_handler/init/prioritized_key_fn'
      )

    jax.monitoring.record_event(
        '/jax/orbax/pytree_checkpoint_handler/init/ocdbt'
    )
    logging.info(
        'Created BasePyTreeCheckpointHandler: use_ocdbt=%s, use_zarr3=%s,'
        ' pytree_metadata_options=%s, array_metadata_store=%s,'
        ' enable_pinned_host_transfer=%s, save_concurrent_bytes: %s,'
        ' restore_concurrent_bytes: %s',
        self._use_ocdbt,
        self._use_zarr3,
        self._pytree_metadata_options,
        self._array_metadata_store,
        self._enable_pinned_host_transfer,
        _format_bytes(self._save_concurrent_bytes),
        _format_bytes(self._restore_concurrent_bytes),
    )
    self._async_io_engine = async_io_engine.AsyncIoEngine()

  def get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    return get_param_names(item)

  def _get_param_infos(
      self,
      item: PyTree,
      directory: epath.Path,
      *,
      use_ocdbt: bool = True,
      use_compression: bool | None = True,
      use_zarr3: Optional[bool] = None,
      ocdbt_target_data_file_size: Optional[int] = None,
      byte_limiter: Optional[limits.ByteLimiter] = None,
      device_host_byte_limiter: Optional[limits.ByteLimiter] = None,
      raise_array_data_missing_error: bool = True,
  ) -> PyTree:
    """Returns parameter information for elements in `item`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.
      use_ocdbt: Whether to use OCDBT for writing or reading.
      use_compression: Whether to use zstd compression
      use_zarr3: Whether to use zarr3.
      ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
        OCDBT data file.
      byte_limiter: ByteLimiter object.
      device_host_byte_limiter: ByteLimiter object for device-to-host transfer.
      raise_array_data_missing_error: See documentation in ParamInfo.

    Returns:
      A PyTree matching `item` of ParamInfo.
    """
    if use_zarr3 is None:
      use_zarr3 = self._use_zarr3
    names = self.get_param_names(item)
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)

    def _param_info(keypath, name, value):
      if isinstance(value, tree_metadata.ValueMetadataEntry):
        skip_deserialize = value.skip_deserialize
      elif isinstance(value, type(PLACEHOLDER)):
        skip_deserialize = True
      else:
        skip_deserialize = False
      return ParamInfo(
          name=name,
          keypath=keypath,
          parent_dir=directory,
          skip_deserialize=skip_deserialize,
          is_ocdbt_checkpoint=use_ocdbt,
          use_compression=use_compression,
          use_zarr3=use_zarr3,
          enable_pinned_host_transfer=self._enable_pinned_host_transfer,
          ocdbt_target_data_file_size=ocdbt_target_data_file_size,
          byte_limiter=byte_limiter,
          device_host_byte_limiter=device_host_byte_limiter,
          ts_context=ts_context,
          value_typestr=type_handler_registry_lib.get_param_typestr(
              value, self._type_handler_registry, self._pytree_metadata_options
          ),
          raise_array_data_missing_error=raise_array_data_missing_error,
          is_prioritized_key_fn=self._is_prioritized_key_fn,
      )

    return jax.tree.map_with_path(
        _param_info, names, item, is_leaf=utils.is_empty_or_leaf
    )

  async def _get_partial_save_batch_requests(
      self,
      directory: epath.Path,
      item: PyTree,
      batch_requests: BatchRequests,
  ) -> BatchRequests:
    flat_item = tree_utils.to_flat_dict(item)
    additions = await partial_save_utils.get_partial_save_additions(
        directory, flat_item, self._pytree_metadata_options
    )
    return _filter_batch_requests(batch_requests, additions)

  async def async_save(
      self,
      directory: epath.Path | path_types.PathAwaitingCreation,
      args: BasePyTreeSaveArgs,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree to a given directory.

    This operation is compatible with a multi-host, multi-device setting. Tree
    leaf values must be supported by the type_handler_registry given in the
    constructor. Standard supported types include Python scalars, `np.ndarray`,
    `jax.Array`, and strings.

    After saving, all files will be located in "directory/".
    A JSON metadata file will be present to store the tree structure.

    Example usage::

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      item = {
          'layer0': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
          'layer1': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
      }
      # Note: save_args may be None if no customization is desired for saved
      # parameters.
      # Eventually calls through to `async_save`.
      ckptr.save(path, item, save_args)

    Args:
      directory: save location directory.
      args: `BasePyTreeSaveArgs` (see below).

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    start_time = time.time()
    initial_ts_metrics = ts.experimental_collect_matching_metrics(
        '/tensorstore/'
    )
    item = args.item
    # Reject only zero-leaf items (empty containers, None). A single falsy leaf
    # (0, '', False, zero-array) is a valid one-leaf tree and must be allowed.
    if not jax.tree.leaves(item) and not item:
      raise ValueError('Found empty item.')
    save_args = args.save_args
    ocdbt_target_data_file_size = args.ocdbt_target_data_file_size
    custom_metadata = args.custom_metadata

    save_args = _fill_missing_save_or_restore_args(item, save_args, mode='save')
    byte_limiter = limits.get_byte_limiter(self._save_concurrent_bytes)

    device_host_concurrent_bytes = self._save_device_host_concurrent_bytes
    if device_host_concurrent_bytes == 'auto':
      self._current_device_host_limit_bytes = (
          self._memory_regulator.update_limit_bytes(
              self._current_device_host_limit_bytes
          )
      )
      device_host_byte_limiter = limits.get_byte_limiter(
          self._current_device_host_limit_bytes
      )
    else:
      device_host_byte_limiter = limits.get_byte_limiter(
          device_host_concurrent_bytes
      )
    param_infos = self._get_param_infos(
        item,
        directory,
        use_ocdbt=self._use_ocdbt,
        ocdbt_target_data_file_size=ocdbt_target_data_file_size,
        byte_limiter=byte_limiter,
        device_host_byte_limiter=device_host_byte_limiter,
        use_compression=self._use_compression,
        use_zarr3=self._use_zarr3,
    )
    # TODO(b/425293362): Add validation for PathAwaitingCreation.
    if isinstance(directory, epath.Path):
      assert all(
          leaf.parent_dir is directory for leaf in jax.tree.leaves(param_infos)
      )

    batch_requests = batched_serialization_requests(
        item,
        param_infos,
        save_args,
        self._type_handler_registry,
    )

    batch_requests_ready_time = time.time()
    if args.partial_save_mode:
      requests_to_save = await self._get_partial_save_batch_requests(
          directory, item, batch_requests
      )
    else:
      requests_to_save = batch_requests

    tree_memory_size = async_io_engine.compute_save_memory_size(
        requests_to_save
    )
    commit_futures = await self._async_io_engine.execute_save(requests_to_save)
    # Flatten to List[future.Future].
    commit_futures, _ = jax.tree.flatten(commit_futures)

    total_serialization_initiated_time = time.time()
    if logging.vlog_is_on(1):
      logging.vlog(1, 'param_info: %s', param_infos)
      logging.vlog(1, 'save_args: %s', save_args)

    save_futures = []
    if multihost.is_primary_host(self._primary_host):
      save_futures.append(
          future.CommitFutureAwaitingContractedSignals(
              self._write_metadata_after_commits(
                  commit_futures,
                  param_infos=param_infos,
                  save_args=save_args,
                  custom_metadata=custom_metadata,
                  use_ocdbt=self._use_ocdbt,
                  use_zarr3=self._use_zarr3,
              ),
              name='write_metadata_after_commits',
          )
      )
    else:
      save_futures += commit_futures


    async_io_engine.log_io_metrics(
        tree_memory_size,
        start_time,
        '/jax/orbax/write/blocking_gbytes_per_sec',
    )
    chained_futures = [
        future.ChainedFuture(
            save_futures,
            functools.partial(
                async_io_engine.log_io_metrics,
                tree_memory_size,
                start_time,
                '/jax/orbax/write/gbytes_per_sec',
                '/jax/orbax/write/gbytes',
                initial_ts_metrics=initial_ts_metrics,
            ),
        )
    ]
    async_save_end_time = time.time()
    logging.info(
        '[process=%s][thread=%s] Initiated Pytree async_save. Time taken:'
        ' %fs (batch_requests_ready=%fs, total_serialization_initiated=%fs,'
        ' others=%fs)',
        multihost.process_index(),
        threading.current_thread().name,
        async_save_end_time - start_time,
        batch_requests_ready_time - start_time,
        total_serialization_initiated_time - batch_requests_ready_time,
        async_save_end_time - total_serialization_initiated_time,
    )
    return chained_futures

  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves the provided item.

    Blocks until both copy and commit complete.

    See async_save.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio_utils.run_sync(async_save(directory, *args, **kwargs))

  async def _maybe_deserialize(
      self,
      item: PyTree,
      metadata: PyTree,
      param_infos: PyTree,
      restore_args: PyTree,
  ) -> Tuple[int, PyTree]:
    """Deserializes values or skips."""
    flat_metadata = tree_utils.to_flat_dict(metadata)
    byte_limiter = limits.get_byte_limiter(self._restore_concurrent_bytes)
    param_infos = jax.tree.map(
        lambda info: info.replace(byte_limiter=byte_limiter),
        param_infos,
    )
    batch_requests = batched_serialization_requests(
        metadata,
        param_infos,
        restore_args,
        self._type_handler_registry,
    )

    deserialized_batches = await self._async_io_engine.execute_restore(
        batch_requests
    )
    tree_memory_size = async_io_engine.compute_restore_memory_size(
        batch_requests, deserialized_batches
    )

    flat_restored = {}
    for request, deserialized in zip(batch_requests, deserialized_batches):
      for key, value in zip(request.keys, deserialized):
        flat_restored[key] = value

    # Add in empty nodes from the metadata tree.
    for key in flat_metadata.keys():
      if key not in flat_restored:
        if type_handlers.is_placeholder(flat_metadata[key]):
          flat_restored[key] = type_handlers.PLACEHOLDER
        else:
          flat_restored[key] = empty_values.get_empty_value_from_typestr(
              flat_metadata[key].value_type, self._pytree_metadata_options
          )

    # Restore using `item` as the target structure. If there are any custom
    # nodes (e.g. optax.EmptyState), these will replace None values in
    # flat_restored.
    return tree_memory_size, tree_utils.from_flat_dict(
        flat_restored, target=item
    )

  def _partial_restore_with_omission(
      self,
      item: PyTree,
      serialized_item: PyTree,
      value_metadata_tree: PyTree,
      restore_args: PyTree,
  ) -> Tuple[PyTree, PyTree]:
    """Restores leaves specified in `item`. Skips omitted leaves."""
    if not self._pytree_metadata_options.support_rich_types:
      # Replace empty containers with scalar values (zeros). During saving,
      # some empty containers (like named tuples) were given
      # ValueMetadataEntries as if they were scalars. We normalize these
      # containers to scalars so that tree_trim is none the wiser.
      serialized_item = jax.tree.map(
          lambda v: 0 if empty_values.is_empty_container(v) else v,
          serialized_item,
          is_leaf=tree_utils.is_empty_or_leaf,
      )

    value_metadata_tree = tree_structure_utils.tree_trim(
        serialized_item, value_metadata_tree, strict=False
    )
    value_metadata_tree = value_metadata_tree.unsafe_structure

    if restore_args is not None:
      restore_args = tree_structure_utils.tree_trim(
          item, restore_args, strict=True
      )

    return value_metadata_tree, restore_args

  def _partial_restore_with_placeholders(
      self, serialized_item: PyTree, value_metadata_tree: PyTree
  ) -> PyTree:
    """Restores leaves from `item`, except for those marked as placeholders."""
    diff = (
        tree_structure_utils.tree_difference(
            serialized_item,
            value_metadata_tree,
            is_leaf=tree_utils.is_empty_or_leaf,
            leaves_equal=lambda a, b: True,
        )
        or {}
    )
    for keypath, value_diff in tree_utils.to_flat_dict(
        diff, is_leaf=lambda x: isinstance(x, tree_structure_utils.Diff)
    ).items():
      if value_diff.lhs is PLACEHOLDER and value_diff.rhs is None:
        parent = value_metadata_tree
        for key in keypath[:-1]:
          parent = parent[key]
        parent[keypath[-1]] = PLACEHOLDER
      else:
        formatted_diff = tree_structure_utils.format_tree_diff(
            diff, source_label='Item', target_label='Metadata'
        )
        raise ValueError(
            'User-provided restore item and on-disk value metadata tree'
            f' structures do not match:\n{formatted_diff}\nIf this mismatch is'
            ' intentional, pass `partial_restore=True` to only restore'
            ' parameters found in `item`.'
        )
    return jax.tree.map(
        lambda v, i: PLACEHOLDER if type_handlers.is_placeholder(i) else v,
        value_metadata_tree,
        serialized_item,
    )

  def restore(
      self,
      directory: epath.Path,
      args: Optional[BasePyTreeRestoreArgs] = None,
  ) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given path.

    In the most basic case, only `directory` is required. The tree will be
    restored exactly as saved, and all leaves will be restored as the correct
    types (assuming the tree metadata is present).

    However, `restore_args` is often required as well. This PyTree gives a
    `RestoreArgs` object (or subclass) for every leaf in the tree. Many types,
    such as string or `np.ndarray` do not require any special options for
    restoration. When restoring an individual leaf as `jax.Array`, however,
    some properties may be required.

    One example is `sharding`, which defines how a `jax.Array` in the restored
    tree should be partitioned. `mesh` and `mesh_axes` can also be used to
    specify `sharding`, but `sharding` is the preferred way of specifying this
    partition since `mesh` and `mesh_axes` only constructs
    `jax.sharding.NamedSharding`. For more information, see `ArrayTypeHandler`
    documentation and JAX sharding documentation.

    Example::

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      restore_args = {
          'layer0': {
              'w': RestoreArgs(),
              'b': RestoreArgs(),
          },
          'layer1': {
              'w': ArrayRestoreArgs(
                  # Restores as jax.Array, regardless of how it was saved.
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  # Warning: may truncate or pad!
                  global_shape=(x, y),
                ),
              'b': ArrayRestoreArgs(
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  global_shape=(x, y),
                ),
          },
      }
      ckptr.restore(path, restore_args=restore_args)

    Providing `item` is typically only necessary when restoring a custom PyTree
    class (or when using transformations). In this case, the restored object
    will take on the same structure as `item`.

    Example::

      @flax.struct.dataclass
      class TrainState:
        layer0: dict[str, jax.Array]
        layer1: dict[str, jax.Array]

      ckptr = Checkpointer(BasePyTreeCheckpointHandler())
      train_state = TrainState(
          layer0={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
          layer1={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
      )
      restore_args = jax.tree.map(_make_restore_args, train_state)
      ckptr.restore(path, item=train_state, restore_args=restore_args)
      # restored tree is of type `TrainState`.

    Args:
      directory: saved checkpoint location directory.
      args: `BasePyTreeRestoreArgs` (see below).

    Returns:
      A PyTree matching the structure of `item`.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `multi_value_fn`.
    """
    start_time = time.time()
    args = args or BasePyTreeRestoreArgs()
    item = args.item
    restore_args = args.restore_args

    logging.vlog(1, 'directory=%s, restore_args=%s', directory, restore_args)
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}'
      )
    # Get value metadata tree and use_zarr3 from serialized pytree metadata.
    internal_tree_metadata = asyncio_utils.run_sync(
        self._metadata_manager.read_metadata_file(
            directory, pytree_metadata_options=self._pytree_metadata_options
        )
    )
    value_metadata_tree = internal_tree_metadata.as_nested_tree()
    if not value_metadata_tree:
      raise ValueError(
          f'Found empty checkpoint PyTree metadata in directory={directory}.'
      )
    use_zarr3 = (
        internal_tree_metadata.use_zarr3
        if internal_tree_metadata.use_zarr3 is not None
        else self._use_zarr3
    )
    use_ocdbt = (
        internal_tree_metadata.use_ocdbt
        if internal_tree_metadata.use_ocdbt is not None
        else type_handlers.is_ocdbt_checkpoint(directory)
    )
    raise_array_data_missing_error = (
        internal_tree_metadata.store_array_data_equal_to_fill_value
    )
    del internal_tree_metadata
    # Prep for restore.
    serialized_item = tree_metadata.serialize_tree(
        item, self._pytree_metadata_options
    )
    if item is None:
      item = value_metadata_tree
    elif args.partial_restore:
      value_metadata_tree, restore_args = self._partial_restore_with_omission(
          item, serialized_item, value_metadata_tree, restore_args
      )
    elif any(
        type_handlers.is_placeholder(leaf) for leaf in jax.tree.leaves(item)
    ):
      value_metadata_tree = self._partial_restore_with_placeholders(
          serialized_item, value_metadata_tree
      )
    else:
      # Deserialize value metadata tree to the same structure as item to allow
      # for comparison with item that contains rich types.
      if self._pytree_metadata_options.support_rich_types:
        value_metadata_tree = tree_utils.deserialize_tree(
            value_metadata_tree, item
        )
      # is_empty_or_leaf is necessary here to treat empty nodes (e.g. empty
      # dicts, lists, custom nodes) as leaves, as they do not contain any
      # actual data to be restored, but are needed to maintain the structure.
      diff = tree_structure_utils.tree_difference(
          serialized_item,
          value_metadata_tree,
          is_leaf=tree_utils.is_empty_or_leaf,
          leaves_equal=lambda a, b: True,
      )
      if diff is not None:
        formatted_diff = tree_structure_utils.format_tree_diff(
            diff, source_label='Item', target_label='Metadata'
        )
        raise ValueError(
            'User-provided restore item and on-disk value metadata tree'
            f' structures do not match:\n{formatted_diff}\nIf this mismatch is'
            ' intentional, pass `partial_restore=True` to only restore'
            ' parameters found in `item`.'
        )
    restore_args = _fill_missing_save_or_restore_args(
        item, restore_args, mode='restore'
    )
    restore_args = tree_metadata.serialize_tree(
        restore_args, self._pytree_metadata_options
    )

    if not self._pytree_metadata_options.support_rich_types:
      value_metadata_tree = tree_utils.deserialize_tree(
          value_metadata_tree, item
      )
    restore_args = tree_utils.deserialize_tree(restore_args, item)

    param_infos = self._get_param_infos(
        item=value_metadata_tree,
        directory=directory,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        raise_array_data_missing_error=raise_array_data_missing_error,
    )
    # Begin restore.
    tree_memory_size, restored_item = asyncio_utils.run_sync(
        self._maybe_deserialize(
            item, value_metadata_tree, param_infos, restore_args
        )
    )

    if args.partial_restore:
      restored_item = jax.tree.map(
          lambda r, i: i if r is type_handlers.PLACEHOLDER else r,
          restored_item,
          item,
          is_leaf=tree_utils.is_empty_or_leaf,
      )

    if logging.vlog_is_on(1):
      logging.vlog(1, 'param_infos: %s', param_infos)
      logging.vlog(1, 'checkpoint_restore_args: %s', restore_args)
      logging.vlog(1, 'restored_item: %s', jax.tree.structure(restored_item))
      logging.vlog(
          1,
          'ts_metrics: %s',
          json.dumps(ts.experimental_collect_matching_metrics('/tensorstore/')),
      )


    async_io_engine.log_io_metrics(
        tree_memory_size,
        start_time,
        '/jax/checkpoint/read/gbytes_per_sec',
        '/jax/checkpoint/read/gbytes',  # device memory usage
    )
    return restored_item

  async def _get_param_infos_with_write_shape(
      self,
      param_infos: PyTree,
      checkpoint_dir: epath.Path,
      array_metadata_store: array_metadata_store_lib.Store,
  ) -> PyTree:
    """Returns `param_infos` updated with `write_shape`.

    Args:
      param_infos: A PyTree of ParamInfo to be updated.
      checkpoint_dir: The checkpoint directory where write_shape metadata is
        saved in ArrayMetadata store.
      array_metadata_store: The ArrayMetadata store to read write_shape metadata
        from.
    """
    if not utils.is_primary_host(self._primary_host):
      return param_infos
    # Extract write_shape from ArrayMetadata for current process_index.
    process_index = multihost.process_index()
    array_metadatas = await array_metadata_store.read(
        checkpoint_dir, process_index=process_index
    )
    if array_metadatas is None:
      jax_array_param_info = type_handlers.any_jax_array_param_info(param_infos)
      if jax_array_param_info is not None:
        raise ValueError(
            f'No ArrayMetadata found for process_index={process_index} in the'
            f' checkpoint directory: {checkpoint_dir}. But input PyTree'
            ' contains at least one jax.Array param_info:'
            f' {jax_array_param_info}.'
        )
      return param_infos

    assert isinstance(array_metadatas, list)
    array_metadatas_cache = {
        array_metadata.param_name: array_metadata
        for array_metadata in array_metadatas
    }

    def update_param_info(param_info: types.ParamInfo) -> types.ParamInfo:
      if not type_handlers.represents_jax_array(param_info):
        return param_info
      if param_info.name not in array_metadatas_cache:
        raise ValueError(
            f'No ArrayMetadata found for param_info: {param_info}, checkpoint'
            f' directory: {checkpoint_dir}, process_index={process_index}.'
        )
      return param_info.replace(
          write_shape=array_metadatas_cache[param_info.name].write_shape
      )

    return jax.tree.map(update_param_info, param_infos)

  async def _write_metadata_after_commits(
      self,
      commit_futures: List[future.Future],
      *,
      param_infos: PyTree,
      save_args: PyTree,
      custom_metadata: tree_types.JsonType | None,
      use_ocdbt: bool,
      use_zarr3: bool,
  ) -> None:
    start_time = time.time()
    if not utils.is_primary_host(self._primary_host):
      return
    for commit_future in commit_futures:
      await asyncio.to_thread(commit_future.result)

    await asyncio.gather(
        *[info.await_path_creation() for info in jax.tree.leaves(param_infos)]
    )
    checkpoint_dir = jax.tree.leaves(param_infos)[0].parent_dir

    commit_time = time.time()
    # `write_shape` is extracted from ArrayMetadata store saved during
    # materialization of commit_futures. Then it is written to the pytree
    # metadata.
    if self._array_metadata_store is not None:
      param_infos = (
          await self._metadata_manager.get_param_infos_with_write_shape(
              param_infos,
              checkpoint_dir,
              array_metadata_store=self._array_metadata_store,
              primary_host=self._primary_host,
          )
      )

    internal_tree_metadata = tree_metadata.InternalTreeMetadata.build(
        param_infos,
        save_args=save_args,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        custom_metadata=custom_metadata,
        pytree_metadata_options=self._pytree_metadata_options,
    )
    await self._metadata_manager.write_metadata_file(
        checkpoint_dir,
        internal_tree_metadata,
        primary_host=self._primary_host,
        pytree_metadata_options=self._pytree_metadata_options,
    )
    end_time = time.time()
    logging.info(
        '[process=%s][thread=%s] Commit + Array metadata written. Time taken:'
        ' %fs (commit=%fs, array_metadata_write=%fs)',
        multihost.process_index(),
        threading.current_thread().name,
        end_time - start_time,
        commit_time - start_time,
        end_time - commit_time,
    )

  def metadata(self, directory: epath.Path) -> tree_metadata.TreeMetadata:
    """Returns tree metadata.

    The result will be a PyTree matching the structure of the saved checkpoint.
    Note that if the item saved was a custom class, the restored metadata will
    be returned as a nested dictionary representation.

    Example::

      {
        'layer0': {
            'w': ArrayMetadata(dtype=jnp.float32, shape=(8, 8), shards=(1, 2)),
            'b': ArrayMetadata(dtype=jnp.float32, shape=(8,), shards=(1,)),
        },
        'step': ScalarMetadata(dtype=jnp.int64),
      }

    If the required metadata file is not present, this method will raise an
    error.

    Args:
      directory: checkpoint location.

    Returns:
      tree containing metadata.
    """
    internal_tree_metadata = asyncio_utils.run_sync(
        self._metadata_manager.read_metadata_file(
            directory, pytree_metadata_options=self._pytree_metadata_options
        )
    )
    return tree_metadata.build_default_tree_metadata(
        internal_tree_metadata.as_custom_metadata(
            directory,
            self._type_handler_registry,
        ),
        custom_metadata=internal_tree_metadata.custom_metadata,
        use_zarr3=internal_tree_metadata.use_zarr3,
    )

  def finalize(self, directory: epath.Path) -> None:
    """Finalization step.

    Called automatically by the Checkpointer/AsyncCheckpointer just before the
    checkpoint is considered "finalized" in the sense of ensuring atomicity. See
    documentation for `type_handlers.merge_ocdbt_per_process_files`.

    Args:
      directory: Path where the checkpoint is located.
    """
    asyncio_utils.run_sync(
        self._metadata_manager.finalize_async(
            directory,
            array_metadata_store=self._array_metadata_store,
            primary_host=self._primary_host,
            array_metadata_validator=self._array_metadata_validator,
            use_zarr3=self._use_zarr3,
            enable_post_merge_validation=self._enable_post_merge_validation,
        )
    )


@register_with_handler(BasePyTreeCheckpointHandler, for_save=True)
@dataclasses.dataclass
class BasePyTreeSaveArgs(CheckpointArgs):
  """Parameters for saving a PyTree.

  Attributes:
    item (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `item`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
    ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
      OCDBT data file.  It only applies when OCDBT is enabled and Zarr3 must be
      turned on.  If left unspecified, default size is 2GB.  A value of 0
      indicates no maximum file size limit.  For best results, ensure
      chunk_byte_size is smaller than this value.  For more details, refer to
      https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
    partial_save_mode: When True, signals that this save is a partial save
      operation. The handler will merge the new data with existing checkpoint
  """

  item: PyTree
  save_args: Optional[PyTree] = None
  ocdbt_target_data_file_size: Optional[int] = None
  custom_metadata: tree_types.JsonType | None = None
  partial_save_mode: bool = False


@register_with_handler(BasePyTreeCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class BasePyTreeRestoreArgs(CheckpointArgs):
  """Parameters for restoring a PyTree.

  Attributes (all optional):
    item: provides the tree structure for the restored item. If not provided,
      will infer the structure from the saved checkpoint. Transformations will
      not be run in this case. Necessary particularly in the case where the
      caller needs to restore the tree as a custom object.
    restore_args: optional object containing additional arguments for
      restoration. It should be a PyTree matching the structure of `item`, or
      if `item` is not provided, then it should match the structure of the
      checkpoint. Each value in the tree should be a `RestoreArgs` object (OR
      a subclass of `RestoreArgs`). Importantly, note that when restoring a
      leaf as a certain type, a specific subclass of `RestoreArgs` may be
      required. `RestoreArgs` also provides the option to customize the
      restore type of an individual leaf.
      partial_restore: If True, only restore the parameters that are specified
      in PyTreeRestoreArgs.
  """

  item: Optional[PyTree] = None
  restore_args: Optional[PyTree] = None
  partial_restore: bool = False
