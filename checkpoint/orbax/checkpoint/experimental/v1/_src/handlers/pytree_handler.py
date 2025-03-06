# Copyright 2024 The Orbax Authors.
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

"""Implementation of `CheckpointableHandler` for PyTrees."""

import dataclasses
from typing import Awaitable, Sequence

from etils import epath
import jax
import numpy as np
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import options
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

PathLike = path_types.PathLike
AsyncResponse = async_types.AsyncResponse
CheckpointableHandler = handler_types.CheckpointableHandler
PyTree = tree_types.PyTree


def _get_legacy_compatible_save_args(
    checkpointable: PyTree,
    create_array_storage_options_fn: options.CreateArrayStorageOptionsFn | None,
) -> PyTree | None:
  """Returns save args that are compatible with the V0 API."""
  if create_array_storage_options_fn is None:
    return None

  def _leaf_get_legacy_compatible_save_args(k, v):
    array_storage_options = create_array_storage_options_fn(k, v)
    save_dtype = (
        np.dtype(array_storage_options.dtype)
        if array_storage_options.dtype
        else None
    )
    return type_handlers.SaveArgs(
        dtype=save_dtype,
        chunk_byte_size=array_storage_options.chunk_byte_size,
        shard_axes=array_storage_options.shard_axes,
    )

  return jax.tree.map_with_path(
      _leaf_get_legacy_compatible_save_args, checkpointable
  )


class PyTreeHandler(
    CheckpointableHandler[
        PyTree,
        PyTree,
        PyTree,
    ]
):
  """An implementation of `CheckpointableHandler` for PyTrees."""

  # TODO(b/398249409): Currently `PyTreeHandler` is not used by higher-level
  # code. Many of the options here need to be moved to the `Context`.
  def __init__(
      self,
      *,
      create_array_storage_options_fn: (
          options.CreateArrayStorageOptionsFn | None
      ) = None,
      save_concurrent_bytes: int | None = None,
      restore_concurrent_bytes: int | None = None,
      use_ocdbt: bool = True,
      use_zarr3: bool = True,
      enable_padding_and_truncation: bool = False,
      ocdbt_target_data_file_size: int | None = None,
      enable_pinned_host_transfer: bool = False,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      type_handler_registry: type_handlers.TypeHandlerRegistry = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
      enable_post_merge_validation: bool = True,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
          tree_metadata.PYTREE_METADATA_OPTIONS
      ),
      array_metadata_validator: array_metadata_store_lib.Validator = (
          array_metadata_store_lib.Validator()
      ),
  ):
    self._handler_impl = base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler(
        save_concurrent_bytes=save_concurrent_bytes,
        restore_concurrent_bytes=restore_concurrent_bytes,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        multiprocessing_options=multiprocessing_options,
        type_handler_registry=type_handler_registry,
        enable_post_merge_validation=enable_post_merge_validation,
        pytree_metadata_options=pytree_metadata_options,
        array_metadata_validator=array_metadata_validator,
    )
    self._create_array_storage_options_fn = create_array_storage_options_fn
    self._ocdbt_target_data_file_size = ocdbt_target_data_file_size
    self._enable_pinned_host_transfer = enable_pinned_host_transfer
    self._multiprocessing_options = multiprocessing_options
    self._enable_padding_and_truncation = enable_padding_and_truncation

  def _finalize(self, directory: path_types.Path):
    if multihost.is_primary_host(self._multiprocessing_options.primary_host):
      self._handler_impl.finalize(directory)

  async def _background_save(
      self,
      directory: path_types.Path,
      *,
      commit_futures: Sequence[future.Future],
      operation_id: str,
  ):
    for f in commit_futures:
      f.result()
    # Global sync to ensure all participating processes have completed their
    # save operations before proceeding to finalize.
    barrier_name = f'save_and_finalize_{operation_id}_commit_complete'
    multihost.sync_global_processes(
        barrier_name, processes=self._multiprocessing_options.active_processes
    )
    # Finalize.
    self._finalize(directory)
    # Global sync to ensure all hosts are aware that the finalize operation
    # has completed before returning to the user.
    barrier_name = f'save_and_finalize_{operation_id}_finalize_complete'
    multihost.sync_global_processes(
        barrier_name, processes=self._multiprocessing_options.active_processes
    )

  async def save(
      self, directory: path_types.PathLike, checkpointable: PyTree
  ) -> Awaitable[None]:
    directory = epath.Path(directory)
    commit_futures = await self._handler_impl.async_save(
        directory,
        args=base_pytree_checkpoint_handler.BasePyTreeSaveArgs(
            item=checkpointable,
            save_args=_get_legacy_compatible_save_args(
                checkpointable, self._create_array_storage_options_fn
            ),
            ocdbt_target_data_file_size=self._ocdbt_target_data_file_size,
            enable_pinned_host_transfer=self._enable_pinned_host_transfer,
        ),
    )
    assert commit_futures

    # TODO(b/398310070): Move operation ID generation to `Context`.
    operation_id = (
        synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id()
    )
    return self._background_save(
        directory, commit_futures=commit_futures, operation_id=operation_id
    )

  async def _background_load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: PyTree | None = None,
  ) -> PyTree:

    def _set_enable_padding_and_truncation(a):
      if not isinstance(a, type_handlers.ArrayRestoreArgs):
        return a
      return dataclasses.replace(
          a,
          strict=not self._enable_padding_and_truncation,
      )

    restore_args = checkpoint_utils.construct_restore_args(
        abstract_checkpointable
    )
    restore_args = jax.tree.map(
        _set_enable_padding_and_truncation, restore_args
    )
    args = base_pytree_checkpoint_handler.BasePyTreeRestoreArgs(
        item=abstract_checkpointable,
        restore_args=restore_args,
    )
    return self._handler_impl.restore(directory, args=args)

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: PyTree | None = None,
  ) -> Awaitable[PyTree]:
    directory = epath.Path(directory)
    return self._background_load(directory, abstract_checkpointable)

  async def metadata(self, directory: path_types.PathLike) -> PyTree:
    directory = epath.Path(directory)
    return self._handler_impl.metadata(directory)
