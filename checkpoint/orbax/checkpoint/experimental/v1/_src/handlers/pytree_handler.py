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

"""Implementation of :py:class:`.CheckpointableHandler` for PyTrees."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Sequence, get_args

from absl import logging
import jax
import numpy as np
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import compatibility
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import protocol_utils
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


Path = path_types.Path
CheckpointableHandler = handler_types.CheckpointableHandler
PyTree = tree_types.PyTree

PYTREE_CHECKPOINTABLE_KEY = 'pytree'


def _get_v0_save_args(
    checkpointable: PyTree,
    create_array_storage_options_fn: (
        options_lib.PyTreeOptions.Saving.CreateArrayStorageOptionsFn | None
    ),
) -> PyTree | None:
  """Returns save args that are compatible with the V0 API."""
  if create_array_storage_options_fn is None:
    return None

  def _leaf_get_v0_save_args(k, v):
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

  return jax.tree.map_with_path(_leaf_get_v0_save_args, checkpointable)


def _create_v0_handler(
    context: context_lib.Context,
    *,
    type_handler_registry: type_handlers.TypeHandlerRegistry,
    array_metadata_validator: array_metadata_store_lib.Validator = array_metadata_store_lib.Validator(),
) -> base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler:
  """Creates a V0 handler from a V1 context."""
  return base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler(
      save_concurrent_bytes=context.array_options.saving.concurrent_bytes,
      restore_concurrent_bytes=context.array_options.loading.concurrent_bytes,
      use_ocdbt=context.array_options.saving.use_ocdbt,
      use_zarr3=context.array_options.saving.use_zarr3,
      multiprocessing_options=v0_options_lib.MultiprocessingOptions(
          primary_host=context.multiprocessing_options.primary_host,
          active_processes=context.multiprocessing_options.active_processes,
          barrier_sync_key_prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
      type_handler_registry=type_handler_registry,
      enable_post_merge_validation=context.array_options.saving.enable_post_merge_validation,
      pytree_metadata_options=context.pytree_options.saving.pytree_metadata_options,
      array_metadata_validator=array_metadata_validator,
      enable_pinned_host_transfer=context.array_options.saving.enable_pinned_host_transfer,
  )


def create_v0_save_args(
    context: context_lib.Context,
    checkpointable: PyTree,
) -> base_pytree_checkpoint_handler.BasePyTreeSaveArgs:
  """Creates v0 CheckpointArgs for saving."""
  return base_pytree_checkpoint_handler.BasePyTreeSaveArgs(
      item=checkpointable,
      save_args=_get_v0_save_args(
          checkpointable,
          context.pytree_options.saving.create_array_storage_options_fn,
      ),
      ocdbt_target_data_file_size=context.array_options.saving.ocdbt_target_data_file_size,
  )


def _restore_type_by_abstract_type(
    abstract_checkpointable: Any,
) -> Any:
  """This is to allow users to override the restored type.

  When users pass in the `value` in the DeserializationParam, the PytreeHandler
  will try to restore to the specified type. T. This only supports the standard
  types supported by Orbax.
  For example:
    - jax.ShapeDtype -> jax.Array
    - NumpyAbstractType -> jax.Array
    - int | float | Type[int] | Type[float] -> int | float | int | float

  Args:
    abstract_checkpointable: The abstract checkpointable that passed in by the
      user.

  Returns:
    Return the restore_type parameter for the V0RestoreArgs.  This is needed to
    determine which LeafHandler will eventually handle this
    abstract_checkpointable.
  """

  if abstract_checkpointable is None:
    ret = None
  elif type_handlers.is_placeholder(abstract_checkpointable):
    ret = type_handlers.PLACEHOLDER
  else:
    if isinstance(abstract_checkpointable, type):
      abstract_type = abstract_checkpointable
    else:
      abstract_type = type(abstract_checkpointable)

    # Make sure test with AbstractArray before AbstractNumpy otherwise Numpy
    # will be matched first.
    if protocol_utils.is_subclass_protocol(
        abstract_type, array_leaf_handler.AbstractArray
    ):
      ret = jax.Array
    elif protocol_utils.is_subclass_protocol(
        abstract_type, numpy_leaf_handler.AbstractNumpy
    ):
      ret = np.ndarray
    elif issubclass(abstract_type, get_args(scalar_leaf_handler.Scalar)):
      ret = abstract_type
    else:
      # this will use registered handler derived from metadata
      ret = None

  logging.vlog(
      1,
      'abstract_checkpointable: %s, restore_type: %s',
      abstract_checkpointable,
      ret,
  )
  return ret


def create_v0_restore_args(
    context: context_lib.Context,
    abstract_checkpointable: PyTree | None,
) -> base_pytree_checkpoint_handler.BasePyTreeRestoreArgs:
  """Creates v0 CheckpointArgs for restoration."""

  if abstract_checkpointable:
    restore_args = jax.tree.map(
        lambda checkpointable: compatibility.V0RestoreArgs(
            restore_type=_restore_type_by_abstract_type(checkpointable),
            abstract_leaf=checkpointable,
        ),
        abstract_checkpointable,
    )
  else:
    restore_args = None

  logging.vlog(1, 'restore_args: %s', restore_args)

  return base_pytree_checkpoint_handler.BasePyTreeRestoreArgs(
      item=abstract_checkpointable,
      restore_args=restore_args,
      partial_restore=context.pytree_options.loading.partial_load,
  )


async def _async_futures(commit_futures: Sequence[future.Future]):
  await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])


class PyTreeHandler(CheckpointableHandler[PyTree, PyTree]):
  """An implementation of :py:class:`.CheckpointableHandler` for PyTrees."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
      array_metadata_validator: array_metadata_store_lib.Validator = (
          array_metadata_store_lib.Validator()
      ),
  ):
    context = context_lib.get_context(context)
    self._context = context
    self._multiprocessing_options = context.multiprocessing_options

    self._leaf_handler_registry = (
        self._context.pytree_options.leaf_handler_registry
        if self._context.pytree_options.leaf_handler_registry is not None
        else registry.StandardLeafHandlerRegistry()
    )

    type_handler_registry = compatibility.get_v0_type_handler_registry(
        self._leaf_handler_registry, self._context
    )

    self._handler_impl = _create_v0_handler(
        context,
        type_handler_registry=type_handler_registry,
        array_metadata_validator=array_metadata_validator,
    )

  async def _finalize(self, directory: path_types.Path):
    if multihost.is_primary_host(self._multiprocessing_options.primary_host):
      await self._handler_impl._finalize_async(directory)  # pylint: disable=protected-access

  async def _background_save(
      self,
      directory: path_types.PathAwaitingCreation,
      *,
      commit_futures: Sequence[future.Future],
      operation_id: str,
  ):
    directory = await directory.await_creation()
    active_processes = self._multiprocessing_options.active_processes or set(
        range(multihost.process_count())
    )
    await _async_futures(commit_futures)
    # Global sync to ensure all participating processes have completed their
    # save operations before proceeding to finalize.
    barrier_name = f'save_and_finalize_{operation_id}_commit_complete'
    await multihost.sync_global_processes(
        barrier_name, processes=active_processes
    )
    # Finalize.
    await self._finalize(directory)
    # Global sync to ensure all hosts are aware that the finalize operation
    # has completed before returning to the user.
    barrier_name = f'save_and_finalize_{operation_id}_finalize_complete'
    await multihost.sync_global_processes(
        barrier_name, processes=active_processes
    )

  async def save(
      self, directory: path_types.PathAwaitingCreation, checkpointable: PyTree
  ) -> Awaitable[None]:

    self._validate_leaves_handleable(checkpointable)

    commit_futures = await self._handler_impl.async_save(
        directory.path,
        args=create_v0_save_args(self._context, checkpointable),
    )
    assert commit_futures

    # TODO(b/398310070): Move operation ID generation to `Context`.
    operation_id = (
        synchronization.OperationIdGenerator.get_current_operation_id()
    )
    # Needed to differentiate between different handlers when we have multiple
    # PyTreeHandlers performing a save.
    operation_id = f'{operation_id}.{directory.path.name}'
    return self._background_save(
        directory, commit_futures=commit_futures, operation_id=operation_id
    )

  async def _background_load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: PyTree | None = None,
  ) -> PyTree:
    return self._handler_impl.restore(
        directory,
        args=create_v0_restore_args(self._context, abstract_checkpointable),
    )

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: PyTree | None = None,
  ) -> Awaitable[PyTree]:
    self._validate_abstract_leaves_handleable(abstract_checkpointable)
    return self._background_load(directory, abstract_checkpointable)

  async def metadata(
      self, directory: path_types.Path
  ) -> metadata_types.PyTreeMetadata:
    v0_metadata = self._handler_impl.metadata(directory).tree

    def _unwrap(metadata):
      # unwrap the V0Metadata to get the V1 metadata
      assert isinstance(metadata, compatibility.V0Metadata)
      return metadata.v1_metadata

    return jax.tree.map(_unwrap, v0_metadata)

  def _validate_leaves_handleable(self, checkpointable: PyTree):
    missing_leaf_types = set()

    def _validate_handleable_leaf(leaf: Any):
      if type_handlers.is_placeholder(leaf):
        return

      leaf_type = type(leaf)
      if not self._leaf_handler_registry.is_handleable(leaf_type):
        missing_leaf_types.add(leaf_type)

    jax.tree.map(
        _validate_handleable_leaf,
        checkpointable,
    )

    if missing_leaf_types:
      raise ValueError(
          'The following leaf types are not registered in the'
          f' `LeafHandlerRegistry`: [{missing_leaf_types}]. Please register a'
          ' `LeafHandler` for each type in the `LeafHandlerRegistry` and'
          ' assign it into the `PyTreeOptions` in the `Context`.'
      )

  def _validate_abstract_leaves_handleable(
      self, abstract_checkpointable: PyTree
  ):
    missing_abstract_leaf_types = set()

    def _validate_handleable_leaf(leaf: Any):
      if type_handlers.is_placeholder(leaf):
        return

      leaf_type = leaf if isinstance(leaf, type) else type(leaf)
      if not self._leaf_handler_registry.is_abstract_handleable(leaf_type):
        missing_abstract_leaf_types.add(leaf_type)

    jax.tree.map(
        _validate_handleable_leaf,
        abstract_checkpointable,
    )

    if missing_abstract_leaf_types:
      raise ValueError(
          'The following abstract leaf types are not registered in the'
          f' `LeafHandlerRegistry`: [{missing_abstract_leaf_types}]. Please'
          ' register a `LeafHandler` for each type in the'
          ' `LeafHandlerRegistry` and assign it into the `PyTreeOptions` in'
          ' the `Context`.'
      )

  def is_handleable(self, checkpointable: Any) -> bool:
    try:
      # If it's a leaf or an empty pytree container, it's not handleable.
      return not jax.tree_util.treedef_is_leaf(
          jax.tree.structure(checkpointable)
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return False

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool:
    return self.is_handleable(abstract_checkpointable)
