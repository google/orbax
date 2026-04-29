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

"""Defines free-function interface for partial saving and finalizing."""

import asyncio
import dataclasses
import time
from typing import Awaitable

from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import utils as ocp_path_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import stateful_checkpointable_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import execution
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY

StatefulCheckpointableHandler = (
    stateful_checkpointable_handler.StatefulCheckpointableHandler
)


@dataclasses.dataclass
class _PartialSavePyTree(handler_types.StatefulCheckpointable):
  """Wraps a PyTree to signal that it should be saved in partial mode."""

  pytree: tree_types.PyTree

  def __post_init__(self):
    self.handler = pytree_handler.PyTreeHandler()

  async def save(
      self, directory: path_types.PathAwaitingCreation
  ) -> Awaitable[None]:
    start_time = time.time()

    operation_id = (
        synchronization.OperationIdGenerator.get_current_operation_id()
    )
    operation_id = f'{operation_id}.{directory.path.name}'

    # pylint: disable=protected-access
    self.handler.validate_leaves_handleable(self.pytree)

    v0_save_args = pytree_handler.create_v0_save_args(
        self.handler._context, self.pytree
    )
    v0_save_args = dataclasses.replace(v0_save_args, partial_save_mode=True)

    commit_futures = await self.handler._handler_impl.async_save(
        directory.path,
        args=v0_save_args,
    )
    assert commit_futures

    return self.handler._background_save(
        directory,
        commit_futures=commit_futures,
        operation_id=operation_id,
        start_time=start_time,
    )
    # pylint: enable=protected-access

  async def load(self, directory: path_types.Path) -> Awaitable[None]:
    raise NotImplementedError('Partial load is not supported via this wrapper.')


def save_pytree(
    path: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    custom_metadata: tree_types.JsonType | None = None,
):
  """Partially saves a PyTree.

  This function allows for incrementally updating a checkpoint. It is designed
  to be called multiple times. The first call initiates a new partial save
  "session" in a temporary location. Subsequent calls will update this session
  by modifying the checkpoint in place.

  The operation is atomic; if it is interrupted, the previous version of the
  partial save will be preserved.

  IMPORTANT: The checkpoint is not finalized at the target `path` until
  :py:func:`.finalize` is called. The intermediate checkpoints are
  temporary and should not be used directly.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  :py:func:`.save_pytree` followed by a single call to :py:func:`~.finalize`::

    path = '/path/to/my/checkpoint'

    # The first call creates a temporary directory:
    # '/path/to/my/checkpoint.partial_save'
    # Note: the exact temporary directory name is an implementation detail that
    # depends on the file system and should not be relied on.
    ocp.partial.save_pytree(path, {'layer1': ..., 'step': 1})

    # A subsequent call reads the previous version and applies new updates
    # to the temporary directory:
    # '/path/to/my/checkpoint.partial_save'
    ocp.partial.save_pytree(path, {'layer2': ..., 'metrics': ...})

    # This call commits the latest version to the final destination at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  ### Additions vs. Replacements

  The provided `pytree` represents a set of updates.
  - If a key in `pytree` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**. In other words, the sets of
    keys of the on-disk PyTree and the provided `pytree` are disjoint.
  - If a key (e.g., 'step') already exists, its value is **replaced**. In other
    words, the sets of keys of the on-disk PyTree and the provided `pytree`
    overlap. Replacements are currently NOT supported. Please reach out to the
    Orbax team if you need this functionality.

  See :py:func:`~.v1.save_pytree` for general
  PyTree saving documentation.

  Args:
    path: The path to save the checkpoint to.
    pytree: A PyTree representing the additions to be applied to the on-disk
      checkpoint.
    custom_metadata: User-provided custom metadata. This will be merged with any
      existing custom metadata. Values from this dictionary will overwrite
      existing values if keys conflict.
  """
  save_pytree_async(
      path,
      pytree,
      custom_metadata=custom_metadata,
  ).result()


def save_pytree_async(
    path: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    custom_metadata: tree_types.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Partially saves a PyTree asynchronously.

  Unlike :py:func:`.save_pytree`, this function returns an
  :py:class:`.AsyncResponse`
  immediately after scheduling the save operation. The actual writing to disk
  happens in a background thread. You can use `response.result()` to block
  until the operation is complete.

  This function allows for incrementally updating a checkpoint. It is designed
  to be called multiple times. The first call initiates a new partial save
  "session" in a temporary location. Subsequent calls will update this session
  by creating a new version that includes all previous changes plus the new
  ones.

  The operation is atomic; if it is interrupted, the previous version of the
  partial save will be preserved.

  IMPORTANT: The checkpoint is not finalized at the target `path` until
  :py:func:`.finalize` is called. The intermediate checkpoints are
  temporary and may be garbage collected in certain environments.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  :py:func:`.save_pytree_async` followed by a single call to
  :py:func:`.finalize`::

    path = '/path/to/my/checkpoint'

    # The first call creates a temporary directory and returns immediately.
    response1 = ocp.partial.save_pytree_async(path, {'layer1': ..., 'step': 1})

    # A subsequent call also returns immediately. Orbax ensures that this
    # operation waits for the first one to complete before starting.
    response2 = ocp.partial.save_pytree_async(
        path, {'layer2': ..., 'metrics': ...}
    )

    # Wait for all async partial saves to complete before finalizing.
    response1.result()
    response2.result()

    # This call commits the latest version to the final destination at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  ### Additions vs. Replacements

  The provided `pytree` represents a set of updates.
  - If a key in `pytree` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**.
  - If a key (e.g., 'step') already exists, its value is **replaced**.
    Replacements are currently NOT supported. Please reach out to the Orbax team
    if you need this functionality.

  See :py:func:`~.v1.save_pytree_async` for general
  PyTree saving documentation.

  Args:
    path: The path to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a :py:class:`.LeafTypeHandler`.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An :py:class:`.AsyncResponse` that can be used to block until the save is
    complete.
    Blocking can be done using `response.result()`, which returns `None`.

  Raises:
    FileExistsError: If a finalized checkpoint already exists at `path`. To
      overwrite, it must be deleted first.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  if path.exists():
    raise FileExistsError(f'Finalized checkpoint already exists at {path}.')

  # By default, the registry associates 'pytree' (PYTREE_CHECKPOINTABLE_KEY)
  # with PyTreeHandler. We want to use StatefulCheckpointableHandler for our
  # wrapper (_PartialSavePyTree) to carry the partial save flag. Since
  # name-based resolution takes priority, we override 'pytree' in a local
  # registry.
  current_reg = ctx.checkpointables_options.registry
  local_reg = registration.local_registry(include_global_registry=False)
  for handler, name in current_reg.get_all_entries():
    if name != PYTREE_CHECKPOINTABLE_KEY:
      local_reg.add(
          handler,
          checkpointable_name=name,
          secondary_typestrs=current_reg.get_secondary_typestrs(handler),
      )
  local_reg.add(
      StatefulCheckpointableHandler,
      checkpointable_name=PYTREE_CHECKPOINTABLE_KEY,
  )

  new_options = options_lib.CheckpointablesOptions(registry=local_reg)
  with context_lib.Context(ctx, checkpointables_options=new_options):
    return execution.save_checkpointables_impl(
        partial_path_lib.add_partial_save_suffix(path),
        {PYTREE_CHECKPOINTABLE_KEY: _PartialSavePyTree(pytree)},
        overwrite=False,
        custom_metadata=custom_metadata,
        async_origin=True,
        partial_save=True,
    )


def finalize(path: path_types.PathLike) -> None:
  """Finalizes a partially-saved checkpoint, making it permanent and readable.

  This function commits all changes made during a partial save session,
  concluding the transaction. It should be called once after all desired
  :py:func:`.save_pytree` operations are complete.

  The finalization process is atomic. It renames the temporary, versioned
  partial save directory to the final target `path`, making the updated
  checkpoint "live".

  IMPORTANT: Until `finalize` is called, the checkpoint at the target `path`
  is not created or modified. All changes are buffered in a temporary location.
  This function is what makes those changes permanent.


  ### Example::
    path = '/path/to/my/checkpoint'

    # These calls write to a temporary, versioned directory, not the final path.
    ocp.partial.save_pytree(path, {'step': 1})
    ocp.partial.save_checkpointables(path, {'metrics': ...})

    # This call performs the atomic rename, making the checkpoint available at
    # '/path/to/my/checkpoint'.
    ocp.partial.finalize(path)

  Args:
    path: The final, target path of the checkpoint to be finalized. This should
      be the same path that was passed to :py:func:`~.save_pytree` calls.

  Raises:
    FileExistsError: If a finalized checkpoint already exists at `path`. To
      overwrite, it must be deleted first.
    FileNotFoundError: If no partial save session is found for the given `path`.
      This can happen if :py:func:`.save_pytree` was not called first.
  """
  context = context_lib.get_context()
  path = context.file_options.path_class(path)
  if partial_path_lib.is_partial_save_path(path):
    final_path = partial_path_lib.remove_partial_save_suffix(path)
    partial_path = path
  else:
    final_path = path
    partial_path = partial_path_lib.add_partial_save_suffix(path)

  async def _finalize_impl():
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_path_existence_start',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=context.operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )
    if await async_path.exists(final_path):
      raise FileExistsError(
          f'Finalized checkpoint already exists at {final_path}.'
      )
    elif not await async_path.exists(partial_path):
      raise FileNotFoundError(
          f'Partial save path {partial_path} does not exist.'
      )

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_path_rename_start',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=context.operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )

    rename_failed = False
    rename_error = None
    if multihost.is_primary_host(context.multiprocessing_options.primary_host):
      try:
        await async_path.rename(partial_path, final_path)
      except OSError as e:
        rename_failed = True
        rename_error = e

    rename_failed = multihost.broadcast_one_to_all(
        rename_failed,
        is_source=multihost.is_primary_host(
            context.multiprocessing_options.primary_host
        ),
    )

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'OcpPartialSaving:finalize_rename_complete',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=context.operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )

    if rename_failed:
      if rename_error is not None:
        raise rename_error
      raise OSError('Partial checkpoint finalization failed during rename.')

  asyncio_utils.run_sync(_finalize_impl())
