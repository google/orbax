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

"""Defines free-function interface for saving."""

# pylint: disable=protected-access

import threading

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def _make_v0_save_args(
    pytree: tree_types.PyTree,
    create_array_storage_options_fn: (
        options_lib.CreateArrayStorageOptionsFn | None
    ),
) -> tree_types.PyTree:
  """Creates v0-compatible `SaveArgs` from v1 storage options."""
  if create_array_storage_options_fn is None:
    save_args = None
  else:
    array_storage_options_pytree = jax.tree.map_with_path(
        create_array_storage_options_fn, pytree
    )
    save_args = jax.tree.map(
        lambda v: ocp.SaveArgs(
            dtype=np.dtype(v.dtype),
            chunk_byte_size=v.chunk_byte_size,
            shard_axes=v.shard_axes,
        ),
        array_storage_options_pytree,
    )
  return save_args


def _get_concurrent_gb(concurrent_bytes: int | None) -> int | None:
  if concurrent_bytes:
    return max(int(concurrent_bytes / 1e9), 1)
  return None


def save_pytree(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    # Generic arguments.
    force: bool = False,
    custom_metadata: ocp.tree.JsonType | None = None,
):
  """Saves a PyTree.

  The operation blocks until complete. For improved performance, consider using
  `save_async` instead.

  Args:
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  context = context_lib.get_context()

  handler_registry = ocp.handlers.create_default_handler_registry(
      pytree=ocp.PyTreeCheckpointHandler(
          use_ocdbt=context.pytree_options.use_ocdbt,
          use_zarr3=context.pytree_options.use_zarr3,
          save_concurrent_gb=_get_concurrent_gb(
              context.pytree_options.save_concurrent_bytes
          ),
      )
  )
  ckptr = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  save_args = _make_v0_save_args(
      pytree, context.pytree_options.create_array_storage_options_fn
  )
  args = ocp.args.Composite(
      pytree=ocp.args.PyTreeSave(
          pytree,
          save_args=save_args,
          ocdbt_target_data_file_size=(
              context.pytree_options.ocdbt_target_data_file_size
          ),
          enable_pinned_host_transfer=(
              context.pytree_options.enable_pinned_host_transfer
          ),
      )
  )
  ckptr.save(directory, args=args, force=force, custom_metadata=custom_metadata)


class _SaveResponse(async_types.AsyncResponse[None]):
  """An `AsyncResponse` representing the result of `save_pytree_async`.

  TODO(cpgaffney): Note that a memory leak is possible if the user does not
  call `result`.
  """

  def __init__(self, checkpointer: ocp.AsyncCheckpointer):
    self._checkpointer = checkpointer
    self._thread = threading.Thread(target=self._wait_for_save)
    self._thread.start()

  def _wait_for_save(self):
    self._checkpointer.wait_until_finished()

  def result(self, timeout: float | None = None) -> None:
    self._thread.join()
    self._checkpointer.close()


def save_pytree_async(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    # Generic arguments.
    force: bool = False,
    custom_metadata: ocp.tree.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Saves a PyTree asynchronously.

  Unlike `save`, this function returns immediately after the save operation is
  scheduled (except for certain operations, like device-to-host copying of
  on-device arrays, which must happen on the main thread). Further writing
  operations continue in a background thread. An `AsyncResponse` is returned
  that can be used to block until the save is complete (using
  `response.result()`). Make sure to wait for completion before attempting to
  load the checkpoint or exiting the program.

  Args:
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  context = context_lib.get_context()

  handler_registry = ocp.handlers.create_default_handler_registry(
      pytree=ocp.PyTreeCheckpointHandler(
          use_ocdbt=context.pytree_options.use_ocdbt,
          use_zarr3=context.pytree_options.use_zarr3,
          save_concurrent_gb=_get_concurrent_gb(
              context.pytree_options.save_concurrent_bytes
          ),
      )
  )
  ckptr = ocp.AsyncCheckpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  save_args = _make_v0_save_args(
      pytree, context.pytree_options.create_array_storage_options_fn
  )
  args = ocp.args.Composite(
      pytree=ocp.args.PyTreeSave(
          pytree,
          save_args=save_args,
          ocdbt_target_data_file_size=(
              context.pytree_options.ocdbt_target_data_file_size
          ),
          enable_pinned_host_transfer=(
              context.pytree_options.enable_pinned_host_transfer
          ),
      )
  )
  directory = epath.Path(directory)
  ckptr.save(directory, args=args, force=force, custom_metadata=custom_metadata)
  return _SaveResponse(ckptr)
