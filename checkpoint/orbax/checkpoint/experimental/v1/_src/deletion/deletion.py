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

"""Defines free-function interface for deletion."""

from orbax.checkpoint.experimental.v1._src.deletion import execution
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import (
    types as async_types,
)


def delete(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None = None,
    missing_ok: bool = False,
) -> bool:
  """Deletes a checkpoint, or a single checkpointable within a checkpoint.

  The operation blocks until complete. For improved performance, consider using
  :py:func:`.delete_async` instead. This function should be called on
  all available controller processes.

  If `checkpointable_name` is None, the entire checkpoint at `path` is deleted.
  Otherwise, only the checkpointable with that name is deleted, and the rest of
  the checkpoint remains loadable. A deleted checkpointable can no longer be
  loaded explicitly, and is omitted from :py:func:`.load_checkpointables` and
  :py:func:`.checkpointables_metadata`.

  Deletion is not atomic. If it is interrupted (e.g. by preemption), the
  checkpoint may be left partially deleted. Call this function again with the
  same arguments to finish the deletion.

  Make sure no other process is reading from or writing to the checkpoint while
  it is being deleted. To delete checkpoints in a training loop, use
  :py:meth:`.training.Checkpointer.delete` instead.

  By default, files are permanently removed. To instead move deleted files to a
  separate location, for cleanup by an external process, configure
  :py:class:`~orbax.checkpoint.v1.options.DeletionOptions` in the active
  :py:class:`.Context`.

  Example usage:
    Delete an entire checkpoint::

      ocp.save('/tmp/my_checkpoint', state)
      ocp.delete('/tmp/my_checkpoint')

    Delete a single checkpointable, keeping the rest of the checkpoint::

      ocp.save_checkpointables(
          '/tmp/my_checkpoint',
          {'model': params, 'optimizer': opt_state},
      )
      ocp.delete('/tmp/my_checkpoint', checkpointable_name='optimizer')
      # 'model' can still be loaded.
      restored = ocp.load_checkpointables('/tmp/my_checkpoint')

    Delete a checkpoint that may not exist::

      ocp.delete('/tmp/my_checkpoint', missing_ok=True)

  Args:
    path: The path of the checkpoint to delete. This must be a single
      checkpoint, not a directory containing multiple checkpoints (e.g. the
      root directory of a training run), and not a checkpointable subdirectory.
    checkpointable_name: The name of the checkpointable to delete. If None, the
      entire checkpoint is deleted.
    missing_ok: If True, does not raise an error if the checkpoint or
      checkpointable does not exist.

  Returns:
    True if a deletion occurred (including finishing an interrupted deletion).
    False if the target did not exist and `missing_ok` is True.

  Raises:
    FileNotFoundError: If the target does not exist and `missing_ok` is False.
    InvalidLayoutError: If `path` is not a supported checkpoint.
    ValueError: If `checkpointable_name` is invalid, or is the only remaining
      checkpointable in the checkpoint. Delete the entire checkpoint instead.
    DeletionRecoveryError: If a previously interrupted deletion of this
      checkpoint conflicts with this request.
  """
  return delete_async(
      path, checkpointable_name=checkpointable_name, missing_ok=missing_ok
  ).result()


def delete_async(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None = None,
    missing_ok: bool = False,
) -> async_types.AsyncResponse[bool]:
  """Deletes a checkpoint, or a single checkpointable, asynchronously.

  See :py:func:`.delete` documentation.

  Unlike :py:func:`.delete`, this function returns immediately after the
  deletion is scheduled. Removal of files continues in a background thread. An
  :py:class:`~.AsyncResponse` is returned that can be used to block until the
  deletion is complete (using `response.result()`). Make sure to wait for
  completion before reusing `path` or exiting the program. This function should
  be called on all available controller processes, with the same arguments.

  Example usage:
    Delete a checkpoint asynchronously::

      response = ocp.delete_async('/tmp/my_checkpoint')
      # Perform other work here...

      # Wait for the deletion to finish
      response.result()

  Args:
    path: The path of the checkpoint to delete. See :py:func:`.delete`.
    checkpointable_name: The name of the checkpointable to delete. If None, the
      entire checkpoint is deleted.
    missing_ok: If True, does not raise an error if the checkpoint or
      checkpointable does not exist.

  Returns:
    An `AsyncResponse` that can be used to block until the deletion is complete.
    Blocking can be done using `response.result()`, which returns True if a
    deletion occurred, and False if the target did not exist and `missing_ok` is
    True.
  """
  return execution.start(
      path,
      checkpointable_name=checkpointable_name,
      missing_ok=missing_ok,
  )
