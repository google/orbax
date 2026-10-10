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
  """Deletes an Orbax checkpoint or one named checkpointable.

  The caller must have exclusive mutation access and finish any reads of the
  selected target before starting deletion. For an active
  training run, prefer ``Checkpointer.delete`` so pending saves and the
  manager's checkpoint listing are coordinated. Offline training-step roots
  and checkpoints written by free functions are both supported.

  Whole-checkpoint deletion is non-atomic and uses existing removal/relocation
  mechanics. Partial deletion keeps recovery metadata inside the checkpoint;
  retry the same request to finish an interrupted operation. No records are
  written outside the checkpoint root. Deletion options come from the active
  Context, including configured GCS relocation. GCS relocation is not atomic:
  if an interrupted move leaves both the item and its destination, resolve the
  conflict before retrying.

  Publishing a partial-deletion record makes the selected item logically
  deleted. Explicit reads of that item fail; discovery loads and metadata
  enumeration exclude it with a warning. Unaffected items remain readable.
  Reads never finish deletion. Retry delete to complete interrupted cleanup.

  This function must be called with matching requests on all participating
  controller processes. A failure is raised locally; peers may time out at the
  next process barrier.

  Args:
    path: One checkpoint root, not a training-run root or item subdirectory.
    checkpointable_name: A named item to remove. None deletes the whole
      checkpoint; specify 'state' to delete only that checkpointable.
    missing_ok: Whether an absent checkpoint or item is a successful no-op.
      Other failures are not suppressed.

  Returns:
    True if this call completed a new or interrupted deletion. False if the
    target was already absent with no pending cleanup and missing_ok is True.
    Failures raise exceptions.

  Raises:
    FileNotFoundError: The target does not exist and missing_ok is False.
    InvalidLayoutError: The path is not a supported checkpoint root.
    ValueError: The name is invalid or selects the final user checkpointable.
    DeletionRecoveryError: A recovery record is invalid or conflicts with this
      request.
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
  """Deletes asynchronously, with the same arguments and scope as ``delete``.

  Validation, process coordination, and partial-deletion record creation run
  before this call returns, using the active Context. Filesystem cleanup runs
  in a background thread. Retain the response and call result()
  to observe completion or errors and release the shared runner's resources;
  leaving Context does not wait for deletion.
  A result timeout does not cancel the operation. Do not reuse the path until
  the operation has finished. Missing targets with missing_ok=True produce a
  response whose result is False. Otherwise, successful completion returns True,
  including completion of an interrupted deletion.

  Args:
    path: One checkpoint root, not a training-run root or item subdirectory.
    checkpointable_name: A named item to remove. None deletes the whole
      checkpoint; specify 'state' to delete only that checkpointable.
    missing_ok: Whether an absent checkpoint or item is a successful no-op.
      Other failures are not suppressed.

  Returns:
    An `AsyncResponse` that can be used to wait for the deletion to complete.
  """
  return execution.start(
      path,
      checkpointable_name=checkpointable_name,
      missing_ok=missing_ok,
  )
