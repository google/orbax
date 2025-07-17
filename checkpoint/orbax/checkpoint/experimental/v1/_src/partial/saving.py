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

"""Defines free-function interface for partial saving and finalizing."""

import os

from etils import epath
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import saving as saving_lib
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = saving_lib.PYTREE_CHECKPOINTABLE_KEY
PARTIAL_SAVE_SUFFIX = '.partial_save'


def _is_partial_save_path(path: path_types.PathLike) -> bool:
  path = epath.Path(path)
  return path.name.endswith(PARTIAL_SAVE_SUFFIX)


def _add_partial_save_suffix(path: path_types.PathLike) -> path_types.Path:
  path = epath.Path(path)
  return path.parent / (path.name + PARTIAL_SAVE_SUFFIX)


def _remove_partial_save_suffix(path: path_types.PathLike) -> path_types.Path:
  path = epath.Path(path)
  return path.parent / path.name.removesuffix(PARTIAL_SAVE_SUFFIX)


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
  `ocp.partial.finalize(path)` is called. The intermediate checkpoints are
  temporary and should not be used directly.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  `partial.save_pytree` followed by a single call to `partial.finalize`.

  ```
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
  ```

  ### Additions vs. Replacements

  The provided `pytree` represents a set of updates.
  - If a key in `pytree` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**. In other words, the sets of
    keys of the on-disk PyTree and the provided `pytree` are disjoint.
  - If a key (e.g., 'step') already exists, its value is **replaced**. In other
    words, the sets of keys of the on-disk PyTree and the provided `pytree`
    overlap. Replacements are currently NOT supported. Please reach out to the
    Orbax team if you need this functionality.

  See `ocp.save_pytree` for general PyTree saving documentation.

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

  Unlike `partial.save_pytree`, this function returns an `AsyncResponse`
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
  `ocp.partial.finalize(path)` is called. The intermediate checkpoints are
  temporary and may be garbage collected in certain environments.

  ### Workflow

  A typical partial save workflow involves one or more calls to
  `partial.save_pytree_async` followed by a single call to `partial.finalize`.

  ```
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
  ```

  ### Additions vs. Replacements

  The provided `pytree` represents a set of updates.
  - If a key in `pytree` (e.g., 'metrics') does not exist in the on-disk
    checkpoint, it is treated as an **addition**.
  - If a key (e.g., 'step') already exists, its value is **replaced**.
    Replacements are currently NOT supported. Please reach out to the Orbax team
    if you need this functionality.

  See `ocp.save_pytree_async` for general PyTree saving documentation.

  Args:
    path: The path to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  return saving_lib._save_checkpointables_impl(  # pylint: disable=protected-access
      path,
      {saving_lib.PYTREE_CHECKPOINTABLE_KEY: pytree},
      overwrite=False,
      custom_metadata=custom_metadata,
      async_origin=True,
      # partial_save=True,
  )


def finalize(path: path_types.PathLike) -> None:
  """Finalizes a partially-saved checkpoint, making it permanent and readable.

  This function commits all changes made during a partial save session,
  concluding the transaction. It should be called once after all desired
  `ocp.partial.save_*` operations are complete.

  The finalization process is atomic. It renames the temporary, versioned
  partial save directory to the final target `path`, making the updated
  checkpoint "live".

  IMPORTANT: Until `finalize` is called, the checkpoint at the target `path`
  is not created or modified. All changes are buffered in a temporary location.
  This function is what makes those changes permanent.


  ### Example
  ```
  path = '/path/to/my/checkpoint'

  # These calls write to a temporary, versioned directory, not the final path.
  ocp.partial.save_pytree(path, {'step': 1})
  ocp.partial.save_checkpointables(path, {'metrics': ...})

  # This call performs the atomic rename, making the checkpoint available at
  # '/path/to/my/checkpoint'.
  ocp.partial.finalize(path)
  ```

  Args:
    path: The final, target path of the checkpoint to be finalized. This should
      be the same path that was passed to `ocp.partial.save_*` calls.

  Raises:
    FileExistsError: If a finalized checkpoint already exists at `path`. To
      overwrite, it must be deleted first.
    FileNotFoundError: If no partial save session is found for the given `path`.
      This can happen if `ocp.partial.save_*` was not called first.
  """
  path = epath.Path(path)

  if _is_partial_save_path(path):
    final_path = _remove_partial_save_suffix(path)
  else:
    final_path = path
    path = _add_partial_save_suffix(path)

  if final_path.exists():
    raise FileExistsError(f'Finalized checkpoint already exists at {path}.')

  if not path.exists():
    raise FileNotFoundError(f'Partial save path {path} does not exist.')

  os.rename(path, final_path)
