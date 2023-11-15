# Copyright 2023 The Orbax Authors.
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

"""Abstract class to manage checkpoints: AbstractCheckpointManager."""

import abc
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

from etils import epath

PyTree = Any
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams


class AbstractCheckpointManager(Protocol):
  """Interface to manage checkpoints.

  Allows a user to save and restore objects for which a Checkpointer
  implementation exists (e.g. PyTreeCheckpointer for PyTrees). The class
  keeps track of multiple checkpointable objects in the following structure::

    path/to/directory/    (top-level directory)
      0/    (step)
        params/    (first saveable)
          ...
        metadata/    (second saveable)
          ...
      1/    (step)
        ...
      2/    (step)
        ...
      ...
  """

  @property
  @abc.abstractmethod
  def directory(self) -> epath.Path:
    """Returns the top-level directory containing checkpoints for all items."""

  @abc.abstractmethod
  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """

  @abc.abstractmethod
  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """

  @abc.abstractmethod
  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """

  @abc.abstractmethod
  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""

  @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """

  @abc.abstractmethod
  def delete(self, step: int):
    """Deletes a step checkpoint."""

  @abc.abstractmethod
  def save(
      self,
      step: int,
      items: Union[Any, Mapping[str, Any]],
      save_kwargs: Optional[Union[SaveParams, Mapping[str, SaveParams]]] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
  ) -> bool:
    """Saves the provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Items and save_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and save_kwargs, a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following::

      {
        'params': PyTree(),
        'metadata': <nested k/v>,
        ...
      }
      Similarly, save_kwargs takes the form:
      {
        'params': {
          <kwargs for PyTreeCheckpointHandler.save>
        },
        'metadata': {
          <kwargs for JsonCheckpointHandler.save>
        }
        ...
      }

    The kwargs under 'params' correspond to PyTreeCheckpointHandler.save. If a
    key is not present in save_kwargs, it is assumed that no kwargs are needed
    for saving that item. If not provided at all, it is assumed that no items
    need extra kwargs for saving.

    Note that if a single Checkpointer was provided at construction time,
    `items` must be a singular saveable object, and `save_kwargs` must be the
    kwargs needed by a single Checkpointer.

    Args:
      step: current step, int
      items: a savable object, or a dictionary of object name to savable object.
      save_kwargs: save kwargs for a single Checkpointer, or a dictionary of
        object name to kwargs needed by the Checkpointer implementation to save
        the object.
      metrics: a dictionary of metric name (string) to numeric value to be
        tracked along with this checkpoint. Required if `options.best_fn` is
        set. Allows users to specify a metric value to determine which
        checkpoints are best and should be kept (in conjunction with
        `options.max_to_keep`).
      force: if `True`, this method will attempt to save a checkpoint regardless
        of the result of `AbstractCheckpointManager.should_save(step)`. By
        default, `save` will only write a checkpoint to disk when the options
        permit, e.g. when `step` is in `options.save_interval_steps` or
        `options.save_on_steps`. Setting `force=True` will not overwrite
        existing checkpoints.

    Returns:
      bool indicating whether a save operation was performed.
    Raises:
      ValueError: if `track_best` was indicated but `metrics` is not provided.
      ValueError: directory creation failed.
      ValueError: if an item is provided for which no `Checkpointer` is
      found.
      ValueError: if the checkpoint already exists.
    """

  @abc.abstractmethod
  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[
          Union[RestoreParams, Mapping[str, RestoreParams]]
      ] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Items and restore_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and restore_kwargs,
    a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following::

      {
        'params': PyTree(),
        'metadata': <nested k/v>,
        ...
      }

    Items may not be provided at all, in which case it the items restored are
    those specified in self._checkpointers, and item=None is provided to
    Checkpointer.restore. Similarly, an item may be omitted from `items`,
    in
    which case item=None will be provided to Checkpointer.restore.

    Similarly, restore_kwargs takes the form::

      {
        'params': {
          'meshes': PyTree(),
          'mesh_axes': PyTree(),
        },
        'metadata': {
          <kwargs for JsonCheckpointHandler.save>
        }
        ...
      }

    The kwargs under 'params' correspond to PyTreeCheckpointHandler.restore. If
    a key is not present in restore_kwargs, it is assumed that no kwargs are
    needed for restoring that item. If not provided at all, it is assumed that
    no items need extra kwargs for restoring.

    Note that if a single Checkpointer was provided at construction time,
    `items` must be a singular saveable object, and `restore_kwargs` must be the
    kwargs needed by a single Checkpointer.

    Args:
      step: current step, int
      items: a restoreable object, or a dictionary of object name to restorable
        object.
      restore_kwargs: restore kwargs for a single Checkpointer, or a dictionary
        of object name to kwargs needed by the Checkpointer implementation to
        restore the object.
      directory: if provided, uses the given directory rather than the
        `directory` property of this class. Can be used to restore checkpoints
        from an independent location.

    Returns:
      A dictionary matching the structure of self._checkpointers, with one
      object returned for each Checkpointer, or a single restored object,
      if a
      single item is being tracked by this manager.
    """

  @abc.abstractmethod
  def item_metadata(self, step: int) -> Union[Any, Mapping[str, Optional[Any]]]:
    """For all Checkpointers, returns any metadata associated with the item.

    Calls the `metadata` method for each Checkpointer and returns a
    mapping of each item name to the restored metadata. If the manager only
    manages a single item, a single metadata will be returned instead.

    Metadata may be None for an individual item.

    Args:
      step: Step for which to retrieve metadata.

    Returns:
      A dictionary mapping name to item metadata, or a single item metadata.
    """

  @abc.abstractmethod
  def metadata(self) -> Mapping[str, Any]:
    """Returns CheckpointManager level metadata if present, empty otherwise."""

  @abc.abstractmethod
  def wait_until_finished(self, join_finalize_thread=True):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.

    Args:
      join_finalize_thread: Whether to join the _finalize_thread. This should
        always be True for external callers.
    """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """

  @abc.abstractmethod
  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
