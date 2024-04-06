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

"""Abstract class to manage checkpoints: AbstractCheckpointManager."""

import abc
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

from etils import epath
from orbax.checkpoint import args as args_lib

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
  def reload(self):
    """Performs disk reads to ensure internal properties are up to date."""

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
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      save_kwargs: Optional[Union[SaveParams, Mapping[str, SaveParams]]] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
      args: Optional[args_lib.CheckpointArgs] = None,
  ) -> bool:
    """Saves the provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    NOTE: The `items` and `save_kwargs` arguments are deprecated, use `args`
    instead. Make sure to configure `CheckpointManager` with `item_names`.

    `args` should be a subclass of
    `orbax.checkpoint.args.CheckpointArgs`, the specific type of which is used
    to indicate what logic is used to save the object. For a typical, PyTree of
    arrays, use `StandardSave`/`StandardRestore`.

    When constructing the `CheckpointManager`, if no `item_names` were provided,
    it is assumed that we are managing a single object. If `item_names` were
    provided, it is assumed that we are managing multiple objects, and `args`
    must be `orbax.checkpoint.args.CompositeArgs`. See below for details.

    Example::

      # Single item
      mngr = ocp.CheckpointManager(directory)
      mngr.save(step, args=ocp.args.StandardSave(my_train_state))

      # Multiple items
      mngr = ocp.CheckpointManager(directory, item_names=('state', 'meta'))
      mngr.save(step, args=ocp.args.Composite(
          state=ocp.args.StandardSave(my_train_state),
          meta=ocp.args.JsonSave(my_metadata)
      ))

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
      args: `CheckpointArgs` which is used to save checkpointable objects with
        the appropriate logic.

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
      args: Optional[args_lib.CheckpointArgs] = None,
  ) -> Union[Any, Mapping[str, Any], args_lib.Composite]:
    """Restores from the given step and provided items.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    NOTE: The `items` and `restore_kwargs` arguments are deprecated, use `args`
    instead. Make sure to configure `CheckpointManager` with `item_names`.
    See `save` docstring for additional details.

    Example::

      # Single item
      mngr = ocp.CheckpointManager(directory)
      mngr.restore(step, args=ocp.args.StandardRestore(abstract_train_state))

      # Multiple items
      mngr = ocp.CheckpointManager(directory, item_names=('state', 'meta'))
      mngr.restore(step, args=ocp.args.Composite(
          state=ocp.args.StandardRestore(abstract_train_state),
          meta=ocp.args.JsonRestore(),
      ))
      # If it is acceptable to restore without providing additional arguments,
      # and if a save has already been performed, it is ok to do the following:
      mngr.restore(step, args=ocp.args.Composite(state=None, meta=None))
      # If a save has not already been performed, there is no way for Orbax to
      # know how to restore the objects. If a save has already been performed,
      # it remembers the logic used to save the objects.

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
      args: `CheckpointArgs` which is used to restore checkpointable objects
        with the appropriate logic.

    Returns:
      If managing a single item, returns a single checkpointable object.
      If managing multiple items, returns ocp.args.Composite, where the keys
      are item names, and values are checkpointable objects.
    """

  @abc.abstractmethod
  def item_metadata(
      self, step: int
  ) -> Union[Any, Mapping[str, Any], args_lib.Composite]:
    """For all Checkpointers, returns any metadata associated with the item.

    Calls the `metadata` method for each Checkpointer and returns a
    mapping of each item name to the restored metadata. If the manager only
    manages a single item, a single metadata will be returned instead.

    To avoid errors due to missing CheckpointHandlers, concrete
    CheckpointManager constructor must allow mapping from item names to
    respective CheckpointHandlers to be input other than via save() and
    restore(). Please note that save() and restore() calls automatically
    map CheckpointHandlers to respective item names and retain it during the
    lifetime of the CheckpointManager instance.

    Example::

      # Single item
      mngr = ocp.CheckpointManager(directory)
      # No calls to save() or restore() before calling item_metadata().
      mngr.item_metadata(step)  # Raises error.

      mngr = ocp.CheckpointManager(directory,
          item_handlers=ocp.StandardCheckpointHandler)
      # No calls to save() or restore() before calling item_metadata().
      metadata = mngr.item_metadata(step)  # Successful.

      # Multiple items
      mngr = ocp.CheckpointManager(directory, item_names=('state', 'extra'))
      # No calls to save() or restore() before calling item_metadata().
      mngr.item_metadata(step)  # Raises error.

      mngr = ocp.CheckpointManager(directory,
        item_names=('state', 'extra'),
        item_handlers={
            'state': ocp.StandardCheckpointHandler,
            'extra': ocp.PytreeCheckpointHandler,
        }
      )
      # No calls to save() or restore() before calling item_metadata().
      metadata = mngr.item_metadata(step)  # Successful.

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
  def metrics(self, step: int) -> Optional[PyTree]:
    """Returns metrics for step, if present."""

  @abc.abstractmethod
  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """

  @abc.abstractmethod
  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
