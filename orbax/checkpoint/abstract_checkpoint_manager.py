# Copyright 2022 The Orbax Authors.
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

"""CheckpointManager interface."""

import abc
from typing import Any, Mapping, Optional, Sequence, Union

SaveParams = Mapping[str, Any]
RestoreParams = SaveParams


class AbstractCheckpointManager(abc.ABC):
  """An interface that manages multiple Checkpointer classes.

  CheckpointManager coordinates save/restore operations across multiple
  Checkpointer classes, and also provides useful methods describing
  checkpoint states.

  For example, CheckpointManager may be responsible for managing a parameter
  state in the form of a PyTree and a dataset iterator state in the form of
  tf.data.Iterator.

  Each item should be handled by a separate Checkpointer.

  For instance, item "a" is handled by Checkpointer A, while item "b" is
  handled by Checkpointer B.
  """

  @abc.abstractmethod
  def save(
      self,
      step: int,
      items: Union[Any, Mapping[str, Any]],
      save_kwargs: Optional[Union[SaveParams, Mapping[str, SaveParams]]] = None
  ) -> bool:
    """Saves the provided items.

    Items may a form similar to the following:
    {
      'params': PyTree(),
      'dataset': tf.data.Iterator(),
      ...
    }
    Each of these values is a saveable item that should be written with a
    specific Checkpointer.

    Similarly, save_kwargs takes the form:
    {
      'params': {
        <kwargs for PyTreeCheckpointHandler.save>
      },
      'dataset': {
        <kwargs for DatasetCheckpointHandler.save>
      }
      ...
    }
    The dict of kwargs for each key in save_kwargs is provided as extra
    arguments to the save method of the corresponding Checkpointer.

    Args:
      step: current step, int
      items: a savable object, or a dictionary of object name to savable object.
      save_kwargs: save kwargs for a single Checkpointer, or a dictionary
        of object name to kwargs needed by the Checkpointer implementation
        to save the object.

    Returns:
      bool indicating whether save was performed or not.
    """
    pass

  @abc.abstractmethod
  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[Union[RestoreParams,
                                     Mapping[str, RestoreParams]]] = None
  ) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    Items may a form similar to the following:
    {
      'params': PyTree(),
      'dataset': tf.data.Iterator(),
      ...
    }
    Each of these values is a restoreable item that should be read with a
    specific Checkpointer. Implementations should support items=None, and
    the ability to restore an item which is not provided in this dict.

    Similarly, restore_kwargs takes the form:
    {
      'params': {
        <kwargs for PyTreeCheckpointHandler.restore>
      },
      'dataset': {
        <kwargs for DatasetCheckpointHandler.restore>
      }
      ...
    }
    The dict of kwargs for each key in restore_kwargs is provided as extra
    arguments to the restore method of the corresponding Checkpointer.

    Args:
      step: current step, int
      items: a restoreable object, or a dictionary of object name to restoreable
        object.
      restore_kwargs: restore kwargs for a single Checkpointer, or a
        dictionary of object name to kwargs needed by the Checkpointer
        implementation to restore the object.

    Returns:
      A dictionary mapping name to restored object, or a single restored object.
    """
    pass

  @abc.abstractmethod
  def structure(self) -> Union[Any, Mapping[str, Any]]:
    """For all Checkpointers, returns the saved structure.

    Calls the `structure` method for each Checkpointer and returns a
    mapping of each item name to the restored structure. If the manager only
    manages a single item, a single structure will be returned instead.

    Note that any items for which the corresponding Checkpointer does not
    have an implemented `structure` method, these items will simply not be
    contained in the result.

    Returns:
      A dictionary mapping name to item structure, or a single item structure.
    """
    pass

  @abc.abstractmethod
  def all_steps(self) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Returns:
      A sequence of steps (integers)
    """
    pass

  @abc.abstractmethod
  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    pass

  @abc.abstractmethod
  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as measured by a metric.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    pass

  @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    """Indicates whether the CheckpointManager should perform a save operation at the current step."""
    pass

  @abc.abstractmethod
  def wait_until_finished(self):
    """Waits for ongoing save operations to complete.

    No-op if save is synchronous.
    """
    pass

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    pass
