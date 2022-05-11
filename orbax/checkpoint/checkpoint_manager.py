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

"""CheckpointManager, a sync implementation of AbstractCheckpointManager."""

import asyncio
import dataclasses
import logging
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.json_checkpointer import JsonCheckpointer
import tensorflow as tf

PyTree = type(jax.tree_structure(None))
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams

_DEFAULT_ITEM_NAME = 'default'
_METRIC_ITEM_NAME = 'metrics'


# TODO(cpgaffney) consider moving to common util.
def _cleanup_tmp_directories(directory: str):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  if jax.process_index() == 0:
    steps = utils.tmp_checkpoint_steps(directory)
    for step in steps:
      if not utils.is_checkpoint_finalized(directory, step):
        tf.io.gfile.rmtree(tf.io.gfile.join(directory, str(step)))

  multihost_utils.sync_global_devices('cleanup_tmp_dirs')


# TODO(cpgaffney) consider moving to common util.
def _create_save_directories(
    step: int, directory: str,
    names: Sequence[str]) -> Mapping[str, CheckpointDirs]:
  """Creates a temporary directory for saving at the given path and step.

  After completion, the filesystem will have the structure:

  directory/
    step/
      name.orbax-checkpoint-tmp-<timestamp>/

  Args:
    step: integer step.
    directory: directory to create.
    names: a list of names, where each name corresponds to a savable object

  Returns:
    A pair of final_dir and tmp_dir (intermediate operations should be written
    to tmp_dir before move to final_dir).
  """
  # Share a timestamp across devices.
  timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))

  dirs = {}
  for name in names:
    final_dir = tf.io.gfile.join(directory, str(step), name)
    tmp_dir = final_dir + utils.TMP_DIR_SUFFIX + f'{timestamp}'

    if tf.io.gfile.exists(final_dir):
      logging.info(
          'Directory already exists for item: %s at step: %d', name, step)

    if jax.process_index() == 0:
      assert not tf.io.gfile.exists(tmp_dir)
      tf.io.gfile.makedirs(tmp_dir)

    dirs[name] = (final_dir, tmp_dir)

  multihost_utils.sync_global_devices('make_dir')

  return dirs


# TODO(cpgaffney) Rely only on async methods when possible.
async def _call_valid_checkpointer_save(ckptr: Checkpointer, *args, **kwargs):
  try:
    return await ckptr.async_save(*args, **kwargs)
  except NotImplementedError:
    return ckptr.save(*args, **kwargs)


async def _call_valid_checkpointer_restore(ckptr: Checkpointer, *args,
                                           **kwargs):
  try:
    return await ckptr.async_restore(*args, **kwargs)
  except NotImplementedError:
    return ckptr.restore(*args, **kwargs)


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  save_interval_steps: the interval at which checkpoints should be saved.
  Ensures checkpoints will only be saved every n steps. Defaults to 1.
  max_to_keep: if provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints.
  best_fn: if set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score
    of the checkpoint. If `max_to_keep` is also set, then the retained
    checkpoints will be kept based on their quality, as measured by this
    function.
  best_mode: one of ['max', 'min']. The best metric is determine on the basis of
    this value.

  """
  save_interval_steps: int = 1
  max_to_keep: Optional[int] = None
  best_fn: Optional[Callable[[PyTree], float]] = None
  best_mode: str = 'max'


class CheckpointManager(AbstractCheckpointManager):
  """A generic, synchronous CheckpointManager implementation.

  Allows a user to save and restore objects for which a Checkpointer
  implementation exists (e.g. PyTreeCheckpointer for PyTrees). The class keeps
  track of multiple checkpointable objects in the following structure:

  path/to/directory/    (top-level directory)
    0/    (step)
      params/    (first saveable)
        ...
      dataset/    (second saveable)
        ...
    1/    (step)
      ...
    2/    (step)
      ...
    ...
  """

  def __init__(
      self,
      directory: str,
      checkpointers: Union[Checkpointer, Mapping[str, Checkpointer]],
      options: Optional[CheckpointManagerOptions] = None,
  ):
    """CheckpointManager constructor.

    Args:
      directory: the top level directory in which to save all files.
      checkpointers: a mapping of object name to Checkpointer object. For
        example, `items` provided to `save` below should have keys matching the
        keys in this argument. Alternatively, a single Checkpointer may be
        provided, in which See below for more details.
     options: CheckpointManagerOptions. May be provided to specify additional
       arugments. If None, uses default values of CheckpointManagerOptions.
    """
    self._directory = directory
    self._single_item = False
    if isinstance(checkpointers, Checkpointer):
      self._single_item = True
      checkpointers = {_DEFAULT_ITEM_NAME: checkpointers}
    elif _METRIC_ITEM_NAME in checkpointers:
      raise ValueError(
          f'Found {_METRIC_ITEM_NAME} in `checkpointers`; this is a reserved key.'
      )
    self._checkpointers = dict(checkpointers)
    self._checkpointers[_METRIC_ITEM_NAME] = JsonCheckpointer(
        filename=_METRIC_ITEM_NAME)
    self._last_checkpoint_step = -1
    self._options = options or CheckpointManagerOptions()
    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')
    self._past_metrics = {}
    # Cleanup directories from previous runs that may not have been finalized.
    _cleanup_tmp_directories(self._directory)

  @property
  def directory(self):
    return self._directory

  def all_steps(self) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Returns:
      A sequence of steps (integers)
    """
    return utils.checkpoint_steps(self.directory)

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    steps = self.all_steps()
    return sorted(steps)[-1] if steps else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    if not self._track_best:
      return self.latest_step()
    steps = self.all_steps()
    if not steps:
      return None
    self._populate_past_metrics()
    return self._steps_sorted_by_metric(steps)[-1]

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    return step > self._last_checkpoint_step and (
        self._last_checkpoint_step == -1 or
        step == self._options.save_interval_steps + self._last_checkpoint_step)

  def save(self,
           step: int,
           items: Union[Any, Mapping[str, Any]],
           save_kwargs: Optional[Union[SaveParams, Mapping[str,
                                                           SaveParams]]] = None,
           metrics: Optional[PyTree] = None,
           force: Optional[bool] = False) -> bool:
    """Saves the provided items.

    Items and save_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and save_kwargs, a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following:
    {
      'params': PyTree(),
      'dataset': tf.data.Iterator(),
      ...
    }
    Similarly, save_kwargs takes the form:
    {
      'params': {
        <kwargs for PyTreeCheckpointer.save>
      },
      'dataset': {
        <kwargs for DatasetCheckpointer.save>
      }
      ...
    }
    The kwargs under 'params' correspond to PyTreeCheckpointer.save. If a
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
      force: if True, forces a save regardless of `self.should_save`. Will
        overwrite any existing files in the checkpoint for `step`.
    Returns:
      bool indicating whether a save operation was performed
    Raises:
      ValueError: if `track_best` was indicated but `metrics` is not provided.
      ValueError: directory creation failed.
      ValueError: if an item is provided for which no `Checkpointer` is found.
    """
    if not force and not self.should_save(step):
      logging.info('Skipping save for step: %d', step)
      return False

    if save_kwargs is None:
      save_kwargs = {}
    if self._single_item:
      items = {_DEFAULT_ITEM_NAME: items}
      save_kwargs = {_DEFAULT_ITEM_NAME: save_kwargs}
    else:
      items = dict(items)

    if self._track_best:
      if metrics is None:
        raise ValueError(
            'Requested `tracked_metric`; must provide `metrics` for save.')
      items[_METRIC_ITEM_NAME] = metrics

    # creates tmp_dir given top directory, step number, and a "collection". All
    # files from the same input object should be saved under this collection.
    dir_pairs = _create_save_directories(step, self._directory,
                                         list(items.keys()))
    if not force and dir_pairs.keys() != items.keys():
      logging.info('Skipping save. Some items at step: %d already exist', step)
      return False

    save_ops = []
    saved_keys = []
    for k, item in items.items():
      if k not in dir_pairs:
        raise ValueError(f'Failed to create directory for item "{k}"')
      if k not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{k}" not found')
      final_dir, tmp_dir = dir_pairs[k]

      logging.info('Saving checkpoint for step %d to %s', step, tmp_dir)
      kwargs = save_kwargs.get(k, {})
      saved_keys += [k]
      save_ops += [
          _call_valid_checkpointer_save(self._checkpointers[k], tmp_dir, item,
                                        **kwargs)
      ]

    # schedule save operations to run concurrently
    async def _save(ops):
      return await asyncio.gather(*ops)

    if save_ops:
      asyncio.run(_save(save_ops))
    multihost_utils.sync_global_devices('CheckpointManager:write')

    # Ensure save operation atomicity
    if jax.process_index() == 0:
      for k in saved_keys:
        final_dir, tmp_dir = dir_pairs[k]
        if not force:
          assert not tf.io.gfile.exists(final_dir)
        if tf.io.gfile.exists(final_dir):
          tf.io.gfile.rmtree(final_dir)
        tf.io.gfile.rename(tmp_dir, final_dir)

    multihost_utils.sync_global_devices('CheckpointManager:save')

    self._last_checkpoint_step = step
    if self._track_best:
      self._past_metrics[step] = metrics
      self._populate_past_metrics()
    if jax.process_index() == 0:
      self._remove_old_checkpoints()

    multihost_utils.sync_global_devices('CheckpointManager:removed_old')
    return True

  def restore(self,
              step: int,
              items: Optional[Union[Any, Mapping[str, Any]]] = None,
              restore_kwargs: Optional[Union[RestoreParams,
                                             Mapping[str,
                                                     RestoreParams]]] = None,
              directory: Optional[str] = None) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    Items and restore_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and retore_kwargs,
    a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following:
    {
      'params': PyTree(),
      'dataset': tf.data.Iterator(),
      ...
    }
    Items may not be provided at all, in which case it the items restored are
    those specified in self._checkpointers, and item=None is provided to
    Checkpointer.restore. Similarly, an item may be ommitted from `items`, in
    which case item=None will be provided to Checkpointer.restore.

    Similarly, restore_kwargs takes the form:
    {
      'params': {
        'meshes': PyTree(),
        'mesh_axes': PyTree(),
      },
      'dataset': {
        <kwargs for DatasetCheckpointer.save>
      }
      ...
    }
    The kwargs under 'params' correspond to PyTreeCheckpointer.restore. If a
    key is not present in restore_kwargs, it is assumed that no kwargs are
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
      object returned for each Checkpointer, or a single restored object, if a
      single item is being tracked by this manager.
    """
    if items is None:
      items = {}
    elif self._single_item:
      items = {_DEFAULT_ITEM_NAME: items}
    if restore_kwargs is None:
      restore_kwargs = {}
    elif self._single_item:
      restore_kwargs = {_DEFAULT_ITEM_NAME: restore_kwargs}

    restored_items = self._restore_impl(
        step, items, restore_kwargs, directory=directory)

    if self._single_item:
      return restored_items[_DEFAULT_ITEM_NAME]
    return restored_items

  def _restore_impl(self,
                    step: int,
                    items: Mapping[str, Any],
                    restore_kwargs: Mapping[str, RestoreParams],
                    directory: Optional[str] = None) -> Mapping[str, Any]:
    """Restores only the provided items, or all items if empty."""
    directory = directory or self.directory
    restore_ops = []
    keys = []

    item_keys_to_restore = items.keys() if len(
        items) else self._checkpointers.keys()
    for k in item_keys_to_restore:
      # No metrics file expected: do not restore
      if k == _METRIC_ITEM_NAME and not self._track_best:
        continue
      if k not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{k}" not found')
      item = items.get(k, None)
      kwargs = restore_kwargs.get(k, {})
      path = tf.io.gfile.join(directory, str(step), k)
      keys += [k]
      restore_ops += [
          _call_valid_checkpointer_restore(self._checkpointers[k], path, item,
                                           **kwargs)
      ]

    # schedule restore ops to run concurrently
    async def _restore(ops):
      return await asyncio.gather(*ops)

    restored = asyncio.run(_restore(restore_ops))

    multihost_utils.sync_global_devices('CheckpointManager:restore')
    return {k: v for k, v in zip(keys, restored)}

  @property
  def _track_best(self):
    """Returns true if we should track the best checkpoints by given metric."""
    return self._options.best_fn is not None

  def _populate_past_metrics(self):
    """Loads metrics for past steps if not already set in `_past_metrics`."""
    for past_step in self.all_steps():
      if past_step not in self._past_metrics:
        # only restore metrics for past_step
        self._past_metrics[past_step] = self._restore_impl(
            past_step, {_METRIC_ITEM_NAME: None}, {})[_METRIC_ITEM_NAME]

  def _steps_sorted_by_metric(self, steps):
    """Sorts `steps` in order of increasing metric quality."""
    return sorted(
        steps,
        key=lambda s: self._options.best_fn(self._past_metrics[s]),
        reverse=(self._options.best_mode == 'min'))

  def _remove_old_checkpoints(self):
    """Keeps the `max_to_keep` most recent checkpoint steps."""
    if not self._options.max_to_keep:
      return
    existing_steps = sorted(self.all_steps())

    if self._track_best:
      # best (to keep) at the end
      existing_steps = self._steps_sorted_by_metric(existing_steps)

    to_remove = len(existing_steps) - self._options.max_to_keep
    if to_remove <= 0:
      return
    for step in existing_steps[:to_remove]:
      # TODO(cpgaffney) optimize.
      tf.io.gfile.rmtree(tf.io.gfile.join(self._directory, str(step)))
