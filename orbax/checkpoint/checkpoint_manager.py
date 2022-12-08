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
import datetime
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
from absl import logging

from etils import epath
import jax
from jax.experimental import multihost_utils
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler


PyTree = type(jax.tree_util.tree_structure(None))
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams

DEFAULT_ITEM_NAME = 'default'
METRIC_ITEM_NAME = 'metrics'
METADATA_ITEM_NAME = 'metadata'


def is_async_checkpointer(checkpointer: AbstractCheckpointer):
  return isinstance(checkpointer, AsyncCheckpointer)


async def _call_valid_checkpointer_save(checkpointer: AbstractCheckpointer,
                                        *args, **kwargs):
  if is_async_checkpointer(checkpointer):
    futures = await checkpointer.async_save(*args, **kwargs)  # pytype: disable=attribute-error
    return await asyncio.gather(*futures)
  else:
    return checkpointer.save(*args, **kwargs)


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  save_interval_steps: the interval at which checkpoints should be saved.
  Ensures checkpoints will only be saved every n steps. Defaults to 1.
  max_to_keep: if provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints.
  keep_time_interval: When more than max_to_keep checkpoints are present,
    an older checkpoint that would ordinarily be deleted will be preserved if it
    has been at least `keep_time_interval` since the previous preserved
    checkpoint. The default setting of `None` does not preserve any checkpoints
    in this way. For example, this may be used to ensure checkpoints are
    retained at a frequency of approximately than one per hour.
  keep_period: If set, will not delete any checkpoint where checkpoint_step %
    keep_period == 0.
  best_fn: if set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score
    of the checkpoint. If `max_to_keep` is also set, then the retained
    checkpoints will be kept based on their quality, as measured by this
    function.
  best_mode: one of ['max', 'min']. The best metric is determine on the basis of
    this value.
  keep_checkpoints_without_metrics: If False, checkpoints with metrics present
    are eligible for cleanup. Otherwise, they will never be deleted.
  step_prefix: if provided, step directories will take the form
    f'{step_prefix}_<step>'. Otherwise, they will simply be an integer <step>.

  """
  save_interval_steps: int = 1
  max_to_keep: Optional[int] = None
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None
  best_fn: Optional[Callable[[PyTree], float]] = None
  best_mode: str = 'max'
  keep_checkpoints_without_metrics: bool = True
  step_prefix: Optional[str] = None

  def __post_init__(self):
    if self.best_mode not in ('min', 'max'):
      msg = ("`CheckpointManagerOptions.best_mode` must be one of None, 'min' "
             "or 'max'. Got {self.dtype}.")
      raise ValueError(msg)


@dataclasses.dataclass
class CheckpointInfo:
  """Metadata about a checkpoint."""
  step: int
  time: datetime.datetime
  metrics: Optional[PyTree]


class CheckpointManager(AbstractCheckpointManager):
  """A generic, synchronous CheckpointManager implementation.

  Allows a user to save and restore objects for which a Checkpointer
  implementation exists (e.g. PyTreeCheckpointer for PyTrees). The class
  keeps track of multiple checkpointable objects in the following structure:

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

  def __init__(
      self,
      directory: epath.PathLike,
      checkpointers: Union[AbstractCheckpointer, Mapping[str,
                                                         AbstractCheckpointer]],
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[Mapping[str, Any]] = None,
  ):
    """CheckpointManager constructor.

    Ex:
    CheckpointManager(
        'path/to/dir/',
        # Multiple items.
        checkpointers = {
            'train_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
            'dataset': Checkpointer(CustomTFDatasetCheckpointHandler())
        },
        metadata={'version': 1.1, 'lang': 'en'},
    )

    CheckpointManager(
        'path/to/dir/',
        # Single item.
        checkpointers = AsyncCheckpointer(PyTreeCheckpointHandler()),
        options = CheckpointManagerOptions(max_to_keep=5, ...),
    )

    Args:
      directory: the top level directory in which to save all files.
      checkpointers: a mapping of object name to Checkpointer object. For
        example, `items` provided to `save` below should have keys matching the
        keys in this argument. Alternatively, a single Checkpointer may be
        provided, in which case `save` and `restore` should always be called
        with a single item rather than a dictionary of items.
        See below for more details.
     options: CheckpointManagerOptions. May be provided to specify additional
       arugments. If None, uses default values of CheckpointManagerOptions.
     metadata: High-level metadata that does not depend on step number, and only
       needs to be saved once.
    """
    self._directory = epath.Path(directory)
    self._single_item = False
    if isinstance(checkpointers, AbstractCheckpointer):
      self._single_item = True
      checkpointers = {DEFAULT_ITEM_NAME: checkpointers}
    elif isinstance(checkpointers, dict):
      if METRIC_ITEM_NAME in checkpointers:
        raise ValueError(
            f'Found {METRIC_ITEM_NAME} in `checkpointers`; this is a reserved'
            ' key.'
        )
    else:
      raise ValueError(
          f'Invalid type for `checkpointers`. Found {checkpointers}.')

    self._checkpointers = checkpointers
    self._options = options or CheckpointManagerOptions()
    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')
    if self._track_best:
      self._checkpointers[METRIC_ITEM_NAME] = Checkpointer(
          JsonCheckpointHandler(filename=METRIC_ITEM_NAME))

    # Cleanup directories from previous runs that may not have been finalized.
    self._cleanup_tmp_directories()
    self._checkpoints = self._create_checkpoints()
    # The distinction between _last_checkpoint and _last_preserved_checkpoint is
    # necessary because some checkpoints may not be kept in the long run, in
    # which case the two values may diverge. _last_preserved_checkpoint cannot
    # be determined until we know which checkpoints will not be saved but
    # ultimately discarded.
    # _last_preserved_checkpoint is used for preserving checkpoints based on
    # time interval. Be cautious if using in a broader context.
    if self._checkpoints:
      self._last_checkpoint = self._checkpoints[-1]
      self._last_preserved_checkpoint = self._checkpoints[-1]
    else:
      self._last_checkpoint = None
      self._last_preserved_checkpoint = None

    self._metadata = None
    if metadata is not None:
      self._save_metadata(metadata)

  @property
  def directory(self) -> epath.Path:
    return self._directory

  def all_steps(self) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Returns:
      A sequence of steps (integers)
    """
    return [ckpt.step for ckpt in self._checkpoints]

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    steps = self.all_steps()
    return max(steps) if steps else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    if not self._track_best:
      return self.latest_step()
    if not self._checkpoints:
      return None
    _, sorted_checkpoints = self._sort_checkpoints_by_metrics(self._checkpoints)
    if not sorted_checkpoints:
      return None
    return sorted_checkpoints[-1].step

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    # Check whether to save an on-demand checkpoint due to preemption.
    if (jax.config.jax_coordination_service and
        multihost_utils.reached_preemption_sync_point(step)):
      return True
    last_checkpoint_step = (
        self._last_checkpoint.step if self._last_checkpoint else None)
    # Ensure current step is between the last step and next step (accounting for
    # save interval). The `last_checkpoint_step` may not be initialized, in
    # which case we should save. Otherwise, step must fall on the specified
    # save interval. This condition accounts for the possibility of saving
    # on preemption, in which case we want to maintain the same save period as
    # if preemption had not happened.
    return last_checkpoint_step is None or (
        last_checkpoint_step < step and
        step % self._options.save_interval_steps == 0)

  def _get_save_directory(self,
                          step: int,
                          directory: epath.Path,
                          key_name: Optional[str] = None) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""
    return utils.get_save_directory(
        step, directory, name=key_name, step_prefix=self._options.step_prefix)

  # TODO(b/256160767) Potentially support updating any item within the
  # checkpoint. Note that metrics is optional because we anticipate adding
  # an additional non-optional `item` argument in the future.
  def update(self, step: int, metrics: Optional[PyTree] = None):
    """Updates a portion of an already existing checkpoint.

    Currently only metrics can be updated.

    Args:
      step: step to update
      metrics: metrics with which to replace the currently saved metrics.

    Raises:
      ValueError: if metrics is None.
      ValueError: if metrics tracking is not enabled.
      ValueError: if the requested step does not exist.
    """
    if step not in self.all_steps():
      raise ValueError('Requested updating a non-existent step.')
    if metrics is None:
      raise ValueError('Must provide metrics to update.')
    if not self._track_best and metrics is not None:
      raise ValueError(
          'Requested update metrics without configuring the CheckpointManager'
          ' to track metrics.'
      )

    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    self.wait_until_finished()

    assert METRIC_ITEM_NAME in self._checkpointers
    final_dir = self._get_save_directory(step, self._directory,
                                         METRIC_ITEM_NAME)
    self._checkpointers[METRIC_ITEM_NAME].save(final_dir, metrics)

    # TODO(b/256160767) Update time if the updated item is something other than
    # metrics.
    self._update_checkpoint_info(step, metrics, update_time=False)
    # If any are async, must wait until all saves are complete before finalize.
    if self._all_checkpointers_are_sync:
      self._finalize()

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
      force: if True, forces a save regardless of `self.should_save`.

    Returns:
      bool indicating whether a save operation was performed.
    Raises:
      ValueError: if `track_best` was indicated but `metrics` is not provided.
      ValueError: directory creation failed.
      ValueError: if an item is provided for which no `Checkpointer` is
      found.
      ValueError: if the checkpoint already exists.
    """
    if not force and not self.should_save(step):
      return False

    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    self.wait_until_finished()

    if step in self.all_steps():
      raise ValueError(f'Checkpoint for step {step} already exists.')

    if save_kwargs is None:
      save_kwargs = {}
    if self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
      save_kwargs = {DEFAULT_ITEM_NAME: save_kwargs}
    else:
      items = dict(items)

    if self._track_best:
      if metrics is None:
        logging.warning('Requested `tracked_metric`; did not provide metrics.')
      else:
        items[METRIC_ITEM_NAME] = metrics

    for k, item in items.items():
      # Gets save dirs given top directory, step number, and a "collection". All
      # files from the same input object should be saved under this collection.
      final_dir = self._get_save_directory(step, self._directory, k)
      if k not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{k}" not found')

      kwargs = save_kwargs.get(k, {})
      self._checkpointers[k].save(final_dir, item, **kwargs)

    self._add_checkpoint_info(step, metrics)
    # If any are async, must wait until all saves are complete before finalize.
    if self._all_checkpointers_are_sync:
      self._finalize()

    return True

  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[Union[RestoreParams,
                                     Mapping[str, RestoreParams]]] = None,
      directory: Optional[epath.PathLike] = None
  ) -> Union[Any, Mapping[str, Any]]:
    """Restores from the given step and provided items.

    Items and restore_kwargs must have a top-level structure matching that of
    self._checkpointers, meaning that for every key in items and restore_kwargs,
    a
    corresponding key must be present in self._checkpointers.

    Items takes a form similar to the following:
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

    Similarly, restore_kwargs takes the form:
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
    if items is None:
      items = {}
    elif self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
    if restore_kwargs is None:
      restore_kwargs = {}
    elif self._single_item:
      restore_kwargs = {DEFAULT_ITEM_NAME: restore_kwargs}

    restored_items = self._restore_impl(
        step, items, restore_kwargs, directory=directory)

    if self._single_item:
      return restored_items[DEFAULT_ITEM_NAME]
    return restored_items

  def _restore_impl(
      self,
      step: int,
      items: Mapping[str, Any],
      restore_kwargs: Mapping[str, RestoreParams],
      directory: Optional[epath.PathLike] = None) -> Mapping[str, Any]:
    """Restores only the provided items, or all items if empty."""
    if directory is None:
      directory = self.directory
    else:
      directory = epath.Path(directory)
    restored = {}
    item_keys_to_restore = items.keys() or self._checkpointers.keys()
    for item_name in item_keys_to_restore:
      path = self._get_save_directory(step, directory, item_name)
      if item_name == METRIC_ITEM_NAME:
        assert self._track_best
        # No metrics file present: not an error.
        if not path.exists():
          logging.warning('Missing metrics for step %d', step)
          continue
      if item_name not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{item_name}" not found')
      item = items.get(item_name, None)
      kwargs = restore_kwargs.get(item_name, {})
      restored[item_name] = self._checkpointers[item_name].restore(
          path, item=item, **kwargs)
    return restored

  def structure(self) -> Union[Any, Mapping[str, Any]]:
    """For all Checkpointers, returns the saved structure.

    Calls the `structure` method for each Checkpointer and returns a
    mapping of each item name to the restored structure. If the manager only
    manages a single item, a single structure will be returned instead.

    Note that any items for which the corresponding Checkpointer does not
    have an implemented `structure` method, these items will simply not be
    contained in the result. If, in this case, there is also only a single item
    managed, None will be returned.

    Returns:
      A dictionary mapping name to item structure, or a single item structure.
    """
    result = {}
    step = self.latest_step()
    if step is None:
      raise ValueError(
          'No existing checkpoint; structure cannot be determined.')
    for name, checkpointer in self._checkpointers.items():
      path = self._get_save_directory(step, self.directory, name)
      if name == METRIC_ITEM_NAME:
        assert self._track_best
        # No metrics file present: not an error.
        if not path.exists():
          logging.warning('Missing metrics for step %d', step)
          continue
      structure = checkpointer.structure(path)
      # If None, then the item has no defined structure, and should be excluded.
      # May be empty, which would simply represent a valid, but empty structure.
      if structure is not None:
        result[name] = structure
    if self._single_item:
      if DEFAULT_ITEM_NAME not in result:
        return None
      return result[DEFAULT_ITEM_NAME]
    return result

  @property
  def _track_best(self):
    """Returns true if we should track the best checkpoints by given metric."""
    return self._options.best_fn is not None

  @property
  def _all_checkpointers_are_sync(self):
    return all(not is_async_checkpointer(checkpointer)
               for checkpointer in self._checkpointers.values())

  def _create_checkpoints(self) -> List[CheckpointInfo]:
    """Create a list of CheckpointInfo for existing checkpoints.

    If none are present, returns empty list.

    Returns:
      a list of CheckpointInfo, sorted by increasing step.
    """
    steps = sorted(utils.checkpoint_steps(self.directory))
    if not steps:
      return []

    tz = datetime.timezone.utc
    times = [
        datetime.datetime.fromtimestamp(
            self._get_save_directory(step, self.directory).stat().mtime, tz=tz)
        for step in steps
    ]

    def get_metrics(step):
      if self._track_best:
        restored = self._restore_impl(step, {METRIC_ITEM_NAME: None}, {})
        if METRIC_ITEM_NAME in restored:
          return restored[METRIC_ITEM_NAME]
      return None

    metrics = [get_metrics(step) for step in steps]

    return [
        CheckpointInfo(step=s, time=t, metrics=m)
        for s, t, m in zip(steps, times, metrics)
    ]

  def _add_checkpoint_info(self, step: int, metrics: Optional[PyTree]):
    self._checkpoints.append(
        CheckpointInfo(step, datetime.datetime.now(tz=datetime.timezone.utc),
                       metrics))
    self._last_checkpoint = self._checkpoints[-1]
    # Only None if this is the very first checkpoint. First checkpoint is
    # always preserved.
    if self._last_preserved_checkpoint is None:
      self._last_preserved_checkpoint = self._checkpoints[-1]

  def _update_checkpoint_info(self, step, metrics, update_time=False):
    for i, info in enumerate(self._checkpoints):
      if info.step == step:
        self._checkpoints[i].metrics = metrics
        if update_time:
          self._checkpoints[i].time = datetime.datetime.now(
              tz=datetime.timezone.utc)
        return
    raise ValueError(
        'Attempting to update a step which is not currently present.')

  def _save_metadata(self, metadata: Mapping[str, Any]):
    checkpointer = Checkpointer(JsonCheckpointHandler())
    path = self.directory / METADATA_ITEM_NAME
    if not path.exists():  # May have been created by a previous run.
      checkpointer.save(path, metadata)

  def metadata(self) -> Mapping[str, Any]:
    if self._metadata is None:
      checkpointer = Checkpointer(JsonCheckpointHandler())
      path = self.directory / METADATA_ITEM_NAME
      self._metadata = checkpointer.restore(path)
    return self._metadata

  def _sort_checkpoints_by_metrics(
      self, checkpoints: List[CheckpointInfo]
  ) -> Tuple[List[CheckpointInfo], List[CheckpointInfo]]:
    """Sorts `checkpoints` in order of increasing metric quality.

    Checkpoints without corresponding metrics set will be at the beginning.

    Args:
      checkpoints: a list of CheckpointInfo.

    Returns:
      Tuple of CheckpointInfo lists:
      (checkpoints_without_metrics, checkpoints_sorted_by_metrics)
    """
    without_metrics = [info for info in checkpoints if info.metrics is None]
    with_metrics = [info for info in checkpoints if info.metrics is not None]

    return without_metrics, sorted(
        with_metrics,
        key=lambda info: self._options.best_fn(info.metrics),
        reverse=(self._options.best_mode == 'min'))

  def _cleanup_tmp_directories(self):
    utils.cleanup_tmp_directories(self.directory)

  def _delete_directory(self, step: int):
    if jax.process_index() == 0:
      # TODO(cpgaffney) Optimize tree removal if possible.
      self._get_save_directory(step, self.directory).rmtree()

  def _remove_old_checkpoints(self):
    """Keeps the `max_to_keep` most recent checkpoint steps."""
    # Must have set max_to_keep or keep_time_interval.
    if not self._options.max_to_keep and not self._options.keep_time_interval:
      return
    # Not enough checkpoints accumulated to consider deletion.
    if len(self._checkpoints) <= self._options.max_to_keep:
      return
    if self._track_best:
      # Best steps (to keep) are at the end, after sorting.
      (
          checkpoints_without_metrics,
          sorted_checkpoints,
      ) = self._sort_checkpoints_by_metrics(self._checkpoints)
    else:
      # checkpoints already sorted by ascending step
      checkpoints_without_metrics = []
      sorted_checkpoints = self._checkpoints

    keep = int(self._options.max_to_keep)
    if self._options.keep_checkpoints_without_metrics:
      maybe_delete = sorted_checkpoints[:-keep]
      active_checkpoints = checkpoints_without_metrics + sorted_checkpoints[
          -keep:]
    else:
      all_checkpoints = checkpoints_without_metrics + sorted_checkpoints
      maybe_delete = all_checkpoints[:-keep]
      active_checkpoints = all_checkpoints[-keep:]

    kept_checkpoints = []
    for info in maybe_delete:
      if (self._options.keep_time_interval is not None and
          self._last_preserved_checkpoint is not None):
        # Preserve if the checkpoint is older than the most recent preserved
        # checkpoint OR if its time is greater than the last preserved time plus
        # plus the given interval.
        if info.time <= self._last_preserved_checkpoint.time:
          kept_checkpoints.append(info)
          continue
        elif (info.time >= self._last_preserved_checkpoint.time +
              self._options.keep_time_interval):
          self._last_preserved_checkpoint = info
          kept_checkpoints.append(info)
          continue

      if (
          self._options.keep_period is not None
          and info.step % self._options.keep_period == 0
      ):
        kept_checkpoints.append(info)
        continue

      self._delete_directory(info.step)

    kept_checkpoints += active_checkpoints
    if self._track_best:
      # Maintain in ascending step order.
      self._checkpoints = sorted(kept_checkpoints, key=lambda info: info.step)
    else:
      self._checkpoints = kept_checkpoints

  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Cleans up old checkpoints.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished. Only at this point
    can old checkpoints be cleaned up, since previously some checkpoints may
    have been incomplete.
    """
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.wait_until_finished()  # pytype: disable=attribute-error
    self._finalize()

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.check_for_errors()  # pytype: disable=attribute-error

  def _finalize(self):
    """Cleans up old checkpoints and synchronizes hosts."""
    self._remove_old_checkpoints()
    multihost_utils.sync_global_devices('CheckpointManager:removed_old')
