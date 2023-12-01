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

"""A class providing functionalities for managing multiple checkpoints."""

import concurrent.futures
import contextlib
import dataclasses
import datetime
import threading
from typing import Any, Callable, Container, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization
import numpy as np
from orbax.checkpoint import utils
# pylint: disable=g-importing-member
from orbax.checkpoint.abstract_checkpoint_manager import AbstractCheckpointManager
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpointer import Checkpointer
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler
from orbax.checkpoint.proto_checkpoint_handler import ProtoCheckpointHandler
# pylint: enable=g-importing-member
PyTree = Any
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams
CheckpointersDict = Mapping[str, AbstractCheckpointer]

DEFAULT_ITEM_NAME = 'default'
DESCRIPTOR_ITEM_NAME = 'descriptor'
METRIC_ITEM_NAME = 'metrics'
METADATA_ITEM_NAME = 'metadata'

RESERVED_ITEM_NAMES = [DESCRIPTOR_ITEM_NAME, METRIC_ITEM_NAME]

_INIT_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


def _metrics_file_exists(metrics_item_path: epath.Path) -> bool:
  """True if item directory AND actual file both exist."""
  return (
      metrics_item_path.exists()
      and (metrics_item_path / METRIC_ITEM_NAME).exists()
  )


def _descriptor_file_exists(descriptor_item_path: epath.Path) -> bool:
  """True if item directory AND actual file both exist."""
  return (
      descriptor_item_path.exists()
      and (descriptor_item_path / f'{DESCRIPTOR_ITEM_NAME}.pbtxt').exists()
  )


class _FinalizeThread(threading.Thread):
  """Thread wrapper that raises an exception if encountered."""

  exception = None

  def run(self):
    try:
      super().run()
    except BaseException as e:  # pylint:disable=broad-exception-caught
      self.exception = e

  def join(self, *args, **kwargs):
    super().join(*args, **kwargs)
    if self.exception:
      exception = self.exception
      self.exception = None
      raise exception


# TODO(b/268051457) Clean up when no longer depended upon by internal users.
def is_async_checkpointer(checkpointer: AbstractCheckpointer):
  return isinstance(checkpointer, AsyncCheckpointer) or isinstance(
      checkpointer,
      serialization.GlobalAsyncCheckpointManagerBase,
  )


# TODO(b/309965339) Set todelete_subdir defaults if directory is on CNS.
@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  save_interval_steps:
    The interval at which checkpoints should be saved.
    Ensures checkpoints will only be saved every n steps. Defaults to 1.
  max_to_keep:
    If provided, specifies the maximum number of checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present. Checkpoints are kept if they meet any of the conditions
    below, such as `keep_time_interval`, `keep_period`, etc. Any remaining
    checkpoints that do not meet these conditions are garbage-collected.
  keep_time_interval:
    When more than max_to_keep checkpoints are present,
    an older checkpoint that would ordinarily be deleted will be preserved if it
    has been at least `keep_time_interval` since the previous preserved
    checkpoint. The default setting of `None` does not preserve any checkpoints
    in this way. For example, this may be used to ensure checkpoints are
    retained at a frequency of approximately than one per hour.
  keep_period:
    If set, will not delete any checkpoint where checkpoint_step %
    keep_period == 0.
  best_fn:
    If set, maintains checkpoints based on the quality of given
    metrics rather than recency. The function should accept a PyTree of metrics,
    and return a scalar value that can be used to determine the quality score
    of the checkpoint. If `max_to_keep` is also set, then the retained
    checkpoints will be kept based on their quality, as measured by this
    function.
  best_mode:
    One of ['max', 'min']. The best metric is determine on the basis of this
    value.
  keep_checkpoints_without_metrics:
    If False, checkpoints without metrics present
    are eligible for cleanup. Otherwise, they will never be deleted.
  step_prefix:
    If provided, step directories will take the form
    f'{step_prefix}_<step>'. Otherwise, they will simply be an integer <step>.
  step_format_fixed_length:
    If set, formats step with n digits (leading zeros).
    This makes sorting steps easier. Otherwise, step has no leading zeros.
  create:
    If True, creates the top-level directory if it does not already exist.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  save_on_steps:
    Optional set of steps at which checkpoints should be saved.
    Useful to save checkpoints on a fixed set of steps that are not multiple of
    `save_interval_steps`.
  single_host_load_and_broadcast:
    If True, calling `all_steps(read=True)` will load on only a single host, and
    will then be broadcast to other hosts. Otherwise, I/O will be performed on
    every host. This can be helpful to reduce QPS to the filesystem if there
    are a large number of hosts.
  todelete_subdir: If set, checkpoints to be deleted will be only renamed into a
    subdirectory with the provided string. Otherwise, they will be directly
    deleted from the file system. Useful if checkpoint deletion is time
    consuming. By default, delete the checkpoint assets. Ignored if file system
    is Google Cloud Storage (directory is prefixed with gs://)
  read_only: If True, then checkpoints save and delete are skipped. However,
    checkpoints restore works as usual.
  """

  save_interval_steps: int = 1
  max_to_keep: Optional[int] = None
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None
  best_fn: Optional[Callable[[PyTree], float]] = None
  best_mode: str = 'max'
  keep_checkpoints_without_metrics: bool = True
  step_prefix: Optional[str] = None
  step_format_fixed_length: Optional[int] = None
  create: bool = True
  cleanup_tmp_directories: bool = False
  save_on_steps: Optional[Container[int]] = None
  single_host_load_and_broadcast: bool = False
  todelete_subdir: Optional[str] = None
  read_only: bool = False

  def __post_init__(self):
    if self.best_mode not in ('min', 'max'):
      msg = (
          "`CheckpointManagerOptions.best_mode` must be one of None, 'min' "
          "or 'max'. Got {self.dtype}."
      )
      raise ValueError(msg)
    if self.max_to_keep is not None and self.max_to_keep < 0:
      raise ValueError('Setting of `max_to_keep` must be None or non-negative.')
    if self.read_only and self.save_interval_steps > 0:
      raise ValueError(
          'CheckpointManagerOptions.save_interval_steps must be 0 as'
          ' read_only=True.'
      )
    if self.read_only and self.max_to_keep is not None:
      raise ValueError(
          'CheckpointManagerOptions.max_to_keep must be None as read_only=True.'
      )
    if self.read_only and self.keep_time_interval is not None:
      raise ValueError(
          'CheckpointManagerOptions.keep_time_interval must be None as'
          ' read_only=True.'
      )
    if self.read_only and self.keep_period is not None:
      raise ValueError(
          'CheckpointManagerOptions.keep_period must be None as read_only=True.'
      )
    if self.read_only and self.create:
      raise ValueError(
          'CheckpointManagerOptions.create must be False as read_only=True.'
      )
    if self.read_only and self.cleanup_tmp_directories:
      raise ValueError(
          'CheckpointManagerOptions.cleanup_tmp_directories must be False as'
          ' read_only=True.'
      )
    if self.read_only and self.save_on_steps is not None:
      raise ValueError(
          'CheckpointManagerOptions.save_on_steps must be None as'
          ' read_only=True.'
      )
    if self.read_only and self.todelete_subdir is not None:
      raise ValueError(
          'CheckpointManagerOptions.todelete_subdir must be None as'
          ' read_only=True.'
      )
    self.save_on_steps = frozenset(self.save_on_steps or ())


@dataclasses.dataclass
class CheckpointInfo:
  """Metadata about a checkpoint."""
  step: int
  time: datetime.datetime
  metrics: Optional[PyTree]
  is_locked: Optional[bool] = None

  def __str__(self) -> str:
    return f'Checkpoint[step={self.step} | time={self.time}]'

  def __eq__(self, other: 'CheckpointInfo') -> bool:
    return self.step == other.step and self.time == other.time


class CheckpointManager(AbstractCheckpointManager):
  """A generic, synchronous AbstractCheckpointManager implementation."""

  def __init__(
      self,
      directory: epath.PathLike,
      checkpointers: Union[AbstractCheckpointer, CheckpointersDict],
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[Mapping[str, Any]] = None,
  ):
    """CheckpointManager constructor.

    Example::

      CheckpointManager(
        'path/to/dir/',
        # Multiple items.
        checkpointers = {
            'train_state': AsyncCheckpointer(PyTreeCheckpointHandler()),
            'dataset': Checkpointer(CustomTFDatasetCheckpointHandler()),
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
        with a single item rather than a dictionary of items. See below for more
        details.
     options: CheckpointManagerOptions. May be provided to specify additional
       arguments. If None, uses default values of CheckpointManagerOptions.
     metadata: High-level metadata that does not depend on step number. If
       `directory` is write enabled then given metadata is saved only once. A
       new CheckpointManager instance with that `directory` does not overwrite
       the existing metadata and ignores the current given metadata. If
       `directory` is read-only then the current given metadata is not saved as
       expected. A CheckpointManager instance with a read-only `directory`
       uses the metadata if already present, otherwise always uses the current
       given metadata.
    """
    jax.monitoring.record_event('/jax/orbax/checkpoint_manager/init')
    self._single_item = False
    if isinstance(checkpointers, AbstractCheckpointer):
      self._single_item = True
      checkpointers = {DEFAULT_ITEM_NAME: checkpointers}
    elif isinstance(checkpointers, dict):
      for item in [k for k in checkpointers if k in RESERVED_ITEM_NAMES]:
        raise ValueError(
            f'Found {item} in `checkpointers`; this is a reserved key.'
        )
    else:
      raise ValueError(
          f'Invalid type for `checkpointers`. Found {checkpointers}.'
      )

    self._checkpointers = checkpointers
    self._options = options or CheckpointManagerOptions()
    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')
    if self._track_best:
      self._checkpointers[METRIC_ITEM_NAME] = Checkpointer(
          JsonCheckpointHandler(filename=METRIC_ITEM_NAME)
      )

    self._directory = epath.Path(directory)
    if self._options.read_only:
      logging.warning('Given directory is read only=%s', self._directory)
    if self._options.create:
      if jax.process_index() == 0 and not self._directory.exists():
        self._directory.mkdir(parents=True)
      utils.sync_global_devices('CheckpointManager:create_directory')

    # Cleanup directories from previous runs that may not have been finalized.
    if self._options.cleanup_tmp_directories:
      self._cleanup_tmp_directories()
    self._checkpoints = self._create_checkpoints()
    self._interval_preserved_checkpoints = (
        self._get_interval_preserved_checkpoints(self._checkpoints)
    )
    if self._checkpoints:
      self._last_checkpoint = self._checkpoints[-1]
    else:
      self._last_checkpoint = None

    if self._options.read_only and not self._metadata_path().exists():
      self._metadata = {} if metadata is None else metadata
    else:
      self._metadata = None
    if metadata is not None and not self._options.read_only:
      self._save_metadata(metadata)

    self._finalize_thread = None
    # Steps that get cleaned up during finalize.
    self._steps_to_remove = []

  @property
  def directory(self) -> epath.Path:
    """See superclass documentation."""
    return self._directory

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """See superclass documentation."""
    if read:
      if self._options.single_host_load_and_broadcast:
        max_steps = len(list(self.directory.iterdir()))
        # Read the step list only from host 0, and then broadcast the list.
        # This minimizes queries on non-leader processes.
        padded_step_list = np.array([-1] * max_steps)
        if jax.process_index() == 0:
          steps = np.array(utils.checkpoint_steps(self.directory))
          assert len(steps) <= max_steps
          padded_step_list[0 : len(steps)] = steps
        padded_step_list = multihost_utils.broadcast_one_to_all(
            padded_step_list
        )
        return [step for step in padded_step_list if step >= 0]
      else:
        return utils.checkpoint_steps(self.directory)
    else:
      return [ckpt.step for ckpt in self._checkpoints]

  def latest_step(self) -> Optional[int]:
    """See superclass documentation."""
    steps = self.all_steps(read=False)
    return max(steps) if steps else None

  def best_step(self) -> Optional[int]:
    """See superclass documentation."""
    if not self._track_best:
      return self.latest_step()
    if not self._checkpoints:
      return None
    _, sorted_checkpoints = self._sort_checkpoints_by_metrics(self._checkpoints)
    if not sorted_checkpoints:
      return None
    return sorted_checkpoints[-1].step

  def reached_preemption(self, step: int) -> bool:
    """See superclass documentation."""
    return utils.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    """See superclass documentation."""
    if self._options.read_only:
      logging.warning('%s is read only, save will be skipped', self.directory)
      return False
    if self.reached_preemption(step):
      return True
    last_checkpoint = self._last_checkpoint
    last_checkpoint_step = last_checkpoint.step if last_checkpoint else None
    # Ensure current step is between the last step and next step (accounting for
    # save interval). The `last_checkpoint_step` may not be initialized, in
    # which case we should save. Otherwise, step must fall on the specified
    # save interval. This condition accounts for the possibility of saving
    # on preemption, in which case we want to maintain the same save period as
    # if preemption had not happened.
    return last_checkpoint_step is None or (
        last_checkpoint_step < step and
        (step % self._options.save_interval_steps == 0 or
         step in self._options.save_on_steps))

  def _get_save_directory(
      self,
      step: int,
      directory: epath.Path,
      key_name: Optional[str] = None,
      tmp_directory: Optional[epath.Path] = None,
  ) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""
    return utils.get_save_directory(
        step,
        directory,
        name=key_name,
        step_prefix=self._options.step_prefix,
        override_directory=tmp_directory,
        step_format_fixed_length=self._options.step_format_fixed_length,
    )

  def _create_tmp_directory(self, directory: epath.Path) -> epath.Path:
    """Creates a tmp directory based on the given directory."""
    return utils.create_tmp_directory(directory)

  def delete(self, step: int):
    """See superclass documentation."""
    if self._options.read_only:
      logging.warning('%s is read only, delete will be skipped', self.directory)
      return
    if step not in self.all_steps():
      raise ValueError(f'Requested deleting a non-existent step: {step}.')
    self._delete_directory(step)
    utils.sync_global_devices('CheckpointManager:deleted_step')
    for i, info in enumerate(self._checkpoints):
      if info.step == step:
        self._checkpoints.pop(i)

  def save(self,
           step: int,
           items: Union[Any, Mapping[str, Any]],
           save_kwargs: Optional[Union[SaveParams, Mapping[str,
                                                           SaveParams]]] = None,
           metrics: Optional[PyTree] = None,
           force: Optional[bool] = False) -> bool:
    """See superclass documentation."""
    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    self.wait_until_finished()

    if not force and not self.should_save(step):
      return False
    if self.reached_preemption(step):
      logging.info('Saving checkpoint at step %d due to preemption.', step)

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

    save_directory = self._get_save_directory(step, self.directory)
    # If a folder for the step to save exists and is not finalized, remove the
    # existing folder.
    if (
        utils.is_gcs_path(save_directory)
        and save_directory.exists()
        and utils.is_tmp_checkpoint(save_directory)
    ):
      logging.warning(
          'Attempting to save at step %s which has an unfinalized checkpoint'
          ' from previous runs. Removing the unfinalized checkpoint before'
          ' saving.',
          step,
      )
      self._delete_directory(step)

    tmp_step_dir = self._create_tmp_directory(save_directory)

    for k, item in items.items():
      # Gets save dirs given top directory, step number, and a "collection". All
      # files from the same input object should be saved under this collection.
      item_dir = self._get_save_directory(
          step, self.directory, k, tmp_directory=tmp_step_dir
      )
      if k not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{k}" not found')
      kwargs = save_kwargs.get(k, {})
      self._checkpointers[k].save(item_dir, item, **kwargs)

    self._add_checkpoint_info(step, metrics)
    self._get_old_steps_to_remove()
    # Sync needed to ensure that old steps to remove are retrieved before
    # actually deleting them during finalize, since retrieval can involve
    # looking at the directory.
    utils.sync_global_devices('CheckpointManager:old_steps_to_remove')

    assert self._finalize_thread is None
    if self._all_checkpointers_are_sync:
      self._finalize(tmp_step_dir)
      utils.sync_global_devices('CheckpointManager:finalize')
    else:
      t = _FinalizeThread(target=self._finalize, args=(tmp_step_dir,))
      t.start()
      self._finalize_thread = t
    return True

  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[Union[RestoreParams,
                                     Mapping[str, RestoreParams]]] = None,
      directory: Optional[epath.PathLike] = None
  ) -> Union[Any, Mapping[str, Any]]:
    """See superclass documentation."""
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
        if not _metrics_file_exists(path):
          logging.warning('Missing metrics for step %d', step)
          continue
      if item_name not in self._checkpointers:
        raise ValueError(f'Checkpointer for item "{item_name}" not found')
      item = items.get(item_name, None)
      kwargs = restore_kwargs.get(item_name, {})
      restored[item_name] = self._checkpointers[item_name].restore(
          path, item=item, **kwargs)

    return restored

  def item_metadata(self, step: int) -> Union[Any, Mapping[str, Optional[Any]]]:
    """See superclass documentation."""
    result = {}
    for name, checkpointer in self._checkpointers.items():
      path = self._get_save_directory(step, self.directory, name)
      if name == METRIC_ITEM_NAME:
        assert self._track_best
        # No metrics file present: not an error.
        if not _metrics_file_exists(path):
          logging.warning('Missing metrics for step %d', step)
          continue
      metadata = checkpointer.metadata(path)
      result[name] = metadata
    if self._single_item:
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
    steps = sorted(self.all_steps(read=True))
    if not steps:
      return []

    def checkpoint_info(step: int) -> CheckpointInfo:
      time = datetime.datetime.fromtimestamp(
          self._get_save_directory(step, self.directory).stat().mtime,
          tz=datetime.timezone.utc,
      )

      metrics = None
      if self._track_best:
        restored = self._restore_impl(step, {METRIC_ITEM_NAME: None}, {})
        if METRIC_ITEM_NAME in restored:
          metrics = restored[METRIC_ITEM_NAME]
      return CheckpointInfo(step=step, time=time, metrics=metrics)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = {step: executor.submit(checkpoint_info, step) for step in steps}
      return [futures[step].result() for step in steps]

  def _get_interval_preserved_checkpoints(
      self, checkpoints: List[CheckpointInfo]
  ) -> List[CheckpointInfo]:
    """Gets which checkpoints should be kept based on keep_time_interval."""
    if not checkpoints:
      return []
    interval_preserved_checkpoints = [checkpoints[0]]
    if self._options.keep_time_interval is not None:
      for info in checkpoints[1:]:
        if (
            info.time
            >= interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          interval_preserved_checkpoints.append(info)
    return interval_preserved_checkpoints

  def _add_checkpoint_info(self, step: int, metrics: Optional[PyTree]):
    self._checkpoints.append(
        CheckpointInfo(step, datetime.datetime.now(tz=datetime.timezone.utc),
                       metrics))
    self._last_checkpoint = self._checkpoints[-1]
    # Only empty if this is the very first checkpoint. First checkpoint is
    # always preserved based on save_time_interval.
    if not self._interval_preserved_checkpoints:
      self._interval_preserved_checkpoints.append(self._checkpoints[-1])

  def _metadata_path(self) -> epath.Path:
    return self.directory / METADATA_ITEM_NAME

  def _save_metadata(self, metadata: Mapping[str, Any]):
    """Saves CheckpointManager level metadata, skips if already present."""
    path = self._metadata_path()
    if not path.exists():  # May have been created by a previous run.
      checkpointer = Checkpointer(JsonCheckpointHandler())
      checkpointer.save(path, metadata)

  def metadata(self) -> Mapping[str, Any]:
    """See superclass documentation."""
    if self._metadata is None:
      path = self._metadata_path()
      if path.exists():
        checkpointer = Checkpointer(JsonCheckpointHandler())
        self._metadata = checkpointer.restore(path)
      else:
        self._metadata = {}
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
    """Deletes step dir or renames it if options.todelete_subdir is set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    if jax.process_index() != 0:
      return

    # Delete if storage is on gcs or todelete_subdir is not set.
    if self._options.todelete_subdir is None or utils.is_gcs_path(
        self.directory
    ):
      self._get_save_directory(step, self.directory).rmtree()
      return

    # Rename step dir.
    rename_dir = self.directory / self._options.todelete_subdir
    if not rename_dir.exists():
      rename_dir.mkdir(parents=True)
    src = self._get_save_directory(step, self.directory)
    dst = self._get_save_directory(step, rename_dir)
    src.replace(dst)

  def _get_old_steps_to_remove(self):
    """Collects checkpoints that should be deleted later."""
    # Must have set max_to_keep in order to remove any checkpoints.
    if self._options.max_to_keep is None:
      return
    # Not enough checkpoints accumulated to consider deletion.
    if len(self._checkpoints) <= self._options.max_to_keep:
      return

    # Exclude the latest checkpoint, since it is not finalized.
    are_locked = utils.are_locked(
        self.directory,
        tuple([info.step for info in self._checkpoints[:-1]]),
        self._options.step_prefix,
        self._options.step_format_fixed_length,
    )
    self._checkpoints[:-1] = [
        dataclasses.replace(info, is_locked=is_locked)
        for info, is_locked in zip(self._checkpoints, are_locked)
    ]

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
      maybe_delete = (
          sorted_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      )
      active_checkpoints = (
          checkpoints_without_metrics + sorted_checkpoints[-keep:]
          if keep > 0
          else []
      )
    else:
      all_checkpoints = checkpoints_without_metrics + sorted_checkpoints
      maybe_delete = all_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      active_checkpoints = all_checkpoints[-keep:] if keep > 0 else []

    kept_checkpoints = []
    self._steps_to_remove = []
    for info in maybe_delete:
      if info.is_locked:
        logging.info(
            'Preserving %s: (Reason: checkpoint is locked).',
            info,
        )
        kept_checkpoints.append(info)
        continue
      if (
          self._options.keep_time_interval is not None
          and self._interval_preserved_checkpoints
      ):
        if info in self._interval_preserved_checkpoints:
          logging.info(
              'Preserving %s: (Reason: older falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.append(info)
          continue
        elif (
            info.time
            >= self._interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          self._interval_preserved_checkpoints.append(info)
          logging.info(
              'Preserving %s: (Reason: latest falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.append(info)
          continue

      if (
          self._options.keep_period is not None
          and info.step % self._options.keep_period == 0
      ):
        logging.info('Preserving %s: (Reason: on keep_period).', info)
        kept_checkpoints.append(info)
        continue

      reason = 'worse metric' if self._track_best else 'old checkpoint'
      logging.info('Deleting %s: (Reason: %s).', info, reason)
      self._steps_to_remove.append(info.step)

    kept_checkpoints += active_checkpoints
    if self._track_best:
      # Maintain in ascending step order.
      self._checkpoints = sorted(kept_checkpoints, key=lambda info: info.step)
    else:
      self._checkpoints = kept_checkpoints

  def _remove_old_steps(self):
    for step in self._steps_to_remove:
      self._delete_directory(step)
    self._checkpoints = [
        info
        for info in self._checkpoints
        if info.step not in self._steps_to_remove
    ]

  def wait_until_finished(self, join_finalize_thread=True):
    """See superclass documentation."""
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.wait_until_finished()  # pytype: disable=attribute-error
    if join_finalize_thread:
      t = self._finalize_thread
      if t is not None:
        self._finalize_thread = None
        try:
          t.join()
        except BaseException as e:  # pylint:disable=broad-exception-caught
          # If an exception occurred in the in finalization of the previous
          # save, we clean up since that checkpoint was never actually saved.
          self._last_checkpoint = (
              self._checkpoints[-2] if len(self._checkpoints) > 1 else None
          )
          self._interval_preserved_checkpoints.remove(self._checkpoints[-1])
          self._checkpoints = self._checkpoints[:-1]
          raise e
        # Additional work is being done on process 0 of the finalize threads.
        # When joining the threads, we must wait for all threads to complete
        # before proceeding.
        utils.sync_global_devices('CheckpointManager:join_finalize_thread')

  def check_for_errors(self):
    """See superclass documentation."""
    for checkpointer in self._checkpointers.values():
      if is_async_checkpointer(checkpointer):
        checkpointer.check_for_errors()  # pytype: disable=attribute-error

  def _finalize_checkpoint(
      self, temp_ckpt_dir: epath.Path
  ) -> Optional[epath.Path]:
    """Moves tmp step checkpoint to final.

    Args:
      temp_ckpt_dir: The temporary checkpoint directory. If not None, only
        finalize the checkpoints in `temp_ckpt_dir`. If None, it will iterate
        through all temp checkpoints in `self.directory` and finalize them all.

    Returns:
      the final checkpoint dir
    """
    final_ckpt_dir = None
    if jax.process_index() == 0:
      try:
        self.check_for_errors()
      except Exception as e:  # pylint: disable=broad-except
        logging.error(
            (
                'Received error: %s from Checkpointer. One or more items may'
                ' not be finalized. Skipping finalization of step checkpoint.'
            ),
            e,
        )
        return None
      step = utils.step_from_checkpoint_name(temp_ckpt_dir.name)
      # If at a preemption step, record the time since the previous checkpoint.
      # This represents training time that would otherwise have been wasted.
      # If another checkpoint has not been previously saved, measures the time
      # since program start.
      if self.reached_preemption(step):
        if len(self._checkpoints) > 1:
          previous_time = self._checkpoints[-2].time
        else:
          previous_time = _INIT_TIME
        assert self._last_checkpoint is not None
        duration = self._last_checkpoint.time - previous_time
        jax.monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/preempt/duration_saved_secs',
            duration.total_seconds(),
        )
      final_ckpt_dir = self._get_save_directory(step, self.directory)
      utils.ensure_atomic_save(temp_ckpt_dir, final_ckpt_dir)
    return final_ckpt_dir

  def _finalize(self, temp_ckpt_dir: epath.Path):
    """Cleans up old checkpoints and synchronizes hosts."""
    self.wait_until_finished(join_finalize_thread=False)
    # If an error is encountered while waiting for commit futures to complete,
    # we will not proceed past this point.
    final_ckpt_dir = self._finalize_checkpoint(temp_ckpt_dir)
    self._remove_old_steps()

  def close(self):
    """See superclass documentation."""
    self.wait_until_finished()
    for c in self._checkpointers.values():
      c.close()


@contextlib.contextmanager
def checkpoint_manager_context(*args, **kwargs):
  """Context manager for CheckpointManager.

  Initializes CheckpointManager and closes the object when the context is
  exited.

  Args:
    *args: Arguments to initialize CheckpointManager.
    **kwargs: Keyword arguments to initialize CheckpointManager.

  Usage::

    with checkpoint_manager_context(
        directory, checkpointers, options) as mngr:
      mngr.save(...)
      mngr.all_steps()

  Yields:
    CheckpointManager
  """
  manager = CheckpointManager(*args, **kwargs)
  try:
    yield manager
  finally:
    manager.close()
