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

"""A class providing functionalities for managing a series of checkpoints."""

import concurrent.futures
import contextlib
import dataclasses
import datetime
import threading
import time
import typing
from typing import Any, Callable, Container, List, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization import serialization
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import abstract_checkpointer
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import async_checkpointer
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_handler
from orbax.checkpoint import checkpointer as checkpointer_lib
from orbax.checkpoint import composite_checkpoint_handler
from orbax.checkpoint import json_checkpoint_handler
from orbax.checkpoint import proto_checkpoint_handler
from orbax.checkpoint import utils
from orbax.checkpoint.path import step as step_lib


PyTree = Any
CheckpointDirs = Tuple[str, str]
SaveParams = Mapping[str, Any]
RestoreParams = SaveParams
AbstractCheckpointer = abstract_checkpointer.AbstractCheckpointer
CheckpointersDict = Mapping[str, AbstractCheckpointer]
AbstractCheckpointManager = (
    abstract_checkpoint_manager.AbstractCheckpointManager
)
AsyncCheckpointer = async_checkpointer.AsyncCheckpointer
Checkpointer = checkpointer_lib.Checkpointer
JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
CheckpointHandler = checkpoint_handler.CheckpointHandler
CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandlersDict = Mapping[str, CheckpointHandler]

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
  return isinstance(
      checkpointer, async_checkpointer.AsyncCheckpointer
  ) or isinstance(
      checkpointer,
      serialization.GlobalAsyncCheckpointManagerBase,
  )


@dataclasses.dataclass
class AsyncOptions:
  """Options used to configure async behavior.

  See `AsyncCheckpointer` for details.
  """

  timeout_secs: int = 300
  barrier_sync_fn: Optional[async_checkpointer.BarrierSyncFn] = None


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
    If set, any existing checkpoints matching checkpoint_step % keep_period == 0
    will not be deleted.
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
  step_name_format:
    NameFormat to build or find steps under input root directory. If provided,
    `step_prefix`, `step_format_fixed_length` are ignored.
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
  enable_async_checkpointing: If True, enables async checkpointing.
  async_options: Used to configure properties of async behavior. See above.
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
  step_name_format: Optional[step_lib.NameFormat] = None
  create: bool = True
  cleanup_tmp_directories: bool = False
  save_on_steps: Optional[Container[int]] = None
  single_host_load_and_broadcast: bool = False
  todelete_subdir: Optional[str] = None
  read_only: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[AsyncOptions] = None

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
      self.save_interval_steps = 0
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' save_interval_steps=0.'
      )
    if self.read_only and self.max_to_keep is not None:
      self.max_to_keep = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting max_to_keep=None.'
      )
    if self.read_only and self.keep_time_interval is not None:
      self.keep_time_interval = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' keep_time_interval=None.'
      )
    if self.read_only and self.keep_period is not None:
      self.keep_period = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting keep_period=None.'
      )
    if self.read_only and self.create:
      self.create = False
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting create=False.'
      )
    if self.read_only and self.cleanup_tmp_directories:
      self.cleanup_tmp_directories = False
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' cleanup_tmp_directories=False.'
      )
    if self.read_only and self.save_on_steps:
      self.save_on_steps = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting save_on_steps=None.'
      )
    if self.read_only and self.todelete_subdir is not None:
      self.todelete_subdir = None
      logging.warning(
          'CheckpointManagerOptions.read_only=True, setting'
          ' todelete_subdir=None.'
      )
    self.save_on_steps = frozenset(self.save_on_steps or ())


@dataclasses.dataclass
class CheckpointInfo:
  """Metadata about a checkpoint."""

  step: int
  time: datetime.datetime
  metrics: Optional[PyTree]
  is_locked: Optional[bool] = None

  def __post_init__(self):
    # Users may provide step as a jax.Array.
    if isinstance(self.step, jax.Array):
      self.step = int(self.step)

  def __str__(self) -> str:
    return f'Checkpoint[step={self.step} | time={self.time}]'

  def __eq__(self, other: 'CheckpointInfo') -> bool:
    return self.step == other.step and self.time == other.time

  def __hash__(self) -> int:
    return hash((self.step, self.time))


def _get_args_for_key(
    handler: CheckpointHandler, item_name: str
) -> Tuple[Type[CheckpointArgs], Type[CheckpointArgs]]:
  if not isinstance(handler, CompositeCheckpointHandler):
    raise ValueError(
        'Expected handler to be a `CompositeCheckpointHandler`, but got'
        f' {type(handler)}.'
    )
  for key, handler in handler._known_handlers.items():  # pylint: disable=protected-access
    if key == item_name:
      return checkpoint_args.get_registered_args_cls(handler)
  raise ValueError(f'Unknown key "{item_name}" in CompositeCheckpointHandler.')


class CheckpointManager(AbstractCheckpointManager):
  """A generic, synchronous AbstractCheckpointManager implementation."""

  def __init__(
      self,
      directory: epath.PathLike,
      checkpointers: Optional[
          Union[AbstractCheckpointer, CheckpointersDict]
      ] = None,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[Mapping[str, Any]] = None,
      item_names: Optional[Sequence[str]] = None,
      item_handlers: Optional[
          Union[CheckpointHandler, CheckpointHandlersDict]
      ] = None,
      primary_host: Optional[int] = 0,
  ):
    """CheckpointManager constructor.

    IMPORTANT: `CheckpointManager` has been refactored to provide a new API.
    Please ensure you have migrated all existing use cases to the newer style by
    May 1st, 2024. Please see
    https://orbax.readthedocs.io/en/latest/api_refactor.html
    for technical details.

    The `CheckpointManager` is ultimately backed by a single `Checkpointer`, to
    which saving and restoring is delegated. Behind step management options,
    metrics-related logic, and other frills, saving and restoring with
    `CheckpointManager` is quite similar to using
    `Checkpointer(CompositeCheckpointHandler)`.

    Example::

      mngr = CheckpointManager(
        'path/to/dir/',
        # Multiple items.
        item_names=('train_state', 'custom_metadata'),
        metadata={'version': 1.1, 'lang': 'en'},
      )
      mngr.save(0, args=args.Composite(
          train_state=args.StandardSave(train_state),
          custom_metadata=args.JsonSave(custom_metadata),
        )
      )
      restored = mngr.restore(0)
      print(restored.train_state)
      print(restored.custom_metadata)
      restored = mngr.restore(0, args=args.Composite(
          train_state=args.StandardRestore(abstract_train_state),
        )
      )
      print(restored.train_state)
      print(restored.custom_metadata)  # Error, not restored

      # Single item, no need to specify `item_names`.
      mngr = CheckpointManager(
        'path/to/dir/',
        options = CheckpointManagerOptions(max_to_keep=5, ...),
      )
      mngr.save(0, args=StandardSave(train_state))
      train_state = mngr.restore(0)
      train_state = mngr.restore(0, args=StandardRestore(abstract_train_state))

    IMPORTANT: Don't forget to use the keyword `args=...` for save and restore!
    Otherwise you will get the legacy API. This will not be necessary forever,
    but only until the legacy API is removed.

    Args:
      directory: the top level directory in which to save all files.
      checkpointers: a mapping of object name to Checkpointer object. For
        example, `items` provided to `save` below should have keys matching the
        keys in this argument. Alternatively, a single Checkpointer may be
        provided, in which case `save` and `restore` should always be called
        with a single item rather than a dictionary of items. See below for more
        details. `item_names` and `checkpointers` are mutually exclusive - do
        not use together. Also, please don't use `checkpointers` and
        `item_handlers` together.
      options: CheckpointManagerOptions. May be provided to specify additional
        arguments. If None, uses default values of CheckpointManagerOptions.
      metadata: High-level metadata that does not depend on step number. If
        `directory` is write enabled then given metadata is saved only once. A
        new CheckpointManager instance with that `directory` does not overwrite
        the existing metadata and ignores the current given metadata. If
        `directory` is read-only then the current given metadata is not saved as
        expected. A CheckpointManager instance with a read-only `directory` uses
        the metadata if already present, otherwise always uses the current given
        metadata.
      item_names: Names of distinct items that may be saved/restored with this
        `CheckpointManager`. `item_names` and `checkpointers` are mutually
        exclusive - do not use together. Also see `item_handlers` below.
      item_handlers: A mapping of item name to `CheckpointHandler`. The mapped
        CheckpointHandler must be registered against the `CheckpointArgs` input
        in save/restore operations. Please don't use `checkpointers` and
        `item_handlers` together. It can be used with or without `item_names`.
        The item name key may or may not be present in `item_names`.
        Alternatively, a single CheckpointHandler may be provided, in which case
        `save` and `restore` should always be called in a single item context.
      primary_host: the host id of the primary host.  Default to 0.  If it's set
        to None, then all hosts will be considered as primary.  It's useful in
        the case that all hosts are only working with local storage.
    """
    jax.monitoring.record_event('/jax/orbax/checkpoint_manager/init')

    self._options = options or CheckpointManagerOptions()
    if self._options.best_mode not in ['min', 'max']:
      raise ValueError('`best_mode` must be one of: "min", "max"')

    self._primary_host = primary_host

    if checkpointers and item_names:
      raise ValueError(
          '`item_names` and `checkpointers` are mutually exclusive - do not use'
          ' together.'
      )
    if checkpointers and item_handlers:
      raise ValueError(
          '`item_handlers` and `checkpointers` are mutually exclusive - do not'
          ' use together.'
      )
    if item_names and isinstance(item_handlers, CheckpointHandler):
      raise ValueError(
          '`item_handlers` in single item mode and `item_names` should not be'
          ' provided together.'
      )
    if checkpointers:
      logging.warning(
          'Configured `CheckpointManager` using deprecated legacy API. Please'
          ' follow the instructions at'
          ' https://orbax.readthedocs.io/en/latest/api_refactor.html to'
          ' migrate by May 1st, 2024.'
      )
      self._single_item = isinstance(checkpointers, AbstractCheckpointer)
      self._checkpointer = self._configure_checkpointer_legacy_init(
          checkpointers, self._options
      )
    else:
      self._single_item = isinstance(item_handlers, CheckpointHandler) or (
          item_names is None and item_handlers is None
      )
      self._checkpointer = self._configure_checkpointer(
          item_names, item_handlers, self._options, self._single_item
      )

    self._directory = epath.Path(directory)
    if self._options.read_only:
      logging.warning('Given directory is read only=%s', self._directory)
    if self._options.create:
      if (
          utils.is_primary_host(self._primary_host)
          and not self._directory.exists()
      ):
        self._directory.mkdir(parents=True)
      utils.sync_global_devices('CheckpointManager:create_directory')


    # Cleanup directories from previous runs that may not have been finalized.
    if self._options.cleanup_tmp_directories:
      self._cleanup_tmp_directories()

    self._checkpoints = self._create_checkpoints()

    if self._options.read_only and not self._metadata_path().exists():
      self._metadata = {} if metadata is None else metadata
    else:
      self._metadata = None
    if metadata is not None and not self._options.read_only:
      self._save_metadata(metadata)

    self._finalize_thread = None
    logging.info(
        'jax.process_index=%s, primary_host=%s. CheckpointManager created: %s',
        jax.process_index(),
        self._primary_host,
        self,
    )

  def _configure_checkpointer_common(
      self,
      handler: CompositeCheckpointHandler,
      options: CheckpointManagerOptions,
      use_async: bool,
  ) -> Checkpointer:
    if use_async:
      if options.async_options is not None:
        return async_checkpointer.AsyncCheckpointer(
            handler,
            timeout_secs=options.async_options.timeout_secs,
            primary_host=self._primary_host,
            barrier_sync_fn=options.async_options.barrier_sync_fn,
        )
      else:
        return async_checkpointer.AsyncCheckpointer(
            handler, primary_host=self._primary_host
        )
    else:
      return Checkpointer(handler, primary_host=self._primary_host)

  def _configure_checkpointer_legacy_init(
      self,
      checkpointers: Union[AbstractCheckpointer, CheckpointersDict],
      options: CheckpointManagerOptions,
  ) -> Checkpointer:
    """Initializes _CompositeCheckpointer with legacy style checkpointers."""
    if self._primary_host != 0:
      raise ValueError(
          f'`primary_host`={self._primary_host} is not supported in legacy API.'
      )

    item_handlers = {}
    if isinstance(checkpointers, Checkpointer):
      use_async = is_async_checkpointer(checkpointers)
      if isinstance(checkpointers, async_checkpointer.AsyncCheckpointer):
        async_timeout = checkpointers._async_manager._timeout_secs  # pylint: disable=protected-access
      else:
        async_timeout = None
      item_handlers[DEFAULT_ITEM_NAME] = checkpointers.handler
    elif isinstance(checkpointers, dict):
      individual_use_async = []
      async_timeout = 0
      for item_name, checkpointer in checkpointers.items():
        if not isinstance(checkpointer, Checkpointer):
          raise ValueError(
              f'Value corresponding to {item_name} in `checkpointers` is not a'
              f' Checkpointer. Found {type(checkpointer)}.'
          )
        individual_use_async.append(is_async_checkpointer(checkpointer))
        if isinstance(checkpointer, async_checkpointer.AsyncCheckpointer):
          async_timeout = max(
              async_timeout, checkpointer._async_manager._timeout_secs  # pylint: disable=protected-access
          )
        if item_name in RESERVED_ITEM_NAMES:
          raise ValueError(
              f'Found {item_name} in `checkpointers`; this is a reserved key.'
          )
        item_handlers[item_name] = checkpointer.handler
      if any(individual_use_async) and not all(individual_use_async):
        logging.error(
            'Orbax `CheckpointManager` is transitioning toward using'
            ' asynchronous saving logic under the hood in all cases. Users that'
            ' configure `CheckpointManager` with some `Checkpointer`s and some'
            ' `AsyncCheckpointer`s will now see asynchronous logic used to save'
            ' all items. This may result in breakages if the code is assuming'
            ' that certain objects will be available immediately after saving.'
            ' Ensure that if you depend on the result of `save` being fully'
            ' written at a particular moment, use `wait_until_finished()`.'
        )
      use_async = any(individual_use_async)
      async_timeout = async_timeout or None
    else:
      raise ValueError(
          f'Invalid type for `checkpointers`. Found {checkpointers}.'
      )

    # if options.best_fn:
    item_handlers[METRIC_ITEM_NAME] = JsonCheckpointHandler(
        filename=METRIC_ITEM_NAME
    )
    options.async_options = options.async_options or AsyncOptions(
        timeout_secs=async_timeout
    )
    return self._configure_checkpointer_common(
        CompositeCheckpointHandler(
            primary_host=self._primary_host,
            **item_handlers,
        ),
        options,
        use_async,
    )


  def _validate_handler(self, handler):
    if (
        hasattr(handler, '_primary_host')
        and handler._primary_host != self._primary_host  # pylint: disable=protected-access
    ):
      raise ValueError(
          f'Inconsistent primary_host, CheckpointManager={self._primary_host}, '
          f'handler[{type(handler)}]={handler._primary_host} '  # pylint: disable=protected-access
      )

  def _configure_checkpointer(
      self,
      item_names: Optional[Sequence[str]],
      item_handlers: Optional[Union[CheckpointHandler, CheckpointHandlersDict]],
      options: CheckpointManagerOptions,
      single_item: bool,
  ) -> Checkpointer:
    """Initializes _CompositeCheckpointer given `item_names`."""
    if self._primary_host is None and item_handlers is None:
      raise ValueError(
          'When primary_host is set to None, item_handlers must be provided to'
          ' match with the primary_host setting.'
      )
    if single_item:
      item_handler = (
          item_handlers
          if isinstance(item_handlers, CheckpointHandler)
          else None
      )
      if item_handler:
        self._validate_handler(item_handler)
      all_item_handlers = {DEFAULT_ITEM_NAME: item_handler}
    else:
      # Initialize all_item_handlers with None or empty.
      if item_names:
        all_item_handlers = {item_name: None for item_name in item_names}
      else:
        all_item_handlers = {}
      # Update all_item_handlers with provided CheckpointHandlers.
      if item_handlers and isinstance(item_handlers, Mapping):
        for item_name, handler in item_handlers.items():
          self._validate_handler(handler)
          all_item_handlers[item_name] = handler

    for item_name in all_item_handlers:
      if item_name in RESERVED_ITEM_NAMES:
        raise ValueError(
            f'Found {item_name} in `checkpointers`; this is a reserved key.'
        )
    if options.best_fn:
      all_item_handlers[METRIC_ITEM_NAME] = JsonCheckpointHandler(
          filename=METRIC_ITEM_NAME, primary_host=self._primary_host
      )
    # CompositeCheckpointHandler defers per-item handler creation until
    # save/restore time.
    return self._configure_checkpointer_common(
        CompositeCheckpointHandler(
            primary_host=self._primary_host,
            **all_item_handlers,
        ),
        options,
        options.enable_async_checkpointing,
    )

  @property
  def directory(self) -> epath.Path:
    """See superclass documentation."""
    return self._directory

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """See superclass documentation."""
    if read:
      logging.warning(
          '`read` option is deprecated. Use `reload` to read from disk.'
      )
      return utils.checkpoint_steps(
          self.directory, self._options.single_host_load_and_broadcast
      )
    return [ckpt.step for ckpt in self._checkpoints]

  def latest_step(self) -> Optional[int]:
    """See superclass documentation."""
    return self._checkpoints[-1].step if self._checkpoints else None

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

  def reload(self):
    """Reloads internal properties.

    Resets internal cache of checkpoint steps, in case the directory managed
    by this object has been updated externally.
    """
    self._checkpoints = self._create_checkpoints()

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
    last_checkpoint_step = self.latest_step()
    # Ensure current step is between the last step and next step (accounting for
    # save interval). The `last_checkpoint_step` may not be initialized, in
    # which case we should save. Otherwise, step must fall on the specified
    # save interval. This condition accounts for the possibility of saving
    # on preemption, in which case we want to maintain the same save period as
    # if preemption had not happened.
    return last_checkpoint_step is None or (
        last_checkpoint_step < step
        and (
            step % self._options.save_interval_steps == 0
            or step in self._options.save_on_steps
        )
    )

  def _get_save_directory(
      self,
      step: int,
      directory: epath.Path,
  ) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""
    step_name_format = (
        self._options.step_name_format
        or step_lib.StandardNameFormat(
            step_prefix=self._options.step_prefix,
            step_format_fixed_length=self._options.step_format_fixed_length,
        )
    )
    return step_lib.build_step_path(directory, step_name_format, step)

  def _get_write_step_directory(
      self, step: int, root_dir: epath.Path
  ) -> epath.Path:
    return self._get_save_directory(step, root_dir)

  def _get_read_step_directory(
      self, step: int, root_dir: epath.Path
  ) -> epath.Path:
    if self._options.step_name_format is not None:
      return self._options.step_name_format.find_step(root_dir, step).path
    else:
      return self._get_save_directory(step, root_dir)

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

  def _validate_args(
      self,
      items: Optional[Union[Any, Mapping[str, Any]]],
      args: Optional[args_lib.CheckpointArgs],
  ):
    if isinstance(items, args_lib.CheckpointArgs):
      raise ValueError(
          'Found an instance of `CheckpointArgs` provided for `items`. This may'
          ' be due to misuse of the newer API - make sure to specify the'
          ' argument keyword (e.g. `args=args`).'
      )
    if args is not None:
      if not isinstance(args, args_lib.CheckpointArgs):
        raise ValueError(
            f'Expected args of type `CheckpointArgs`; found {type(args)}.'
        )
      if self._single_item:
        if isinstance(args, args_lib.Composite):
          raise ValueError(
              'Cannot provide `args` of type `Composite` when dealing with a'
              ' single checkpointable object.'
          )
      else:
        if not isinstance(args, args_lib.Composite):
          raise ValueError(
              'Must provide `args` of type `Composite` when dealing with'
              ' multiple checkpointable objects.'
          )

  def save(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      save_kwargs: Optional[Union[SaveParams, Mapping[str, SaveParams]]] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
      args: Optional[args_lib.CheckpointArgs] = None,
  ) -> bool:
    """See superclass documentation."""

    if items is None and args is None:
      raise ValueError('Must provide `args` for `save`.')
    self._validate_args(items, args)

    if not force and not self.should_save(step):
      return False
    if self.reached_preemption(step):
      logging.info('Saving checkpoint at step %d due to preemption.', step)

    # Wait for ongoing saves to complete. Only applicable if some of the
    # checkpointers are AsyncCheckpointers.
    # Must happen after `should_save` to avoid blocking callers.
    wait_for_prev_start_time = time.time()
    self.wait_until_finished()
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/wait_for_prev_duration_secs',
        time.time() - wait_for_prev_start_time,
    )

    if step in self.all_steps():
      raise ValueError(f'Checkpoint for step {step} already exists.')

    if items is None:
      items = {}
    if save_kwargs is None:
      save_kwargs = {}
    if self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
      save_kwargs = {DEFAULT_ITEM_NAME: save_kwargs}

    if self._track_best and metrics is None:
      logging.warning('Requested `tracked_metric`; did not provide metrics.')

    if args is None:
      args_dict = {}
      for key, item in items.items():
        save_ckpt_arg_cls, _ = _get_args_for_key(
            self._checkpointer.handler,
            key,
        )
        extra_args = save_kwargs[key] if key in save_kwargs else {}
        extra_args = extra_args or {}
        args_dict[key] = save_ckpt_arg_cls(item, **extra_args)  # pytype: disable=wrong-arg-count
      args = args_lib.Composite(**args_dict)
    else:
      if self._single_item:
        args = args_lib.Composite(**{DEFAULT_ITEM_NAME: args})
      else:
        if not isinstance(args, args_lib.Composite):
          raise ValueError(
              f'Expected args of type `Composite`; found {type(args)}.'
          )
        args = typing.cast(args_lib.Composite, args)

    args_dict = dict(args.items())
    if metrics is not None and self._track_best:
      args_dict['metrics'] = args_lib.JsonSave(metrics)
    args = args_lib.Composite(**args_dict)

    save_directory = self._get_write_step_directory(step, self.directory)
    # If a folder for the step to save exists and is not finalized, remove the
    # existing folder.
    if utils.is_gcs_path(self.directory):
      if (
          utils.is_primary_host(self._primary_host)
          and save_directory.exists()
          and utils.is_tmp_checkpoint(save_directory)
      ):
        logging.warning(
            'Attempting to save on GCS at step %s which has an unfinalized'
            ' checkpoint from previous runs. Removing the unfinalized'
            ' checkpoint before saving.',
            step,
        )
        self._delete_directory(step)
      utils.sync_global_devices('CheckpointManager:delete_unfinalized_step_gcs')

    self._checkpointer.save(save_directory, args=args)

    self._add_checkpoint_info(step, metrics)
    get_old_steps_start_time = time.time()
    steps_to_remove = self._get_old_steps_to_remove()
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/get_old_steps_duration_secs',
        time.time() - get_old_steps_start_time,
    )
    self._checkpoints = [
        info for info in self._checkpoints if info.step not in steps_to_remove
    ]
    # Sync needed to ensure that old steps to remove are retrieved before
    # actually deleting them during finalize, since retrieval can involve
    # looking at the directory.
    utils.sync_global_devices('CheckpointManager:old_steps_to_remove')

    assert self._finalize_thread is None
    if is_async_checkpointer(self._checkpointer):
      logging.info('Beginning async checkpoint finalize.')
      t = _FinalizeThread(
          target=self._finalize, args=(save_directory, steps_to_remove)
      )
      t.start()
      self._finalize_thread = t
    else:
      self._finalize(save_directory, steps_to_remove)
      logging.info('Finished synchronous save.')
      utils.sync_global_devices('CheckpointManager:finalize')
    return True

  def restore(
      self,
      step: int,
      items: Optional[Union[Any, Mapping[str, Any]]] = None,
      restore_kwargs: Optional[
          Union[RestoreParams, Mapping[str, RestoreParams]]
      ] = None,
      directory: Optional[epath.PathLike] = None,
      args: Optional[args_lib.CheckpointArgs] = None,
  ) -> Union[Any, Mapping[str, Any]]:
    """See superclass documentation."""
    directory = directory or self.directory
    directory = epath.Path(directory)
    self._validate_args(items, args)

    if items is None:
      items = {}
    elif self._single_item:
      items = {DEFAULT_ITEM_NAME: items}
    if restore_kwargs is None:
      restore_kwargs = {}
    elif self._single_item:
      restore_kwargs = {DEFAULT_ITEM_NAME: restore_kwargs}

    if args is None:
      args_dict = {}
      item_keys = set(items.keys()) | set(restore_kwargs.keys())
      for key in item_keys:
        _, restore_ckpt_arg_cls = _get_args_for_key(
            self._checkpointer.handler,
            key,
        )
        item = items[key] if key in items else None
        extra_args = restore_kwargs[key] if key in restore_kwargs else {}
        extra_args = extra_args or {}
        args_dict[key] = restore_ckpt_arg_cls(item, **extra_args)  # pytype: disable=wrong-arg-count
      args = args_lib.Composite(**args_dict)
    else:
      if self._single_item:
        args = args_lib.Composite(**{DEFAULT_ITEM_NAME: args})
      else:
        args = typing.cast(args_lib.Composite, args)

    restore_directory = self._get_read_step_directory(step, directory)
    restored = self._checkpointer.restore(restore_directory, args=args)
    if self._single_item:
      return restored[DEFAULT_ITEM_NAME]
    return restored

  def item_metadata(self, step: int) -> Union[Any, args_lib.Composite]:
    """See superclass documentation."""
    # TODO(b/321751056): Move the validation to CompositeCheckpointHandler by
    # changing the current metadata() biz logic.
    if isinstance(self._checkpointer.handler, CompositeCheckpointHandler):
      items_missing_handlers = []
      for (
          item_name,
          handler,
      ) in self._checkpointer.handler._known_handlers.items():  # pylint: disable=protected-access
        if handler is None:
          items_missing_handlers.append(item_name)
      if items_missing_handlers:
        raise ValueError(
            'No mapped CheckpointHandler found for items:'
            f' {items_missing_handlers}. Please see documentation of'
            ' `item_handlers` in CheckpointManager.'
        )

    result = self._checkpointer.metadata(
        self._get_read_step_directory(step, self.directory)
    )
    if self._single_item:
      return result[DEFAULT_ITEM_NAME]
    return result

  def metrics(self, step: int) -> Optional[PyTree]:
    if self._track_best:
      try:
        restored = self._checkpointer.restore(
            self._get_read_step_directory(step, self.directory),
            args=args_lib.Composite(
                **{METRIC_ITEM_NAME: args_lib.JsonRestore()}
            ),
        )
        return restored[METRIC_ITEM_NAME]
      except FileNotFoundError:
        logging.warning('Missing metrics for step %d', step)
        return None
    else:
      return None

  @property
  def _track_best(self):
    """Returns true if we should track the best checkpoints by given metric."""
    return self._options.best_fn is not None

  def _create_checkpoints(self) -> List[CheckpointInfo]:
    """Create a list of CheckpointInfo for existing checkpoints.

    If none are present, returns empty list.

    Returns:
      a list of CheckpointInfo, sorted by increasing step.
    """
    start = time.time()
    steps = utils.checkpoint_steps(
        self.directory, self._options.single_host_load_and_broadcast
    )
    steps.sort()  # Prefer in-place sort.

    if not steps:
      logging.info('No checkpoint steps found in %s', self.directory)
      checkpoint_infos = []
    else:

      def checkpoint_info(step: int) -> CheckpointInfo:
        timestamp = datetime.datetime.fromtimestamp(
            self._get_read_step_directory(step, self.directory).stat().mtime,
            tz=datetime.timezone.utc,
        )
        return CheckpointInfo(
            step=step, time=timestamp, metrics=self.metrics(step)
        )

      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            step: executor.submit(checkpoint_info, step) for step in steps
        }
        checkpoint_infos = [futures[step].result() for step in steps]

    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/read/load_all_step_metadata_duration_secs',
        time.time() - start,
    )
    logging.info(
        'Found %d checkpoint steps in %s', len(checkpoint_infos), self.directory
    )
    return checkpoint_infos

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
        CheckpointInfo(
            step, datetime.datetime.now(tz=datetime.timezone.utc), metrics
        )
    )

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
        reverse=(self._options.best_mode == 'min'),
    )

  def _cleanup_tmp_directories(self):
    utils.cleanup_tmp_directories(self.directory)

  def _delete_directory(self, step: int):
    """Deletes step dir or renames it if options.todelete_subdir is set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    if not utils.is_primary_host(self._primary_host):
      return

    # Delete if storage is on gcs or todelete_subdir is not set.
    if self._options.todelete_subdir is None or utils.is_gcs_path(
        self.directory
    ):
      self._get_read_step_directory(step, self.directory).rmtree()
      logging.info('Deleted step %d.', step)
      return

    # Rename step dir.
    rename_dir = self.directory / self._options.todelete_subdir
    if not rename_dir.exists():
      rename_dir.mkdir(parents=True)
    src = self._get_read_step_directory(step, self.directory)
    dst = self._get_write_step_directory(step, rename_dir)
    src.replace(dst)
    logging.info('Renamed step %d (todelete_subdir option specified).', step)

  def _get_old_steps_to_remove(self) -> List[int]:
    """Returns checkpoints that should be deleted."""
    # Must have set max_to_keep in order to remove any checkpoints.
    if self._options.max_to_keep is None:
      return []
    # Not enough checkpoints accumulated to consider deletion.
    if len(self._checkpoints) <= self._options.max_to_keep:
      return []

    # This isn't a duration but there isn't a general counter that we can use so
    # we abuse a duration metric to count the number of steps examined.
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/old_steps_examined_count',
        len(self._checkpoints),
    )

    # Exclude the latest checkpoint, since it is not finalized.
    step_name_format = (
        self._options.step_name_format
        or step_lib.StandardNameFormat(
            step_prefix=self._options.step_prefix,
            step_format_fixed_length=self._options.step_format_fixed_length,
        )
    )
    are_locked = utils.are_locked(
        self.directory,
        steps=tuple([info.step for info in self._checkpoints[:-1]]),
        step_name_format=step_name_format,
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
      active_checkpoints = set(
          checkpoints_without_metrics + sorted_checkpoints[-keep:]
          if keep > 0
          else []
      )
    else:
      all_checkpoints = checkpoints_without_metrics + sorted_checkpoints
      maybe_delete = all_checkpoints[:-keep] if keep > 0 else sorted_checkpoints
      active_checkpoints = set(all_checkpoints[-keep:] if keep > 0 else [])

    interval_preserved_checkpoints = self._get_interval_preserved_checkpoints(
        self._checkpoints
    )
    kept_checkpoints = set()
    for info in maybe_delete:
      if info.is_locked:
        logging.info(
            'Preserving %s: (Reason: checkpoint is locked).',
            info,
        )
        kept_checkpoints.add(info)
        continue
      if (
          self._options.keep_time_interval is not None
          and interval_preserved_checkpoints
      ):
        if info in interval_preserved_checkpoints:
          logging.info(
              'Preserving %s: (Reason: older falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.add(info)
          continue
        elif (
            info.time
            >= interval_preserved_checkpoints[-1].time
            + self._options.keep_time_interval
        ):
          interval_preserved_checkpoints.append(info)
          logging.info(
              'Preserving %s: (Reason: latest falling on keep_time_interval).',
              info,
          )
          kept_checkpoints.add(info)
          continue

      if (
          self._options.keep_period is not None
          and info.step % self._options.keep_period == 0
      ):
        logging.info('Preserving %s: (Reason: on keep_period).', info)
        kept_checkpoints.add(info)
        continue

    kept_checkpoints.update(active_checkpoints)

    steps_to_remove = []
    for info in self._checkpoints:
      if info not in kept_checkpoints:
        reason = 'worse metric' if self._track_best else 'old checkpoint'
        logging.info('Deleting %s: (Reason: %s).', info, reason)
        steps_to_remove.append(info.step)
    return steps_to_remove

  def _wait_for_checkpointers(self):
    if is_async_checkpointer(self._checkpointer):
      self._checkpointer.wait_until_finished()  # pytype: disable=attribute-error

  def wait_until_finished(self):
    """See superclass documentation."""
    t = self._finalize_thread
    if t is not None:
      self._finalize_thread = None
      try:
        t.join()
      except BaseException as e:  # pylint:disable=broad-exception-caught
        # If an exception occurred in the in finalization of the previous
        # save, we clean up since that checkpoint was never actually saved.
        assert self._checkpoints
        self._checkpoints = self._checkpoints[:-1]
        raise e
      # Additional work is being done on process 0 of the finalize threads.
      # When joining the threads, we must wait for all threads to complete
      # before proceeding.
      utils.sync_global_devices('CheckpointManager:join_finalize_thread')

  def check_for_errors(self):
    """See superclass documentation."""
    if is_async_checkpointer(self._checkpointer):
      self._checkpointer.check_for_errors()  # pytype: disable=attribute-error

  def _finalize_checkpoint(self, step: int):
    """Executes final actions just before the checkpoint write completes.

    * Logs error if any.
    * Records duration saved due to preemption if any.

    Args:
      step: finalized checkpoint step.
    """
    if utils.is_primary_host(self._primary_host):
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
      # If at a preemption step, record the time since the previous checkpoint.
      # This represents training time that would otherwise have been wasted.
      # If another checkpoint has not been previously saved, measures the time
      # since program start.
      if self.reached_preemption(step):
        if len(self._checkpoints) > 1:
          previous_time = self._checkpoints[-2].time
        else:
          previous_time = _INIT_TIME
        assert self._checkpoints
        duration = self._checkpoints[-1].time - previous_time
        jax.monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/preempt/duration_saved_secs',
            duration.total_seconds(),
        )

  def _finalize(self, directory: epath.Path, steps_to_remove: List[int]):
    """Cleans up old checkpoints and synchronizes hosts."""
    self._wait_for_checkpointers()
    # If an error is encountered while waiting for commit futures to complete,
    # we will not proceed past this point.
    self._finalize_checkpoint(utils.step_from_checkpoint_name(directory.name))
    remove_steps_start_time = time.time()
    for step in steps_to_remove:
      self._delete_directory(step)
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/remove_steps_duration_secs',
        time.time() - remove_steps_start_time,
    )

  def close(self):
    """See superclass documentation."""
    self.wait_until_finished()
    self._checkpointer.close()


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
