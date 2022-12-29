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

"""Utility functions for Orbax."""
import asyncio
import functools
import os
import time
from typing import Any, Iterator, List, Optional, Tuple

from absl import logging
from etils import epath
import flax.serialization
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp-'
# TODO(b/260759189): Deprecate this prefix when no longer in use by JAX MG.
_AGGREGATED_PREFIX = 'AGGREGATED://'
# Used in a msgpack checkpoint file to denote a leaf value that has been written
# individually. Typically, this may indicate an array that was written using
# Tensorstore rather than its value being directly stored in the msgpack file.
# To avoid duplication, we replace the value with a placeholder prefix and other
# relevant information (see functions below).
_PLACEHOLDER_PREFIX = 'PLACEHOLDER://'
_COMMIT_SUCCESS_FILE = 'commit_success.txt'
_GCS_PATH_PREFIX = 'gs://'
_LAST_CHECKPOINT_WRITE_TIME = time.time()
CheckpointDirs = Tuple[str, str]
PyTree = type(jax.tree_util.tree_structure(None))


def sync_global_devices(name: str):
  """Thin wrapper to provide additional features support."""
  multihost_utils.sync_global_devices(name)


def _wrap(func):
  """Wraps a function to make it async."""

  @functools.wraps(func)
  async def run(*args, loop=None, executor=None, **kwargs):
    if loop is None:
      loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

  return run


# TODO(cpgaffney): This functionality should be provided by an external library.
def async_makedirs(
    path: epath.Path,
    *args,
    parents: bool = False,
    exist_ok: bool = True,
    **kwargs,
):
  return _wrap(path.mkdir)(*args, parents=parents, exist_ok=exist_ok, **kwargs)


def async_write_bytes(path: epath.Path, data: Any):
  return _wrap(path.write_bytes)(data)


def register_ts_spec_for_serialization():
  # Register functions with flax.serialization to handle `ts.Spec`.
  def is_dict(s):
    return isinstance(s, (dict, flax.core.FrozenDict))

  flax.serialization.register_serialization_state(
      ts.Spec,
      ty_to_state_dict=lambda t: t.to_json(),
      # The parameter may have been written to tensorstore or msgpack.
      # If the former, a dict of the spec will be stored. If the latter it will
      # be the value itself.
      ty_from_state_dict=lambda t, s: ts.Spec(s) if is_dict(s) else s,
      override=True)


def leaf_is_placeholder(leaf: Any) -> bool:
  """Determines if `leaf` represents a placeholder for a non-aggregated value.
  """
  return isinstance(leaf, str) and (leaf.startswith(_PLACEHOLDER_PREFIX) or
                                    leaf.startswith(_AGGREGATED_PREFIX))


def leaf_placeholder(name: str) -> str:
  """Constructs value to act as placeholder for non-aggregated value."""
  return _PLACEHOLDER_PREFIX + name


def name_from_leaf_placeholder(placeholder: str) -> str:
  """Gets the param name from a placeholder with the correct prefix."""
  if not leaf_is_placeholder(placeholder):
    msg = ('Requested name from placeholder, but value did not contain required'
           ' prefix.')
    raise ValueError(msg)
  if placeholder.startswith(_AGGREGATED_PREFIX):
    return placeholder.replace(_AGGREGATED_PREFIX, '', 1)
  elif placeholder.startswith(_PLACEHOLDER_PREFIX):
    return placeholder.replace(_PLACEHOLDER_PREFIX, '', 1)
  else:
    raise ValueError('Found placeholder beginning with unexpected prefix.')


def is_supported_aggregation_type(value: Any) -> bool:
  """Determines if the value is supported for aggregation."""
  return isinstance(value,
                    (str, int, float, np.number, np.ndarray, jnp.ndarray))


def pytree_structure(directory: epath.PathLike) -> PyTree:
  """Reconstruct state dict from saved model format in `directory`."""
  directory = epath.Path(directory)

  def add_nested_key(subtree, nested_key, key_name):
    if not nested_key:
      return subtree

    current = nested_key[0]

    if len(nested_key) == 1:
      assert current not in subtree
      subtree[current] = leaf_placeholder(key_name)
      return subtree

    subkeys = nested_key[1:]
    if current not in subtree:
      subtree[current] = {}
    subtree[current] = add_nested_key(subtree[current], subkeys, key_name)
    return subtree

  keys = directory.iterdir()
  tree = {}
  for k in keys:
    tree = add_nested_key(tree, k.name.split('.'), k.name)
  return tree


def _rebuild_ts_specs(tree):
  """Converts any ts_spec dict leaves to ts.Spec object."""

  def is_leaf(x):
    if isinstance(x, dict):
      return set(x.keys()) >= {'driver', 'kvstore'}
    return False

  return jax.tree_util.tree_map(
      lambda x: ts.Spec(x) if isinstance(x, dict) else x, tree, is_leaf=is_leaf)


def msgpack_restore(msgpack):
  """Restores tree serialized using Flax. Converts ts_spec dict to ts.Spec."""
  state_dict = flax.serialization.msgpack_restore(msgpack)
  return _rebuild_ts_specs(state_dict)


def to_state_dict(pytree):
  """Converts tree to state_dict. Converts ts_spec dict to ts.Spec."""
  state_dict = flax.serialization.to_state_dict(pytree)
  return _rebuild_ts_specs(state_dict)


def cleanup_tmp_directories(directory: epath.PathLike):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  directory = epath.Path(directory)
  if jax.process_index() == 0:
    tmp_files = tmp_checkpoints(directory)
    for tmp_file in tmp_files:
      (directory / tmp_file).rmtree()

  sync_global_devices('cleanup_tmp_dirs')


def is_gcs_path(path: epath.Path):
  return os.fspath(path).startswith(_GCS_PATH_PREFIX)


def get_save_directory(step: int,
                       directory: epath.PathLike,
                       name: Optional[str] = None,
                       step_prefix: Optional[str] = None) -> epath.Path:
  """Returns the standardized path to a save directory for a single item."""
  directory = epath.Path(directory)
  if step_prefix is None:
    result = directory / str(step)
  else:
    result = directory / f'{step_prefix}_{step}'
  if name is not None:
    result /= name
  return result


def create_tmp_directory(final_dir: epath.PathLike) -> epath.Path:
  """Creates a temporary directory for saving at the given path."""
  # Share a timestamp across devices.
  final_dir = epath.Path(final_dir)
  # Renames are not atomic in GCS. Save directly to final_dir and rely on commit
  # completion file to indicate success.
  if is_gcs_path(final_dir):
    # Sync needed to prevent an error since caller may think the directory
    # exists from a previous save, rather than just having been created.
    sync_global_devices('create_tmp_directory:pre')
    tmp_dir = final_dir
  else:
    timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
    tmp_dir = epath.Path(final_dir.parent) / (
        final_dir.name + TMP_DIR_SUFFIX + f'{timestamp}')

  if jax.process_index() == 0:
    assert not tmp_dir.exists()
    tmp_dir.mkdir(parents=True)

  sync_global_devices('create_tmp_directory')

  return tmp_dir


def ensure_atomic_save(temp_ckpt_dir: epath.Path, final_ckpt_dir: epath.Path):
  """Finalizes atomic save by renaming tmp_dir or writing a success file."""
  if temp_ckpt_dir == final_ckpt_dir:
    (final_ckpt_dir / _COMMIT_SUCCESS_FILE
    ).write_text(f'Checkpoint commit was successful to {final_ckpt_dir}')
  else:
    logging.info('Renaming %s to %s', temp_ckpt_dir, final_ckpt_dir)
    temp_ckpt_dir.rename(final_ckpt_dir)
    logging.info('Finished saving checkpoint to `%s`.', final_ckpt_dir)


def record_saved_duration(checkpoint_start_time: float):
  """Record program duration that is accounted for by this checkpoint.

  For the very first checkpoint, this is the interval between program init and
  current checkpoint start time.

  Note that we use the checkpoint start time instead of end time. The saved
  duration should not include prallel training duration while the async
  checkpoint is being written in the background.

  Args:
    checkpoint_start_time: Start time of current checkpoint.
  """
  global _LAST_CHECKPOINT_WRITE_TIME
  # Note: for the very first checkpoint, this is the interval between program
  # init and the current checkpoint start time.
  duration_since_last_checkpoint = (
      checkpoint_start_time - _LAST_CHECKPOINT_WRITE_TIME)
  # TODO(hanyangtay): Remove version guard.
  if jax.version.__version_info__ > (0, 3, 25):
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/duration_since_last_checkpoint_secs',
        duration_since_last_checkpoint)
  _LAST_CHECKPOINT_WRITE_TIME = checkpoint_start_time


def on_commit_callback(temp_ckpt_dir: epath.Path, final_ckpt_dir: epath.Path,
                       checkpoint_start_time: float):
  """Finalize atomic save and record training duration saved in a checkpoint."""
  ensure_atomic_save(temp_ckpt_dir, final_ckpt_dir)
  record_saved_duration(checkpoint_start_time)
  logging.info('Finished saving checkpoint to `%s`.', final_ckpt_dir)


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


def is_checkpoint_item_finalized(path: epath.PathLike) -> bool:
  """Determines if the checkpoint item path is finalized.

  NOT TO BE CONFUSED WITH is_checkpoint_finalized. That method works on the step
  level, while this method works on the item level.

  Path takes the form:
  <directory>/<step>/<item1>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
    # Checkpoint files
    ...
  OR
  <directory>/<step>/<item2>/  # finalized
    ...

  Alternatively:
  gs://<directory>/<step>/<item1>/  # finalized
    commit_success.txt
    ...
  OR
  gs://<directory>/<step>/<item2>/  # not finalized
    ...

  Args:
    path: Path to item directory.

  Returns:
    True if the checkpoint item is finalized.

  Raises:
    ValueError if the provided path is not a directory. Valid checkpoint paths
    must be a directory.
  """
  path = epath.Path(path)
  if not path.is_dir():
    raise ValueError(f'Path {path} is not a directory.')
  if is_gcs_path(path) and not (path / _COMMIT_SUCCESS_FILE).exists():
    return False
  if TMP_DIR_SUFFIX in path.name:
    return False
  return True


def is_checkpoint_step_finalized(path: epath.PathLike) -> bool:
  """Determines if the checkpoint path is finalized.

  NOT TO BE CONFUSED WITH is_checkpoint_item_finalized. That method works on the
  per-item level, while this method works on the per-step level.

  Path takes the form:
  <directory>/<step>/
    <item1>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
      # Checkpoint files
      ...
    <item2>  # finalized
      ...

  Alternatively:
  gs://<directory>/<step>/
    <item1>  # finalized
      commit_success.txt
      ...
    <item2>  # not finalized
      ...


  # not finalized
  <directory>/checkpoint_<step>.orbax-checkpoint-tmp-<timestamp>/
    checkpoint
    a/
      0.0
      .zarray
    b/
      ...

  <directory>/checkpoint_<step>/  # finalized
    checkpoint
    ...


  Args:
    path: Path to step directory.

  Returns:
    True if the checkpoint is finalized.

  Raises:
    ValueError if the provided path is not a directory. Valid checkpoint paths
    must be a directory.
  """
  path = epath.Path(path)
  if not path.is_dir():
    raise ValueError(f'Path {path} is not a directory.')
  for subpath in path.iterdir():
    if not is_checkpoint_item_finalized(subpath):
      return False
  return True


def _is_step_checkpoint(path: epath.Path) -> bool:
  """Determines if the path resembles an Orbax step directory.

  Note that this is not foolproof, and users should not add extra files to the
  checkpoint directory beyond what is done by CheckpointManager.

  Args:
    path: path to check.

  Returns:
    bool indicating whether the path resembles an Orbax step directory.
  """
  name = os.fspath(path.name)
  # Path must be a directory and either a digit, or end in '_' + digit.
  return path.is_dir() and (name.isdigit() or name.split('_')[-1].isdigit())


def _step_from_name(name: str) -> int:
  if name.isdigit():
    return int(os.fspath(name))
  elif name.split('_')[-1].isdigit():
    return int(name.split('_')[-1])
  else:
    raise ValueError('Unrecognized name format.')


def checkpoint_steps(checkpoint_dir: epath.PathLike) -> List[int]:
  """Returns a list of finalized checkpoint steps in the directory."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [
      _step_from_name(s.name)
      for s in checkpoint_dir.iterdir()
      if _is_step_checkpoint(s) and is_checkpoint_finalized(s)
  ]


def is_checkpoint_finalized(path: epath.PathLike) -> bool:
  """Branches to step_finalized/item_finalized depending on the path."""
  path = epath.Path(path)
  if not path.is_dir():
    raise ValueError(f'Checkpoint path {path} must be a directory.')
  if _is_step_checkpoint(path):
    return is_checkpoint_step_finalized(path)
  else:
    return is_checkpoint_item_finalized(path)


def tmp_checkpoints(checkpoint_dir: epath.PathLike) -> List[str]:
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [
      s.name
      for s in checkpoint_dir.iterdir()
      if s.is_dir() and not is_checkpoint_finalized(s)
  ]


def _wait_for_new_checkpoint(checkpoint_dir: epath.Path,
                             last_checkpoint_step: Optional[int],
                             seconds_to_sleep: int = 1,
                             timeout: Optional[int] = None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint_step: The last checkpoint step used or `None` if we're
      expecting a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint step, or None if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    steps = checkpoint_steps(checkpoint_dir)
    checkpoint_step = sorted(steps)[-1] if steps else None
    if checkpoint_step is None or checkpoint_step == last_checkpoint_step:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint step: %d', checkpoint_step)
      return checkpoint_step


def checkpoints_iterator(checkpoint_dir: epath.PathLike,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None) -> Iterator[int]:
  """Continuously yield new checkpoint files as they appear.

  Based on the equivalent TF method.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  Warning: If CheckpointManager is running in a different process for training
  and is cleaning up old checkpoints (via the `max_to_keep` argument), steps
  returned by this function may not be valid after being clean up by another
  process. In this case, `max_to_keep` should be increased (suggested value: 5)

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    Integer step numbers of the latest checkpoints as they arrive.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  checkpoint_step = None
  while True:
    new_checkpoint_step = 0
    if jax.process_index() == 0:
      new_checkpoint_step = _wait_for_new_checkpoint(
          checkpoint_dir, checkpoint_step, timeout=timeout) or -1
    # None cannot be broadcast
    new_checkpoint_step = multihost_utils.broadcast_one_to_all(
        np.int32(new_checkpoint_step))
    if new_checkpoint_step == -1:
      if not timeout_fn:
        # timed out
        logging.info('Timed-out waiting for a checkpoint.')
        return
      if timeout_fn():
        # The timeout_fn indicated that we are truly done.
        return
      else:
        # The timeout_fn indicated that more checkpoints may come.
        continue
    start = time.time()
    checkpoint_step = new_checkpoint_step
    yield checkpoint_step
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
