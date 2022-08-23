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
import logging
import os
import time
from typing import Iterator, List, Optional, Tuple, Union

from etils import epath
import flax.serialization
import jax
from jax.experimental import multihost_utils
import numpy as np
import tensorstore as ts

TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp-'
_COMMIT_SUCCESS_FILE = 'commit_success.txt'
_GCS_PATH_PREFIX = 'gs://'
CheckpointDirs = Tuple[str, str]
PyTree = type(jax.tree_util.tree_structure(None))
Path = Union[str, epath.Path]


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
def async_makedirs(path,
                   *args,
                   parents: bool = True,
                   exist_ok: bool = True,
                   **kwargs):
  return _wrap(path.mkdir)(*args, parents=parents, exist_ok=exist_ok, **kwargs)


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


def rmtree(path: epath.Path):
  """Recursively removes non-empty directory."""
  for child in path.iterdir():
    if child.is_file():
      child.unlink()
    else:
      rmtree(child)
  path.rmdir()


Leaf = epath.Path


def pytree_structure(directory: Path) -> PyTree:
  """Reconstruct state dict from saved model format in `directory`."""
  directory = epath.Path(directory)

  def add_nested_key(subtree, nested_key, full_name):
    if not nested_key:
      return subtree

    current = nested_key[0]

    if len(nested_key) == 1:
      assert current not in subtree
      subtree[current] = full_name
      return subtree

    subkeys = nested_key[1:]
    if current not in subtree:
      subtree[current] = {}
    subtree[current] = add_nested_key(subtree[current], subkeys, full_name)
    return subtree

  keys = directory.iterdir()
  tree = {}
  for k in keys:
    tree = add_nested_key(tree, k.name.split('.'), k)
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


def cleanup_tmp_directories(directory: Path):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  directory = epath.Path(directory)
  if jax.process_index() == 0:
    tmp_files = tmp_checkpoints(directory)
    for tmp_file in tmp_files:
      rmtree(directory / tmp_file)

  multihost_utils.sync_global_devices('cleanup_tmp_dirs')


def is_gcs_path(path: epath.Path):
  return os.fspath(path).startswith(_GCS_PATH_PREFIX)


def get_save_directory(step: int,
                       directory: Path,
                       name: Optional[str] = None) -> epath.Path:
  """Returns the standardized path to a save directory for a single item."""
  directory = epath.Path(directory)
  result = directory / str(step)
  if name is not None:
    result /= name
  return result


def create_tmp_directory(final_dir: Path) -> epath.Path:
  """Creates a temporary directory for saving at the given path."""
  # Share a timestamp across devices.
  final_dir = epath.Path(final_dir)
  # Renames are not atomic in GCS. Save directly to final_dir and rely on commit
  # completion file to indicate success.
  if is_gcs_path(final_dir):
    return final_dir
  timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  tmp_dir = epath.Path(final_dir.parent) / (
      final_dir.name + TMP_DIR_SUFFIX + f'{timestamp}')

  if jax.process_index() == 0:
    assert not tmp_dir.exists()
    tmp_dir.mkdir(parents=True)

  multihost_utils.sync_global_devices('make_dir')

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


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


def is_checkpoint_finalized(path: Path) -> bool:
  """Determines if the checkpoint path is finalized.

  Path takes the form:
  <directory>/<step>/
    <name1>.orbax-checkpoint-tmp-<timestamp>  # not finalized
    <name2>  # finalized

  Alternatively:
  gs://<directory>/<step>/
    <name1>  # finalized
      commit_success.txt
      ...
    <name2>  # not finalized
      ...

  Args:
    path: Path to step directory.

  Returns:
    True if the checkpoint is finalized.
  """
  path = epath.Path(path)
  for subpath in path.iterdir():
    if is_gcs_path(subpath) and not (subpath / _COMMIT_SUCCESS_FILE).exists():
      return False
    if TMP_DIR_SUFFIX in subpath.name:
      return False
  return True


def checkpoint_steps(checkpoint_dir: Path) -> List[int]:
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [
      int(os.fspath(s.name))
      for s in checkpoint_dir.iterdir()
      if is_checkpoint_finalized(s) and os.fspath(s.name).isdigit()
  ]


def tmp_checkpoints(checkpoint_dir: Path) -> List[str]:
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [
      s.name for s in checkpoint_dir.iterdir() if not is_checkpoint_finalized(s)
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


def checkpoints_iterator(checkpoint_dir: Path,
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
