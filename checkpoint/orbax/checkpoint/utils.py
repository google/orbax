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

"""Utility functions for Orbax.

TODO(b/266449081) Increase unit test coverage.
"""
import asyncio
import concurrent.futures
import functools
import os
import re
import time
from typing import Any, List, Optional, Tuple, Union

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np

TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp-'
# prefix_1000.orbax-checkpoint-tmp-1010101
# OR
# 1000.orbax-checkpoint-tmp-1010101
TMP_DIR_STEP_PATTERN = r'.*?_*?(\d+)\.orbax-checkpoint-tmp-\d+'
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
_LOCK_ITEM_NAME = 'LOCKED'
_LAST_CHECKPOINT_WRITE_TIME = time.time()
CheckpointDirs = Tuple[str, str]
PyTree = Any




def should_skip_device_sync() -> bool:
  return False


def sync_global_devices(name: str):
  """Thin wrapper to provide additional features support."""
  if should_skip_device_sync():
    return
  multihost_utils.sync_global_devices(name)


def broadcast_one_to_all(pytree: PyTree) -> PyTree:
  """Thin wrapper to provide additional features support."""
  return multihost_utils.broadcast_one_to_all(pytree)


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


def async_exists(path: epath.Path):
  return _wrap(path.exists)()


class EmptyNode:
  pass


def is_empty_or_leaf(x: Any) -> bool:
  try:
    children, _ = jax._src.tree_util.flatten_one_level(x)  # pylint: disable=protected-access
  except ValueError:
    return True  # Cannot flatten x; means it must be a leaf
  return not children


def get_key_name(key: Any) -> Union[int, str]:
  """Returns the name of a JAX Key."""
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(key, jax.tree_util.DictKey):
    return str(key.key)
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return key.name
  elif isinstance(key, jax.tree_util.FlattenedIndexKey):
    return key.key
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def _is_dict_key(key) -> bool:
  return isinstance(key, (jax.tree_util.DictKey, jax.tree_util.GetAttrKey))


def _is_sequence_key(key) -> bool:
  return isinstance(
      key, (jax.tree_util.FlattenedIndexKey, jax.tree_util.SequenceKey)
  )


def _raise_unsupported_key_error(key):
  raise ValueError(f'Unsupported KeyEntry: {key}.')


def _extend_list(ls, idx, nextvalue):
  assert idx <= len(ls)
  if idx == len(ls):
    ls.append(nextvalue)
  return ls


def to_flat_dict(
    tree: PyTree, sep: Optional[str] = None, keep_empty_nodes: bool = False
) -> PyTree:
  """Converts a tree into a flattened dictionary.

  The nested keys are flattened to a tuple.

  Example::

    tree = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    to_flat_dict(tree)
    {
      ('foo',): 1,
      ('bar', 'a'): 2,
    }

  Args:
    tree: A PyTree to be flattened.
    sep: If provided, keys will be returned as `sep`-separated strings.
      Otherwise, keys are returned as tuples.
    keep_empty_nodes: If True, empty nodes are not filtered out.

  Returns:
    A flattened dictionary and the tree structure.
  """

  def tuple_path_from_keypath(keypath: Tuple[Any, ...]) -> Tuple[str, ...]:
    return tuple([str(get_key_name(k)) for k in keypath])

  flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=is_empty_or_leaf if keep_empty_nodes else None
  )
  flat_dict = {tuple_path_from_keypath(k): v for k, v in flat_with_keys}
  if sep is not None:
    flat_dict = {sep.join(k): v for k, v in flat_dict.items()}
  return flat_dict


def serialize_tree(tree: PyTree, keep_empty_nodes: bool = False) -> PyTree:
  """Transforms a PyTree to a serializable format.

  Args:
    tree: The tree to serialize.
    keep_empty_nodes: If true, does not filter out empty nodes.

  Returns:
    The serialized PyTree.
  """
  flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=is_empty_or_leaf if keep_empty_nodes else None
  )
  # Accesses the first path element (arbitrary), first tuple element
  # (keypath tuple), first key in keypath (outermost key in the PyTree).
  outerkey = flat_with_keys[0][0][0]
  if _is_dict_key(outerkey):
    result = {}
  elif _is_sequence_key(outerkey):
    result = []
  else:
    result = None
    _raise_unsupported_key_error(outerkey)

  for keypath, value in flat_with_keys:
    subtree = result
    for i, key in enumerate(keypath):
      if i == 0:
        assert isinstance(key, type(outerkey))
      if i == len(keypath) - 1:
        if _is_dict_key(key):
          assert isinstance(subtree, dict)
          subtree[get_key_name(key)] = value
        elif _is_sequence_key(key):
          assert isinstance(subtree, list)
          idx = get_key_name(key)
          subtree = _extend_list(subtree, idx, value)
      else:
        nextkey = keypath[i + 1]
        if _is_dict_key(nextkey):
          nextvalue = {}
        elif _is_sequence_key(nextkey):
          nextvalue = []
        else:
          nextvalue = None
          _raise_unsupported_key_error(nextkey)

        if _is_dict_key(key):
          assert isinstance(subtree, dict)
          name = get_key_name(key)
          if name not in subtree:
            subtree[name] = nextvalue
          subtree = subtree[name]
        elif _is_sequence_key(key):
          assert isinstance(subtree, list)
          idx = get_key_name(key)
          subtree = _extend_list(subtree, idx, nextvalue)
          subtree = subtree[idx]
        else:
          _raise_unsupported_key_error(key)

  return result


def deserialize_tree(
    serialized: PyTree, target: PyTree, keep_empty_nodes: bool = False
) -> PyTree:
  """Deserializes a PyTree to the same structure as `target`."""

  def _reconstruct_from_keypath(keypath, _):
    result = serialized
    for key in keypath:
      key_name = get_key_name(key)
      # Special case to support Pax.
      if not isinstance(result, list) and key_name not in result:
        key_name = str(key_name)
      result = result[key_name]
    return result

  return jax.tree_util.tree_map_with_path(
      _reconstruct_from_keypath,
      target,
      is_leaf=is_empty_or_leaf if keep_empty_nodes else None,
  )


def from_flat_dict(
    flat_dict: PyTree,
    target: Optional[PyTree] = None,
    sep: Optional[str] = None,
) -> PyTree:
  """Reconstructs the original tree object from a flattened dictionary.

  Args:
    flat_dict: A dictionary conforming to the return value of `to_flat_dict`.
    target: A reference PyTree. The returned value will conform to this
      structure. If not provided, an unflattened dict will be returned with the
      inferred structure of the original tree, without necessarily matching it
      exactly. Note, if not provided, the keys in `flat_dict` need to match
      `sep`.
    sep: separator used for nested keys in `flat_dict`.

  Returns:
    A dict matching the structure of `tree` with the values of `flat_dict`.
  """
  if target is None:
    result = {}
    for k, v in flat_dict.items():
      subtree = result
      if sep is None:
        assert isinstance(k, tuple)
        tuple_k = k
      else:
        tuple_k = tuple(k.split(sep))
      for i, name in enumerate(tuple_k):
        if i == len(k) - 1:
          assert name not in subtree
          subtree[name] = v
        else:
          if name not in subtree:
            subtree[name] = {}
          subtree = subtree[name]
    return result
  else:
    flat_structure = to_flat_dict(target, sep=sep)
    # Ensure that the ordering of `flat_dict` keys matches that of `target`.
    # Necessary for later unflattening.
    flat_dict = {k: flat_dict[k] for k in flat_structure.keys()}
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(target), flat_dict.values()
    )


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


def is_supported_empty_aggregation_type(value: Any) -> bool:
  """Determines if the *empty* value is supported for aggregation."""
  return isinstance(value, (dict, list, type(None))) and not value


def is_supported_aggregation_type(value: Any) -> bool:
  """Determines if the value is supported for aggregation."""
  return isinstance(
      value,
      (str, int, float, np.number, np.ndarray, bytes, jax.Array),
  ) or is_supported_empty_aggregation_type(value)


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


def cleanup_tmp_directories(directory: epath.PathLike,
                            primary_host: Optional[int] = 0):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  directory = epath.Path(directory)
  if primary_host is None or jax.process_index() == primary_host:
    logging.info('Cleaning up existing temporary directories at %s.', directory)
    tmp_files = tmp_checkpoints(directory)
    for tmp_file in tmp_files:
      (directory / tmp_file).rmtree()

  sync_global_devices('cleanup_tmp_dirs')


def is_gcs_path(path: epath.Path):
  return os.fspath(path).startswith(_GCS_PATH_PREFIX)


def get_tmp_directory(path: epath.Path) -> epath.Path:
  """Returns a tmp directory for the given path. Does not create it."""
  if is_gcs_path(path):
    return path
  now = time.time()
  sec = int(now)
  usec = int((now - sec) * 1000000)
  timestamp = broadcast_one_to_all((np.int32(sec), np.int32(usec)))
  return epath.Path(path.parent) / (
      path.name + TMP_DIR_SUFFIX + f'{timestamp[0]}{timestamp[1]:06}'
  )


def get_save_directory(
    step: int,
    directory: epath.PathLike,
    name: Optional[str] = None,
    step_prefix: Optional[str] = None,
    override_directory: Optional[epath.PathLike] = None,
    step_format_fixed_length: Optional[int] = None,
) -> epath.Path:
  """Returns the standardized path to a save directory for a single item.

  Args:
    step: Step number.
    directory: Top level checkpoint directory.
    name: Item name ('params', 'state', 'dataset', etc.).
    step_prefix: Prefix applied to `step` (e.g. 'checkpoint').
    override_directory: If provided, step, directory, and step_prefix are
      ignored.
    step_format_fixed_length: Uses a fixed number of digits with leading zeros
      to represent the step number. If None, there are no leading zeros.

  Returns:
    A directory.
  """
  if directory is None:
    raise ValueError('Directory cannot be None.')
  directory = epath.Path(directory)
  step_str = (
      f'{step:0{step_format_fixed_length}d}'
      if step_format_fixed_length is not None
      else str(step)
  )
  if override_directory is not None:
    result = epath.Path(override_directory)
  else:
    result = (
        directory / step_str
        if step_prefix is None
        else directory / f'{step_prefix}_{step_str}'
    )
  if name is not None:
    result /= name
  return result


def create_tmp_directory(final_dir: epath.PathLike,
                         primary_host: Optional[int] = 0) -> epath.Path:
  """Creates a temporary directory for saving at the given path."""
  # Share a timestamp across devices.
  final_dir = epath.Path(final_dir)
  # Renames are not atomic in GCS. Save directly to final_dir and rely on commit
  # completion file to indicate success.
  if is_gcs_path(final_dir):
    tmp_dir = final_dir
  else:
    tmp_dir = get_tmp_directory(final_dir)

  if tmp_dir.exists():
    if is_gcs_path(tmp_dir):
      raise FileExistsError(
          f'Attempted to create temporary directory {tmp_dir} which already'
          ' exists.'
      )
    else:
      raise AssertionError(
          f'Attempted to create temporary directory {tmp_dir} which already'
          ' exists. This condition should never arise on non-GCS'
          ' filesystems.'
      )
  # Sync after existence check to ensure that the directory has not just been
  # created by another host.
  sync_global_devices('create_tmp_directory:pre')

  if primary_host is None or jax.process_index() == primary_host:
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
  """A callback to run on completion of checkpoint save operation.

  Args:
    temp_ckpt_dir: A temporary checkpoint directory, where the checkpoint data
      is currently saved.
    final_ckpt_dir: A directory that represents the finalized name of the
      checkpoint. Should not exist yet if atomicity is ensured via `rename`, but
      may exist if atomicity is ensured by writing a commit success file.
    checkpoint_start_time: The time at which checkpoint saving began.
  """
  ensure_atomic_save(temp_ckpt_dir, final_ckpt_dir)
  record_saved_duration(checkpoint_start_time)
  logging.info('Finished saving checkpoint to `%s`.', final_ckpt_dir)


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


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


def step_from_checkpoint_name(name: str) -> int:
  """Returns the step from a checkpoint name. Also works for tmp checkpoints."""
  if name.isdigit():
    return int(os.fspath(name))
  elif name.split('_')[-1].isdigit():
    split = name.split('_')
    if len(split) == 2 and split[0]:
      return int(split[-1])
  elif tmp_match := re.match(TMP_DIR_STEP_PATTERN, name):
    return int(tmp_match.group(1))
  raise ValueError(f'Unrecognized name format: {name}.')


def checkpoint_steps_paths(
    checkpoint_dir: epath.PathLike,
) -> List[epath.Path]:
  """Returns a list of finalized checkpoint paths in the directory."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  if not checkpoint_dir.exists():
    raise ValueError(f'Path {checkpoint_dir} does not exist.')

  def check_step_dir(step_dir: epath.Path) -> bool:
    return _is_step_checkpoint(step_dir) and is_checkpoint_finalized(step_dir)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        step_dir: executor.submit(check_step_dir, step_dir)
        for step_dir in checkpoint_dir.iterdir()
    }
    return [step_dir for step_dir, future in futures.items() if future.result()]


def checkpoint_steps(checkpoint_dir: epath.PathLike) -> List[int]:
  """Returns a list of finalized checkpoint steps in the directory."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [
      step_from_checkpoint_name(s.name)
      for s in checkpoint_steps_paths(checkpoint_dir)
  ]


def any_checkpoint_step(checkpoint_dir: epath.PathLike) -> Optional[int]:
  """Returns any finalized checkpoint step in the directory or None.

  This avoids iterating over the entire directory.

  Args:
    checkpoint_dir: Checkpoint directory.

  Returns:
    Any finalized checkpoint step in the directory or None.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  for s in checkpoint_dir.iterdir():
    if _is_step_checkpoint(s) and is_checkpoint_finalized(s):
      return step_from_checkpoint_name(s.name)
  return None


def is_checkpoint_finalized(path: epath.PathLike) -> bool:
  """Determines if the given path represents a finalized checkpoint.

  Path takes the form::

    path/to/my/dir/<name>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
    path/to/my/dir/<name>/  # finalized

  Alternatively::

    gs://path/to/my/dir/<name>/  # finalized
      commit_success.txt
      ...
    gs://<path/to/my/dir/<name>/  # not finalized
      ...

  Args:
    path: Directory.

  Returns:
    True if the checkpoint is finalized.

  Raises:
    ValueError if the provided path is not a directory. Valid checkpoint paths
    must be a directory.
  """
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'Path {path} does not exist.')
  if not path.is_dir():
    raise ValueError(f'Path {path} is not a directory. Not a valid checkpoint')
  if is_gcs_path(path) and not (path / _COMMIT_SUCCESS_FILE).exists():
    return False
  if TMP_DIR_SUFFIX in path.name:
    return False
  return True


def is_tmp_checkpoint(path: epath.PathLike) -> bool:
  """Determines whether a directory is a tmp checkpoint path."""
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'Path {path} does not exist.')
  if not path.is_dir():
    return False
  if is_gcs_path(path) and not (path / _COMMIT_SUCCESS_FILE).exists():
    return True
  if TMP_DIR_SUFFIX in path.name:
    return True
  return False


def tmp_checkpoints(checkpoint_dir: epath.PathLike) -> List[str]:
  """Returns a list of tmp checkpoints in the directory."""
  checkpoint_dir = epath.Path(checkpoint_dir)
  return [s.name for s in checkpoint_dir.iterdir() if is_tmp_checkpoint(s)]


def fully_replicated_host_local_array_to_global_array(
    arr: jax.Array,
) -> jax.Array:
  """Converts a host local array from to global jax.Array.

  In most cases, the local array is expected to have been produced by pmap.

  Args:
    arr: Host local array

  Returns:
    A global array.
  """
  if not arr.is_fully_replicated:
    raise ValueError('Array must be fully replicated.')
  global_shape = arr.device_buffers[0].shape
  # Create a 1D mesh to create fully replicated global jax.Array.
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',)),
      jax.sharding.PartitionSpec(None),
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(arr.device_buffers, key=lambda x: x.device().id)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, dbs)


def lockdir(directory: epath.Path) -> epath.Path:
  """Constructs a directory used to indicate that a checkpoint step is `locked`."""
  return directory / _LOCK_ITEM_NAME


async def _async_is_locked(directory: epath.Path) -> bool:
  """(Async) determines whether a checkpoint step is considered `locked`."""
  parent_dir_exists = await async_exists(directory)
  if not parent_dir_exists:
    raise ValueError(f'Parent directory {directory} does not exist.')
  return await async_exists(lockdir(directory))


def is_locked(directory: epath.Path) -> bool:
  """Determines whether a checkpoint step is considered `locked`."""
  return asyncio.run(_async_is_locked(directory))


def are_locked(
    directory: epath.Path,
    steps: Tuple[int],
    step_prefix: Optional[str],
    step_format_fixed_length: Optional[int],
) -> List[bool]:
  """In parallel, determines whether the steps is considered `locked`."""

  def _get_save_directory(step):
    return get_save_directory(
        step,
        directory,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )

  async def _run_in_parallel(ops):
    return await asyncio.gather(*ops)

  ops = [_async_is_locked(_get_save_directory(step)) for step in steps]
  return asyncio.run(_run_in_parallel(ops))
