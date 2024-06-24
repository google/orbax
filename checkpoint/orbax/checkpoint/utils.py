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

"""Utility functions for Orbax.

NOTE: Functions in this file are deprecated in favor of corresponding functions
in sub-modules. Please use those functions instead, and do not add new
functions here.

TODO(b/266449081) Increase unit test coverage.
"""

import functools
import time
from typing import Any, Optional, Tuple

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import multihost
from orbax.checkpoint import tree as tree_utils
from orbax.checkpoint.metadata import checkpoint
from orbax.checkpoint.metadata import sharding as sharding_metadata
from orbax.checkpoint.metadata import value as value_metadata
from orbax.checkpoint.path import step as step_lib
from orbax.checkpoint.path import utils as path_utils


TMP_DIR_SUFFIX = step_lib.TMP_DIR_SUFFIX
# TODO(b/260759189): Deprecate this prefix when no longer in use by JAX MG.
_AGGREGATED_PREFIX = 'AGGREGATED://'
# Used in a msgpack checkpoint file to denote a leaf value that has been written
# individually. Typically, this may indicate an array that was written using
# Tensorstore rather than its value being directly stored in the msgpack file.
# To avoid duplication, we replace the value with a placeholder prefix and other
# relevant information (see functions below).
_PLACEHOLDER_PREFIX = 'PLACEHOLDER://'
PyTree = Any

sync_global_processes = multihost.sync_global_processes
sync_global_devices = multihost.sync_global_processes
broadcast_one_to_all = multihost.broadcast_one_to_all
reached_preemption = multihost.reached_preemption
is_primary_host = multihost.is_primary_host


_wrap = path_utils._wrap  # pylint: disable=protected-access
async_makedirs = path_utils.async_makedirs
async_write_bytes = path_utils.async_write_bytes
async_exists = path_utils.async_exists
lockdir = path_utils.lockdir
is_locked = path_utils.is_locked
are_locked = path_utils.are_locked

is_gcs_path = step_lib.is_gcs_path
checkpoint_steps = step_lib.checkpoint_steps
any_checkpoint_step = step_lib.any_checkpoint_step
is_checkpoint_finalized = step_lib.is_checkpoint_finalized
is_tmp_checkpoint = step_lib.is_tmp_checkpoint
tmp_checkpoints = step_lib.tmp_checkpoints
cleanup_tmp_directories = step_lib.cleanup_tmp_directories
get_tmp_directory = step_lib.get_tmp_directory
get_save_directory = step_lib.get_save_directory
create_tmp_directory = step_lib.create_tmp_directory
record_saved_duration = step_lib.record_saved_duration
step_from_checkpoint_name = step_lib.step_from_checkpoint_name
checkpoint_steps_paths = step_lib.checkpoint_steps_paths

deserialize_tree = tree_utils.deserialize_tree
from_flat_dict = tree_utils.from_flat_dict
from_flattened_with_keypath = tree_utils.from_flattened_with_keypath
serialize_tree = tree_utils.serialize_tree
to_flat_dict = tree_utils.to_flat_dict
is_sequence_key = tree_utils.is_sequence_key
is_dict_key = tree_utils.is_dict_key
tuple_path_from_keypath = tree_utils.tuple_path_from_keypath
get_key_name = tree_utils.get_key_name
is_empty_node = tree_utils.is_empty_node
is_empty_or_leaf = tree_utils.is_empty_or_leaf


def leaf_is_placeholder(leaf: Any) -> bool:
  """Determines if `leaf` represents a placeholder for a non-aggregated value."""
  return isinstance(leaf, str) and (
      leaf.startswith(_PLACEHOLDER_PREFIX)
      or leaf.startswith(_AGGREGATED_PREFIX)
  )


def leaf_placeholder(name: str) -> str:
  """Constructs value to act as placeholder for non-aggregated value."""
  return _PLACEHOLDER_PREFIX + name


def name_from_leaf_placeholder(placeholder: str) -> str:
  """Gets the param name from a placeholder with the correct prefix."""
  if not leaf_is_placeholder(placeholder):
    msg = (
        'Requested name from placeholder, but value did not contain required'
        ' prefix.'
    )
    raise ValueError(msg)
  if placeholder.startswith(_AGGREGATED_PREFIX):
    return placeholder.replace(_AGGREGATED_PREFIX, '', 1)
  elif placeholder.startswith(_PLACEHOLDER_PREFIX):
    return placeholder.replace(_PLACEHOLDER_PREFIX, '', 1)
  else:
    raise ValueError('Found placeholder beginning with unexpected prefix.')


def all_leaves_are_placeholders(tree: PyTree) -> bool:
  """Determines if all leaves in `tree` are placeholders."""
  return all(
      leaf_is_placeholder(leaf) for leaf in jax.tree.leaves(tree)
  )


def is_supported_empty_aggregation_type(value: Any) -> bool:
  """Determines if the *empty* value is supported for aggregation."""
  # Check isinstance first to avoid `not` checks on jax.Arrays (raises error).
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
    # Sharding file stores sharding data that is only used by orbax. Therefore,
    # it shouldn't be included here. See b/279969796 for more details.
    if k.name == '_sharding':
      continue
    if k.name == '_METADATA':
      continue
    tree = add_nested_key(tree, k.name.split('.'), k.name)
  return tree


# TODO(b/337137764): Move to step.py and fix
# learning/gemini/pax/core/checkpoint_managers_test.py
def ensure_atomic_save(
    temp_ckpt_dir: epath.Path,
    final_ckpt_dir: epath.Path,
    checkpoint_metadata_store: Optional[
        checkpoint.CheckpointMetadataStore
    ] = None,
):
  """Finalizes atomic save by renaming tmp_dir or writing a success file.

  Updates checkpoint metadata with commit_timestamp_nsecs.

  Args:
    temp_ckpt_dir: directory path containing uncommitted checkpoint.
    final_ckpt_dir: directory path which will contain committed checkpoint.
    checkpoint_metadata_store: optional `CheckpointMetadataStore` instance. If
      present then it is used to update `CheckpointMetadata` related to
      committed data present in `final_ckpt_dir`.
  """
  if temp_ckpt_dir == final_ckpt_dir:
    commit_success_file = final_ckpt_dir / step_lib._COMMIT_SUCCESS_FILE  # pylint: disable=protected-access
    commit_success_file.write_text(
        f'Checkpoint commit was successful to {final_ckpt_dir}'
    )
  else:
    logging.info('Renaming %s to %s', temp_ckpt_dir, final_ckpt_dir)
    temp_ckpt_dir.rename(final_ckpt_dir)
  if checkpoint_metadata_store:
    checkpoint_metadata_store.update(
        checkpoint_path=final_ckpt_dir,
        commit_timestamp_nsecs=time.time_ns(),
    )


# TODO(b/337137764): Move to step.py and fix
# learning/gemini/pax/core/checkpoint_managers_test.py
def on_commit_callback(
    temp_ckpt_dir: epath.Path,
    final_ckpt_dir: epath.Path,
    checkpoint_start_time: float,
    checkpoint_metadata_store: Optional[
        checkpoint.CheckpointMetadataStore
    ] = None,
):
  """To commit save operation, atomically finalizes step dir.

  Delegates to `ensure_atomic_save(...)`.

  Records save duration and lineage-logs step dir.

  Args:
    temp_ckpt_dir: A temporary checkpoint directory, where the checkpoint data
      is currently saved.
    final_ckpt_dir: A directory that represents the finalized name of the
      checkpoint. Should not exist yet if atomicity is ensured via `rename`, but
      may exist if atomicity is ensured by writing a commit success file.
    checkpoint_start_time: The time at which checkpoint saving began.
    checkpoint_metadata_store: `CheckpointMetadataStore` to update commit
      timestamp in step level metadata.
  """
  # If not provided then use checkpoint_metadata_store with blocking writes.
  checkpoint_metadata_store = (
      checkpoint_metadata_store
      or checkpoint.checkpoint_metadata_store(
          enable_write=True, blocking_write=True
      )
  )
  ensure_atomic_save(temp_ckpt_dir, final_ckpt_dir, checkpoint_metadata_store)
  step_lib.record_saved_duration(checkpoint_start_time)
  logging.info('Finished saving checkpoint to `%s`.', final_ckpt_dir)


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


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
  # input `arr` is fully replicated, so it's shape is the global shape.
  global_shape = arr.addressable_data(0).shape

  # Create a 1D mesh to create fully replicated global jax.Array.
  mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
  partition_spec = (
      jax.sharding.PartitionSpec(None)
      if global_shape
      else jax.sharding.PartitionSpec()
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(
      [shard.data for shard in arr.addressable_shards],
      key=lambda x: list(x.devices())[0].id,
  )
  return jax.make_array_from_single_device_arrays(
      global_shape, jax.sharding.NamedSharding(mesh, partition_spec), dbs
  )


def to_shape_dtype_struct(x, dtype=None, scalar_dtype=None):
  """Get ShapeDtypeStruct from array."""
  if isinstance(x, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(
        shape=x.shape,
        dtype=dtype if dtype is not None else x.dtype,
        sharding=x.sharding
        if isinstance(x.sharding, jax.sharding.Sharding)
        else x.sharding.to_jax_sharding(),
    )
  elif isinstance(x, jax.Array):
    dtype = dtype or x.dtype
    return jax.ShapeDtypeStruct(x.shape, dtype, sharding=x.sharding)
  elif isinstance(x, np.ndarray):
    dtype = dtype or x.dtype
    return jax.ShapeDtypeStruct(x.shape, dtype)
  elif is_scalar(x):
    if scalar_dtype is not None:
      return scalar_dtype(x)
    return x
  elif isinstance(x, value_metadata.Metadata):
    if not isinstance(x, value_metadata.ArrayMetadata):
      raise ValueError(f'Unexpected Metadata type: {type(x)}.')
    dtype = dtype or x.dtype
    return jax.ShapeDtypeStruct(
        shape=x.shape,
        dtype=dtype,
        sharding=x.sharding.to_jax_sharding()
        if isinstance(x.sharding, sharding_metadata.ShardingMetadata)
        else x.sharding,
    )
  else:
    raise ValueError(f'Unexpected type: {type(x)}.')


def _sum(x, replica_axis_index):
  return jax.tree.map(
      functools.partial(jnp.sum, axis=replica_axis_index), x
  )


@functools.partial(jax.jit, static_argnums=0)
def fake_zero_data(sharding, x):
  x = jnp.zeros_like(x)
  return jax.lax.with_sharding_constraint(x, sharding)


def broadcast_one_replica_to_all(
    in_tree: Tuple[PyTree, ...],
    global_mesh: jax.sharding.Mesh,
    per_replica_shardings: Tuple[Optional[jax.sharding.NamedSharding], ...],
    replica_axis_index: int,
    is_source: bool,
) -> Tuple[PyTree, ...]:
  """One replica reads the data and broadcasts to others."""
  num_replicas = global_mesh.devices.shape[replica_axis_index]
  replica_axis_name = global_mesh.axis_names[replica_axis_index]

  def pre_jit(x, per_replica_sharding):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_replica_sharding, x)
    inp = jnp.expand_dims(inp, axis=replica_axis_index)
    in_spec = jax.sharding.PartitionSpec(
        *x.sharding.spec[:replica_axis_index],
        replica_axis_name,
        *x.sharding.spec[replica_axis_index:],
    )
    global_shape = (
        x.shape[:replica_axis_index]
        + (num_replicas,)
        + x.shape[replica_axis_index:]
    )
    global_sharding = jax.sharding.NamedSharding(global_mesh, in_spec)
    return jax.make_array_from_single_device_arrays(
        global_shape, global_sharding, [s.data for s in inp.addressable_shards]
    )

  out_sharding = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec)
      ),
      in_tree,
  )
  in_tree_sharded = jax.tree.map(
      pre_jit, in_tree, per_replica_shardings
  )
  # Delete immediately to conserve memory.
  jax.tree.map(lambda x: x.delete(), in_tree)
  out_tree = jax.jit(
      functools.partial(_sum, replica_axis_index=replica_axis_index),
      out_shardings=out_sharding,
  )(in_tree_sharded)
  jax.block_until_ready(out_tree)
  return out_tree


def get_primary_replica_ids_and_pids(
    replica_axis_idx: int,
    mesh: jax.sharding.Mesh,
    primary_replica_id: int,
):
  """Returns the primary replica ids and process ids."""
  replica_devices = np.take(
      mesh.devices,
      primary_replica_id,
      axis=replica_axis_idx,
  ).flatten()
  pids = set([d.process_index for d in replica_devices])
  ids = set([d.id for d in replica_devices])
  return ids, pids
