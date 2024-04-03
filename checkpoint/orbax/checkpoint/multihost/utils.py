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

"""Orbax utils related to multihost functionality."""

import functools
import time
from typing import Any, Set, Optional
import zlib
from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
import numpy as np




def _psum(x: Any) -> Any:
  return jax.tree.map(functools.partial(jnp.sum, axis=0), x)


def broadcast_one_to_some(
    in_tree: Any,
    *,
    is_source: Optional[bool] = None,
    processes: Optional[Set[int]] = None,
) -> Any:
  """Broadcast data from a source host to some or all other hosts.

  The function should only be called by participating processes - i.e. those
  appearing in `processes` if specified, or any process if not specified.

  Inspired by JAX multihost_utils.

  Args:
    in_tree: pytree of arrays - each array *must* have the same shape across the
      hosts.
    is_source: Whether the current process is the source of the broadcast. If
      None, an arbitrary process within `processes` will be selected as the
      source for the broadcast.
    processes: Set of participating processes. Assumed to be all processes if
      None.

  Returns:
    A pytree matching in_tree where the leaves now all contain the data from the
    first host.
  """
  processes = processes or set(range(jax.process_count()))
  if is_source is None:
    primary_process = next(iter(processes))
    is_source = jax.process_index() == primary_process
  if jax.process_index() not in processes:
    raise ValueError(
        'Attempted to broadcast from one host to other hosts, but the current'
        f' process: {jax.process_index()} was not present in the provided list'
        f' of processes: {processes}.'
    )
  devices: np.ndarray = np.array(
      [d for d in jax.devices() if d.process_index in processes]
  ).reshape(len(processes), jax.local_device_count())

  global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))
  pspec = jax.sharding.PartitionSpec('processes')

  def pre_jit(x):
    if is_source:
      inp = x
    else:
      inp = np.zeros_like(x)
    inp = np.expand_dims(inp, axis=0)
    return multihost_utils.host_local_array_to_global_array(
        inp, global_mesh, pspec
    )

  def post_jit(x):
    return np.asarray(x.addressable_data(0))

  in_tree = jax.tree.map(pre_jit, in_tree)
  out_tree = jax.jit(
      _psum,
      out_shardings=jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec()
      ),
  )(in_tree)
  return jax.tree.map(post_jit, out_tree)


def broadcast_one_to_all(in_tree, is_source: Optional[bool] = None):
  """Broadcast data from a source host to all other hosts."""
  return broadcast_one_to_some(in_tree, is_source=is_source)


def should_skip_process_sync() -> bool:
  return False


def _assert_tree_leaves_all_equal(
    in_tree: Any,
    fail_message: str = '',
    *,
    processes: Optional[Set[int]],
):
  """Verifies that all the hosts have the same tree of values."""
  expected = broadcast_one_to_some(in_tree, processes=processes)
  if not jax.tree_util.tree_all(
      jax.tree_util.tree_map(lambda *x: np.all(np.equal(*x)), in_tree, expected)
  ):
    raise AssertionError(
        f'{fail_message} Expected: {expected}; got: {in_tree}.'
    )


def sync_global_processes(name: str, processes: Optional[Set[int]] = None):
  """Barrier to sync concurrent processes.

  Args:
    name: barrier name.
    processes: If None, expects to wait across all processes and devices.
      Otherwise, creates a barrier only across devices associated with the given
      processes.
  """
  if should_skip_process_sync():
    logging.info('Skipping global process sync, barrier name: %s', name)
    return
  logging.debug('sync_global_processes: %s', name)
  sync_start_time = time.time()
  h = np.uint32(zlib.crc32(name.encode()))
  _assert_tree_leaves_all_equal(
      h, f"sync_global_processes name mismatch ('{name}')", processes=processes
  )
  # This may end up just being too noisy given how many barriers there are, but
  # it does represent how long different processes waited around waiting for
  # other processes to reach a barrier.
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/sync_global_devices_duration_sec',
      time.time() - sync_start_time,
  )


def reached_preemption(step: int) -> bool:
  """Returns True if a preemption sync point has been reached."""
  return multihost_utils.reached_preemption_sync_point(step)
