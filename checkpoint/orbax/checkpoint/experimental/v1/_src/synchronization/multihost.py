# Copyright 2025 The Orbax Authors.
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

"""Orbax utils related to multihost_utils functionality."""

import threading
import time
from typing import Collection, Optional
from absl import logging
import jax
from jax.experimental import multihost_utils
from orbax.checkpoint.experimental.v1._src.synchronization import signaling_client

# Default timeout in seconds.
_DEFAULT_BARRIER_TIMEOUT = 300




def coordination_timeout() -> int:
  """Returns the coordination timeout in seconds."""
  return _DEFAULT_BARRIER_TIMEOUT


def should_skip_process_sync(processes: Collection[int] | None = None) -> bool:
  if processes and len(processes) == 1 and process_index() in processes:
    return True
  if jax.process_count() == 1:
    return True
  return False


def _unique_barrier_key(key: str) -> str:
  """Function that can be overridden for testing purposes."""
  return key


def unique_barrier_key(
    key: str,
    *,
    prefix: str | None = None,
    suffix: str | None = None,
) -> str:
  """Constructs a key given an optional prefix and suffix."""
  if prefix is not None:
    key = f'{prefix}_{key}'
  if suffix is not None:
    key = f'{key}.{suffix}'
  return key


async def sync_global_processes(
    key: str,
    *,
    timeout: int | None = None,
    processes: Collection[int] | None = None,
    record_event_name: str = '/jax/checkpoint/sync_global_devices_duration_sec',
):
  """Barrier to sync concurrent processes.

  NOTE: The barrier name must be unique, i.e. no process should wait on the
  same barrier name multiple times.

  Args:
    key: barrier name. Must be unique.
    timeout: timeout in seconds.
    processes: If None, expects to wait across all processes and devices.
      Otherwise, creates a barrier only across devices associated with the given
      processes.
    record_event_name: The name of the event to record the duration of the
      synchronization.
  """
  if should_skip_process_sync(processes):
    logging.vlog(
        1,
        '[process=%s][thread=%s] Skipping global process sync, barrier'
        ' name: %s',
        process_index(),
        threading.current_thread().name,
        key,
    )
    return
  sync_start_time = time.time()
  logging.vlog(
      1,
      '[process=%s][thread=%s] Waiting at barrier: %s with processes: %s',
      process_index(),
      threading.current_thread().name,
      key,
      processes
  )

  timeout = timeout or coordination_timeout()
  if timeout <= 0:
    raise ValueError(f'Timeout must be positive, but got {timeout} seconds.')
  client = signaling_client.get_signaling_client()
  key = _unique_barrier_key(key)
  if processes is not None:
    if process_index() not in processes:
      raise ValueError(
          'Attempted to create a barrier across a subset of processes, but the'
          f' current process: {process_index()} was not present in the provided'
          f' list of processes: {processes}.'
      )
    processes = list(processes)
  await client.wait_at_barrier(key, timeout_secs=timeout, process_ids=processes)
  logging.vlog(
      1,
      '[process=%s][thread=%s] Done waiting at barrier: %s',
      process_index(),
      threading.current_thread().name,
      key,
  )

  # This may end up just being too noisy given how many barriers there are, but
  # it does represent how long different processes waited around waiting for
  # other processes to reach a barrier.
  jax.monitoring.record_event_duration_secs(
      record_event_name,
      time.time() - sync_start_time,
  )


def is_primary_host(primary_host: int | None):
  if primary_host is None or primary_host == process_index():
    return True
  return False


def process_count() -> int:
  return jax.process_count()


def process_index() -> int:
  # Note that jax.process_index() does not return the same thing as
  # global_state.process_id. We rely on the latter to work with barriers over a
  # subset of processes.
  return jax._src.distributed.global_state.process_id  # pylint: disable=protected-access


def broadcast_one_to_all(in_tree, is_source: Optional[bool] = None):
  """Broadcast data from a source host to all other hosts."""
  if is_source is None:
    is_source = process_index() == 0
  return multihost_utils.broadcast_one_to_all(in_tree, is_source=is_source)
