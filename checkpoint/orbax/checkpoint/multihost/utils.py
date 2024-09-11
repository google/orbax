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

import threading
import time
from typing import List, Optional, Protocol, Set
from absl import flags
from absl import logging
import jax
from jax.experimental import multihost_utils
import numpy as np

# Default timeout in seconds.
_DEFAULT_BARRIER_TIMEOUT = 1200

DIRECTORY_CREATION_TIMEOUT = 360
DIRECTORY_DELETION_TIMEOUT = 360

# Used in unit tests with multiple parallel test cases.
_TEST_CASE_INDEX = None

# Map from runtime process index to distributed process index.
_RUNTIME_TO_DISTRIBUTED_ID: List[int] = None

EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID = flags.DEFINE_bool(
    'experimental_orbax_use_distributed_process_id',
    False,
    'If True, uses jax._src.distributed.global_state.process_id instead of'
    ' jax.process_index().',
)




def is_runtime_to_distributed_ids_initialized() -> bool:
  return _RUNTIME_TO_DISTRIBUTED_ID is not None


def initialize_runtime_to_distributed_ids():
  """Initializes the process index mapping.

  Note that this function is for experimental purposes only. It is intended to
  be called exactly once per processes.

  TODO(b/325293150): Remove once the bug is resolved.
  """
  global _RUNTIME_TO_DISTRIBUTED_ID
  client = _get_jax_distributed_client()

  # Index is distributed id.
  # Value is runtime id.
  _RUNTIME_TO_DISTRIBUTED_ID = [0 for _ in range(jax.process_count())]
  own_runtime_id = jax.process_index()
  own_distributed_id = jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
  dir_key = 'jax/process_id/'
  key = dir_key + str(own_runtime_id)
  client.key_value_set(key, str(own_distributed_id))
  client.wait_at_barrier('orbax_global_discovery', timeout_in_ms=5000)
  ids = client.key_value_dir_get(dir_key)
  for key, distributed_id in ids:
    runtime_id = int(key.split('/')[-1])
    _RUNTIME_TO_DISTRIBUTED_ID[runtime_id] = int(distributed_id)
  logging.info(
      '[process=%s][thread=%s] runtime_to_distributed_id: %s',
      process_index(),
      threading.current_thread().name,
      _RUNTIME_TO_DISTRIBUTED_ID,
  )


def runtime_to_distributed_ids() -> List[int]:
  if _RUNTIME_TO_DISTRIBUTED_ID is None:
    raise ValueError('Please call initialize_runtime_to_distributed_ids().')
  return _RUNTIME_TO_DISTRIBUTED_ID


def runtime_to_distributed_process_id(pid: int) -> int:
  """Converts a distributed process index to a runtime process index."""
  if _RUNTIME_TO_DISTRIBUTED_ID is None:
    raise ValueError('Please call initialize_runtime_to_distributed_ids().')
  return _RUNTIME_TO_DISTRIBUTED_ID[pid]


def broadcast_one_to_all(in_tree, is_source: Optional[bool] = None):
  """Broadcast data from a source host to all other hosts."""
  if is_source is None:
    is_source = process_index() == 0
  return multihost_utils.broadcast_one_to_all(in_tree, is_source=is_source)


def should_skip_process_sync() -> bool:
  if jax.process_count() == 1:
    return True
  return False


def _get_jax_distributed_client():
  client = jax._src.distributed.global_state.client  # pylint: disable=protected-access
  if client is None:
    raise ValueError(
        'Distributed system is not available; please initialize it via '
        '`jax.distributed.initialize()` at the start of your program.'
    )
  return client


class BarrierSyncFn(Protocol):
  """Protocol for a barrier synchronization callable."""

  def __call__(self, *, key: str, timeout_ms: int) -> None:
    """Blocks on a barrier identified by key with the given timeout."""
    ...


def get_barrier_sync_fn(
    *,
    processes: Optional[Set[int]] = None,
) -> BarrierSyncFn:
  """Provides a barrier synchronization function for JAX processes.

  Barriers with different sync keys are safe to use from independent background
  threads.

  Args:
    processes: If None, expects to wait across all processes and devices.
      Otherwise, creates a barrier only across devices associated with the given
      processes.

  Returns:
    No-op function if there is a single JAX process.
    A barrier synchronization callable which accepts two arguments:
      - "key": [str] unique barrier id;
      - "timeout_ms": [int] timeout to use for waiting on the barrier.
    Should be called from all JAX processes with the same sync key and will
    block until either 1) all processes have reached the barrier or
    2) the timeout is exceeded.
  """
  if jax.process_count() == 1:
    return lambda **kwargs: None

  client = _get_jax_distributed_client()
  barrier_processes = processes or set(range(jax.process_count()))
  if process_index() not in barrier_processes:
    raise ValueError(
        'Attempted to create a barrier across a subset of processes, but the'
        f' current process: {process_index()} was not present in the provided'
        f' list of processes: {barrier_processes}.'
    )

  # Use distributed ids.
  if processes is None:
    barrier_processes = None
  else:
    # Don't map ids anymore if we are using distributed ids.
    barrier_processes = list(barrier_processes)

  def _fn(*, key: str, timeout_ms: int) -> None:
    key = _unique_barrier_key(key)
    logging.info(
        '[process=%s][thread=%s] Waiting at barrier: %s',
        process_index(),
        threading.current_thread().name,
        key,
    )
    if processes is None:
      client.wait_at_barrier(key, timeout_ms)
    else:
      logging.vlog(
          1,
          '[process=%s][thread=%s] Barrier processes: %s',
          process_index(),
          threading.current_thread().name,
          barrier_processes,
      )
      client.wait_at_barrier(key, timeout_ms, process_ids=barrier_processes)
    logging.info(
        '[process=%s][thread=%s] Done waiting at barrier: %s',
        process_index(),
        threading.current_thread().name,
        key,
    )

  return _fn


def _unique_barrier_key(key: str) -> str:
  """Function that can be overridden for testing purposes."""
  return key


def unique_barrier_key(
    key: str,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
  """Constructs a key given an optional prefix and suffix."""
  if prefix is not None:
    key = f'{prefix}_{key}'
  if suffix is not None:
    key = f'{key}.{suffix}'
  return key


def sync_global_processes(
    name: str,
    *,
    timeout: Optional[int] = None,
    processes: Optional[Set[int]] = None,
    barrier_sync_fn: Optional[BarrierSyncFn] = None,
):
  """Barrier to sync concurrent processes.

  NOTE: The barrier name must be unique, i.e. no process should wait on the
  same barrier name multiple times.

  Args:
    name: barrier name. Must be unique.
    timeout: timeout in seconds.
    processes: If None, expects to wait across all processes and devices.
      Otherwise, creates a barrier only across devices associated with the given
      processes.
    barrier_sync_fn: Used as the implementation for the synchronization. If not
      provided, a default implementation is used.
  """
  if should_skip_process_sync():
    logging.info(
        '[process=%s][thread=%s] Skipping global process sync, barrier'
        ' name: %s',
        process_index(),
        threading.current_thread().name,
        name,
    )
    return
  timeout = timeout or _DEFAULT_BARRIER_TIMEOUT
  sync_start_time = time.time()
  # Temporarily default to existing behavior to minimize risk of breakage.
  if processes is None:
    key = _unique_barrier_key(name)
    logging.info(
        '[process=%s][thread=%s] Waiting with jax/sync_global_devices("%s")',
        process_index(),
        threading.current_thread().name,
        key,
    )
    multihost_utils.sync_global_devices(key)
    logging.info(
        '[process=%s][thread=%s] Done waiting with'
        ' jax/sync_global_devices("%s")',
        process_index(),
        threading.current_thread().name,
        key,
    )
  else:
    barrier_sync_fn = barrier_sync_fn or get_barrier_sync_fn(
        processes=processes
    )
    barrier_sync_fn(key=name, timeout_ms=timeout * 1000)
  # This may end up just being too noisy given how many barriers there are, but
  # it does represent how long different processes waited around waiting for
  # other processes to reach a barrier.
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/sync_global_devices_duration_sec',
      time.time() - sync_start_time,
  )


def _maybe_log_reached_preemption(
    step: int, preemption_sync_point_reached: bool
):
  if not preemption_sync_point_reached:
    return
  jax.monitoring.record_event('/jax/orbax/write/preemption')
  logging.warning(
      '[process=%s][thread=%s] Reached preemption sync point, step=%s',
      process_index(),
      threading.current_thread().name,
      step,
  )


def reached_preemption(step: int) -> bool:
  """Returns True if a preemption sync point has been reached."""

  preemption_sync_point_reached = multihost_utils.reached_preemption_sync_point(
      step
  )
  _maybe_log_reached_preemption(step, preemption_sync_point_reached)
  return preemption_sync_point_reached


def is_primary_host(primary_host: Optional[int]):
  if primary_host is None or primary_host == process_index():
    return True
  return False


def process_index() -> int:
  """Customized logic for obtaining JAX process index."""
  try:
    experimental_orbax_use_distributed_process_id = (
        EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.value
    )
  except Exception:  # pylint: disable=broad-exception-caught
    logging.log_first_n(
        logging.INFO,
        '[thread=%s] Failed to get flag value for'
        ' EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.',
        1,
        threading.current_thread().name,
    )
    experimental_orbax_use_distributed_process_id = False
  if experimental_orbax_use_distributed_process_id:
    logging.log_first_n(
        logging.INFO,
        '[thread=%s] Using distributed process id.',
        1,
        threading.current_thread().name,
    )
    return jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
  else:
    return jax.process_index()


def unique_processes_from_devices(device_array: np.ndarray) -> Set[int]:
  get_pids_from_devices = np.vectorize(
      lambda d: runtime_to_distributed_process_id(d.process_index)
  )
  return set(get_pids_from_devices(device_array).flat)
