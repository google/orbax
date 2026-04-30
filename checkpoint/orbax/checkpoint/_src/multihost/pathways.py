# Copyright 2026 The Orbax Authors.
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

"""Pathways-specific multihost utilities."""

import collections
from collections.abc import Sequence
import functools
import re

from absl import logging
import jax


_WARNED_REPR_PATTERNS = set()


def _extract_int_from_repr(
    device: jax.Device,
    pattern: str,
) -> int | None:
  """Extracts an integer from the device repr using the given regex pattern."""
  match = re.search(pattern, repr(device))
  if match:
    if pattern not in _WARNED_REPR_PATTERNS:
      _WARNED_REPR_PATTERNS.add(pattern)
      logging.warning(
          'Pathways worker-key inference fell back to repr parsing for '
          'pattern=%r. Sample device=%r',
          pattern,
          device,
      )
    return int(match.group(1))
  return None


def _get_device_task_index(device: jax.Device) -> int | None:
  if (
      hasattr(device, 'virtual_task_index')
      and device.virtual_task_index is not None
  ):
    return int(device.virtual_task_index)
  task_index = _extract_int_from_repr(device, r'logical_task=(\d+)')
  if task_index is not None:
    return task_index
  if hasattr(device, 'task_id') and device.task_id is not None:
    return int(device.task_id)
  return _extract_int_from_repr(device, r'vtask=(\d+)')


def _get_device_slice_index(device: jax.Device) -> int | None:
  if hasattr(device, 'slice_index') and device.slice_index is not None:
    return int(device.slice_index)
  return None


def _get_device_worker_key(device: jax.Device) -> tuple[int, ...]:
  """Returns a tuple uniquely identifying the worker/VM for `device`."""
  task_index = _get_device_task_index(device)
  slice_index = _get_device_slice_index(device)

  if task_index is not None and slice_index is not None:
    return (task_index, slice_index)
  if task_index is not None:
    return (task_index,)
  if slice_index is not None:
    msg = (
        'Pathways worker-key inference requires a task identifier; '
        'slice_index alone is ambiguous:'
        f' {device!r}'
    )
    logging.error(msg)
    raise ValueError(msg)

  msg = (
      'Unable to infer Pathways worker key from device attributes/repr:'
      f' {device!r}'
  )
  logging.error(msg)
  raise ValueError(msg)


def group_devices_by_worker(
    devices: Sequence[jax.Device],
) -> dict[tuple[int, ...], list[jax.Device]]:
  """Groups devices by their worker/VM.

  Pathways runtimes expose worker identity via device task/slice attributes
  because ``device.process_index`` is not unique per worker there.

  Args:
    devices: A sequence of JAX devices.

  Returns:
    A dict mapping worker keys to lists of devices belonging to that
    worker. Order is by first device occurrence.
  """
  worker_devices = collections.defaultdict(list)
  for d in devices:
    key = _get_device_worker_key(d)
    worker_devices[key].append(d)
  logging.info(
      'Grouped %d devices into %d Pathways workers: %s',
      len(devices),
      len(worker_devices),
      sorted(worker_devices),
  )
  return dict(worker_devices)


def _get_worker_count_key(device: jax.Device) -> tuple[tuple[int, ...], bool]:
  """Returns a best-effort worker key for `worker_count` compatibility."""
  attrs = []
  warn = False

  task_index = _get_device_task_index(device)
  if task_index is not None:
    attrs.append(task_index)
  else:
    warn = True

  slice_index = _get_device_slice_index(device)
  if slice_index is not None:
    attrs.append(slice_index)
  else:
    warn = True

  return tuple(attrs), warn


@functools.lru_cache(maxsize=1)
def worker_count(global_mesh: jax.sharding.Mesh | None) -> int:
  """Gets the number of Pathways workers.

  Args:
    global_mesh: The global mesh of active devices. If None is provided,
      `jax.devices()` will be used.

  Returns:
    The number of Pathways workers in the mesh.
  """
  global_mesh = global_mesh or jax.sharding.Mesh(jax.devices(), 'x')
  devices = global_mesh.devices.flatten()
  workers = set()
  warn = False
  for d in devices:
    worker_key, missing_metadata = _get_worker_count_key(d)
    workers.add(worker_key)
    warn = warn or missing_metadata

  if warn:
    logging.warning(
        'worker_count() may not be accurate; task or slice metadata was '
        'missing from some Pathways devices: %s',
        devices,
    )
  return len(workers)
