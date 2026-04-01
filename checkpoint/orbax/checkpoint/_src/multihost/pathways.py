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

from collections.abc import Sequence
import functools
import re
from typing import Optional

from absl import logging
import jax


def _extract_int_from_repr(
    device: jax.Device,
    pattern: str,
) -> Optional[int]:
  match = re.search(pattern, repr(device))
  if match:
    return int(match.group(1))
  return None


def _get_device_task_index(device: jax.Device) -> Optional[int]:
  if (
      hasattr(device, 'virtual_task_index')
      and device.virtual_task_index is not None
  ):
    return int(device.virtual_task_index)
  if hasattr(device, 'task_id') and device.task_id is not None:
    return int(device.task_id)
  task_index = _extract_int_from_repr(device, r'logical_task=(\d+)')
  if task_index is not None:
    return task_index
  return _extract_int_from_repr(device, r'vtask=(\d+)')


def _get_device_slice_index(device: jax.Device) -> Optional[int]:
  if hasattr(device, 'slice_index') and device.slice_index is not None:
    return int(device.slice_index)
  return _extract_int_from_repr(device, r'slice=(\d+)')


def _get_device_worker_key(device: jax.Device) -> tuple[int, ...]:
  """Returns a tuple uniquely identifying the worker/VM for `device`."""
  task_index = _get_device_task_index(device)
  slice_index = _get_device_slice_index(device)

  if task_index is not None and slice_index is not None:
    return (task_index, slice_index)
  if task_index is not None:
    return (task_index,)
  if slice_index is not None:
    return (slice_index,)

  logging.warning(
      'Unable to infer Pathways worker key from device attributes/repr: %s',
      repr(device),
  )
  # Fallback for non-Pathways environments.
  return (device.process_index,)


def group_devices_by_worker(
    devices: Sequence[jax.Device],
) -> dict[tuple[int, ...], list[jax.Device]]:
  """Groups devices by their worker/VM.

  On Pathways Single Controller, device.process_index is 0 for all devices,
  so this function uses (virtual_task_index, slice_index) to identify unique
  workers. Falls back to process_index for non-Pathways environments.

  Args:
    devices: A sequence of JAX devices.

  Returns:
    A dict mapping worker keys to lists of devices belonging to that
    worker. Order is by first device occurrence.
  """
  worker_devices = {}
  for d in devices:
    key = _get_device_worker_key(d)
    worker_devices.setdefault(key, []).append(d)
  return worker_devices


def compute_distributed_to_device_ids(
    devices: Sequence[jax.Device],
) -> list[list[int]]:
  """Returns a per-worker list of sorted device IDs, in slice-major order.

  This is the Pathways equivalent of
  ``multihost.distributed_to_device_ids()``, which relies on the JAX
  distributed key-value store (unavailable on Single Controller).

  Args:
    devices: All devices in the current process view.

  Returns:
    A list of ``list[int]``, one entry per worker, each containing the
    sorted device IDs belonging to that worker.
  """
  worker_groups = group_devices_by_worker(devices)
  sorted_worker_groups = sorted(
      worker_groups.items(),
      key=lambda item: (item[0][1], item[0][0])
      if len(item[0]) == 2
      else item[0],
  )
  return [sorted(d.id for d in wdevs) for _, wdevs in sorted_worker_groups]


@functools.lru_cache(maxsize=1)
def worker_count(global_mesh: Optional[jax.sharding.Mesh]) -> int:
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
    attrs = []
    if hasattr(d, 'virtual_task_index'):
      attrs.append(d.virtual_task_index)
    else:
      # virtual_task_index is not exposed, so get it from repr.
      if match := re.findall(r'vtask=(\d+),', repr(d)):
        attrs.append(int(match[0]))
      else:
        warn = True
    if hasattr(d, 'slice_index'):
      attrs.append(d.slice_index)
    else:
      warn = True
    workers.add(tuple(attrs))

  if warn:
    logging.warning(
        'Worker_count() may not be accurate, vtask or slice_index not found in'
        ' devices: %s',
        devices,
    )

  return len(workers)
