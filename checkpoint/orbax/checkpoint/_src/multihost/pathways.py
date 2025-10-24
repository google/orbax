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

"""Pathways-specific multihost utilities."""

import functools
import jax


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
  for d in devices:
    attrs = []
    if hasattr(d, 'virtual_task_index'):
      attrs.append(d.virtual_task_index)
    if hasattr(d, 'slice_index'):
      attrs.append(d.slice_index)
    workers.add(tuple(attrs))
  return len(workers)
