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

"""Runs the Orbax benchmark framework on SimDevice: simulated TPUs on CPU.

SimDevice is a PJRT plugin that fabricates a TPU cluster of any size without
allocating device memory or doing math. A benchmark YAML opts in with a
`simdevice:` block (see `configs.SimDeviceConfig`); `activate` must then run
before any JAX backend starts, i.e. before the first `jax.devices()`.
"""

import atexit
import os
import tempfile

from absl import logging
import jax
from orbax.checkpoint._src.testing.benchmarks.core import configs
import yaml

_active_config: configs.SimDeviceConfig | None = None


def render_topology_yaml(config: configs.SimDeviceConfig) -> str:
  """Returns the SimDevice topology YAML for `config`.

  The file is shared by every simulated host. Per-host identity (process id,
  barrier address) is set through SimDevice's environment variables instead,
  because a barrier address in the YAML would override them.

  Args:
    config: The simulated cluster.
  """
  return yaml.safe_dump({
      'device': {'type': config.device_type},
      'topology': {'mesh_dimensions': list(config.topology)},
      'orchestration': {'num_processes': config.num_processes},
  })


def activate(config: configs.SimDeviceConfig) -> None:
  """Makes SimDevice the JAX backend for this process.

  Args:
    config: The simulated cluster.

  Raises:
    RuntimeError: If called twice, or after a JAX backend has started.
  """
  global _active_config
  if _active_config is not None:
    raise RuntimeError('SimDevice is already active.')

  # SimDevice reads the file when the JAX backend starts, later in the run, so
  # it must outlive this function.
  with tempfile.NamedTemporaryFile(
      'w', prefix='simdevice_', suffix='.yaml', delete=False
  ) as f:
    f.write(render_topology_yaml(config))
  topology_path = f.name
  atexit.register(os.remove, topology_path)
  os.environ['SIMDEVICE_CONFIG_PATH'] = topology_path
  if config.fabricated_fill_seed is not None:
    os.environ['SIMDEVICE_FABRICATED_FILL_SEED'] = str(
        config.fabricated_fill_seed
    )

  # Registers the plugin and puts it first in JAX_PLATFORMS. Raises if a JAX
  # backend has already started; a failed import must not fall back to CPU.
  import simdevice  # pylint: disable=g-import-not-at-top,unused-import

  # JAX reads JAX_PLATFORMS when it is imported, which is before this runs.
  jax.config.update('jax_platforms', os.environ['JAX_PLATFORMS'])

  _active_config = config
  logging.info(
      'SimDevice active: %s (topology file %s).', config, topology_path
  )
