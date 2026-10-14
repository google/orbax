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

With `num_processes > 1`, each simulated host is its own process.
`run_local_cluster` starts them as child processes: all of them on a single
machine, or an even share of them on each task of a multi-task launch (e.g.
Borg), which sets `SIMDEVICE_TASK_ID` and `SIMDEVICE_NUM_TASKS`. A launcher
can also start every simulated host itself and set `SIMDEVICE_PROCESS_ID`.
"""

import atexit
from collections.abc import Sequence
import os
import subprocess
import sys
import tempfile
import time

from absl import flags
from absl import logging
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import configs
import portpicker
import yaml

_PROCESS_ID_ENV = 'SIMDEVICE_PROCESS_ID'
# Set by a multi-task launcher: this task's index and the number of tasks. The
# launcher also sets the barrier and JAX coordinator addresses (task 0).
_TASK_ID_ENV = 'SIMDEVICE_TASK_ID'
_NUM_TASKS_ENV = 'SIMDEVICE_NUM_TASKS'
_LAUNCHER_ADDRESS_ENVS = (
    'SIMDEVICE_BARRIER_ADDRESS',
    'JAX_COORDINATOR_ADDRESS',
)
# Multi-process launchers set one of these: the environment variables for
# `jax.distributed`, or the flag for  JAX on Borg.
_JAX_PROCESS_ENVS = ('JAX_PROCESS_ID', 'JAX_COORDINATOR_ADDRESS')
_JAX_CONTROLLER_FLAG = 'jax_controller_address'
_POLL_INTERVAL_SECONDS = 1.0

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

  if config.num_processes > 1:
    # Orbax's default barrier is a device all-gather, whose result SimDevice
    # fabricates. Use the JAX coordination service's barrier instead.
    barrier_flag = multihost.EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_BARRIER.name
    flags.FLAGS[barrier_flag].value = True

  _active_config = config
  logging.info(
      'SimDevice active: %s (topology file %s).', config, topology_path
  )


def _started_as_one_of_many() -> bool:
  """Returns whether a launcher started this process as one of several."""
  if any(name in os.environ for name in _JAX_PROCESS_ENVS):
    return True
  return _JAX_CONTROLLER_FLAG in flags.FLAGS and bool(
      flags.FLAGS[_JAX_CONTROLLER_FLAG].value
  )


def needs_local_cluster(config: configs.SimDeviceConfig) -> bool:
  """Returns whether this process must start the simulated hosts itself.

  True for a multi-process config when nothing has assigned this process a
  simulated host yet.

  Args:
    config: The simulated cluster.

  Raises:
    RuntimeError: If a launcher started this process as one of several but set
      neither `SIMDEVICE_PROCESS_ID` nor `SIMDEVICE_TASK_ID`. Each process
      would otherwise start its own local cluster and pass on its own.
  """
  if config.num_processes == 1 or _PROCESS_ID_ENV in os.environ:
    return False
  if _TASK_ID_ENV not in os.environ and _started_as_one_of_many():
    raise RuntimeError(
        'This process was started as one of several, but neither'
        f' {_PROCESS_ID_ENV} nor {_TASK_ID_ENV} is set. The launcher must set'
        ' one of them, e.g. to %task% on Borg.'
    )
  return True


def _local_process_ids(config: configs.SimDeviceConfig) -> range:
  """Returns the ids of the simulated hosts this process starts.

  Args:
    config: The simulated cluster.

  Raises:
    ValueError: If the simulated hosts do not split evenly across the
      launcher's tasks.
  """
  if _TASK_ID_ENV not in os.environ:
    return range(config.num_processes)
  task_id = int(os.environ[_TASK_ID_ENV])
  num_tasks = int(os.environ[_NUM_TASKS_ENV])
  if config.num_processes % num_tasks:
    raise ValueError(
        f'simdevice.num_processes ({config.num_processes}) must be a multiple'
        f' of the number of launcher tasks ({num_tasks}).'
    )
  per_task = config.num_processes // num_tasks
  return range(task_id * per_task, (task_id + 1) * per_task)


def _cluster_env() -> dict[str, str]:
  """Returns the barrier and JAX coordinator settings for the children.

  A multi-task launcher provides them (they point at task 0); on a single
  machine this process picks free local ports.

  Raises:
    ValueError: If a multi-task launcher did not provide them.
  """
  if _TASK_ID_ENV in os.environ:
    missing = [
        name for name in _LAUNCHER_ADDRESS_ENVS if name not in os.environ
    ]
    if missing:
      raise ValueError(f'{_TASK_ID_ENV} is set, but {missing} are not.')
    return {}
  barrier_port = portpicker.pick_unused_port()
  return {
      'SIMDEVICE_BARRIER_ADDRESS': f'localhost:{barrier_port}',
      'SIMDEVICE_BARRIER_PORT': str(barrier_port),
      'JAX_COORDINATOR_ADDRESS': f'localhost:{portpicker.pick_unused_port()}',
  }


def run_local_cluster(
    config: configs.SimDeviceConfig, command: Sequence[str] | None = None
) -> None:
  """Runs one local child process per simulated host and waits for them.

  On a multi-task launch, this task runs its share of the simulated hosts.
  Each child gets its simulated host id, the SimDevice barrier address and the
  JAX coordinator address through the environment. If any child fails, the
  others are stopped rather than left waiting at a barrier.

  Args:
    config: The simulated cluster.
    command: The command each child runs. If None, this process's own command
      line, so each child reruns the same benchmark as one simulated host.

  Raises:
    RuntimeError: If any simulated host fails.
  """
  if command is None:
    command = sys.orig_argv
  process_ids = _local_process_ids(config)
  cluster_env = dict(os.environ, **_cluster_env())
  children = []
  for process_id in process_ids:
    env = dict(
        cluster_env,
        SIMDEVICE_PROCESS_ID=str(process_id),
        JAX_PROCESS_ID=str(process_id),
        JAX_NUM_PROCESSES=str(config.num_processes),
    )
    children.append(subprocess.Popen(command, env=env))
  logging.info(
      'Started simulated hosts %s of %d (coordinator %s, barrier %s).',
      list(process_ids),
      config.num_processes,
      cluster_env['JAX_COORDINATOR_ADDRESS'],
      cluster_env['SIMDEVICE_BARRIER_ADDRESS'],
  )

  try:
    while True:
      exit_codes = [child.poll() for child in children]
      failures = {
          process_id: code
          for process_id, code in zip(process_ids, exit_codes)
          if code not in (None, 0)
      }
      if failures:
        raise RuntimeError(
            f'Simulated hosts failed (process id: exit code): {failures}.'
        )
      if all(code == 0 for code in exit_codes):
        return
      time.sleep(_POLL_INTERVAL_SECONDS)
  finally:
    for child in children:
      if child.poll() is None:
        child.kill()
        child.wait()


def check_process_count() -> None:
  """Raises if JAX did not start one process per simulated host.

  `jax.distributed` setup failures are otherwise logged and the run continues
  as a single process, which on SimDevice would look like a passing run.

  Raises:
    RuntimeError: If SimDevice is active and the process count is wrong.
  """
  if _active_config is None:
    return
  if jax.process_count() != _active_config.num_processes:
    raise RuntimeError(
        f'SimDevice expects {_active_config.num_processes} processes, but JAX'
        f' has {jax.process_count()}; jax.distributed did not initialize.'
    )
