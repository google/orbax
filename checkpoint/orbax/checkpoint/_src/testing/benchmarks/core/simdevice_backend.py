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

With `num_processes > 1`, each simulated host is its own process. A launcher
that starts them (e.g. on Borg) sets `SIMDEVICE_PROCESS_ID` for each one; on a
single machine, `run_local_cluster` starts them as child processes.
"""

import atexit
from collections.abc import Sequence
import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time

from absl import flags
from absl import logging
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import configs
import yaml

_PROCESS_ID_ENV = 'SIMDEVICE_PROCESS_ID'
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
    RuntimeError: If a launcher started this process as one of several but did
      not set `SIMDEVICE_PROCESS_ID`. Each process would otherwise start its
      own local cluster and pass on its own.
  """
  if config.num_processes == 1 or _PROCESS_ID_ENV in os.environ:
    return False
  if _started_as_one_of_many():
    raise RuntimeError(
        f'This process was started as one of several, but {_PROCESS_ID_ENV}'
        ' is not set. The launcher must set it to the simulated host id, e.g.'
        ' %task% on Borg.'
    )
  return True


def _pick_unused_ports(count: int) -> list[int]:
  """Returns `count` distinct TCP ports that are free on this machine now."""
  with contextlib.ExitStack() as stack:
    sockets = []
    # Keep every socket open until all ports are read, so they are distinct.
    for _ in range(count):
      sock = stack.enter_context(socket.socket(socket.AF_INET))
      sock.bind(('localhost', 0))
      sockets.append(sock)
    return [sock.getsockname()[1] for sock in sockets]


def run_local_cluster(
    config: configs.SimDeviceConfig, command: Sequence[str] | None = None
) -> None:
  """Runs one local child process per simulated host and waits for them.

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
  coordinator_port, barrier_port = _pick_unused_ports(2)
  coordinator = f'localhost:{coordinator_port}'
  children = []
  for process_id in range(config.num_processes):
    env = dict(
        os.environ,
        SIMDEVICE_PROCESS_ID=str(process_id),
        SIMDEVICE_BARRIER_ADDRESS=f'localhost:{barrier_port}',
        SIMDEVICE_BARRIER_PORT=str(barrier_port),
        JAX_COORDINATOR_ADDRESS=coordinator,
        JAX_PROCESS_ID=str(process_id),
        JAX_NUM_PROCESSES=str(config.num_processes),
    )
    children.append(subprocess.Popen(command, env=env))
  logging.info(
      'Started %d simulated hosts (coordinator %s, barrier port %d).',
      config.num_processes,
      coordinator,
      barrier_port,
  )

  try:
    while True:
      exit_codes = [child.poll() for child in children]
      failures = {
          process_id: code
          for process_id, code in enumerate(exit_codes)
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
