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

"""Reads where this task sits in a multi-task launch from its environment.

Each platform tells a task who it is in its own way. `resolve` reads the
environment variables of the platforms the benchmarks run on and returns one
`LaunchEnv`, or None for a single-task launch (one machine):

1. Explicit variables, set by XManager and the GCE VM launcher:
   `SIMDEVICE_TASK_ID`, `SIMDEVICE_NUM_TASKS`, `JAX_COORDINATOR_ADDRESS`,
   `SIMDEVICE_BARRIER_ADDRESS` and optionally `SIMDEVICE_BARRIER_PORT`.
2. JobSet variables, set on GKE CPU pods started by XPK: `JOBSET_NAME`,
   `REPLICATED_JOB_NAME`, `JOB_INDEX`, `JOB_COMPLETION_INDEX`,
   `PROCESSES_IN_JOB` and `JAX_PROCESS_COUNT`, all of them. Task 0 listens on
   the default ports, or on `JAX_COORDINATOR_PORT` for the JAX coordinator.
"""

from collections.abc import Mapping
import dataclasses
import os

TASK_ID_ENV = 'SIMDEVICE_TASK_ID'
NUM_TASKS_ENV = 'SIMDEVICE_NUM_TASKS'
COORDINATOR_ADDRESS_ENV = 'JAX_COORDINATOR_ADDRESS'
BARRIER_ADDRESS_ENV = 'SIMDEVICE_BARRIER_ADDRESS'
BARRIER_PORT_ENV = 'SIMDEVICE_BARRIER_PORT'

_JOBSET_ENVS = (
    'JOBSET_NAME',
    'REPLICATED_JOB_NAME',
    'JOB_INDEX',
    'JOB_COMPLETION_INDEX',
    'PROCESSES_IN_JOB',
    'JAX_PROCESS_COUNT',
)
_COORDINATOR_PORT_ENV = 'JAX_COORDINATOR_PORT'
DEFAULT_COORDINATOR_PORT = 1234
DEFAULT_BARRIER_PORT = 50051


@dataclasses.dataclass(frozen=True)
class LaunchEnv:
  """Where this task sits in a multi-task launch.

  Attributes:
    source: The platform the values came from, for logging.
    task_id: This task's index, from 0.
    num_tasks: The number of tasks in the launch.
    coordinator_address: The JAX coordinator address, on task 0.
    barrier_address: The SimDevice barrier address, on task 0.
    barrier_port: The port the SimDevice barrier listens on, if the launcher
      chose one.
  """

  source: str
  task_id: int
  num_tasks: int
  coordinator_address: str
  barrier_address: str
  barrier_port: int | None = None

  def __post_init__(self):
    if self.num_tasks < 1 or not 0 <= self.task_id < self.num_tasks:
      raise ValueError(
          f'Invalid task {self.task_id} of {self.num_tasks} from'
          f' {self.source} environment.'
      )


def _from_explicit(environ: Mapping[str, str]) -> LaunchEnv:
  required = (NUM_TASKS_ENV, COORDINATOR_ADDRESS_ENV, BARRIER_ADDRESS_ENV)
  missing = [name for name in required if name not in environ]
  if missing:
    raise ValueError(f'{TASK_ID_ENV} is set, but {missing} are not.')
  barrier_port = environ.get(BARRIER_PORT_ENV)
  return LaunchEnv(
      source='explicit',
      task_id=int(environ[TASK_ID_ENV]),
      num_tasks=int(environ[NUM_TASKS_ENV]),
      coordinator_address=environ[COORDINATOR_ADDRESS_ENV],
      barrier_address=environ[BARRIER_ADDRESS_ENV],
      barrier_port=int(barrier_port) if barrier_port else None,
  )


def _from_jobset(environ: Mapping[str, str]) -> LaunchEnv:
  """Reads the task from the JobSet variables XPK sets on CPU pods.

  Args:
    environ: The environment, with all of `_JOBSET_ENVS` set.

  Returns:
    The task, with task 0's coordinator and barrier at pod 0 of the JobSet.
  """
  task_id = int(environ['JOB_INDEX']) * int(
      environ['PROCESSES_IN_JOB']
  ) + int(environ['JOB_COMPLETION_INDEX'])
  if task_id == 0:
    # Task 0 talks to itself.
    task0_host = 'localhost'
  else:
    # The first pod of the first replicated job, through the JobSet's headless
    # service.
    jobset = environ['JOBSET_NAME']
    task0_host = f'{jobset}-{environ["REPLICATED_JOB_NAME"]}-0-0.{jobset}'
  coordinator_port = environ.get(
      _COORDINATOR_PORT_ENV, str(DEFAULT_COORDINATOR_PORT)
  )
  return LaunchEnv(
      source='JobSet',
      task_id=task_id,
      num_tasks=int(environ['JAX_PROCESS_COUNT']),
      coordinator_address=f'{task0_host}:{coordinator_port}',
      barrier_address=f'{task0_host}:{DEFAULT_BARRIER_PORT}',
      barrier_port=DEFAULT_BARRIER_PORT,
  )


def resolve(environ: Mapping[str, str] | None = None) -> LaunchEnv | None:
  """Returns this task's place in the launch, or None for a single task.

  Args:
    environ: The environment to read; `os.environ` by default.

  Raises:
    ValueError: If `SIMDEVICE_TASK_ID` is set without the other explicit
      variables, or the task id is out of range.
  """
  if environ is None:
    environ = os.environ
  if TASK_ID_ENV in environ:
    return _from_explicit(environ)
  # JobSet sets some of these on every pod; XPK adds the rest on CPU pods.
  if all(name in environ for name in _JOBSET_ENVS):
    return _from_jobset(environ)
  return None
