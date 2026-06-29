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

"""Deleter that dispatches to Pathways workers with Remote Python."""

from typing import Sequence
import jax
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.path import deleter as deleter_lib
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types


CheckpointDeleter = deleter_lib.CheckpointDeleter


class _PathwaysDeleter(deleter_lib.CheckpointDeleter):
  """Deleter that dispatches to Pathways workers with Remote Python."""

  def __init__(
      self,
      deleter: deleter_lib.StandardCheckpointDeleter,
      global_mesh: jax.sharding.Mesh | None,
  ):
    self._global_mesh = global_mesh or jax.sharding.Mesh(jax.devices(), 'x')
    self._deleter = deleter
    self._dispatcher = dispatchers.RemotePythonDispatcher()

  def delete(self, step: int) -> None:
    """Deletes a step.

    Args:
      step: The step to delete.
    """

    def _delete(
        input_arrays: jax.Array,
        step: int,
    ):
      del input_arrays
      self._deleter.delete(step)

    jax.block_until_ready(
        self._dispatcher.dispatch(
            _delete,
            input_arrays=dispatchers.get_dummy_input_array(
                self._global_mesh.devices.flatten().tolist(),
            ),
            func_kwargs={'step': step},
        )
    )

  def delete_steps(self, steps: Sequence[int]) -> None:
    """Deletes a sequence of steps.

    Args:
      steps: The steps to delete.
    """
    def _delete(
        input_arrays: jax.Array,
        steps: Sequence[int],
    ):
      del input_arrays
      self._deleter.delete_steps(steps)

    jax.block_until_ready(
        self._dispatcher.dispatch(
            _delete,
            input_arrays=dispatchers.get_dummy_input_array(
                self._global_mesh.devices.flatten().tolist(),
            ),
            func_kwargs={'steps': steps},
        )
    )

  def close(self) -> None:
    """Performs any cleanup before closing this deleter."""
    self._deleter.close()


def create_checkpoint_deleter(
    directory: path_types.Path,
    *,
    global_mesh: jax.sharding.Mesh | None = None,
    name_format: step_lib.NameFormat[step_lib.Metadata],
    todelete_subdir: str | None = None,
) -> CheckpointDeleter:
  return _PathwaysDeleter(
      deleter_lib.StandardCheckpointDeleter(
          directory,
          name_format=name_format,
          todelete_subdir=todelete_subdir,
          primary_host=None,
      ),
      global_mesh,
  )
