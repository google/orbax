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

"""A class providing emergency checkpoint management.


This class is experimental; do not use without specific approval.
"""

import dataclasses
from typing import Any, Sequence
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import utils
from orbax.checkpoint.path import step as step_lib


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler


@dataclasses.dataclass
class LocalCheckpointOptions:
  """Optional CheckpointManager arguments for saving local checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to local storage.
    Ensures checkpoints will only be saved every m steps. Defaults to 10.
  max_to_keep:
    If provided, specifies the maximum number of local checkpoints to
    keep. Older checkpoints are removed. When set, no more than `max_to_keep`
    checkpoints will be present at any one time. This option has a slightly
    different meaning than it normally does in Orbax: this should be treated
    as a hard cap on the number of checkpoints concurrently present, rather
    than a threshold beyond which checkpoints start to be deleted.
  """

  save_interval_steps: int = 10
  max_to_keep: int | None = 2


@dataclasses.dataclass
class PersistentCheckpointOptions:
  """Optional CheckpointManager arguments for saving persistent checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to persistent storage.
    Ensures checkpoints will only be saved every n steps. Defaults to 1000.
  max_to_keep:
    If provided, specifies the maximum number of persistent checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present.
  """

  save_interval_steps: int = 1000
  max_to_keep: int | None = None


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  local:
    Options relevant to the local checkpoints.
    See `LocalCheckpointOptions`.
  persistent:
    Options relevant to the persistent checkpoints. See
    `PersistentCheckpointOptions`.
  step_name_format:
    NameFormat to build or find steps under input root directory. If provided,
    `step_prefix`, `step_format_fixed_length` are ignored.
  create:
    If True, creates the top-level directory if it does not already exist.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  async_options: Used to configure properties of async behavior. See above.
  """

  local: LocalCheckpointOptions = dataclasses.field(
      default_factory=LocalCheckpointOptions
  )
  persistent: PersistentCheckpointOptions = dataclasses.field(
      default_factory=PersistentCheckpointOptions
  )

  step_name_format: step_lib.NameFormat | None = None
  create: bool = True
  cleanup_tmp_directories: bool = False
  async_options: checkpoint_manager.AsyncOptions | None = None


class LocalCheckpointManager(checkpoint_manager.CheckpointManager):
  """A checkpoint manager that checkpoints to local storage."""

  # TODO(b/330585086) Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  def __init__(
      self,
      directory: epath.PathLike,
      # TODO(b/330585086) Support arbitrary items beyond state. We will have
      # to evaluate whether arbitrary items can be a good fit for local
      # checkpointing, given restore+broadcast requirements.
      state_handler: CheckpointHandler,
      *,
      options: CheckpointManagerOptions | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    # TODO(b/330585086): Fully support options.
    options = options or CheckpointManagerOptions()
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        create=options.create,
        cleanup_tmp_directories=options.cleanup_tmp_directories,
        async_options=options.async_options,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            primary_host=None
        ),
    )

    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=state_handler,
    )

  def _is_equal_on_all_hosts(self, value: int | float) -> bool:
    """return true if all `values` are equal on all hosts."""

    global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        ['x', 'y'],
    )
    pspecs = jax.sharding.PartitionSpec('x', None)

    arr = multihost_utils.host_local_array_to_global_array(
        np.array([value]),
        global_mesh,
        pspecs,
    )

    # calculate the global range, eg. (max - min)
    @jax.jit
    def global_ptp(x):
      return jnp.ptp(x)

    ptp = global_ptp(arr)
    return ptp.addressable_data(0) == 0

  def latest_step(self) -> int | None:
    """Returns the latest step saved in the local storage.

    TODO(b/330585086) Currently only returns latest step if all hosts have the
    same step, otherwise, None.Still needs to identify the latest step if not
    all hosts have the same step.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    local_latest = super().latest_step() or -1

    if self._is_equal_on_all_hosts(local_latest):
      return local_latest if local_latest != -1 else None
    else:
      return None


class CheckpointManager(abstract_checkpoint_manager.AbstractCheckpointManager):
  """A checkpoint manager that checkpoints to local and/or persistent storage."""

  # TODO(b/330585086) Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      # TODO(b/330585086) Support arbitrary items beyond state. We will have
      # to evaluate whether arbitrary items can be a good fit for local
      # checkpointing, given restore+broadcast requirements.
      local_state_handler: CheckpointHandler,
      persistent_state_handler: CheckpointHandler,
      *,
      options: CheckpointManagerOptions | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    # TODO(b/330585086): Fully support options.
    options = options or CheckpointManagerOptions()

    self._local_checkpoint_manager = LocalCheckpointManager(
        local_directory,
        local_state_handler,
        options=options,
        metadata=metadata,
    )
    # TODO(b/330585086): Build options for persistent CheckpointManager.
    persistent_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.persistent.save_interval_steps,
        max_to_keep=options.persistent.max_to_keep,
        step_name_format=options.step_name_format,
        create=options.create,
        cleanup_tmp_directories=options.cleanup_tmp_directories,
        async_options=options.async_options,
    )
    self._persistent_checkpoint_manager = checkpoint_manager.CheckpointManager(
        persistent_directory,
        options=persistent_options,
        metadata=metadata,
        item_handlers=persistent_state_handler,
        # TODO(b/330585086): Use the appropriate primary_host.
    )

  @property
  def directory(self) -> epath.Path:
    raise NotImplementedError()

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    # TODO(b/330585086) Implement.
    raise NotImplementedError('Implement: b/330585086.')

  def latest_step(self) -> int | None:
    """Returns the latest step saved.

    Includes steps located in local as well as persistent storage.

    TODO(b/330585086) Currently only returns latest step returned by the local
    checkpoint manager. Still needs to fall back to persistent storage if no
    latest step can be identified in local storage.

    Returns:
      A step (int) or None if no steps are present.
    """
    return self._local_checkpoint_manager.latest_step()

  def best_step(self) -> int | None:
    """Returns the best step saved, as defined by `options.best_fn`.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    raise NotImplementedError(
        'Metrics tracking not yet implemented for emergency.CheckpointManager.'
    )

  def reload(self):
    """Performs disk reads to ensure internal properties are up to date."""
    self._local_checkpoint_manager.reload()
    self._persistent_checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return utils.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    return self._local_checkpoint_manager.should_save(
        step
    ) or self._persistent_checkpoint_manager.should_save(step)

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: args_lib.CheckpointArgs | None = None,
      metrics: PyTree | None = None,
      force: bool | None = False,
  ) -> bool:
    return self._local_checkpoint_manager.save(
        step, args=args, metrics=metrics, force=force
    )

  def restore(
      self,
      step: int,
      args: args_lib.CheckpointArgs | None = None,
      directory: epath.PathLike | None = None,
  ) -> Any | args_lib.Composite:
    return self._local_checkpoint_manager.restore(
        step, args=args, directory=directory
    )

  def item_metadata(self, step: int) -> Any | args_lib.Composite:
    raise NotImplementedError(
        'Item metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metadata(self) -> dict[str, Any]:
    """Returns CheckpointManager level metadata if present, empty otherwise."""
    raise NotImplementedError(
        'Metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metrics(self, step: int) -> PyTree | None:
    """Returns metrics for step, if present."""
    raise NotImplementedError(
        'Metrics not yet implemented for emergency.CheckpointManager.'
    )

  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """
    self._local_checkpoint_manager.wait_until_finished()
    self._persistent_checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    self._local_checkpoint_manager.check_for_errors()
    self._persistent_checkpoint_manager.check_for_errors()

  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
    self._local_checkpoint_manager.close()
    self._persistent_checkpoint_manager.close()
