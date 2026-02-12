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

"""Handles persistent storage logic (GCS/S3) for P2P checkpointing."""

from typing import Any, final

from absl import logging
from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import constants
from orbax.checkpoint.experimental.emergency.p2p import utils

_PRIMARY_REPLICA_ID = 0
PyTree = Any


def _create_persistent_handler(
    mp_options: checkpoint_manager.MultiprocessingOptions,
) -> ocp.PyTreeCheckpointHandler:
  """Creates a PyTreeCheckpointHandler for persistent storage.

  Args:
    mp_options: Multiprocessing options for the checkpoint handler.

  Returns:
    A PyTreeCheckpointHandler configured for persistent storage.
  """
  registry = type_handler_registry.create_type_handler_registry((
      jax.Array,
      type_handlers.ArrayHandler(
          primary_host=mp_options.primary_host,
          replica_id=_PRIMARY_REPLICA_ID,
          use_replica_parallel=False,
      ),
  ))
  return ocp.PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=mp_options,
      type_handler_registry=registry,
  )


@final
class PersistentCheckpointManager:
  """Manages saving/restoring from slow persistent storage."""

  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      *,
      replica_axis_index: int,
      options: emergency_checkpoint_manager.CheckpointManagerOptions,
  ):
    self._directory = epath.Path(directory)
    self._global_mesh = global_mesh
    self._replica_axis_index = replica_axis_index
    self._process_index = multihost.process_index()
    self._replica_id = multislice.process_replica_id(
        self._process_index,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    self._in_primary_slice = multislice.in_replica(
        self._process_index,
        global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=_PRIMARY_REPLICA_ID,
    )

    replica_devices = multislice.replica_devices(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=self._replica_id,
    )
    primary_host = multislice.primary_process_in_replica(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=self._replica_id,
    )
    active_processes = multihost.unique_processes_from_devices(replica_devices)
    mp_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=primary_host,
        active_processes=active_processes,
        barrier_sync_key_prefix=f'persistent_fallback_{self._replica_id}',
    )

    internal_options = checkpoint_manager.CheckpointManagerOptions(
        create=False,
        multiprocessing_options=mp_options,
        step_name_format=options.step_name_format,
        save_interval_steps=options.persistent.save_interval_steps,
        max_to_keep=options.persistent.max_to_keep,
        enable_async_checkpointing=True,
    )

    item_handlers = dict(state=_create_persistent_handler(mp_options))
    if utils.pygrain() is not None:
      item_handlers['data_iter'] = utils.pygrain().PyGrainCheckpointHandler()

    self._manager = checkpoint_manager.CheckpointManager(
        self._directory,
        options=internal_options,
        item_handlers=item_handlers,
    )

  @property
  def directory(self) -> epath.Path:
    return self._directory

  def latest_step(self) -> int | None:
    return self._manager.latest_step()

  def save(
      self,
      step: int,
      args: p2p_args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    if self._in_primary_slice:
      return self._manager.save(step, args=args, force=force)
    return True

  def restore(
      self,
      step: int,
      args: p2p_args_lib.Composite,
  ) -> p2p_args_lib.Composite:
    """Restores a checkpoint from persistent storage.

    Args:
      step: The step number to restore.
      args: A Composite object containing the abstract state to restore.

    Returns:
      The restored state as a Composite object.
    """
    assert self._manager is not None
    logging.info(
        'Restoring step %s from persistent storage on slice %d...',
        step,
        self._replica_id,
    )
    abstract_state = args.state

    sharding_tree = jax.tree.map(lambda x: x.sharding, abstract_state)
    # TODO(exlin): Enable SingleReplicaRestore.
    restore_args_obj = args_lib.PyTreeRestore(
        item=abstract_state,
        restore_args=checkpoint_utils.construct_restore_args(
            abstract_state, sharding_tree
        ),
    )
    restore_kwargs = {'state': restore_args_obj}
    if constants.DATA_ITER_KEY in args:
      restore_kwargs[constants.DATA_ITER_KEY] = args.data_iter
    return self._manager.restore(
        step, args=p2p_args_lib.Composite(**restore_kwargs)
    )

  def delete(self, step: int):
    if self._in_primary_slice:
      self._manager.delete(step)

  def wait_until_finished(self):
    self._manager.wait_until_finished()

  def check_for_errors(self):
    self._manager.check_for_errors()

  def close(self):
    self._manager.close()
