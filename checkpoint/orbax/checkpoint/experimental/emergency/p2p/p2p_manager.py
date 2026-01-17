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

"""Internal manager implementation for peer-to-peer redistribution logic."""

from typing import Optional

from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.logging import abstract_logger
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emc


class _P2PCheckpointManager(checkpoint_manager.CheckpointManager):
  """Internal manager for P2P redistribution.

  This class is not intended to be instantiated directly. It manages
  local-only storage and redistribution logic, relying on the composite
  p2p.CheckpointManager to handle persistent failover.
  """

  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      replica_id: int,
      *,
      options: Optional[emc.CheckpointManagerOptions] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    options = options or emc.CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._replica_axis_index = options.replica_axis_index

    # Identify devices and processes for this specific replica
    replica_devices = multislice.replica_devices(
        self._global_mesh,
        replica_id=replica_id,
        replica_axis_index=self._replica_axis_index,
    )
    local_replica_processes = multihost.unique_processes_from_devices(
        replica_devices
    )

    # Configure multiprocessing to isolate this replica.
    # We set primary_host=None as P2P typically implies decentralized handling.
    barrier_sync_key_prefix = f'p2p-{replica_id}'
    multiprocessing_options = ocp.options.MultiprocessingOptions(
        primary_host=None,
        active_processes=local_replica_processes,
        barrier_sync_key_prefix=barrier_sync_key_prefix,
    )

    manager_options = ocp.CheckpointManagerOptions(
        step_name_format=options.step_name_format,
        multiprocessing_options=multiprocessing_options,
        create=False,
        # Always clean up local tmp directories explicitly
        cleanup_tmp_directories=False,
        single_host_load_and_broadcast=False,
        enable_background_delete=False,
        save_root_metadata=False,
        enable_per_process_directory_creation=True,
    )

    # Create a registry that handles Arrays without replica parallelism
    local_registry = type_handler_registry.create_type_handler_registry((
        jax.Array,
        type_handlers.ArrayHandler(
            primary_host=None, replica_id=None, use_replica_parallel=False
        ),
    ))

    handler = ocp.PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=True,
        multiprocessing_options=multiprocessing_options,
        type_handler_registry=local_registry,
    )

    super().__init__(
        directory,
        options=manager_options,
        item_handlers=handler,
        logger=logger,
    )
