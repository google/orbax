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

"""Internal checkpoint manager for local P2P storage logic."""

from typing import Any, Sequence, final

from absl import logging
from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import type_handlers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint.experimental.emergency.p2p import constants


@final
class LocalCheckpointManager:
  """Wrapper around Orbax CheckpointManager for local P2P shards."""

  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      *,
      options: checkpoint_manager.CheckpointManagerOptions,
  ):
    self._directory = epath.Path(directory)
    self._global_mesh = global_mesh
    self._process_index = multihost.process_index()

    barrier_sync_key_prefix = f'p2p_shard_{self._process_index}'
    mp_options = ocp.options.MultiprocessingOptions(
        primary_host=None,  # Symmetric read/write
        active_processes={self._process_index},  # Only I write to my shard
        barrier_sync_key_prefix=barrier_sync_key_prefix,
    )

    p2p_specific_options = checkpoint_manager.CheckpointManagerOptions(
        step_name_format=options.step_name_format,
        save_interval_steps=options.save_interval_steps,
        max_to_keep=options.max_to_keep,
        should_save_fn=options.should_save_fn,
        multiprocessing_options=mp_options,
        create=False,
        cleanup_tmp_directories=False,
        enable_background_delete=True,
        enable_per_process_directory_creation=True,
    )

    local_registry = type_handler_registry.create_type_handler_registry((
        jax.Array,
        type_handlers.ArrayHandler(
            primary_host=None, replica_id=None, use_replica_parallel=False
        ),
    ))

    handler = ocp.PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=True,
        multiprocessing_options=mp_options,
        type_handler_registry=local_registry,
    )

    self._manager = checkpoint_manager.CheckpointManager(
        self._directory,
        options=p2p_specific_options,
        item_handlers=dict(state=handler),
    )

  @property
  def directory(self) -> epath.Path:
    return self._directory

  def _detect_process_index(self, step: int) -> int | None:
    """Inspects the disk to find which process index created this step."""
    step_path = self._directory / str(step)
    if not step_path.exists():
      return None

    # Check for standard Orbax/OCDBT structure
    # P2P checkpoint requires 'state' item in CompositeArgs
    try:
      item_path = step_path / constants.STATE_SUBDIR
      if item_path.exists():
        for path in item_path.glob(f'{constants.PROCESS_SUBDIR_PREFIX}*'):
          if path.is_dir():
            # Format: ocdbt.process_0, ocdbt.process_12, etc.
            return int(path.name.split('_')[-1])
    except (ValueError, IndexError, OSError) as e:
      logging.warning('Could not detect process index for step %d: %s', step, e)

    return None

  def scan_stored_steps(self) -> tuple[int | None, Sequence[int]]:
    """Identifies available steps and the stored process index (from latest)."""
    if not self._directory.exists():
      return None, []

    steps = self._manager.all_steps()
    if not steps:
      return None, []

    latest = steps[-1]
    detected_index = self._detect_process_index(latest)

    return detected_index, steps

  def restore(
      self,
      step: int,
      *,
      directory: epath.PathLike | None = None,
  ) -> Any:
    """Restores the checkpoint, enforcing process identity check."""
    # No need to check for P2P restore directory
    if directory is None:
      # 1. Fast Fail: Verify Process Identity
      stored_index = self._detect_process_index(step)

      if stored_index is not None and stored_index != self._process_index:
        error_msg = (
            f'Process Mismatch: Local checkpoint at step {step} belongs to'
            f' Process {stored_index}, but current process is'
            f' {self._process_index}. Aborting local restore to trigger'
            ' P2P/Persistent fallback.'
        )
        logging.warning(error_msg)
        raise ValueError(error_msg)

    # 2. Delegate to Orbax
    return self._manager.restore(step, directory=directory)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._manager, name)

  def close(self):
    self._manager.close()
