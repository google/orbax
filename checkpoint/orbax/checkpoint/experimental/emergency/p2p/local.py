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
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import utils


@final
class LocalCheckpointManager:
  """Wrapper around Orbax CheckpointManager for local P2P shards."""

  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      *,
      options: emergency_checkpoint_manager.CheckpointManagerOptions,
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
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        should_save_fn=options.local.should_save_fn,
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

  def scan_stored_steps(self) -> tuple[int | None, Sequence[int]]:
    """Identifies available steps and the stored process index (from latest)."""
    if not self._directory.exists():
      logging.warning('Local scan: Directory not found: %s.', self._directory)
      return None, []

    steps = self._manager.all_steps()
    if not steps:
      logging.warning('Local scan: No steps found in %s.', self._directory)
      return None, []

    latest = steps[-1]
    detected_index = utils.detect_process_index(self._directory, latest)

    if detected_index is None:
      raise ValueError(
          f'Failed to detect process index for step {latest} in'
          f' {self._directory}. Checkpoint may be malformed.'
      )
    logging.info(
        'Local scan: Found steps=%s in %s (owner=%d).',
        steps,
        self._directory,
        detected_index,
    )
    return detected_index, steps

  def save(
      self,
      step: int,
      args: p2p_args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    """Saves the checkpoint."""
    return self._manager.save(step, args=args, force=force)

  def restore(
      self,
      step: int,
      args: p2p_args_lib.Composite,
      *,
      directory: epath.PathLike | None = None,
  ) -> p2p_args_lib.Composite:
    """Restores the checkpoint, enforcing process identity check."""
    # No need to check for P2P restore directory
    if directory is None:
      # 1. Fast Fail: Verify Process Identity
      stored_index = utils.detect_process_index(self._directory, step)

      if stored_index != self._process_index:
        error_msg = (
            f'Process Mismatch: Local checkpoint at step {step} belongs to'
            f' Process {stored_index}, but current process is'
            f' {self._process_index}. Aborting local restore to trigger'
            ' P2P/Persistent fallback.'
        )
        raise ValueError(error_msg)

    # 2. Delegate to Orbax
    restored = self._manager.restore(
        step,
        args=args,
        directory=directory,
    )
    return restored

  def __getattr__(self, name: str) -> Any:
    return getattr(self._manager, name)

  def close(self):
    self._manager.close()
