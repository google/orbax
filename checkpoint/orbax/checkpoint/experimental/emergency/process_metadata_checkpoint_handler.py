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

"""ProcessMetadataCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import List, Optional, Tuple

from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency import mesh_consistency

CheckpointArgs = checkpoint_args.CheckpointArgs
PreviousDistributedToDeviceIds = List[List[int]]
PreviousDeviceIds = List[int]
register_with_handler = checkpoint_args.register_with_handler


class ProcessMetadataCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Saves processmetadata."""

  def __init__(
      self,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      use_colocated_python: bool = False,
  ):
    """Initializes ProcessMetadataCheckpointHandler."""
    self._multiprocessing_options = multiprocessing_options
    self._use_colocated_python = use_colocated_python

  async def async_save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ) -> Optional[List[future.Future]]:
    """Saves the given process metadata."""
    if self._use_colocated_python:
      dir_str = str(directory)
      device_ids = [int(d.id) for d in args.global_mesh.devices.flatten()]

      # Calculate distributed_to_device_ids natively for Single-Controller
      devices = jax.devices()
      dist_to_dev = [
          sorted(d.id for d in devices if d.process_index == pid)
          for pid in range(jax.process_count())
      ]

      # Setup SPMD dummy array across all remote hosts
      cpu_devices = colocated_python.colocated_cpu_devices(devices)
      unique_cpu_devices = list(
          {dev.process_index: dev for dev in cpu_devices}.values()
      )
      cpu_mesh = jax.sharding.Mesh(np.array(unique_cpu_devices), ('d',))
      replicated_sharding = jax.sharding.NamedSharding(
          cpu_mesh, jax.sharding.PartitionSpec()
      )
      dummy_in = jax.device_put(
          jnp.array(True, dtype=jnp.bool), replicated_sharding
      )

      def _save_fn(dummy_arg):
        mesh_consistency.write_process_metadata(
            epath.Path(dir_str),
            device_ids,
            dist_to_dev,
        )

        # Return dummy_arg to satisfy strict JAX SPMD output device matching
        return dummy_arg

      # Wrap and specialize for native SPMD async execution
      wrapped = colocated_python.colocated_python(_save_fn)
      wrapped = wrapped.specialize(out_specs_fn=lambda x: x)

      async def _do_save_coro():
        def _run_all():
          # JAX's C++ backend natively dispatches this concurrently to all hosts
          jax.block_until_ready(wrapped(dummy_in))

        # Run the blocking JAX execution in the asyncio executor
        # to avoid blocking the main event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_all)

      return [
          future.CommitFutureAwaitingContractedSignals(
              _do_save_coro(),
              name='process_metadata_ch_save',
          )
      ]
    else:
      return [
          future.CommitFutureAwaitingContractedSignals(
              mesh_consistency.save_process_metadata(
                  directory,
                  args.global_mesh,
                  multihost.distributed_to_device_ids(),
              ),
              name='process_metadata_ch_save',
          )
      ]

  def save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ):

    async def _internal_async_save(directory, args):
      commit_futures = await self.async_save(directory, args)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio_utils.run_sync(_internal_async_save(directory, args))

  def restore(
      self,
      directory: epath.Path,
      args: ProcessMetadataRestoreArgs,
  ) -> Tuple[PreviousDistributedToDeviceIds, PreviousDeviceIds]:
    """Restores metadata from directory."""
    if getattr(self, '_use_colocated_python', False):
      dir_str = str(directory)

      def _read_fn(dummy_in):
        res1, res2 = mesh_consistency.read_process_metadata(epath.Path(dir_str))

        payload = json.dumps([res1, res2]).encode('utf-8')
        arr = jnp.array(list(payload), dtype=jnp.uint8)

        # Explicitly put the array on the exact same device as the dummy input
        return jax.device_put(arr, dummy_in.sharding)

      # In Single-Controller, pick the first remote worker to read the file
      cpu_devices = colocated_python.colocated_cpu_devices(jax.devices())
      my_cpu_dev = cpu_devices[0]

      dummy_in = jax.device_put(jnp.zeros((), dtype=jnp.int32), my_cpu_dev)
      wrapped = colocated_python.colocated_python(_read_fn)

      # Since we provide an input array, JAX can infer execution placement
      arr = wrapped(dummy_in)

      payload_bytes = np.asarray(arr).tobytes()
      res1, res2 = json.loads(payload_bytes.decode('utf-8'))
      return res1, res2

    else:
      return mesh_consistency.read_process_metadata(directory)


@register_with_handler(ProcessMetadataCheckpointHandler, for_save=True)
@dataclasses.dataclass
class ProcessMetadataSaveArgs(CheckpointArgs):
  """Parameters for saving process metadata.

  Attributes:
    global_mesh: the global mesh.
  """

  global_mesh: jax.sharding.Mesh


@register_with_handler(ProcessMetadataCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class ProcessMetadataRestoreArgs(CheckpointArgs):
  """Parameters for restoring process metadata."""
