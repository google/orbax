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

"""Tests cancellation of Orbax V0 Checkpointer in a multiprocess setting."""

import os
import shutil
import time

import jax
from jax import sharding
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing import multiprocess_test


NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh
PartitionSpec = sharding.PartitionSpec


class CancelMultiprocessTest(multiprocess_test.MultiProcessTest):
  """Tests cancellation of Orbax V0 Checkpointer in a multiprocess setting."""

  def test_cancel_multiprocess_saves(self):
    path = '/tmp/orbax_cancel_multiprocess_test'
    if jax.process_index() == 0 and os.path.exists(path):
      shutil.rmtree(path)

    # Sync all

    # Create a properly sharded global jax array that is not "host local"
    mesh = Mesh(jax.devices(), ('x',))
    named_sharding = NamedSharding(
        mesh,
        PartitionSpec(
            'x',
        ),
    )
    shape = (10000, 10000)
    large_array = jax.device_put(
        jnp.ones(shape, dtype=jnp.float32), named_sharding
    )
    large_array.block_until_ready()

    options = ocp.CheckpointManagerOptions(enable_async_checkpointing=True)
    mngr = ocp.CheckpointManager(path, options=options)

    step = 1

    save_start = time.time()
    print(f'[Process {jax.process_index()}] Save called in main thread...')
    mngr.save(step, args=ocp.args.StandardSave({'array': large_array}))
    save_end = time.time()
    print(
        f'[Process {jax.process_index()}] Save returned. Duration:'
        f' {save_end - save_start:.2f} seconds.'
    )

    time.sleep(0.01)

    print(f'[Process {jax.process_index()}] Triggering cancellation...')
    if hasattr(mngr, '_checkpointer') and hasattr(
        mngr._checkpointer, 'cancel'  # pylint: disable=protected-access
    ):
      mngr._checkpointer.cancel()  # pylint: disable=protected-access

    wait_start = time.time()
    mngr.wait_until_finished()
    wait_end = time.time()
    print(
        f'[Process {jax.process_index()}] wait_until_finished took'
        f' {wait_end - wait_start:.2f} seconds.'
    )

    mngr.close()


if __name__ == '__main__':
  multiprocess_test.main()
