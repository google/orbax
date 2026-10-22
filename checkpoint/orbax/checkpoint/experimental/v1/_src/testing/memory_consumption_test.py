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

"""Tests for memory consumption during checkpoint saving in V1."""

import tracemalloc
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp


class MemoryConsumptionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='memory_consumption_test').full_path
    )

  def test_deepcopy_host_arrays_default(self):
    ctx = ocp.Context()
    self.assertTrue(ctx.memory.deepcopy_host_arrays)

  def test_deepcopy_host_arrays_memory_consumption(self):
    # 20MB array: 5,000,000 float32 elements * 4 bytes = 20 MB.
    value = np.ones((5000, 1000), dtype=np.float32)
    pytree = {'arr': value}

    def _measure_memory(deepcopy_host_arrays: bool) -> int:
      path = self.directory / f'test_deepcopy_{deepcopy_host_arrays}'
      ctx = ocp.Context()
      ctx.memory.deepcopy_host_arrays = deepcopy_host_arrays

      with ctx:
        tracemalloc.start()
        response = ocp.save_async(path, pytree)  # pyrefly: ignore[bad-argument-type]
        _, peak = tracemalloc.get_traced_memory()
        response.result()
        tracemalloc.stop()
        return peak

    peak_true = _measure_memory(deepcopy_host_arrays=True)
    peak_false = _measure_memory(deepcopy_host_arrays=False)

    self.assertGreater(peak_true, peak_false)
    self.assertGreaterEqual(peak_true - peak_false, int(value.nbytes * 0.9))

  def test_copy_device_arrays_for_async_transfer_default(self):
    ctx = ocp.Context()
    self.assertFalse(ctx.memory.copy_device_arrays_for_async_transfer)

  @parameterized.named_parameters(
      ('unbatched', None),
      ('batched', 1024),
  )
  def test_copy_device_arrays_for_async_transfer(
      self, transfer_concurrent_bytes
  ):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    values = {
        'a': np.arange(16 * mesh.size, dtype=np.float32),
        'b': np.arange(16 * mesh.size, dtype=np.float32) * 10,
    }
    pytree = jax.tree.map(lambda v: jax.device_put(v, sharding), values)
    path = self.directory / f'copy_device_arrays_{transfer_concurrent_bytes}'

    ctx = ocp.Context()
    ctx.memory.copy_device_arrays_for_async_transfer = True
    ctx.memory.transfer_concurrent_bytes = transfer_concurrent_bytes
    with ctx, mock.patch.object(jnp, 'copy', wraps=jnp.copy) as mock_copy:
      response = ocp.save_pytree_async(path, pytree)
      # Donating `pytree` reuses its device buffers while the background
      # transfer of the on-device copies may still be pending.
      pytree = jax.jit(
          lambda t: jax.tree.map(lambda x: x + 100, t), donate_argnums=0
      )(pytree)
      response.result()

    # Every leaf is copied on device, and the checkpoint holds the values from
    # before the donation.
    self.assertEqual(mock_copy.call_count, 2)
    restored = ocp.load_pytree(path)
    for key, value in values.items():
      np.testing.assert_array_equal(restored[key], value)


if __name__ == '__main__':
  absltest.main()
