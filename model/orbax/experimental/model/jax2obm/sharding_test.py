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

from absl.testing import absltest
import jax
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import sharding


class ShardingTest(absltest.TestCase):
  """Tests for sharding."""

  def test_jax_named_sharding_to_op_sharding_with_none(self):
    self.assertIsNone(sharding.jax_named_sharding_to_op_sharding(None, 2))

  def test_jax_named_sharding_to_op_sharding(self):
    if jax.device_count() < 8:
      raise ValueError('Test requires 8 devices.')
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('x', 'y')
    )
    spec = jax.sharding.PartitionSpec('x', 'y')
    named_sharding = jax.sharding.NamedSharding(mesh, spec)
    op_sharding = sharding.jax_named_sharding_to_op_sharding(named_sharding, 2)

    self.assertIsInstance(op_sharding, obm.OpSharding)

    expected_op_sharding = obm.OpSharding(
        type=obm.OpSharding.Type.OTHER,
        tile_assignment_dimensions=[2, 4],
        iota_reshape_dims=[8],
        iota_transpose_perm=[0],
    )

    self.assertEqual(
        op_sharding,
        expected_op_sharding,
    )


if __name__ == '__main__':
  absltest.main()
