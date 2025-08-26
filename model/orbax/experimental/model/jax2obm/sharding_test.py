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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import sharding


class ShardingTest(absltest.TestCase):

  def test_hlo_sharding_to_op_sharding_with_none(self):
    self.assertIsNone(sharding.hlo_sharding_to_op_sharding(None))

  def test_hlo_sharding_to_op_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('Test requires at least 4 devices.')
    mesh = jax.sharding.Mesh(
        devices=jnp.asarray(jax.devices()).reshape(2, 2),
        axis_names=('x', 'y'),
    )
    # From jax.sharding.NamedSharding._to_xla_hlo_sharding
    hlo_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x', 'y')
    )._to_xla_hlo_sharding(num_dimensions=2)

    op_sharding = sharding.hlo_sharding_to_op_sharding(hlo_sharding)

    expected_op_sharding = obm.OpSharding()
    expected_op_sharding.ParseFromString(
        hlo_sharding.to_proto().SerializeToString()
    )
    self.assertEqual(op_sharding, expected_op_sharding)


if __name__ == '__main__':
  absltest.main()
