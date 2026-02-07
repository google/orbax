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

  def test_hlo_sharding_to_op_sharding_with_none(self):
    self.assertIsNone(sharding.hlo_sharding_to_op_sharding(None))

  def test_hlo_sharding_to_op_sharding(self):
    if jax.device_count() < 8:
      raise ValueError('Test requires 8 devices.')
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('x', 'y')
    )
    spec = jax.sharding.PartitionSpec('x', 'y')
    named_sharding = jax.sharding.NamedSharding(mesh, spec)
    hlo_sharding = named_sharding._to_xla_hlo_sharding(2)

    op_sharding = sharding.hlo_sharding_to_op_sharding(hlo_sharding)

    self.assertIsInstance(op_sharding, obm.OpSharding)
    self.assertEqual(
        op_sharding.SerializeToString(),
        hlo_sharding.to_proto().SerializeToString(),
    )


if __name__ == '__main__':
  absltest.main()
