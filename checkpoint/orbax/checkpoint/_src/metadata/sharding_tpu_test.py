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

import json
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata


class ShardingTpuTest(parameterized.TestCase):

  def test_named_sharding(self):
    # Convert from `jax.sharding.NamedSharding` to `NamedShardingMetadata`
    jax_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)),
        jax.sharding.PartitionSpec(None),
    )
    metadata = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        jax_sharding
    )
    metadata_json = metadata.to_serialized_string()

    # restored it back to jax.sharding.NamedSharding
    restored_sharding_metadata = (
        sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
            json.loads(metadata_json)
        )
    )
    restored_sharding = restored_sharding_metadata.to_jax_sharding()

    self.assertIsInstance(restored_sharding, jax.sharding.NamedSharding)
    self.assertEqual(restored_sharding, jax_sharding)

    # test some more complicated mesh
    mesh2 = np.asarray(
        (jax.devices()[2], jax.devices()[3], jax.devices()[5], jax.devices()[7])
    ).reshape(1, 2, 2)
    jax_sharding2 = jax.sharding.NamedSharding(
        jax.sharding.Mesh(mesh2, ("x", "y", "z")),
        jax.sharding.PartitionSpec("x", "y"),
    )
    metadata2 = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        jax_sharding2
    )
    metadata_json2 = metadata2.to_serialized_string()

    # restored it back to jax.sharding.NamedSharding
    restored_sharding_metadata2 = (
        sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
            json.loads(metadata_json2)
        )
    )
    restored_sharding2 = restored_sharding_metadata2.to_jax_sharding()

    self.assertIsInstance(restored_sharding2, jax.sharding.NamedSharding)
    self.assertEqual(restored_sharding2, jax_sharding2)
    self.assertNotEqual(restored_sharding2, jax_sharding)

  def test_backward_compatibility(self):
    """Test jax.sharding can still restore from metadata that misses 'device_mesh'."""
    jax_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)),
        jax.sharding.PartitionSpec(None),
    )
    new_metadata = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        jax_sharding
    )
    new_json_str = new_metadata.to_serialized_string()
    new_json_object = json.loads(new_json_str)

    old_json_object = new_json_object.copy()
    del old_json_object["device_mesh"]
    old_metadata = (
        sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
            old_json_object
        )
    )
    old_sharding = old_metadata.to_jax_sharding()

    self.assertEqual(jax_sharding, old_sharding)

  @parameterized.parameters(
      ((8,), ("x",), None),
      ((8,), ("x",), ("x",)),
      (
          (4, 2),
          ("x", "y"),
          None,
      ),
      ((4, 2), ("x", "y"), ("x",)),
      ((4, 2), ("x", "y"), ("y",)),
      ((4, 2), ("x", "y"), ("x", "y")),
      ((2, 2, 2), ("x", "y", "z"), None),
      ((2, 2, 2), ("x", "y", "z"), ("x",)),
      ((2, 2, 2), ("x", "y", "z"), ("y",)),
      ((2, 2, 2), ("x", "y", "z"), ("z",)),
      ((2, 2, 2), ("x", "y", "z"), ("x", "y")),
      ((2, 2, 2), ("x", "y", "z"), ("x", "y", "z")),
      ((1, 2, 4), ("x", "y", "z"), ("x", "y", "z")),
  )
  def test_named_shardings(self, shape, axis, psec):
    # Convert from `jax.sharding.NamedSharding` to `NamedShardingMetadata`
    jax_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(jax.devices()).reshape(shape), axis),
        jax.sharding.PartitionSpec(psec),
    )
    metadata = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        jax_sharding
    )
    metadata_json = metadata.to_serialized_string()

    # restored it back to jax.sharding.NamedSharding
    restored_sharding_metadata = (
        sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
            json.loads(metadata_json)
        )
    )
    restored_sharding = restored_sharding_metadata.to_jax_sharding()

    self.assertIsInstance(restored_sharding, jax.sharding.NamedSharding)
    self.assertEqual(restored_sharding, jax_sharding)


if __name__ == "__main__":
  absltest.main()
