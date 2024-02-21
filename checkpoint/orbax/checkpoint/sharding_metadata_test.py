# Copyright 2024 The Orbax Authors.
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
from orbax.checkpoint import sharding_metadata
import pytest


class TestShardingMetadata(absltest.TestCase):

  def test_convert_between_jax_named_sharding_and_sharding_metadata(self):
    # Convert from `jax.sharding.NamedSharding` to `NamedShardingMetadata`
    jax_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)),
        jax.sharding.PartitionSpec(None),
    )
    expected_named_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([1]), axis_names=(["x"]), partition_spec=(None,)
    )
    converted_named_sharding_metadata = sharding_metadata.from_jax_sharding(
        jax_sharding
    )

    self.assertIsInstance(
        converted_named_sharding_metadata,
        sharding_metadata.NamedShardingMetadata,
    )
    self.assertEqual(
        converted_named_sharding_metadata, expected_named_sharding_metadata
    )

    # Convert from `NamedShardingMetadata` to `jax.sharding.NamedSharding`
    converted_jax_sharding = converted_named_sharding_metadata.to_jax_sharding()
    self.assertIsInstance(converted_jax_sharding, jax.sharding.NamedSharding)
    self.assertEqual(converted_jax_sharding, jax_sharding)

  def test_convert_between_jax_single_device_sharding_and_sharding_metadata(
      self,
  ):
    # Convert from `jax.sharding.SingleDeviceSharding` to
    # `SingleDeviceShardingMetadata`
    jax_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(backend="cpu")[0]
    )
    expected_single_device_sharding_metadata = (
        sharding_metadata.SingleDeviceShardingMetadata(device_str="TFRT_CPU_0")
    )
    converted_single_device_sharding_metadata = (
        sharding_metadata.from_jax_sharding(jax_sharding)
    )

    self.assertIsInstance(
        converted_single_device_sharding_metadata,
        sharding_metadata.SingleDeviceShardingMetadata,
    )
    self.assertEqual(
        converted_single_device_sharding_metadata,
        expected_single_device_sharding_metadata,
    )

    # Convert from `SingleDeviceShardingMetadata` to
    # `jax.sharding.SingleDeviceSharding`
    converted_jax_sharding = (
        converted_single_device_sharding_metadata.to_jax_sharding()
    )
    self.assertIsInstance(
        converted_jax_sharding, jax.sharding.SingleDeviceSharding
    )
    self.assertEqual(converted_jax_sharding, jax_sharding)

  def test_convert_to_jax_sharding_unsupported_types(self):
    jax_sharding = jax.sharding.PositionalSharding(jax.devices())
    with pytest.raises(NotImplementedError):
      sharding_metadata.from_jax_sharding(jax_sharding)

if __name__ == "__main__":
  absltest.main()
