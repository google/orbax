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
from orbax.checkpoint.metadata import sharding as sharding_metadata


class TestShardingMetadata(absltest.TestCase):

  def test_convert_between_jax_named_sharding_and_sharding_metadata(self):
    # Convert from `jax.sharding.NamedSharding` to `NamedShardingMetadata`
    jax_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)),
        jax.sharding.PartitionSpec(None),
    )
    expected_named_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([1]),
        axis_names=(["x"]),
        partition_spec=(None,),
        device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
            jax_sharding.mesh
        ),
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

  def test_convert_between_jax_positional_sharding_and_sharding_metadata(
      self,
  ):
    # Convert from `jax.sharding.PositionalSharding` to
    # `PositionalShardingMetadata`
    jax_sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(
        [1, -1]
    )
    expected_positional_sharding_metadata = (
        sharding_metadata.PositionalShardingMetadata(
            jax_sharding.shape, jax_sharding.memory_kind
        )
    )
    converted_positional_sharding_metadata = (
        sharding_metadata.from_jax_sharding(jax_sharding)
    )
    self.assertIsInstance(
        converted_positional_sharding_metadata,
        sharding_metadata.PositionalShardingMetadata,
    )
    self.assertEqual(
        converted_positional_sharding_metadata,
        expected_positional_sharding_metadata,
    )

    # Convert from `PositionalShardingMetadata` to
    # `jax.sharding.PositionalSharding`
    converted_jax_sharding = (
        converted_positional_sharding_metadata.to_jax_sharding()
    )
    self.assertIsInstance(
        converted_jax_sharding, jax.sharding.PositionalSharding
    )
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

  def test_convert_between_named_sharding_string_to_named_sharding_metadata(
      self,
  ):
    # Convert from `NamedShardingMetadata` to `str`
    named_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([1]), axis_names=(["x"]), partition_spec=(None,)
    )
    expected_named_sharding_string = (
        '{"sharding_type": "NamedSharding", "shape": [1], "axis_names": ["x"],'
        ' "partition_spec": [null]}'
    )
    named_sharding_string = named_sharding_metadata.to_serialized_string()
    self.assertEqual(named_sharding_string, expected_named_sharding_string)

    # Convert from `str` to `NamedShardingMetadata`
    converted_named_sharding_metadata = (
        sharding_metadata.from_serialized_string(named_sharding_string)
    )
    self.assertIsInstance(
        converted_named_sharding_metadata,
        sharding_metadata.NamedShardingMetadata,
    )
    self.assertEqual(converted_named_sharding_metadata, named_sharding_metadata)

  def test_positional_sharding_string_to_metadata(
      self,
  ):
    # Convert from `PositionalShardingMetadata` to `str`
    positional_sharding_metadata = sharding_metadata.PositionalShardingMetadata(
        shape=np.array([1, 2])
    )
    expected_positional_sharding_string = (
        '{"sharding_type": "PositionalSharding", "shape": [1, 2]}'
    )
    positional_sharding_string = (
        positional_sharding_metadata.to_serialized_string()
    )
    self.assertEqual(
        positional_sharding_string, expected_positional_sharding_string
    )

    # Convert from `str` to `PositionalShardingMetadata`
    converted_positional_sharding_metadata = (
        sharding_metadata.from_serialized_string(positional_sharding_string)
    )
    self.assertIsInstance(
        converted_positional_sharding_metadata,
        sharding_metadata.PositionalShardingMetadata,
    )
    self.assertEqual(
        converted_positional_sharding_metadata,
        positional_sharding_metadata,
    )

  def test_positional_sharding_string_to_metadata_with_memory_kind(
      self,
  ):
    # Convert from `PositionalShardingMetadata` to `str`
    positional_sharding_metadata = sharding_metadata.PositionalShardingMetadata(
        shape=np.array([1, 2]), memory_kind="foo"
    )
    expected_positional_sharding_string = (
        '{"sharding_type": "PositionalSharding", "shape": [1, 2],'
        ' "memory_kind": "foo"}'
    )
    positional_sharding_string = (
        positional_sharding_metadata.to_serialized_string()
    )
    self.assertEqual(
        positional_sharding_string, expected_positional_sharding_string
    )

    # Convert from `str` to `PositionalShardingMetadata`
    converted_positional_sharding_metadata = (
        sharding_metadata.from_serialized_string(positional_sharding_string)
    )
    self.assertIsInstance(
        converted_positional_sharding_metadata,
        sharding_metadata.PositionalShardingMetadata,
    )
    self.assertEqual(
        converted_positional_sharding_metadata,
        positional_sharding_metadata,
    )

  def test_single_device_sharding_string_to_metadata(
      self,
  ):
    # Convert from `SingleDeviceShardingMetadata` to `str`
    single_device_sharding_metadata = (
        sharding_metadata.SingleDeviceShardingMetadata(device_str="TFRT_CPU_0")
    )
    expected_single_device_sharding_string = (
        '{"sharding_type": "SingleDeviceSharding", "device_str": "TFRT_CPU_0"}'
    )
    single_device_sharding_string = (
        single_device_sharding_metadata.to_serialized_string()
    )
    self.assertEqual(
        single_device_sharding_string, expected_single_device_sharding_string
    )

    # Convert from `str` to `SingleDeviceShardingMetadata`
    converted_single_device_sharding_metadata = (
        sharding_metadata.from_serialized_string(single_device_sharding_string)
    )
    self.assertIsInstance(
        converted_single_device_sharding_metadata,
        sharding_metadata.SingleDeviceShardingMetadata,
    )
    self.assertEqual(
        converted_single_device_sharding_metadata,
        single_device_sharding_metadata,
    )

  def test_convert_to_jax_sharding_unsupported_types(self):
    jax_sharding = jax.sharding.GSPMDSharding.get_replicated(jax.devices())
    warning_message = (
        r"Conversion for <class '.*\.GSPMDSharding'>"
        " has not been implemented."
    )
    with self.assertLogs(level="WARNING") as log_output:
      sharding_metadata.from_jax_sharding(jax_sharding)
      self.assertNotEmpty(log_output[0])
      self.assertRegex(log_output[0][0].message, warning_message)


if __name__ == "__main__":
  absltest.main()
