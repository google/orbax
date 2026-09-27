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
import jax
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.sharding_utils import make_single_device_sharding


class _SerializedOnlyShardingMetadata(sharding_metadata.ShardingMetadata):
  """Implements only the abstract methods, like an external subclass."""

  @classmethod
  def from_jax_sharding(cls, jax_sharding):
    raise NotImplementedError()

  def to_jax_sharding(self):
    raise NotImplementedError()

  @classmethod
  def from_deserialized_dict(cls, deserialized_dict):
    raise NotImplementedError()

  def to_serialized_string(self):
    return '{"sharding_type": "Custom", "shape": [2]}'


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
        axis_types=(jax.sharding.AxisType.Auto,),
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

  def test_named_sharding_with_explicit_axis_type_roundtrip(self):
    # Create jax.sharding.NamedSharding with Explicit AxisType
    axis_types = (jax.sharding.AxisType.Explicit,)
    mesh = jax.sharding.Mesh(jax.devices(), ("x",), axis_types=axis_types)
    jax_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None),
    )

    # Convert to NamedShardingMetadata
    named_sharding_metadata = sharding_metadata.from_jax_sharding(jax_sharding)
    self.assertEqual(named_sharding_metadata.axis_types, axis_types)

    # Serialize to string
    serialized_string = named_sharding_metadata.to_serialized_string()
    self.assertIn('"axis_types": ["AxisType.Explicit"]', serialized_string)

    # Deserialize from string
    deserialized_metadata = sharding_metadata.from_serialized_string(
        serialized_string
    )
    self.assertIsInstance(
        deserialized_metadata, sharding_metadata.NamedShardingMetadata
    )
    self.assertEqual(deserialized_metadata, named_sharding_metadata)

    # Convert back to jax.sharding.NamedSharding
    converted_jax_sharding = deserialized_metadata.to_jax_sharding()
    self.assertIsInstance(
        converted_jax_sharding, jax.sharding.NamedSharding
    )
    self.assertEqual(converted_jax_sharding, jax_sharding)

  def test_convert_between_jax_single_device_sharding_and_sharding_metadata(
      self,
  ):
    # Convert from `jax.sharding.SingleDeviceSharding` to
    # `SingleDeviceShardingMetadata`
    jax_sharding = make_single_device_sharding(
        jax.local_devices(backend="cpu")[0]
    )
    expected_single_device_sharding_metadata = (
        sharding_metadata.SingleDeviceShardingMetadata(device_str="cpu:0")
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
    positional_sharding_string = (
        '{"sharding_type": "PositionalSharding", "shape": [1, 2]}'
    )

    with self.assertRaisesRegex(
        ValueError, "PositionalSharding has been deprecated"
    ):
      sharding_metadata.from_serialized_string(positional_sharding_string)

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

  def test_named_sharding_to_json_dict(self):
    devices = [sharding_metadata.DeviceMetadata(id=i) for i in range(4)]
    named_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([2, 2]),
        axis_names=["data", "model"],
        partition_spec=(("data", "model"), None),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        device_mesh=sharding_metadata.DeviceMetadataMesh(
            mesh=[devices[:2], devices[2:]]
        ),
    )
    expected = {
        "sharding_type": "NamedSharding",
        "shape": [2, 2],
        "axis_names": ["data", "model"],
        "axis_types": ["AxisType.Explicit", "AxisType.Explicit"],
        "partition_spec": [["data", "model"], None],
        "device_mesh": {
            "mesh": [[{"id": 0}, {"id": 1}], [{"id": 2}, {"id": 3}]]
        },
    }

    self.assertEqual(named_sharding_metadata.to_json_dict(), expected)
    self.assertEqual(
        named_sharding_metadata.to_serialized_string(), json.dumps(expected)
    )

  def test_single_device_sharding_to_json_dict(self):
    single_device_sharding_metadata = (
        sharding_metadata.SingleDeviceShardingMetadata(device_str="cpu:0")
    )
    self.assertEqual(
        single_device_sharding_metadata.to_json_dict(),
        {"sharding_type": "SingleDeviceSharding", "device_str": "cpu:0"},
    )

  def test_default_to_json_dict_parses_serialized_string(self):
    self.assertEqual(
        _SerializedOnlyShardingMetadata().to_json_dict(),
        {"sharding_type": "Custom", "shape": [2]},
    )


if __name__ == "__main__":
  absltest.main()
