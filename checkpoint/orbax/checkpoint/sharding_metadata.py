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

"""ShardingMetadata representing Sharding property."""

import abc
import dataclasses
import enum
import json
import logging
from typing import List, Optional, Tuple, Union
import jax
import numpy as np

PartitionSpecElement = Union[None, str, Tuple[str, ...]]

_PARTITION_SPEC = 'partition_spec'
_SHARDING = '_sharding'
_SHARDING_TYPE = 'sharding_type'
_DEVICE_STR = 'device_str'
_MESH_AXES = 'axis_names'
_MESH_SHAPE = 'shape'


class ShardingTypes(enum.Enum):
  NAMED_SHARDING = 'NamedSharding'
  SINGLE_DEVICE_SHARDING = 'SingleDeviceSharding'
  POSITIONAL_SHARDING = 'PositionalSharding'
  GSPMD_SHARDING = 'GSPMDSharding'


def _convert_jax_partition_spec_to_partition_spec_elements(
    jax_spec: jax.sharding.PartitionSpec,
) -> 'NamedShardingMetadata.partition_spec':
  """Converts `jax.sharding.PartitionSpec` to `NamedShardingMetadata.partition_spec`."""

  converted_spec = []
  for element in jax_spec:
    if element is None:
      converted_element = None
    elif isinstance(element, str):
      converted_element = element
    elif isinstance(element, (tuple, list)):
      converted_element = tuple(element)
    else:
      raise ValueError(f'Unsupported element type: {type(element)}')
    converted_spec.append(converted_element)
  return tuple(converted_spec)


@dataclasses.dataclass
class ShardingMetadata(abc.ABC):
  """ShardingMetadata representing Sharding property.

  This ShardingMetadata only represents the following `jax.sharding.Sharding`:
    jax.sharding.NamedSharding
    jax.sharding.SingleDeviceSharding
    jax.sharding.GSPMDSharding
    jax.sharding.PositionalSharding
  """

  @classmethod
  @abc.abstractmethod
  def from_jax_sharding(cls, jax_sharding) -> 'ShardingMetadata':
    """Converts `jax.sharding.Sharding` to `ShardingMetadata`."""

  @abc.abstractmethod
  def to_jax_sharding(self) -> jax.sharding.Sharding:
    """Converts `ShardingMetadata` to `jax.sharding.Sharding`."""

  @classmethod
  @abc.abstractmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> 'ShardingMetadata':
    """Converts serialized_string in the form of `dict[str, str]` to `ShardingMetadata`."""

  @abc.abstractmethod
  def to_serialized_string(self) -> str:
    """Converts `ShardingMetadata` to `serialized_string`."""


@dataclasses.dataclass
class NamedShardingMetadata(ShardingMetadata):
  """NamedShardingMetadata representing `jax.sharding.NamedSharding`."""
  shape: np.ndarray
  axis_names: List[str]
  partition_spec: Tuple[
      PartitionSpecElement, ...
  ]  # Each element is either ``None``, a string, or a tuple of strings.

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.NamedSharding
  ) -> 'NamedShardingMetadata':
    return cls(
        shape=np.array(list(jax_sharding.mesh.shape.values())),
        axis_names=list(jax_sharding.mesh.axis_names),
        partition_spec=_convert_jax_partition_spec_to_partition_spec_elements(
            jax_sharding.spec
        ),
    )

  def to_jax_sharding(self) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(
            np.array(jax.devices()).reshape(self.shape),
            axis_names=self.axis_names,
        ),
        spec=jax.sharding.PartitionSpec(*self.partition_spec),
    )

  @classmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> 'NamedShardingMetadata':
    if (
        _MESH_SHAPE in deserialized_dict
        and _MESH_AXES in deserialized_dict
        and _PARTITION_SPEC in deserialized_dict
    ):
      shape = np.array(deserialized_dict[_MESH_SHAPE])
      axis_names = list(deserialized_dict[_MESH_AXES])
      partition_spec = tuple(deserialized_dict[_PARTITION_SPEC])
      return cls(
          shape=shape,
          axis_names=axis_names,
          partition_spec=partition_spec,
      )
    else:
      raise ValueError(
          f'Sharding data not found in deserialized_dict: {deserialized_dict}'
      )

  def to_serialized_string(self) -> str:
    sharding_data = {}
    sharding_data[_SHARDING_TYPE] = ShardingTypes.NAMED_SHARDING.value
    sharding_data[_MESH_SHAPE] = self.shape.tolist()
    sharding_data[_MESH_AXES] = self.axis_names
    sharding_data[_PARTITION_SPEC] = self.partition_spec
    return json.dumps(sharding_data)

  def __repr__(self):
    return (
        f'NamedShardingMetadata(shape={self.shape},'
        f' axis_names={self.axis_names}, partition_spec={self.partition_spec})'
    )

  def __eq__(self, other):
    return (
        np.array_equal(self.shape, other.shape)
        and self.axis_names == other.axis_names
        and self.partition_spec == other.partition_spec
    )


@dataclasses.dataclass
class SingleDeviceShardingMetadata(ShardingMetadata):
  """SingleDeviceShardingMetadata representing `jax.sharding.SingleDeviceSharding`."""

  device_str: str

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.SingleDeviceSharding
  ) -> 'SingleDeviceShardingMetadata':
    return cls(device_str=str(next(iter(jax_sharding.device_set))))

  def to_jax_sharding(self) -> jax.sharding.SingleDeviceSharding:
    device_map = {str(device): device for device in jax.local_devices()}
    device_str = self.device_str
    if device := device_map.get(device_str, None):
      return jax.sharding.SingleDeviceSharding(device)
    else:
      raise ValueError(
          f'Device {device_str} was not found in jax.local_devices().'
      )

  @classmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> 'SingleDeviceShardingMetadata':
    if (
        _DEVICE_STR in deserialized_dict
        and deserialized_dict[_DEVICE_STR] is not None
    ):
      return cls(device_str=deserialized_dict[_DEVICE_STR])
    raise ValueError(
        f'Device str not found in deserialized_dict: {deserialized_dict}'
    )

  def to_serialized_string(self) -> str:
    sharding_data = {}
    sharding_data[_SHARDING_TYPE] = ShardingTypes.SINGLE_DEVICE_SHARDING.value
    sharding_data[_DEVICE_STR] = self.device_str
    return json.dumps(sharding_data)

  def __repr__(self):
    return f'SingleDeviceShardingMetadata(device_str={self.device_str})'

  def __eq__(self, other):
    return self.device_str == other.device_str


@dataclasses.dataclass
class GSPMDShardingMetadata(ShardingMetadata):
  pass


@dataclasses.dataclass
class PositionalShardingMetadata(ShardingMetadata):
  pass


def from_jax_sharding(jax_sharding) -> Optional[ShardingMetadata]:
  """Converts `jax.sharding.Sharding` to `ShardingMetadata`."""
  if isinstance(jax_sharding, jax.sharding.NamedSharding):
    return NamedShardingMetadata.from_jax_sharding(jax_sharding)
  elif isinstance(jax_sharding, jax.sharding.SingleDeviceSharding):
    return SingleDeviceShardingMetadata.from_jax_sharding(jax_sharding)
  else:
    logging.warning(
        'Conversion for %s has not been implemented.', type(jax_sharding)
    )


def from_serialized_string(serialized_str) -> ShardingMetadata:
  """Converts `serialized_string` to `ShardingMetadata`."""
  deserialized_dict = json.loads(serialized_str)
  if deserialized_dict[_SHARDING_TYPE] == ShardingTypes.NAMED_SHARDING.value:
    return NamedShardingMetadata.from_deserialized_dict(deserialized_dict)
  elif (
      deserialized_dict[_SHARDING_TYPE]
      == ShardingTypes.SINGLE_DEVICE_SHARDING.value
  ):
    return SingleDeviceShardingMetadata.from_deserialized_dict(
        deserialized_dict
    )
  else:
    raise NotImplementedError(
        f'Conversion for {deserialized_dict[_SHARDING_TYPE]} has not been'
        ' implemented.'
    )


def get_sharding_or_none(serialized_string):
  try:
    return from_serialized_string(serialized_string.item()).to_jax_sharding()
  except ValueError as e:
    logging.error(e)
