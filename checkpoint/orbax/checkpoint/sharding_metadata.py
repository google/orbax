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
from typing import List, Optional, Tuple, Union
import jax
import numpy as np

PartitionSpecElement = Union[None, str, Tuple[str, ...]]


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
    elif isinstance(element, tuple):
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

  def to_serialized_string(self) -> Optional[str]:
    pass


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

  def __repr__(self):
    return (
        f'NamedShardingMetadata(shape={self.shape},'
        f' axis_names={self.axis_names}, partition_spec={self.partition_spec})'
    )

  def __eq__(self, other):
    return (
        self.shape == other.shape
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


def from_jax_sharding(jax_sharding) -> ShardingMetadata:
  """Converts `jax.sharding.Sharding` to `ShardingMetadata`."""
  if isinstance(jax_sharding, jax.sharding.NamedSharding):
    return NamedShardingMetadata.from_jax_sharding(jax_sharding)
  elif isinstance(jax_sharding, jax.sharding.SingleDeviceSharding):
    return SingleDeviceShardingMetadata.from_jax_sharding(jax_sharding)
  else:
    raise NotImplementedError(
        f'Conversion for {type(jax_sharding)} has not been implemented.'
    )
