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

import dataclasses
from typing import List, Optional, Tuple, Union
import jax
import numpy as np

PartitionSpecElement = Union[None, str, Tuple[str, ...]]


@dataclasses.dataclass
class ShardingMetadata:
  """ShardingMetadata representing Sharding property.

  This ShardingMetadata only represents the following `jax.sharding.Sharding`:
    jax.sharding.NamedSharding
    jax.sharding.SingleDeviceSharding
    jax.sharding.GSPMDSharding
    jax.sharding.PositionalSharding
  """

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.Sharding
  ) -> 'Optional[ShardingMetadata]':
    pass

  def to_jax_sharding(self) -> Optional[jax.sharding.Sharding]:
    pass

  @classmethod
  def from_serialized_string(
      cls, serialized_str: str
  ) -> 'Optional[ShardingMetadata]':
    pass

  def to_serialized_string(self) -> Optional[str]:
    pass


@dataclasses.dataclass
class NamedShardingMetadata(ShardingMetadata):
  shape: np.ndarray
  axis_names: List[str]
  partition_spec: Tuple[
      PartitionSpecElement, ...
  ]  # Each element is either ``None``, a string, or a tuple of strings.

  def __repr__(self):
    pass


@dataclasses.dataclass
class SingleDeviceShardingMetadata(ShardingMetadata):
  device_str: str

  def __repr__(self):
    pass


@dataclasses.dataclass
class GSPMDShardingMetadata(ShardingMetadata):
  pass


@dataclasses.dataclass
class PositionalShardingMetadata(ShardingMetadata):
  pass
