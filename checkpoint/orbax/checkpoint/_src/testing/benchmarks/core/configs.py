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

"""Configuration dataclasses for Orbax benchmark tests."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class CheckpointConfig:
  """Configuration for the test checkpoint data to be generated or loaded.

  Attributes:
      path: The path to the checkpoint data to be used in the test. If not
        provided, the checkpoint will be generated using the `spec` attribute.
      spec: A dictionary defining the structure and type of the PyTree to be
        generated. Example:
        {
            'params': {
                'dtype': 'float32',
                'shape': [1024, 1024],
                'sharding': ['data', 'model']  # PartitionSpec
            },
            'step': 'int'
        }
  """

  path: str | None = None
  spec: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MeshConfig:
  """A structured configuration for creating a complex jax.sharding.Mesh.

  This class captures the user's high-level intent for a distributed
  hardware topology, which is then translated into a concrete device mesh.

  Attributes:
      mesh_axes: A list of names for all dimensions of parallelism, e.g.,
        ['data', 'fsdp', 'tensor'].
      ici_parallelism: A dictionary mapping axis names to their parallelism
        degree *within* a slice (Inter-Chip Interconnect).
          Example: {'tensor': 4, 'fsdp': 2}
      dcn_parallelism: A dictionary mapping axis names to their parallelism
        degree *across* slices (Data Center Network). This typically contains a
        single entry for the data-parallel axis.
          Example: {'data': 2}
      allow_split_physical_axes: If True, we will split physical axes if
        necessary to produce the desired device mesh.
      process_is_granule: If True, treat processes as the units of the
        slower/outer network.
  """
  mesh_axes: list[str]
  ici_parallelism: dict[str, int] = dataclasses.field(default_factory=dict)
  dcn_parallelism: dict[str, int] = dataclasses.field(default_factory=dict)
  allow_split_physical_axes: bool = False
  process_is_granule: bool = False
