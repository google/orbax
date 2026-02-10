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

"""Configuration dataclasses for Orbax benchmark tests."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class CheckpointConfig:
  """Configuration for the test checkpoint data to be generated or loaded.

  `sharding_config_path` points to a file in the following format::

    {
      "params.params.decoder.decoder_norm.scale": {
        "shape": [
            4096
        ],
        "dtype": "float32",
        "sharding": {
            "mesh": {
                "shape": [
                    8,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1
                ],
                "axes": [
                    "data",
                    "stage",
                    "fsdp",
                    "fsdp_transpose",
                    "sequence",
                    "context",
                    "context_autoregressive",
                    "tensor",
                    "tensor_transpose",
                    "tensor_sequence",
                    "expert",
                    "autoregressive"
                ]
            },
            "spec": [
                [
                    "tensor",
                    "tensor_transpose"
                ]
            ]
        }
      },
    }

  Attributes:
      path: The path to the checkpoint data to be used in the test. If not
        provided, the checkpoint will be generated using the `spec` attribute.
      random_seed: The random seed to use for generating the checkpoint data. If
        not provided, default random seed will be used.
      spec: A dictionary defining the structure and type of the PyTree to be
        generated. Example: { 'params': { 'dtype': 'float32', 'shape': [1024,
        1024], 'sharding': ['data', 'model']  # PartitionSpec }, 'step': 'int' }
      sharding_config_path: A path to a file containing sharding specifications,
        used alongside `path`. See above.
  """

  path: str | None = None
  random_seed: int = 0
  spec: dict[str, Any] | None = None
  sharding_config_path: str | None = None

  def __post_init__(self):
    if self.path is None and self.spec is None:
      raise ValueError('Either path or spec must be provided.')
    if self.path is not None and self.spec is not None:
      raise ValueError('Only one of path or spec can be provided.')
    if self.sharding_config_path is not None and self.path is None:
      raise ValueError(
          'If `sharding_config_path` is provided, `path` must also be provided.'
      )


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
          Example: {'data': 2} If None, an ordinary device mesh will be used,
            rather than a hybrid device mesh (intended for multi-replica
            workloads)
      allow_split_physical_axes: If True, we will split physical axes if
        necessary to produce the desired device mesh.
      process_is_granule: If True, treat processes as the units of the
        slower/outer network.
  """

  mesh_axes: list[str]
  ici_parallelism: dict[str, int] = dataclasses.field(default_factory=dict)
  dcn_parallelism: dict[str, int] | None = None
  allow_split_physical_axes: bool = False
  process_is_granule: bool = False
