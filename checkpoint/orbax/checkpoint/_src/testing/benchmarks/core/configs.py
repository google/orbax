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

from collections.abc import Sequence
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
      load_with_colocated_python: If True, the checkpoint will be loaded with
        colocated Python.  Only effective in Pathways and `path` is specified.
  """

  path: str | None = None
  random_seed: int = 0
  spec: dict[str, Any] | None = None
  sharding_config_path: str | None = None
  load_with_colocated_python: bool = False

  def __post_init__(self):
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


@dataclasses.dataclass(frozen=True)
class SimDeviceConfig:
  """A simulated TPU cluster to run the benchmarks on, using SimDevice.

  Set by a top-level `simdevice:` block in the benchmark YAML. Only the shape
  of the block is checked here; SimDevice itself rejects an unknown
  `device_type` or chips that do not split evenly across processes.

  Attributes:
      device_type: The SimDevice hardware profile, e.g. 'v6e' or 'v7x'.
      topology: The chip mesh dimensions: exactly three positive integers.
      num_processes: The number of simulated hosts. SimDevice splits the chips
        evenly between them.
      fabricated_fill_seed: If set, host reads of buffers SimDevice never
        computed return random values seeded with it instead of zeros, so
        checkpoints have a realistic size on disk. Must fit in 32 bits.
  """

  device_type: str
  topology: Sequence[int]
  num_processes: int = 1
  fabricated_fill_seed: int | None = None

  def __post_init__(self):
    if not isinstance(self.device_type, str) or not self.device_type:
      raise ValueError(
          f'device_type must be a non-empty string, got {self.device_type!r}.'
      )
    if (
        not isinstance(self.topology, Sequence)
        or len(self.topology) != 3
        or any(not isinstance(d, int) or d < 1 for d in self.topology)
    ):
      raise ValueError(
          f'topology must be three positive integers, got {self.topology!r}.'
      )
    # Frozen, so convert the YAML list to a tuple to keep the config hashable.
    object.__setattr__(self, 'topology', tuple(self.topology))
    if not isinstance(self.num_processes, int) or self.num_processes < 1:
      raise ValueError(
          'num_processes must be a positive integer, got'
          f' {self.num_processes!r}.'
      )
    if self.fabricated_fill_seed is not None and (
        not isinstance(self.fabricated_fill_seed, int)
        or not 0 <= self.fabricated_fill_seed < 2**32
    ):
      raise ValueError(
          'fabricated_fill_seed must be an integer in [0, 2**32), got'
          f' {self.fabricated_fill_seed!r}.'
      )
