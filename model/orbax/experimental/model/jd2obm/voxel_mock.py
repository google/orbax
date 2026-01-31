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

"""Fake VoxelSpec definition for testing purposes."""

from collections.abc import Sequence, Set
import dataclasses
from unittest import mock

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np


DTypeLike = str | np.dtype | type[np.generic]


# TODO(b/431506483): Replace with real voxel module when implemented.
@dataclasses.dataclass
class VoxelSpec:
  """Temporary specification for Voxel tensors.

  This is a temporary implementation of `VoxelSpec` used for Orbax Model (OBM)
  integration, as the real Voxel specification is not yet implemented. It
  assumes `np.dtype` for data types to simplify OBM converter development.

  Attributes:
    shape: Tensor dimensions as a tuple of integers.
    dtype: Tensor data type as a `np.dtype`.
  """

  shape: Sequence[int]
  dtype: DTypeLike

  def __post_init__(self):
    object.__setattr__(self, 'shape', tuple(self.shape))
    try:
      dtype = np.dtype(self.dtype)
      object.__setattr__(self, 'dtype', dtype)
    except TypeError as e:
      raise ValueError(
          f'Invalid dtype: {self.dtype!r} cannot be converted to np.dtype.'
      ) from e


class VoxelModule:
  """A mock VoxelModule for testing."""

  def __init__(self):
    self._assets: Set[str] = set()

  def set_assets(self, assets: Set[str]) -> None:
    self._assets = assets

  def export_assets(self) -> Set[str]:
    return self._assets

  def get_output_signature(
      self, input_signature: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    for leaf in jax.tree_util.tree_leaves(input_signature):
      assert isinstance(leaf, VoxelSpec)
    # This mock assumes input_signature is a sequence for simplicity. The
    # actual implementation of the Voxel module may use any appropriate
    # structure for its input signature.
    d1, d2 = input_signature[0]['input'].shape
    return {
        'output': {
            'feature1': VoxelSpec(shape=(d1, d2), dtype=jnp.float32),
            'feature2': VoxelSpec(shape=(d1, d2 * 4), dtype=jnp.int32),
        }
    }

  def export_plan(self):
    plan_proto = mock.Mock()
    plan_proto.SerializeToString.return_value = b'test plan'
    return plan_proto


# Define `__all__` to explicitly declare the public API of this module.
# This controls what `from jd2obm import *` imports and helps linters.
__all__ = [
    'VoxelSpec',
    'VoxelModule',
]
