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

from collections.abc import Sequence
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np


# TODO(b/431506483): Replace with real voxel module when implemented.
class VoxelSpec:
  """Temporary specification for Voxel tensors.

  This is a temporary implementation of `VoxelSpec` used for Orbax Model (OBM)
  integration, as the real Voxel specification is not yet implemented. It
  assumes `np.dtype` for data types to simplify OBM converter development.

  Attributes:
    shape: Tensor dimensions as a tuple of integers.
    dtype: Tensor data type as a `np.dtype`.
  """

  def __init__(self, shape: Sequence[int], dtype: Any):
    self.shape = tuple(shape)
    try:
      self.dtype: np.dtype = np.dtype(dtype)
    except TypeError as e:
      raise ValueError(
          f'Invalid dtype: {dtype} cannot be converted to np.dtype.'
      ) from e

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, VoxelSpec):
      return NotImplemented
    return self.shape == other.shape and self.dtype == other.dtype

  def __repr__(self) -> str:
    return f'VoxelSpec(shape={self.shape}, dtype={self.dtype})'


class VoxelModule:
  """A mock VoxelModule for testing."""

  def __init__(self):
    self._assets: set[str] = set()

  def set_assets(self, assets: set[str]):
    self._assets = assets

  def export_assets(self) -> set[str]:
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
