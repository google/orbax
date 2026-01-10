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

"""Utilities for converting Voxel signatures to OBM."""

import pprint
import jax
import numpy as np
from orbax.experimental.model import core as obm
from .learning.brain.experimental.jax_data.python import voxel_tensor_spec


def _obm_to_voxel_dtype(t):
  if isinstance(t, obm.ShloDType):
    return obm.shlo_dtype_to_np_dtype(t)
  return t


def obm_spec_to_voxel_signature(
    spec: obm.Tree[obm.ShloTensorSpec],
) -> voxel_tensor_spec.VoxelSchemaTree:
  try:
    return jax.tree_util.tree_map(
        lambda x: voxel_tensor_spec.VoxelTensorSpec(
            shape=x.shape, dtype=obm.shlo_dtype_to_np_dtype(x.dtype)
        ),
        spec,
    )
  except Exception as err:
    raise ValueError(
        'Failed to convert OBM spec of type'
        f' {type(spec)} to Voxel:\n{pprint.pformat(spec)}'
    ) from err


def _voxel_to_obm_dtype(t) -> obm.ShloDType:
  if not isinstance(t, np.dtype):
    raise ValueError(f'Expected a numpy.dtype, got {t!r} of type {type(t)}')
  return obm.np_dtype_to_shlo_dtype(t)


def voxel_signature_to_obm_spec(
    signature: voxel_tensor_spec.VoxelSchemaTree,
) -> obm.Tree[obm.ShloTensorSpec]:
  try:
    return jax.tree_util.tree_map(
        lambda x: obm.ShloTensorSpec(
            shape=x.shape, dtype=_voxel_to_obm_dtype(x.dtype)
        ),
        signature,
    )
  except Exception as err:
    raise ValueError(
        'Failed to convert voxel signature of type'
        f' {type(signature)} to OBM:\n{pprint.pformat(signature)}'
    ) from err
