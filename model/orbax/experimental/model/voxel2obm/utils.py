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

import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.voxel2obm.voxel_mock import VoxelSpec


VoxelSignature = obm.Tree[VoxelSpec]


def _obm_to_voxel_dtype(t):
  if isinstance(t, obm.ShloDType):
    return obm.shlo_dtype_to_np_dtype(t)
  return t


def _obm_to_voxel_spec(spec: obm.ShloTensorSpec) -> VoxelSpec:
  return VoxelSpec(shape=spec.shape, dtype=_obm_to_voxel_dtype(spec.dtype))


def obm_spec_to_voxel_signature(
    spec: obm.Tree[obm.ShloTensorSpec],
) -> VoxelSignature:
  try:
    return obm.tree_util.tree_map(_obm_to_voxel_spec, spec)
  except Exception as err:
    raise ValueError(
        f'Failed to convert OBM spec {spec} of type {type(spec)} to Voxel.'
    ) from err


def _voxel_to_obm_dtype(t) -> obm.ShloDType:
  if not isinstance(t, np.dtype):
    raise ValueError(f'Expected a numpy.dtype, got {t} of type {type(t)}')
  return obm.np_dtype_to_shlo_dtype(t)


def _voxel_to_obm_spec(spec: VoxelSpec) -> obm.ShloTensorSpec:
  print(
      'returned value: ',
      obm.ShloTensorSpec(
          shape=spec.shape, dtype=_voxel_to_obm_dtype(spec.dtype)
      ),
  )
  return obm.ShloTensorSpec(
      shape=spec.shape, dtype=_voxel_to_obm_dtype(spec.dtype)
  )


def voxel_signature_to_obm_spec(
    signature: VoxelSignature,
) -> obm.Tree[obm.ShloTensorSpec]:
  print('line 81, input voxel signature: ', signature)
  try:
    return obm.tree_util.tree_map(_voxel_to_obm_spec, signature)
  except Exception as err:
    raise ValueError(
        f'Failed to convert voxel signature {signature} of type'
        f' {type(signature)} to OBM.'
    ) from err
