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

"""Utilities for converting TF signatures to OBM."""

from typing import Any

from jax import tree_util as jax_tree_util
import numpy as np
from orbax.experimental.model import core as obm
import tensorflow as tf


def tf_dtype_to_obm(t: tf.DType) -> obm.ShloDType:
  """Converts a TensorFlow dtype to an OBM ShloDType.

  Args:
    t: The TensorFlow dtype to convert.

  Returns:
    The corresponding OBM ShloDType.

  Raises:
    ValueError: If the TensorFlow dtype cannot be converted to an OBM ShloDType.
  """
  if t == tf.string:
    return obm.ShloDType.str
  # need special handling for bfloat16 since numpy doesn't have a bfloat16
  # dtype.
  if t == tf.bfloat16:
    return obm.ShloDType.bf16
  if t in (tf.resource, tf.variant):
    raise ValueError(f"Can't convert TF dtype {t} to OBM.")
  np_dtype = t.as_numpy_dtype()
  try:
    np_dtype = np.dtype(np_dtype)
  except Exception as err:
    raise ValueError(
        f'Failed to create a numpy.dtype object from {np_dtype} of type '
        f'{type(np_dtype)} . The original TF dtype was {t} of type {type(t)} .'
    ) from err
  return obm.np_dtype_to_shlo_dtype(np_dtype)


def tf_tensor_spec_to_obm(
    spec: tf.TensorSpec | tf.Tensor,
) -> obm.ShloTensorSpec:
  """Converts a tf.TensorSpec or tf.Tensor to an obm.ShloTensorSpec.

  Args:
    spec: The tf.TensorSpec or tf.Tensor to convert.

  Returns:
    The corresponding obm.ShloTensorSpec.

  Raises:
    ValueError: If the dtype of the input spec cannot be converted to an OBM
      ShloDType.
  """

  if spec.shape.rank is None:
    obm_shape = None
  else:
    obm_shape = tuple(spec.shape.as_list())

  return obm.ShloTensorSpec(
      shape=obm_shape,
      dtype=tf_dtype_to_obm(spec.dtype),
      name=spec.name,
  )


TfSignature = obm.Tree[Any]


def tf_signature_to_obm_spec(tree: TfSignature) -> obm.Tree[obm.ShloTensorSpec]:
  try:
    return jax_tree_util.tree_map(tf_tensor_spec_to_obm, tree)
  except Exception as err:
    raise ValueError(
        f'Failed to convert TF signature {tree} of type {type(tree)} to OBM.'
    ) from err
