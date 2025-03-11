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

"""Utilities for converting TF signatures to OBM."""

from typing import Any

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
  if t in (tf.resource, tf.variant):
    raise ValueError(
        f'Can\'t convert TF dtype {t} to OBM.'
    )
  np_dtype = t.as_numpy_dtype()
  try:
    np_dtype = np.dtype(np_dtype)
  except Exception as err:
    raise ValueError(
        f'Failed to create a numpy.dtype object from {np_dtype} of type '
        f'{type(np_dtype)} . The original TF dtype was {t} of type {type(t)} .'
    ) from err
  return obm.np_dtype_to_shlo_dtype(np_dtype)


def tf_tensor_spec_to_obm(spec: Any) -> obm.ShloTensorSpec:
  # ConcreteFunction.structured_outputs returns `SymbolicTensor`s, not
  # `TensorSpec`s, so we need to also check for `SymbolicTensor`.
  if not (isinstance(spec, tf.TensorSpec) or tf.is_symbolic_tensor(spec)):
    raise ValueError(
        f'Expected a tf.TensorSpec or a SymbolicTensor, got {spec} of type'
        f' {type(spec)}'
    )
  return obm.ShloTensorSpec(shape=spec.shape, dtype=tf_dtype_to_obm(spec.dtype))


TfSignature = obm.Tree[Any]


def tf_signature_to_obm_spec(tree: TfSignature) -> obm.Tree[obm.ShloTensorSpec]:
  try:
    return obm.tree_util.tree_map(tf_tensor_spec_to_obm, tree)
  except Exception as err:
    raise ValueError(
        f'Failed to convert TF signature {tree} of type {type(tree)} to OBM.'
    ) from err


def get_input_signature(
    concrete_function: tf.types.experimental.ConcreteFunction,
) -> TfSignature:
  return concrete_function.structured_input_signature


def get_output_signature(
    concrete_function: tf.types.experimental.ConcreteFunction,
) -> TfSignature:
  return concrete_function.structured_outputs
