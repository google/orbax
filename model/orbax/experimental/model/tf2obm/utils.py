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


def tf_tensor_spec_to_obm(spec: Any) -> obm.ShloTensorSpec:
  # ConcreteFunction.structured_outputs returns `SymbolicTensor`s, not
  # `TensorSpec`s, so we need to also check for `SymbolicTensor`.
  if not (isinstance(spec, tf.TensorSpec) or tf.is_symbolic_tensor(spec)):
    raise ValueError(
        f'Expected a tf.TensorSpec or a SymbolicTensor, got {spec} of type'
        f' {type(spec)}'
    )
  return obm.ShloTensorSpec(
      shape=spec.shape, dtype=tf_dtype_to_obm(spec.dtype), name=spec.name
  )


TfSignature = obm.Tree[Any]


def tf_signature_to_obm_spec(tree: TfSignature) -> obm.Tree[obm.ShloTensorSpec]:
  try:
    return jax_tree_util.tree_map(tf_tensor_spec_to_obm, tree)
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
  """Gets the output signature from a concrete function.

  Args:
    concrete_function: The concrete function to get the output signature from.

  Returns:
    The output signature as a PyTree of `tf.TensorSpec`s.

  Raises:
    ValueError: If the structured_outputs cannot be converted to
    `tf.TensorSpec`.
  """
  try:
    # The structured_outputs are `SymbolicTensor`s with "name" that we don't
    # need. To make a unified path to obm.ShloTensorSpec, we convert them to
    # `TensorSpec`s (without name) first.
    output_signature = jax_tree_util.tree_map(
        lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype),
        concrete_function.structured_outputs,
    )
  except Exception as err:
    raise ValueError(
        'Failed to convert TF structured_outputs'
        f' {concrete_function.structured_outputs} to tf.TensorSpec.'
    ) from err
  return output_signature
