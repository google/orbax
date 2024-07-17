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

"""Utilities for Orbax export."""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import inspect
import os
from typing import Any, Callable, Optional, Union
from absl import logging
import jax
from jax import export as jax_export
import jaxtyping
import tensorflow as tf

ConfigProto = Any
PyTree = jaxtyping.PyTree
SignatureDef = Any

_FILE_TYPE = 'jax_exported'


@dataclasses.dataclass
class TensorSpecWithDefault:
  """Extends tf.TensorSpec to hold a default value.

  Constraints due to Python function calling conventions:
    - For a python function parameter, all corresponding tensor values in the
      signature must have a TensorSpecWithDefault or none of them should.
    - Parameters with default values should be ordered before non-default ones.
  """

  tensor_spec: tf.TensorSpec
  default_val: Any
  is_primary: bool = False

  def __post_init__(self):
    if self.default_val is None:
      raise ValueError('Use TensorSpec if no defaults are needed.')

    # Has to be a Tensor to be available for TF1 style signatures.
    if not isinstance(self.default_val, tf.Tensor):
      self.default_val = tf.convert_to_tensor(
          self.default_val, dtype=self.tensor_spec.dtype
      )

    if not tf.TensorSpec.from_tensor(
        self.default_val,
        name=self.tensor_spec.name,
    ).is_subtype_of(self.tensor_spec):
      raise ValueError(
          f'TensorSpec {self.tensor_spec} is not compatible with'
          f' the default value {self.default_val}'
      )


NestedTfTensorSpec = jaxtyping.PyTree[
    Union[tf.TensorSpec, TensorSpecWithDefault]
]


def remove_signature_defaults(input_signature: PyTree) -> PyTree:
  """Removes TensorSpecWithDefault from an input_signature."""

  def strip_fn(x):
    if isinstance(x, TensorSpecWithDefault):
      return x.tensor_spec
    else:
      return x

  return jax.tree_util.tree_map(
      strip_fn,
      input_signature,
  )


def _get_defaults(input_signature: Sequence[PyTree]) -> list[PyTree]:
  """Returns a list of default values corresponding with each parameter."""
  default_values = []
  for parameter in input_signature:
    leaves = jax.tree_util.tree_leaves(parameter)
    if not any(isinstance(x, TensorSpecWithDefault) for x in leaves):
      default_values.append(inspect.Parameter.empty)
    else:
      if any(isinstance(x, tf.TensorSpec) for x in leaves):
        raise ValueError(
            'TensorSpecWithDefault must be defined for each tensor in the'
            ' structure for the Python arg.'
        )
      default_values.append(
          jax.tree_util.tree_map(lambda x: x.default_val, parameter)
      )
  return default_values


def with_default_args(
    tf_fn: Callable[..., Any],
    input_signature: Sequence[PyTree],
) -> tf.types.experimental.PolymorphicFunction:
  """Creates a TF function with default args specified in `input_signature`.

  Args:
    tf_fn: the TF function.
    input_signature: the input signature. Even leaf is a tf.TensorSpec, or a
      orbax.export.TensorSpecWithDefault if the default value is specified.

  Returns:
    A tf function with default arguments.
  """
  tf_input_signature = remove_signature_defaults(input_signature)
  tf_fn_with_input_signature = tf.function(
      tf_fn,
      input_signature=tf_input_signature,
      jit_compile=False,
      autograph=False,
  )
  default_values = _get_defaults(input_signature)
  if all(v is inspect.Parameter.empty for v in default_values):
    return tf_fn_with_input_signature

  # Generate a new Python function signature with default values.
  old_parameters = (
      tf_fn_with_input_signature.function_spec.function_type.parameters.values()
  )
  parameters = [
      inspect.Parameter(parameter.name, parameter.kind, default=value)
      for parameter, value in zip(old_parameters, default_values)
  ]
  py_signature_with_defaults = inspect.Signature(parameters)

  # Create a fn_with_defaults that upholds py_signature_with_defaults.
  def fn_with_defaults(*args, **kwargs):
    bound_args = py_signature_with_defaults.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return tf_fn(*bound_args.args, **bound_args.kwargs)

  fn_with_defaults.__signature__ = py_signature_with_defaults

  # Generate a tf.function and return.
  return tf.function(
      func=fn_with_defaults,
      input_signature=tf_input_signature,
      jit_compile=False,
      autograph=False,
  )


def _runtime_batch_size(x: tf.Tensor) -> tf.Tensor:
  """Gets the runtime batch size of a tensor."""
  return tf.shape(x)[0]


@tf.function(autograph=True)
def _repeat_to_batch(
    x: tf.Tensor,
    primary_batch_size: tf.Tensor,
    tensor_name: Optional[str] = None,
) -> tf.Tensor:
  """Repeats a tensor to match a primary batch size."""
  input_batch_size = _runtime_batch_size(x)
  if input_batch_size == 1:
    x = tf.repeat(x, primary_batch_size, axis=0)
  else:
    tf.assert_equal(
        input_batch_size,
        primary_batch_size,
        f'The batch size of a non-primary input tensor (name={tensor_name})'
        ' must be 1 or the same as that of the primary tensors.',
    )
  return x


def make_auto_batching_function(
    input_signature: Sequence[PyTree],
) -> Callable[..., Any]:
  """Creates an auto-batching function from input signature.

  An auto-batching function is a function whose input tensors can have either
  a batch size of "b" or 1, and whose output tensors have a batch size of "b",
  where "b" is the batch size of the primary input tensors.

  Requirements:
    - All input tensors must have a leading batch dimension.
    - There must be at least one primary tensor. A primary tensor is a tensor
    whose tensor spec is either a `tf.TensorSpec` or a `TensorSpecWithDefault`
    whose
    is_primary attribute is True.
    - All primary tensors must have the same batch size.
    - All non-primary tensors must have a batch size of 1, or the same as the
    primary batch size.

  Example:
    >>> input_signature = (
    >>>     tf.TensorSpec([None], tf.int32, name='primary'),
    >>>     TensorSpecWithDefault(
    >>>         tf.TensorSpec([None], tf.int32, name='optional'), [1]
    >>>     ),
    >>> )
    >>> batching_fn = utils.make_auto_batching_function(input_signature)
    >>> batching_fn(tf.constant([0, 0]), tf.constant([1]))
    (<tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 0], dtype=int32)>,
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 1], dtype=int32)>)

  Args:
    input_signature: a sequence of PyTrees whose leaf node is `tf.Tensor` or
      `TensorSpecWithDefault`.

  Returns:
    A TF function whose output tensors all have the same batch size.
  """
  flat_sig, sig_treedef = jax.tree_util.tree_flatten(tuple(input_signature))
  is_primary_tensor = [
      isinstance(x, tf.TensorSpec) or x.is_primary for x in flat_sig
  ]
  tensor_names = [
      x.name if isinstance(x, tf.TensorSpec) else x.tensor_spec.name
      for x in flat_sig
  ]

  if not any(is_primary_tensor):
    raise ValueError(
        'No primary input tensors. A primary tensor is a tensor whose tensor'
        ' spec is either a `tf.TensorSpec` or a `TensorSpecWithDefault` whose'
        ' `is_primary` attribute is True. Got'
        f' input_signature={input_signature}`'
    )

  def auto_batching_fn(*args):
    flat_args, arg_treedef = jax.tree_util.tree_flatten(args)
    assert arg_treedef == sig_treedef, (arg_treedef, sig_treedef)

    primary_batch_size = None
    for tensor, is_primary in zip(flat_args, is_primary_tensor):
      if is_primary:
        primary_batch_size = _runtime_batch_size(tensor)
        break
    assert primary_batch_size is not None

    batched = []
    for tensor, is_primary, name in zip(
        flat_args, is_primary_tensor, tensor_names
    ):
      if is_primary:
        tf.assert_equal(
            primary_batch_size,
            _runtime_batch_size(tensor),
            'All primary input tensors must have the same batch size.',
        )
      else:
        tensor = _repeat_to_batch(tensor, primary_batch_size, name)
      batched.append(tensor)
    return jax.tree_util.tree_unflatten(arg_treedef, batched)

  return with_default_args(auto_batching_fn, input_signature)


class CallableSignatures:
  """Holds TF SignatureDefs as python callables."""

  def __init__(
      self,
      sess: tf.compat.v1.Session,
      signature_defs: Mapping[str, SignatureDef],
  ):
    callable_signatures = {}
    for name, signature_def in signature_defs.items():

      def call(signature_def, **inputs):
        output_tensor_keys = list(signature_def.outputs.keys())
        feed_dict = {
            sess.graph.get_tensor_by_name(signature_def.inputs[k].name): (
                v.numpy() if isinstance(v, tf.Tensor) else v
            )
            for k, v in inputs.items()
        }
        fetches = [
            sess.graph.get_tensor_by_name(signature_def.outputs[k].name)
            for k in output_tensor_keys
        ]
        outputs = sess.run(fetches, feed_dict)
        return dict(zip(output_tensor_keys, outputs))

      callable_signatures[name] = functools.partial(call, signature_def)

    self._sess = sess
    self._signatures = callable_signatures

  @classmethod
  def from_saved_model(
      cls, model_dir: str, tags: list[str], sess_config: ConfigProto = None
  ):
    """Loads a SavedModel and reconsruct its signatures as python callables.

    The signatures of the object loaded by the ``tf.saved_model.load`` API
    doesn't support default values, hence one can use this class to load the
    model in TF1 and reconstruct the signatures. Example:

    >>> loaded = CallableSignatures.from_saved_model(model_dir, ['serve'])
    >>> outputs = loaded.signature['serving_default'](**inputs)

    The TF2 version of this example is

    >>> loaded_tf2 = tf.saved_model.load(model_dir, ['serve'])
    >>> outputs = loaded_tf2.signatures['serving_default'](**inputs)

    But the callables in `loaded_tf2.signatures` doesn't have any default
    inputs.

    Args:
      model_dir: SavedModel directory.
      tags: Tags to identify the metagraph to load. Same as the `tags` argument
        in tf.saved_model.load.
      sess_config: (Optional.) A
        [`ConfigProto`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
        protocol buffer with configuration options for the session.

    Returns:
      A mapping of signature names to the callables.
    """
    with tf.Graph().as_default():
      sess = tf.compat.v1.Session(config=sess_config)
    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, tags, model_dir)
    return cls(sess, meta_graph_def.signature_def)

  @property
  def signatures(self):
    """Returns a mapping for signature names to python callables."""
    return self._signatures


def _save_jax_exported_to_disk(
    exp: jax_export.Exported,
    bin_file_path: str,
    vjp_order: int = 0,
) -> None:
  if tf.io.gfile.exists(bin_file_path):
    raise ValueError(f'File {bin_file_path} already exists.')
  with tf.io.gfile.GFile(bin_file_path, 'wb') as f:
    f.write(exp.serialize(vjp_order=vjp_order))


def _load_jax_exported_from_disk(bin_file_path: str) -> jax_export.Exported:
  if not tf.io.gfile.exists(bin_file_path):
    raise ValueError(f'File {bin_file_path} does not exist.')
  with tf.io.gfile.GFile(bin_file_path, 'rb') as f:
    exp = jax_export.deserialize(bytearray(f.read()))
    return exp


def save_jax_exported_map(
    dir_path: str,
    jax_exported_map: Mapping[str, jax_export.Exported],
):
  """Saves the orbax.export JaxExported Map to disk."""
  if tf.io.gfile.exists(dir_path):
    raise ValueError(f'Directory {dir_path} already exists.')

  tf.io.gfile.makedirs(dir_path)
  for method_key, jax_exported in jax_exported_map.items():
    file_path = os.path.join(dir_path, f'{method_key}.{_FILE_TYPE}')
    _save_jax_exported_to_disk(jax_exported, os.path.join(dir_path, file_path))
  logging.info('Saved JaxExported Map to %s successfully.', dir_path)


def load_jax_exported_map(dir_path: str) -> Mapping[str, jax_export.Exported]:
  """Loads the orbax.export ApplyFn JaxExported Map from disk.

  Args:
    dir_path: The directory path to load the ApplyFn Map.

  Returns:
    A map of method_key to JaxExported object.
  """
  jax_exported_map = {}

  if not tf.io.gfile.exists(dir_path):
    raise ValueError(f'Directory {dir_path} does not exist.')

  for method_key in tf.io.gfile.listdir(dir_path):
    if not method_key.endswith(f'.{_FILE_TYPE}'):
      continue
    jax_exported = _load_jax_exported_from_disk(
        os.path.join(dir_path, method_key)
    )
    jax_exported_map[method_key[: -len(f'.{_FILE_TYPE}')]] = jax_exported
  if not jax_exported_map:
    raise ValueError(f'No .{_FILE_TYPE} files found in {dir_path}.')
  logging.info('Loaded ApplyFn JaxExported Map from %s successfully.', dir_path)
  return jax_exported_map
