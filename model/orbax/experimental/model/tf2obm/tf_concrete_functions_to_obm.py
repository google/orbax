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

"""Converts TF concrete functions to OBM functions (allowing TF resources)."""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, NamedTuple, Tuple

from jax import tree_util as jax_tree_util
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm import utils
import tensorflow as tf


TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE = (
    'application/protobuf;'
    ' type=orbax_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)
TF_CONCRETE_FUNCTION_HANDLE_VERSION = '0.0.1'

_INPUT_NAME_PREFIX = 'input'
_OUTPUT_NAME_PREFIX = 'output'


def is_pair(tree: utils.TfSignature) -> bool:
  return isinstance(tree, Sequence) and len(tree) == 2


def is_args_kwargs_pattern(tree: utils.TfSignature) -> bool:
  return (
      is_pair(tree)
      and isinstance(tree[0], Sequence)
      and isinstance(tree[1], dict)
  )


_NamesAndSequence = Tuple[Sequence[str], Sequence[Any]]


def tf_concrete_function_name_to_obm_function(
    name: str,
    *,
    input_signature: utils.TfSignature | None = None,
    output_signature: utils.TfSignature | None = None,
    fn: tf.types.experimental.ConcreteFunction | None = None,
) -> obm.SerializableFunction:
  """Converts a TensorFlow (TF) concrete function name (with input/output signatures) to an Orbax Model (OBM) function.

  The OBM function is essentially a name pointing into a TF SavedModel where the
  concrete function is actually stored.

  Only one of `fn` and the `(input_signature, output_signature)` pair should be
  provided.

  Args:
    name: a name used in `save_tf_concrete_functions` to identify a concrete
      function.
    input_signature: the input signature of the concrete function.
    output_signature: the output signature of the concrete function.
    fn: the concrete function itself.

  Returns:
    An OBM function.
  """
  if fn is not None:
    if input_signature is not None:
      raise ValueError(
          'Both `fn` and `input_signature` are provided. Please provide only '
          'one of them.'
      )
    if output_signature is not None:
      raise ValueError(
          'Both `fn` and `output_signature` are provided. Please provide only '
          'one of them.'
      )
    input_signature = utils.get_input_signature(fn)
    output_signature = utils.get_output_signature(fn)

  input_names, _, _ = _get_flat_signature(input_signature, _INPUT_NAME_PREFIX)
  output_names, _, _ = _get_flat_signature(
      output_signature, _OUTPUT_NAME_PREFIX
  )
  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
          fn_name=name,
          input_names=list(input_names),
          output_names=list(output_names),
      ).SerializeToString(),
      mime_type=TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE,
      version=TF_CONCRETE_FUNCTION_HANDLE_VERSION,
  )
  return obm.SerializableFunction(
      body=obm.UnstructuredDataWithExtName(
          proto=unstructured_data,
          ext_name='pb',
      ),
      input_signature=utils.tf_signature_to_obm_spec(input_signature),
      output_signature=utils.tf_signature_to_obm_spec(output_signature),
  )


SAVED_MODEL_MIME_TYPE = 'application/x.tensorflow-saved-model'
SAVED_MODEL_VERSION = '1.0'


def tf_saved_model_as_obm_supplemental(path: str) -> obm.UnstructuredData:
  """An Orbax Model global supplemental pointing to a TF SavedModel (path).

  Args:
    path: the path to the TF SavedModel folder. This can be a path relative to
      the Orbax Model folder, such as 'tf_saved_model/', or an absolute path.

  Returns:
    An `obm.UnstructuredData` which can be used as an Orbax Model global
    supplemental.
  """
  return obm.manifest_pb2.UnstructuredData(
      file_system_location=obm.manifest_pb2.FileSystemLocation(
          string_path=path
      ),
      mime_type=SAVED_MODEL_MIME_TYPE,
      version=SAVED_MODEL_VERSION,
  )


_StrDict = Dict[str, Any]


def _make_dict_only_signature(
    names: Sequence[str], leaves: Sequence[tf.TensorSpec]
) -> _StrDict:
  """Creates a dict-only signature from names and leaves.

  Args:
    names: A sequence of names for the leaves.
    leaves: A sequence of tensor specs.

  Returns:
    A dict-only signature where keys are names and values are tensor specs.
  """
  # The TensorSpec values in `leaves` may have their `name` attribute
  # set. Some SavedModel loaders (e.g. TFRT) may use this `name`
  # instead of the key in the returned dict to construct the input signature. We
  # clear the `name` attribute to avoid that.
  return {
      k: tf.TensorSpec(
          shape=v.shape,
          dtype=v.dtype,
          name=None,
      )
      for k, v in zip(names, leaves)
  }


def _tree_to_dict(
    tree: utils.TfSignature | None,
    names: Sequence[str] | None,
) -> _StrDict | None:
  """Converts a TF signature tree to a dictionary if names are provided.

  If `names` is None or `tree` is None, returns None. If `names` is provided,
  `tree` is flattened and a dictionary is created by mapping names to leaves.

  Args:
    tree: The TF signature tree.
    names: An optional sequence of names for leaves.

  Returns:
    A dictionary mapping names to leaves, or None.
  """
  if not names or tree is None:
    return None
  flat = jax_tree_util.tree_leaves(tree)
  return dict(zip(names, flat))


def _dict_to_tree(
    d: _StrDict | None,
    names: Sequence[str] | None,
    tree_def: jax_tree_util.PyTreeDef,
) -> utils.TfSignature | None:
  """Converts a dictionary to a TF signature tree if names are provided.

  If `names` is None, `d` is returned unchanged. If `names` is provided,
  leaves are extracted from `d` in the order of `names` and unflattened into a
  tree using `tree_def`.

  Args:
    d: A dictionary mapping names to leaves.
    names: An optional sequence of names for leaves.
    tree_def: The PyTreeDef for unflattening.

  Returns:
    A TF signature tree.
  """
  if not names or not d:
    return None
  flat = tuple(d[name] for name in names)
  return jax_tree_util.tree_unflatten(tree_def, flat)


def _to_args_kwargs_pattern(
    tree: utils.TfSignature,
) -> Tuple[Sequence[Any], Dict[str, Any]]:
  """Converts a TF signature tree to the '(args, kwargs)' pattern.

  Args:
    tree: a TF signature tree.

  Returns:
    A tuple `(args, kwargs)`, where `args` is a sequence of positional
    arguments, and `kwargs` is a dict of keyword arguments.

  Raises:
    ValueError: if the tree cannot be converted to the '(args, kwargs)' pattern.
  """
  if is_args_kwargs_pattern(tree):
    return tree[0], tree[1]
  elif isinstance(tree, Sequence):
    return tree, {}
  elif isinstance(tree, dict):
    return (), tree
  else:
    raise ValueError(
        f"Can`t convert this tree to the '(args, kwargs)' pattern: {tree}"
    )


# The flattened form of a signature: a named tuple of (names, leaves, treedef).
class SignatureFlat(NamedTuple):
  """The flattened form of a signature.

  Attributes:
    names: A sequence of names for the leaves.
    leaves: A sequence of tensor specs.
    tree_def: The PyTreeDef for unflattening.
  """

  names: Sequence[str]
  leaves: Sequence[tf.TensorSpec]
  tree_def: jax_tree_util.PyTreeDef


# We choose to rely solely on a concrete function's TF signature to
# determine its argument names, not using any other information (such
# as the argument names in the original Python `def`, or the `name`
# field in `TensorSpec`). Currently in TF SavedModel, if a concrete
# function's TF signature is a list, SavedModel may use the argument
# names in the original Python `def` to generate a keyword-based
# version of this function (which is needed for Servomatic which only
# supports keyword-based calling conventions). We think relying on
# this SavedModel behavior is a mistake and the user should make the
# TF signature a dict instead if they want to serve the function on
# Servomatic. If we find that there are too many users relying on this
# SavedModel behavior, we can revisit the decision here.
def _get_flat_signature(
    signature: utils.TfSignature, name_prefix: str
) -> SignatureFlat:
  """Gets the flattened signature.

  Args:
    signature: The TF signature.
    name_prefix: The prefix for generating names.

  Returns:
    A SignatureFlat object `(names, leaves, treedef)`.
  """
  leaves, tree_def = jax_tree_util.tree_flatten(signature)
  names = tuple(f'{name_prefix}_{i}' for i in range(len(leaves)))
  return SignatureFlat(names, leaves, tree_def)


def to_keyword_only_fn(
    f: tf.types.experimental.ConcreteFunction,
) -> tf.types.experimental.ConcreteFunction:
  """Wraps a function into one whose inputs and outputs are keyword-only.

  Args:
    f: a TF concrete function.

  Returns:
    The wrapped function (also a TF concrete function).
  """
  input_names, input_leaves, input_def = _get_flat_signature(
      utils.get_input_signature(f), _INPUT_NAME_PREFIX
  )
  output_names, _, _ = _get_flat_signature(
      utils.get_output_signature(f), _OUTPUT_NAME_PREFIX
  )

  if input_names is None and output_names is None:
    return f

  new_input_signature = _make_dict_only_signature(input_names, input_leaves)
  # Note that `new_input_signature`, `input_names` or `output_names`
  # may still be None here.

  @tf.function(
      autograph=False,
  )
  def new_f(**input_dict):
    args, kwargs = _to_args_kwargs_pattern(
        _dict_to_tree(
            input_dict,
            input_names,
            input_def,
        )
    )
    output = f(*args, **kwargs)
    new_output = _tree_to_dict(
        output,
        output_names,
    )
    return new_output

  return new_f.get_concrete_function(
      **new_input_signature,
  )


def save_tf_concrete_functions(
    path: str,
    concrete_functions: Mapping[str, tf.types.experimental.ConcreteFunction],
    trackable_resources: Any = None,
) -> None:
  """Saves TensorFlow (TF) concrete functions with names to a TF SavedModel.

  The TF SavedModel will have enough information to let one retrieve (and call)
  the concrete functions by their names.

  Args:
    path: the absolute path to save the TF SavedModel.
    concrete_functions: a mapping from names to concrete functions.
    trackable_resources: a nested structure (i.e. PyTree) of
      `tf.saved_model.experimental.TrackableResource`s that are used in
      `concrete_functions`. All TF resources the concrete functions use
      (directly or indirectly) must be present in this structure. Otherwise, an
      "untracked resource" error will be raised.
  """
  # We are using saved_model.save(signatures=...)
  # (i.e. serving_signatures) to save concrete functions, but
  # serving_signatures only supports functions with keyword-only
  # arguments and outputs (where all arguments and outputs must be
  # tensors), so we need to wrap our concrete function into such a
  # conforming form, and save the information gap separately (in
  # tf_concrete_function_name_to_obm_function).
  concrete_functions = {
      k: to_keyword_only_fn(v) for k, v in concrete_functions.items()
  }

  tf_module = tf.Module()
  if trackable_resources is not None:
    tf_module.resources = trackable_resources
  tf.saved_model.save(tf_module, path, signatures=concrete_functions)


TF_SAVED_MODEL_SUPPLEMENTAL_NAME = 'tensorflow_saved_model'
