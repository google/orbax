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

"""Converts TF concrete functions to OBM functions (allowing TF resources)."""

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, TypeVar

from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm.utils import get_input_signature
from orbax.experimental.model.tf2obm.utils import get_output_signature
from orbax.experimental.model.tf2obm.utils import tf_signature_to_obm_spec
from orbax.experimental.model.tf2obm.utils import TfSignature
import tensorflow as tf


TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE = 'tensorflow_concrete_function_handle'
TF_CONCRETE_FUNCTION_HANDLE_VERSION = '0.0.1'


def is_pair(tree: TfSignature) -> bool:
  return isinstance(tree, Sequence) and len(tree) == 2


def is_args_kwargs_pattern(tree: TfSignature) -> bool:
  return (
      is_pair(tree)
      and isinstance(tree[0], Sequence)
      and isinstance(tree[1], dict)
  )


def _is_str_tensor_spec_dict(tree: TfSignature) -> bool:
  if not isinstance(tree, dict):
    return False
  for k, v in tree.items():
    if not isinstance(k, str):
      return False
    if not isinstance(v, tf.TensorSpec):
      return False
  return True


def _is_dict_only(tree: TfSignature) -> bool:
  if _is_str_tensor_spec_dict(tree):
    return True
  elif is_args_kwargs_pattern(tree):
    return not tree[0] and _is_str_tensor_spec_dict(tree)
  return False


_NamesAndSequence = Tuple[Sequence[str] | None, Sequence[Any] | None]


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
def _generate_names(tree: TfSignature, prefix: str = '') -> _NamesAndSequence:
  if _is_dict_only(tree):
    # If the input signature is dict-only, the function shouldn't be
    # called with positional arguments anyway, so we don't generate
    # names and just return None.
    return None, None
  flat = obm.tree_util.flatten(tree)
  return tuple(f'{prefix}_{i}' for i in range(len(flat))), flat


def _get_input_names(tree: TfSignature) -> _NamesAndSequence:
  return _generate_names(tree, prefix='input')


def _get_output_names(tree: TfSignature) -> _NamesAndSequence:
  return _generate_names(tree, prefix='output')


_T0 = TypeVar('_T0')
_T1 = TypeVar('_T1')


def optional_map(f: Callable[[_T0], _T1], a: _T0 | None) -> _T1 | None:
  if a is None:
    return None
  return f(a)


def _to_optional_str_list(
    seq: Sequence[str] | None,
) -> tf_concrete_function_handle_pb2.StrList | None:
  return optional_map(
      lambda s: tf_concrete_function_handle_pb2.StrList(elements=s), seq
  )


def tf_concrete_function_name_to_obm_function(
    name: str,
    *,
    input_signature: TfSignature | None = None,
    output_signature: TfSignature | None = None,
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
    input_signature = get_input_signature(fn)
    output_signature = get_output_signature(fn)

  input_names, _ = _get_input_names(input_signature)
  output_names, _ = _get_output_names(output_signature)
  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
          fn_name=name,
          input_names=_to_optional_str_list(input_names),
          output_names=_to_optional_str_list(output_names),
      ).SerializeToString(),
      mime_type=TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE,
      version=TF_CONCRETE_FUNCTION_HANDLE_VERSION,
  )
  return obm.SerializableFunction(
      body=obm.UnstructuredDataWithExtName(
          proto=unstructured_data,
          ext_name='pb',
      ),
      input_signature=tf_signature_to_obm_spec(input_signature),
      output_signature=tf_signature_to_obm_spec(output_signature),
  )


SAVED_MODEL_MIME_TYPE = 'tensorflow_saved_model'
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
    tree: TfSignature,
    get_names_fn: Callable[[TfSignature], _NamesAndSequence],
) -> Tuple[_StrDict | None, Sequence[str] | None]:
  """Converts a TF signature to a dict-only signature.

  Args:
    tree: a TF signature.
    get_names_fn: a function that takes a TF signature and returns a sequence of
      names for the flattened signature and the flattened signature itself.

  Returns:
    A dict-only signature and the sequence of names.
  """
  names, flat = get_names_fn(tree)
  if names is None:
    return None, None
  d = dict(zip(names, flat))
  # `d`'s values (of type TensorSpec) may have their `name` attribute
  # set. Some SavedModel loaders (e.g. TFRT) may use this `name`
  # instead of the key in `d` to construct the input signature. We
  # clear the `name` attribute to avoid that.
  d = {
      k: tf.TensorSpec(
          shape=v.shape,
          dtype=v.dtype,
          name=None,
      )
      for k, v in d.items()
  }
  return d, names


def _tree_to_dict(
    tree: TfSignature,
    names: Sequence[str] | None,
) -> _StrDict:
  if names is None:
    return tree
  flat = obm.tree_util.flatten(tree)
  return dict(zip(names, flat))


def _dict_to_tree(
    d: _StrDict,
    names: Sequence[str] | None,
    tree_pattern: TfSignature,
) -> TfSignature:
  if names is None:
    return d
  flat = tuple(d[name] for name in names)
  return obm.tree_util.unflatten(tree_pattern, flat)


def _to_args_kwargs_pattern(
    tree: TfSignature,
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


def _to_keyword_only_fn(
    f: tf.types.experimental.ConcreteFunction,
) -> tf.types.experimental.ConcreteFunction:
  """Wraps a function into one whose inputs and outputs are keyword-only.

  Args:
    f: a TF concrete function.

  Returns:
    The wrapped function (also a TF concrete function).
  """
  input_signature = get_input_signature(f)
  output_signature = get_output_signature(f)
  new_input_signature, input_names = _make_dict_only_signature(
      input_signature, _get_input_names
  )
  output_names, _ = _get_output_names(output_signature)
  if input_names is None and output_names is None:
    return f

  @tf.function(
      autograph=False,
  )
  def new_f(**input_dict):
    args, kwargs = _to_args_kwargs_pattern(
        _dict_to_tree(
            input_dict,
            input_names,
            input_signature,
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
      k: _to_keyword_only_fn(v) for k, v in concrete_functions.items()
  }

  tf_module = tf.Module()
  if trackable_resources is not None:
    tf_module.resources = trackable_resources
  tf.saved_model.save(tf_module, path, signatures=concrete_functions)


TF_SAVED_MODEL_SUPPLEMENTAL_NAME = 'tensorflow_saved_model'
