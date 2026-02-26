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

"""Converts TF concrete functions to OBM functions (allowing TF resources)."""

from collections.abc import Mapping, Sequence
import copy
import os
import tempfile
from typing import Any, Dict, NamedTuple, Tuple

from jax import tree_util as jax_tree_util
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm._src import utils
import tensorflow as tf

from .learning.brain.contrib.tpu_modeling.inference_converter_v2 import converter_options_v2_pb2
from .learning.brain.contrib.tpu_modeling.inference_converter_v2.python import converter
from tensorflow.core.protobuf import meta_graph_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2  # pylint: disable=g-direct-tensorflow-import

TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE = (
    'application/protobuf;'
    ' type=orbax_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)
TF_CONCRETE_FUNCTION_HANDLE_VERSION = '0.0.1'
OBM_TF_SAVED_MODEL_SUB_DIR = 'tf_saved_model/'
TF_SAVED_MODEL_SUPPLEMENTAL_NAME = 'tensorflow_saved_model'
SAVED_MODEL_MIME_TYPE = 'application/x.tensorflow-saved-model'
SAVED_MODEL_VERSION = '1.0'

_OUTPUT_NAME_PREFIX = 'output'


def _is_args_kwargs_pattern(tree: utils.TfSignature) -> bool:
  return (
      isinstance(tree, Sequence)
      and len(tree) == 2
      and isinstance(tree[0], Sequence)
      and isinstance(tree[1], dict)
  )


def convert_function(
    fn_name: str,
    fn: tf.types.experimental.ConcreteFunction,
    converter_options: (
        converter_options_v2_pb2.ConverterOptionsV2 | None
    ) = None,
    trackable_resources: Any | None = None,
) -> obm.SerializableFunction:
  """Converts the TF concrete function to an OBM function.

  The resulting OBM function is effectively a reference to a function
  stored in TF SavedModel by `save_tf_functions`.

  Args:
    fn_name: The name to be used in the OBM manifest to refer to the TF
      function.
    fn: The TF concrete function.
    converter_options: The converter options to use for the TF SavedModel. If
      set, the TF SavedModel will be converted using Inference Converter V2 in
      order to get the correct types for the input and output signatures.
    trackable_resources: Trackable resources used by the function.

  Returns:
    The OBM function referring to the original TF function in the TF SavedModel.
  """

  input_names, _, _ = _flat_input_signature(fn)
  output_names = _output_names(fn)

  if converter_options is not None:
    converterted_signature_def = _get_converted_function_signature_def(
        fn_name, fn, trackable_resources, converter_options
    )
    input_signature = _copy_types_from_signature_def(
        fn.structured_input_signature,
        converterted_signature_def.inputs,
        input_names,
    )
    output_signature = _copy_types_from_signature_def(
        get_output_signature(fn),
        converterted_signature_def.outputs,
        output_names,
    )
  else:
    input_signature = fn.structured_input_signature
    output_signature = get_output_signature(fn)

  unstructured_data = obm.manifest_pb2.UnstructuredData(
      inlined_bytes=tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
          fn_name=fn_name,
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


def tf_saved_model_as_obm_supplemental(subdir: str) -> obm.UnstructuredData:
  """An Orbax Model global supplemental pointing to a TF SavedModel (path).

  Args:
    subdir: the path to the TF SavedModel dir. This can be a path relative to
      the Orbax Model dir, such as 'tf_saved_model/', or an absolute path.

  Returns:
    An `obm.UnstructuredData` which can be used as an Orbax Model global
    supplemental.
  """
  return obm.manifest_pb2.UnstructuredData(
      file_system_location=obm.manifest_pb2.FileSystemLocation(
          string_path=subdir
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
  if _is_args_kwargs_pattern(tree):
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


def _flat_input_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> SignatureFlat:
  """Returns the flattened input signature of the given function."""
  leaves, tree_def = jax_tree_util.tree_flatten(fn.structured_input_signature)
  # The argument names in SavedModel's SignatureDef may not match the names in
  # the input signature due to internal name mangling, hence we're looking
  # it up in the FunctionDef.
  input_names = [arg.name for arg in fn.function_def.signature.input_arg]
  if len(input_names) < len(leaves):
    # There could be more arguments in the FunctionDef than in the input
    # signature, because it also contains the captured inputs appended
    # to the flattened list of the input arguments.
    raise ValueError(
        f'The number of input arguments in FunctionDef ({len(input_names)}) is'
        ' smaller than the number of leaves in the flattened input signature'
        f' ({len(leaves)})'
    )
  return SignatureFlat(input_names[: len(leaves)], leaves, tree_def)


def _output_name_for_key(key: Any) -> str:
  if isinstance(key, jax_tree_util.SequenceKey):
    return f'{_OUTPUT_NAME_PREFIX}_{key.idx}'
  elif isinstance(key, jax_tree_util.DictKey):
    # The order is stable as guaranteed by `jax.tree.flatten`.
    return f'{key.key}'
  elif isinstance(key, jax_tree_util.GetAttrKey):
    return f'{key.name}'
  raise ValueError(f'Invalid output key type: {key}')


def _output_name(path: Sequence[Any]) -> str:
  """Returns the output name based on its path in the output signature."""
  if not path:
    # Scalar return value (single tensor).
    return f'{_OUTPUT_NAME_PREFIX}_0'

  # Multiple levels of nesting is normally not suppported for
  # TF concrete function outputs. However, we already
  # support the case of nested sturctures in Orbax TF export,
  # so we will explicitly support nested structures here.
  return '.'.join(_output_name_for_key(k) for k in path)


def _output_names(
    fn: tf.types.experimental.ConcreteFunction,
) -> Sequence[str]:
  """Returns the flattened output signature of the given function."""
  leaves_with_path = jax_tree_util.tree_leaves_with_path(fn.structured_outputs)
  if not leaves_with_path:
    return []
  paths, _ = zip(*leaves_with_path)
  return [_output_name(path) for path in paths]


def get_output_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> utils.TfSignature:
  """Returns the output signature of the TF function.

  Tensor names in the output signature match the output names of the TF function
  in the TF SavedModel.

  Args:
    fn: A concrete TF function.
  """
  output_names_iter = iter(list(_output_names(fn)))

  return jax_tree_util.tree_map(
      lambda t: tf.TensorSpec(
          shape=t.shape, dtype=t.dtype, name=next(output_names_iter)
      ),
      fn.structured_outputs,
  )


def to_keyword_only_fn(
    f: tf.types.experimental.ConcreteFunction,
) -> tf.types.experimental.ConcreteFunction:
  """Wraps a function into one whose inputs and outputs are keyword-only.

  Args:
    f: a TF concrete function.

  Returns:
    The wrapped function (also a TF concrete function).
  """
  input_names, input_leaves, input_def = _flat_input_signature(f)
  output_names = _output_names(f)

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


def save_tf_functions(
    model_dir: str,
    fns: Mapping[str, tf.types.experimental.ConcreteFunction],
    *,
    trackable_resources: Any = None,
    converter_options: (
        converter_options_v2_pb2.ConverterOptionsV2 | None
    ) = None,
    tf_saved_model_sub_dir: str = OBM_TF_SAVED_MODEL_SUB_DIR,
) -> dict[str, obm.GlobalSupplemental]:
  """Saves TensorFlow (TF) concrete functions with names to a TF SavedModel.

  The TF SavedModel will have enough information to let one retrieve (and call)
  the concrete functions by their names. If `converter_options` is set, the TF
  SavedModel will be converted using Inference Converter V2 with the given
  options.

  Args:
    model_dir: The OBM model directory.
    fns: The mapping from names to concrete functions.
    trackable_resources: The nested structure (i.e. PyTree) of
      `tf.saved_model.experimental.TrackableResource`s that are used in
      `concrete_functions`. All TF resources the concrete functions use
      (directly or indirectly) must be present in this structure. Otherwise, an
      "untracked resource" error will be raised. If tf.Module is passed, it will
      be used to create the saved model.
    converter_options: The converter options to use for the TF SavedModel. If
      set, the TF SavedModel will be converted using Inference Converter V2.
    tf_saved_model_sub_dir: The sub-directory name for the TF SavedModel,
      relative to `model_dir`.

  Returns:
    The single-entry dictionary with the resulting TF supplemental.
  """
  # We are using saved_model.save(signatures=...)
  # (i.e. serving_signatures) to save concrete functions, but
  # serving_signatures only supports functions with keyword-only
  # arguments and outputs (where all arguments and outputs must be
  # tensors), so we need to wrap our concrete function into such a
  # conforming form, and save the information gap separately (in
  # convert_function).
  wrapped_fns = {k: to_keyword_only_fn(v) for k, v in fns.items()}

  tf_module = tf.Module()
  if isinstance(trackable_resources, tf.Module):
    # tf.Module may contain variables and other resources that cannot be
    # easily accessed piecemeal. The caller can pass the tf.Module directly
    # and it will be used to create the saved model; all nested resources
    # will be included.
    tf_module = trackable_resources
  elif trackable_resources is not None:
    tf_module.resources = trackable_resources

  target_path = os.path.join(model_dir, tf_saved_model_sub_dir)
  if converter_options is not None:
    # Inference Converter V2 modifies the converter_options in place, so we
    # need to deepcopy it to avoid modifying the original options and keep
    # them re-usable.
    converter_options_copy = copy.deepcopy(converter_options)
    pre_conversion_path = os.path.join(model_dir, 'tmp_tf_saved_model')
    tf.saved_model.save(
        tf_module,
        pre_conversion_path,
        signatures=wrapped_fns,
        # Function aliases are used by the Inference Converter V2 to
        # identify XLA functions.
        options=tf.saved_model.SaveOptions(function_aliases=wrapped_fns),
    )
    converter.ConvertSavedModel(
        pre_conversion_path,
        target_path,
        converter_options_copy,
    )
    tf.io.gfile.rmtree(pre_conversion_path)
  else:
    tf.saved_model.save(tf_module, target_path, signatures=wrapped_fns)

  return {
      TF_SAVED_MODEL_SUPPLEMENTAL_NAME: obm.GlobalSupplemental(
          tf_saved_model_as_obm_supplemental(tf_saved_model_sub_dir)
      )
  }


def _copy_types_from_signature_def(
    original_signature: Any,
    signature_def_args: Mapping[str, meta_graph_pb2.TensorInfo],
    arg_names: Sequence[str],
) -> Any:
  """Copies types from TF SignatureDef to the original signature.

  Args:
    original_signature: The original signature that needs new types.
    signature_def_args: The TF SignatureDef arguments to copy types from.
    arg_names: The argument names of the original TF function. They are used to
      infer the input order in the original signature.

  Returns:
    The original signature with types copied from the signature_def for the
    corresponding input names.

  Raises:
    ValueError: If any of the argument names is not found in the SignatureDef.
  """

  arg_names_iter = iter(arg_names)

  def _copy_type(t: Any) -> Any:
    arg_name = next(arg_names_iter)
    if arg_name not in signature_def_args:
      raise ValueError(
          f'Argument name {arg_name!r} not found in SignatureDef: '
          f'{signature_def_args.keys()!r}'
      )

    if not isinstance(t, tf.TensorSpec):
      return t

    return tf.TensorSpec(
        shape=t.shape,
        dtype=tf.as_dtype(signature_def_args[arg_name].dtype),
        name=arg_name,
    )

  return jax_tree_util.tree_map(
      _copy_type,
      original_signature,
  )


def _get_converted_function_signature_def(
    fn_name: str,
    fn: tf.types.experimental.ConcreteFunction,
    trackable_resources: Any,
    converter_options: converter_options_v2_pb2.ConverterOptionsV2,
) -> meta_graph_pb2.SignatureDef:
  """Saves the function, converts it, returns its SignatureDef.

  Args:
    fn_name: The name of the function in the SavedModel.
    fn: The concrete function to save.
    trackable_resources: The trackable resources to save.
    converter_options: The converter options to use for the TF SavedModel.

  Returns:
    The SignatureDef of the converted function.
  """

  opts_copy = copy.deepcopy(converter_options)
  # There is no need to convert the checkpoint in this case, since we are only
  # interested in the signature.
  opts_copy.bfloat16_optimization_options.experimental.convert_checkpoint = (
      False
  )

  with tempfile.TemporaryDirectory() as temp_dir:
    save_tf_functions(
        temp_dir,
        {fn_name: fn},
        trackable_resources=trackable_resources,
        converter_options=opts_copy,
    )

    converted_model_path = os.path.join(temp_dir, OBM_TF_SAVED_MODEL_SUB_DIR)
    with open(os.path.join(converted_model_path, 'saved_model.pb'), 'rb') as f:
      saved_model_proto = saved_model_pb2.SavedModel.FromString(f.read())

    return saved_model_proto.meta_graphs[0].signature_def[fn_name]
