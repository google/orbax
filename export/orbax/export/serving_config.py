# Copyright 2023 The Orbax Authors.
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

"""ServingConfig class."""
import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence, Text, Union

import jax
import tensorflow as tf

PyTree = Any


@dataclasses.dataclass
class TensorSpecWithDefault:
  """Extends tf.TensorSpec to hold a default value."""

  tensor_spec: tf.TensorSpec
  default_val: Any

  def __post_init__(self):
    if self.default_val is not None:
      if not tf.TensorSpec.from_tensor(
          tf.convert_to_tensor(
              self.default_val,
              self.tensor_spec.dtype,
          ),
          name=self.tensor_spec.name,
      ).is_subtype_of(self.tensor_spec):
        raise ValueError(
            f'TensorSpec {self.tensor_spec} is not compatible with'
            f' the default value {self.default_val}'
        )


@dataclasses.dataclass
class ServingConfig:
  """Configuration for constructing a serving signature for a JaxModule.

  A ServingConfig is to be bound with a JaxModule to form an end-to-end serving
  signature.
  """
  # The key of the serving signature or a sequence of keys mapping to the same
  # serving signature.
  signature_key: Union[str, Sequence[str]]
  # The input signature for `tf_preprocessor` (or the JaxModule method if there
  # is no `tf_preprocessor`). If not specified, this will be infered from
  # `tf_preprocessor`, in which case `tf_preprocessor` must be a tf.function
  # with `input_signature` annotation. See
  # https://www.tensorflow.org/api_docs/python/tf/function#input_signatures.
  input_signature: Optional[Sequence[PyTree]] = None
  # Optional pre-precessing function written in TF.
  tf_preprocessor: Optional[Callable[..., Any]] = None
  # Optional post-processing function written in TF.
  tf_postprocessor: Optional[Callable[..., Any]] = None
  # A nested structure of tf.saved_model.experimental.TrackableResource that are
  # used in `tf_preprocessor` and/or `tf_postprocessor`. If a TrackableResource
  # an attritute of the `tf_preprocessor` (or `tf_postprocessor`), and the
  # `tf_preprocessor` (or `tf_postprocessor`) is a tf.module,
  # the TrackableResource does not need to be in `extra_trackable_resources`.
  extra_trackable_resources: Any = None
  # Specify the key of the JAX method of the `JaxModule` to be bound
  # with this serving config. If unspecified, the `JaxModule` should have
  # exactly one method which will be used.
  method_key: Optional[str] = None

  def get_signature_keys(self) -> Sequence[str]:
    if isinstance(self.signature_key, str):
      return [self.signature_key]
    else:
      return self.signature_key

  def get_input_signature(self, required=True) -> Any:
    """Gets the input signature from the explict one or tf_preprocessor."""
    input_signature = self.input_signature
    if input_signature is None:
      if (
          hasattr(self.tf_preprocessor, 'input_signature')
          and self.tf_preprocessor.input_signature is not None
      ):
        input_signature = self.tf_preprocessor.input_signature

    if required and input_signature is None:
      raise ValueError(
          f'Neithr the ServingConfig (key={self.signature_key}) nor its'
          ' `tf_preprocessor` sets an `input_signature`, please set'
          ' `input_signature` explictly.',
      )
    return input_signature

  def get_infer_step(
      self, infer_step_fns: Union[Callable[..., Any],
                                  Mapping[str, Callable[..., Any]]]
  ) -> Callable[..., Any]:
    """Finds the right inference fn to be bound with the ServingConfig.

    Args:
      infer_step_fns: the method_key/infer_step dict. Ususally the user can pass
        `JaxModule.methods` here.

    Returns:
      method: the corresponding jax method of current ServingConfig.
    """
    if callable(infer_step_fns):
      return infer_step_fns

    method_key = self.method_key
    if method_key is None:
      if len(infer_step_fns) != 1:
        raise ValueError(
            '`method_key` is not specified in ServingConfig '
            f'"{self.signature_key}" and the infer_step_fns has more than one '
            f' methods: {list(infer_step_fns)}. Please specify '
            '`method_key` explictly.')
      (method,) = infer_step_fns.values()  # this is a tuple-destructuring
      return method
    else:
      if method_key not in infer_step_fns:
        raise ValueError(
            f'Method key "{method_key}" is not found in the infer_step_fns. '
            f'Available method keys: {list(infer_step_fns.keys())}.')
      return infer_step_fns[method_key]

  def bind(
      self,
      infer_step_fns: Union[
          Callable[[PyTree], PyTree], Mapping[str, Callable[[PyTree], PyTree]]
      ],
      require_numpy: bool = True,
  ) -> Mapping[str, Callable[..., Mapping[Text, Any]]]:
    """Returns an e2e inference function by binding a inference step function.

    Args:
      infer_step_fns:  An inference step function of a mapping of method key to
        inference step function. If it is a mapping, the function whose key
        matches the `method_key` of this ServingConfig will be used.  If Users
        only provide infer_step function, all `method_key`s use same infer_step
        function.
      require_numpy: Decide convert tf tensor to numpy after tf preprocess and
        tf postprocess. As a rule of thumb,  if infer_step is jax function, set
        it to True. if infer_step if tf function, set it to False.
    Return:
      func_map:  The mapping of serving signature to the inference function
        bound with the pre- and post-processors of this ServingConfig.
    """

    def make_inference_fn(infer_step):
      """Bind the preprocess, method and postproess together."""
      preprocessor = tf.function(self.tf_preprocessor or (lambda *a: a))
      postprocessor = tf.function(self.tf_postprocessor or (lambda *a: a))

      def inference_fn(*inputs):
        if self.tf_preprocessor:
          preprocessed_inputs = preprocessor(*inputs)
          if require_numpy:
            preprocessed_inputs = jax.tree_util.tree_map(
                lambda x: x.numpy(), preprocessed_inputs
            )
        else:
          preprocessed_inputs = inputs

          if len(preprocessed_inputs) != 1:
            raise ValueError(
                'JaxModule only takes single arg as the input, but got'
                f' len(inputs)={len(inputs)} from the preprocessor or input'
                ' signature. Please pack all inputs into one PyTree by'
                ' modifying the `input_signature` (if no `tf_preprocessor`) or'
                ' the ServingConfig.tf_preprocessor.'
            )

          preprocessed_inputs = preprocessed_inputs[0]

        # Currently Jax Module only takes 1 input
        outputs = infer_step(preprocessed_inputs)
        if self.tf_postprocessor:
          outputs = postprocessor(outputs)
          if require_numpy:
            outputs = jax.tree_util.tree_map(lambda x: x.numpy(), outputs)
        return outputs

      return inference_fn

    func_map = {}
    infer_fn_with_processors = make_inference_fn(
        self.get_infer_step(infer_step_fns))
    for key in self.get_signature_keys():
      func_map[key] = infer_fn_with_processors

    return func_map


def _split_tensor_specs(
    input_signature: PyTree,
) -> tuple[
    list[tf.TensorSpec],
    list[TensorSpecWithDefault],
    Callable[[list[Any], list[Any]], PyTree],
]:
  """Splits the TensorSpecs in the input signature into required and optional group.

  Args:
    input_signature: a PyTree of tensors specs.

  Returns:
    A tuple with:
      * Required group: a flattened list of `tf.TensorSpec`s.
      * Optional group: a flattened list of `TensorSpecWithDefault`s.
      * A function that unflattens the two output lists back to the original
      PyTree.
  """
  flat_tf_specs, treedef = jax.tree_util.tree_flatten(input_signature)

  required = []
  required_index = []
  optional = []
  optional_index = []

  for i, spec in enumerate(flat_tf_specs):
    if isinstance(spec, TensorSpecWithDefault):
      if spec.default_val is not None:
        optional.append(spec)
        optional_index.append(i)
      else:
        required.append(spec.tensor_spec)
        required_index.append(i)
    else:
      required.append(spec)
      required_index.append(i)

  def unflatten(required: list[Any], optional: list[Any]) -> PyTree:
    merged = [None for _ in range(len(required) + len(optional))]
    for i, index in enumerate(required_index):
      merged[index] = required[i]
    for i, index in enumerate(optional_index):
      merged[index] = optional[i]
    assert None not in merged
    return jax.tree_util.tree_unflatten(treedef, merged)

  return required, optional, unflatten


def with_default_args(
    tf_fn: Callable[..., Any],
    input_signature: Sequence[PyTree],
) -> tf.types.experimental.GenericFunction:
  """Creates a TF function with default args specified in `input_signature`.

  Args:
    tf_fn: the TF function.
    input_signature: the input signature. Even leaf is a tf.TensorSpec, or a
      orbax.export.TensorSpecWithDefault if the default value is specified.

  Returns:
    A tf function with default arguments. Note that the structure of the input
    signature may be different from the original one.
  """

  required_specs, optional_specs, unflatten = _split_tensor_specs(
      input_signature
  )
  if not optional_specs:
    return tf.function(
        tf_fn,
        input_signature=input_signature,
        jit_compile=False,
        autograph=False,
    )

  default_vals = [
      tf.convert_to_tensor(x.default_val, dtype=x.tensor_spec.dtype)
      for x in optional_specs
  ]
  optional_ts = [ts.tensor_spec for ts in optional_specs]

  @tf.function(
      input_signature=[required_specs, optional_ts],
      jit_compile=False,
      autograph=False,
  )
  def wrapped_fn(required, optional=default_vals):
    inputs = unflatten(required, optional)
    return tf_fn(*inputs)

  return wrapped_fn
