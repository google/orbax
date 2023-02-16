# Copyright 2022 The Orbax Authors.
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

PyTree = jax.tree_util.PyTreeDef


@dataclasses.dataclass
class ServingConfig:
  """Configuration for constructing a serving signature for a JaxModule.

  A ServingConfig is to be bound with a JaxModule to form an end-to-end serving
  signature.
  """
  # The key of the serving signature or a sequence of keys mapping to the same
  # serving signature.
  signature_key: Union[str, Sequence[str]]
  # The input signature. Will infer input_signature is specified from
  # `tf_preprocessor` by default.
  input_signature: Optional[Sequence[Any]] = None
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
      infer_step_fns: Union[Callable[[PyTree], PyTree],
                            Mapping[str, Callable[[PyTree], PyTree]]],
      require_numpy=True) -> Mapping[str, Callable[..., Mapping[Text, Any]]]:
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

      def inference_fn(*preprocessed_inputs):
        if self.tf_preprocessor:
          inputs = tf.function(self.tf_preprocessor)(*preprocessed_inputs)
          if require_numpy:
            inputs = jax.tree_util.tree_map(lambda x: x.numpy(), inputs)
        else:
          inputs = preprocessed_inputs

          if len(preprocessed_inputs) != 1:
            raise ValueError('Currently does not accept multiple args, '
                             f'got len(inputs)={len(inputs)}.')

          inputs = preprocessed_inputs[0]

        # Currently Jax Module only takes 1 input
        outputs = infer_step(inputs)
        if self.tf_postprocessor:
          outputs = tf.function(self.tf_postprocessor)(outputs)
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
