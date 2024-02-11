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

"""The definition of ValidationManager class."""
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union

from absl import logging
import jax
from orbax.export import jax_module
from orbax.export.serving_config import ServingConfig
from orbax.export.validate.validation_job import ValidationJob
from orbax.export.validate.validation_job import ValidationSingleJobResult
from orbax.export.validate.validation_report import ValidationReport
from orbax.export.validate.validation_report import ValidationReportOption
import tensorflow as tf


def _is_flat_dict(x):
  """Checks if x is flat dict."""
  if not isinstance(x, Mapping):
    return False
  return all(
      jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(v))
      for v in x.values()
  )


def _is_flat_sequence(x):
  """Checks if x is flat list."""
  if not isinstance(x, Sequence):
    return False
  return all(
      jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(v)) for v in x
  )


class ValidationManager:
  """Validate the JaxModule and its output tf saved model."""

  def __init__(
      self,
      module: Union[
          jax_module.JaxModule,
          Mapping[str, Callable[[jax_module.PyTree], jax_module.PyTree]],
      ],
      serving_configs: Sequence[ServingConfig],
      model_inputs: Union[Sequence[Any], Mapping[str, Sequence[Any]]],
  ):
    """Create the ValidationManager ojbect.

    Args:
      module: the JaxModule object.
      serving_configs: the ServingConfig Sequence.
      model_inputs: The inputs for saved TF SavedModel. It support two formats:
        (1) A mapping of signature key to a sequences batch inputs; or (2) a
        sequence of batch inputs to validate all signatures.
    """
    if isinstance(module, jax_module.JaxModule):
      self._jax_methods = module.jax_methods
    else:
      logging.warn(
          'Using Mapping[str, Callable] to initialize ValidationManager is'
          ' deprecated. Use JaxModule instead.'
      )
      self._jax_methods = module
    self._serving_configs = serving_configs
    self._model_inputs = model_inputs

  def _create_baseline_fns(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map from signature keys to validation functions."""
    validation_func_map = {}
    for sc in self._serving_configs:
      validation_func_map.update((sc.bind(self._jax_methods)))
    return validation_func_map

  def _create_candidate_fns(
      self, loaded_model: Any
  ) -> Mapping[str, Callable[..., Any]]:
    """Returns a map from signature keys to candidate functions.

    Args:
      loaded_model: The user should provided the loaded_model. For CPU,
        `loaded_model=tf.saved_model.load(tf_model_path, ['serve'])` for TPU,
        `loaded_model=tf.saved_model.load(tf_model_path, ['serve', 'tpu'])`
    """
    loaded_model_signatures = loaded_model.signatures
    candidate_func_map = {}

    def make_candidate_inference_fn(signature_key):
      def inference_fn(*inputs):
        if len(inputs) != 1:
          raise ValueError(
              'Currently does not accept multiple args, '
              f'got len(inputs)={len(inputs)}.'
          )
        real_inputs = inputs[0]
        real_inputs = jax.tree_util.tree_map(tf.convert_to_tensor, real_inputs)
        self.check_input(real_inputs, batch_mode=False)
        if isinstance(real_inputs, Mapping):
          outputs = loaded_model_signatures[signature_key](**real_inputs)
        elif isinstance(real_inputs, Sequence):
          outputs = loaded_model_signatures[signature_key](*real_inputs)
        else:
          outputs = loaded_model_signatures[signature_key](real_inputs)
        outputs = jax.tree_util.tree_map(lambda x: x.numpy(), outputs)
        return outputs

      return inference_fn

    for sc in self._serving_configs:
      for key in sc.get_signature_keys():
        candidate_func_map[key] = make_candidate_inference_fn(key)

    return candidate_func_map

  def _create_input_map(self) -> Mapping[str, Sequence[Any]]:
    """Converts batch input into Mapping[signature_key, batch input] if need."""
    if isinstance(self._model_inputs, Mapping):
      return self._model_inputs

    model_inputs = {}
    for sc in self._serving_configs:
      for key in sc.get_signature_keys():
        model_inputs[key] = self._model_inputs
    return model_inputs

  def validate(
      self,
      loaded_model: Any,
      with_xprof: bool = False,
      report_option: ValidationReportOption | None = None,
  ) -> Mapping[str, ValidationReport]:
    """Validates the baseline and candidate function map."""
    candidate_fns = self._create_candidate_fns(loaded_model)
    baseline_fns = self._create_baseline_fns()
    input_map = self._create_input_map()
    results = {}

    if not report_option:
      report_option = ValidationReportOption()

    for sc in self._serving_configs:
      for key in sc.get_signature_keys():
        validation_job = ValidationJob(
            baseline_fns[key], candidate_fns[key], input_map[key], with_xprof
        )
        baseline_result = validation_job.calc_baseline_result()
        candidate_result = validation_job.calc_candidate_result()
        # Always convert list to Dict
        baseline_result.maybe_convert_result_to_dict()
        candidate_result.maybe_convert_result_to_dict()
        self.check_output(baseline_result, candidate_result)
        results[key] = ValidationReport(
            baseline_result, candidate_result, report_option
        )
    return results

  @classmethod
  def check_input(
      cls, inputs: Union[Any, Sequence[Any]], batch_mode: bool = True
  ) -> None:
    """check model input format.

    Args:
      inputs: model inputs. If batch_mode == True, inputs should be a list.
      batch_mode: it decide `inputs` is a list of the input or a single input.
    """
    if batch_mode:
      if not isinstance(inputs, Sequence):
        raise ValueError('Batch inputs should be a python list.')
      test_inputs = inputs[0]
    else:
      test_inputs = inputs
    if not _is_flat_dict(test_inputs) and not _is_flat_sequence(test_inputs):
      logging.warning(
          (
              'Recommend Orbax validate inputs format as flat_dict or'
              ' flat_list, so it can generate the consistent tf.SavedModel'
              ' signature. For arbitrary format, we assume it as atomic data'
              ' type, it may fail. Got inputs format %s'
          ),
          type(test_inputs),
      )

  @classmethod
  def check_output(
      cls,
      baseline_result: ValidationSingleJobResult,
      candidate_result: ValidationSingleJobResult,
  ) -> None:
    """check model output format."""
    baseline_outputs = baseline_result.outputs
    candidate_outputs = candidate_result.outputs

    if not _is_flat_dict(baseline_outputs[0]):
      err_message = (
          'Currently ValidationReport only accept  flat dict outputs. '
          f' But we got {type(baseline_outputs[0])}'
      )
      raise ValueError(err_message)

    baseline_flat = jax.tree_util.tree_leaves(baseline_outputs)
    candidate_flat = jax.tree_util.tree_leaves(candidate_outputs)
    if len(baseline_flat) != len(candidate_flat):
      raise ValueError(
          'baseline and candidate has different output length.'
          f'len(baseline) = {len(baseline_flat)},'
          f'len(candidate) = {len(candidate_flat)}.'
      )
