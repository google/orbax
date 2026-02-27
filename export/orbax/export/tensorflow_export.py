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

"""Export class that implements the save and load abstract class defined in Export Base for use with the TensorFlow SavedModel export format."""

from collections.abc import Callable, Mapping, Sequence
import os
import tempfile
from typing import Any

from absl import logging
from etils.epy import reraise_utils
from orbax.export import config
from orbax.export import export_base
from orbax.export import jax_module
from orbax.export import serving_config as osc
from orbax.export import utils
import tensorflow as tf

obx_export_config = config.config
maybe_reraise = reraise_utils.maybe_reraise


class TensorFlowExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using TensorFlow SavedModel."""

  def __init__(
      self,
      module: jax_module.JaxModule,
      serving_configs: Sequence[osc.ServingConfig],
  ):
    self._tf_module = tf.Module()
    self._computation_module = module
    self._serving_signatures = {}
    self._process_serving_configs(
        serving_configs,
        obx_export_config.obx_export_tf_preprocess_only,  # pytype: disable=attribute-error
    )

  def save(
      self,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves the model.

    Args:
      model_path: The path to save the model.
      **kwargs: Additional arguments to pass to the `save` method. Accepted
        arguments are `save_options`,`serving_signatures`, and
        `tree_verity_options`.
    """

    logging.info('Exporting model using TensorFlow SavedModel.')
    save_options = (
        kwargs['save_options']
        if 'save_options' in kwargs and kwargs['save_options'] is not None
        else tf.saved_model.SaveOptions()
    )
    if not isinstance(save_options, tf.saved_model.SaveOptions):
      raise ValueError(
          'save_options must be of type tf.saved_model.SaveOptions. Got type: '
          f'{type(save_options)}'
      )

    save_options.experimental_custom_gradients = (
        self._computation_module.with_gradient
    )

    serving_signatures = dict(self._serving_signatures)
    signature_overrides = (
        kwargs['signature_overrides']
        if 'signature_overrides' in kwargs and kwargs['signature_overrides']
        else {}
    )

    if signature_overrides:
      serving_signatures.update(signature_overrides)

    converter_options = kwargs.get('inference_converter_options')
    if converter_options:
      with tempfile.TemporaryDirectory() as tmpdir:
        tf_model_path = tmpdir
        tf.saved_model.save(
            self._tf_module,
            tf_model_path,
            serving_signatures,
            options=save_options,
        )
    else:
      tf_model_path = model_path
      tf.saved_model.save(
          self._tf_module,
          tf_model_path,
          serving_signatures,
          options=save_options,
      )

  def load(self, model_path: str, **kwargs: Any) -> Any:
    """Loads the model previously saved in the TensorFlow SavedModel format."""
    logging.info('Loading model using TensorFlow SavedModel.')
    return tf.saved_model.load(model_path, **kwargs)

  def tf_export_module(self) -> tf.Module:
    """Returns the tf.Module that was exported."""
    return self._tf_module

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""

    return self._serving_signatures

  def _process_serving_configs(
      self,
      serving_configs: Sequence[osc.ServingConfig],
      obx_export_tf_preprocess_only: bool,
  ):
    """Processes the serving functions into their TF wrapped concrete functions.

    The function will use the serving_configs and the methods defined in the
    provided module to populate the serving_signatures map with the concrete
    inference functions.

    In addition, if trackable resources are provided in the serving_configs,
    they will be added to the module's tf_trackable_resources property.

    Args:
      serving_configs: a sequence of which each element is a `ServingConfig`
        corresponding to a serving signature of the exported SavedModel.
      obx_export_tf_preprocess_only: a boolean indicating whether to export only
        the preprocessor.
    """
    tf_trackable_resources = []
    for sc in serving_configs:
      with maybe_reraise(f'Failed exporting signature_key={sc.signature_key} '):
        if obx_export_tf_preprocess_only:
          if not sc.tf_preprocessor:
            raise ValueError(
                'serving_config.tf_preprocessor must be provided when'
                ' in `obx_export_tf_preprocess_only` mode.'
            )

          def tf_preprocessor(*inputs):
            return tf.nest.flatten(sc.tf_preprocessor(*inputs))  # pylint: disable=cell-var-from-loop

          preprocessor = utils.with_default_args(
              tf_preprocessor, sc.get_input_signature()
          )
          inference_fn = preprocessor
        else:
          method = sc.get_infer_step(self._computation_module.methods)
          inference_fn = utils.make_e2e_inference_fn(method, sc)

        keys = sc.get_signature_keys()

        for key in keys:
          if key in self._serving_signatures:
            raise ValueError(
                f'Duplicated key "{sc.signature_key}" in `serving_configs`.'
            )
          self._serving_signatures[key] = inference_fn

        if sc.extra_trackable_resources is not None:
          tf_trackable_resources.append(sc.extra_trackable_resources)

      if len(serving_configs) == 1:
        # Make this module callable. Once exported, it can be loaded back in
        # python and the nested input structure will be preservered. In
        # contrast, signatures will flatten the TensorSpecs of the to kwargs.
        self._tf_module.__call__ = inference_fn

    # Use a single tf.Module to track all the trackable resources.
    self._tf_module.tf_trackable_resources = tf_trackable_resources
    self._tf_module.tf_var_leaves = getattr(
        self._computation_module.export_module(), '_tf_var_leaves'
    )
