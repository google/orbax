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

"""Manage the exporting of a JAXModule."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Optional

from etils.epy import reraise_utils
from orbax.export import config
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_export
from orbax.export import serving_config as osc
from orbax.export import tensorflow_export
from orbax.export import utils
from orbax.export.modules import obm_module
import tensorflow as tf


obx_export_config = config.config
maybe_reraise = reraise_utils.maybe_reraise


class ExportManager:
  """Exports a JAXModule with pre- and post-processors."""

  def __init__(
      self,
      module: jax_module.JaxModule,
      serving_configs: Sequence[osc.ServingConfig],
      version: constants.ExportModelType = constants.ExportModelType.TF_SAVEDMODEL,
  ):
    """ExportManager constructor.

    Args:
      module: the `JaxModule` to be exported.
      serving_configs: a sequence of which each element is a `ServingConfig`
        cooresponding to a serving signature of the exported SavedModel.
      version: the version of the export format to use. Defaults to
        TF_SAVEDMODEL.
    """
    if version != module.export_version():
      raise ValueError(
          '`version` and `module.export_version()`'
          f' must be the same. The former is {version}. The latter is '
          f'{module.export_version()}.'
      )
    # TODO(b/363033166): Skip this step for OBM once TF isolation is done.
    self._module = tf.Module()
    self._module.computation_module = module
    self._serving_signatures = {}
    if version == constants.ExportModelType.ORBAX_MODEL:
      self.serialization_functions = obm_export.ObmExport()
      obm_module_ = module.orbax_module()
      if not isinstance(obm_module_, obm_module.ObmModule):
        raise ValueError(
            'module.orbax_module() must return an `ObmModule`. '
            f'Got type: {type(obm_module_)}'
        )
      # TODO(bdwalker): Let `ObmExport.__init__() do this `build()` step.
      obm_module_.build(serving_configs)
    else:
      self.serialization_functions = tensorflow_export.TensorFlowExport()
      # TODO(bdwalker): Let `TensorFlowExport.__init__() do this
      #   `process_serving_configs()` step.
      process_serving_configs(
          serving_configs,
          obx_export_config.obx_export_tf_preprocess_only,  # pytype: disable=attribute-error
          self._module,
          self._serving_signatures,
      )

  @property
  def tf_module(self) -> tf.Module:
    """Returns the tf.module maintained by the export manager."""
    return self._module

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    return self._serving_signatures

  def save(
      self,
      model_path: str,
      save_options: Optional[tf.saved_model.SaveOptions] = None,
      signature_overrides: Optional[Mapping[str, Callable[..., Any]]] = None,
  ):
    """Saves the JAX model to a Savemodel.

    Args:
      model_path: a directory in which to write the SavedModel.
      save_options: an optional tf.saved_model.SaveOptions for configuring save
        options.
      signature_overrides: signatures to override the self-maintained ones, or
        additional signatures to export.
    """
    serving_signatures = dict(self._serving_signatures)
    if signature_overrides:
      serving_signatures.update(signature_overrides)

    self.serialization_functions.save(
        jax_module=self._module,
        model_path=model_path,
        save_options=save_options,
        serving_signatures=serving_signatures,
    )

  def load(self, model_path: str, **kwargs: Any):
    loaded = self.serialization_functions.load(model_path, **kwargs)
    return loaded


def make_e2e_inference_fn(
    model_fn: Callable[..., Any],
    serving_config: osc.ServingConfig,
) -> Callable[..., Any]:
  """Creates an concrete end-to-end inference tf.function.

  Args:
    model_fn: a callable in TF context for the numeric computation.
    serving_config: a ServingConfig that defines the input sigature,
      pre-processor and post-processor of the inference function.

  Returns:
    A tf.function for end-to-end inference.
  """
  infer_step_func_map = serving_config.bind(model_fn, require_numpy=False)
  signature_key = serving_config.get_signature_keys()[0]
  return utils.with_default_args(
      infer_step_func_map[signature_key], serving_config.get_input_signature()
  )


def process_serving_configs(
    serving_configs: Sequence[osc.ServingConfig],
    obx_export_tf_preprocess_only: bool,
    module: tf.Module,
    serving_signatures: Dict[str, Callable[..., Any]],
):
  """Processes the serving functions into their TF wrapped concrete functions.

  The function will use the serving_configs and the methods defined in the
  provided module to populate the serving_signatures map with the concrete
  inference functions.

  In addition, if trackable resources are provided in the serving_configs,
  they will be added to the module's tf_trackable_resources property.

  Args:
    serving_configs: a sequence of which each element is a `ServingConfig`
      cooresponding to a serving signature of the exported SavedModel.
    obx_export_tf_preprocess_only: a boolean indicating whether to export only
      the preprocessor.
    module: A tf module  that will provide the method definitions. The module
      should have a JaxModule set as a computation_module property.
    serving_signatures: a map of signature keys to serving functions. This map
      will be populated by this function.
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
        method = sc.get_infer_step(module.computation_module.methods)
        inference_fn = make_e2e_inference_fn(method, sc)

      if isinstance(sc.signature_key, str):
        keys = [sc.signature_key]
      else:
        keys = sc.signature_key

      for key in keys:
        if key in serving_signatures:
          raise ValueError(
              f'Duplicated key "{sc.signature_key}" in `serving_configs`.'
          )
        serving_signatures[key] = inference_fn

      if sc.extra_trackable_resources is not None:
        tf_trackable_resources.append(sc.extra_trackable_resources)

    if len(serving_configs) == 1:
      # Make this module callable. Once exported, it can be loaded back in
      # python and the nested input structure will be preservered. In
      # contrast, signatures will flatten the TensorSpecs of the to kwargs.
      module.__call__ = inference_fn

  module.tf_trackable_resources = tf_trackable_resources
