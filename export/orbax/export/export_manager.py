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

"""Manage the exporting of a JAXModule."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from orbax.export import dtensor_utils
from orbax.export import export_manager_base
from orbax.export import jax_module
from orbax.export import utils
from orbax.export.serving_config import ServingConfig  # pylint: disable=g-importing-member
import tensorflow as tf
from tensorflow.experimental import dtensor


class ExportManager(export_manager_base.ExportManagerBase):
  """Exports a JAXModule with pre- and post-processors."""

  def __init__(
      self,
      module: jax_module.JaxModuleProtocol,
      serving_configs: Sequence[ServingConfig],
  ):
    """ExportManager constructor.

    Args:
      module: the `JaxModule` to be exported.
      serving_configs: a sequence of which each element is a `ServingConfig`
        cooresponding to a serving signature of the exported SavedModel.
    """
    # Creates a new tf.Module wrapping the JaxModule and extra trackable
    # resources.
    self._module = tf.Module()
    self._module.computation_module = module
    self._serving_signatures = {}
    tf_trackable_resources = []

    for sc in serving_configs:
      method = sc.get_infer_step(module.methods)
      inference_fn = make_e2e_inference_fn(method, sc)
      if isinstance(sc.signature_key, str):
        keys = [sc.signature_key]
      else:
        keys = sc.signature_key
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
        self.tf_module.__call__ = inference_fn

    self._module.tf_trackable_resources = tf_trackable_resources

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
    save_options = save_options or tf.saved_model.SaveOptions()
    save_options.experimental_custom_gradients = (
        self._module.computation_module.with_gradient
    )

    serving_signatures = dict(self.serving_signatures)
    if signature_overrides:
      serving_signatures.update(signature_overrides)

    tf.saved_model.save(
        self.tf_module, model_path, serving_signatures, options=save_options
    )

    if dtensor_utils.get_current_dtensor_mesh():
      # TODO(b/261191533): we can remove this once tf.saved_model.save is aware
      # of SPMD saving.
      dtensor.barrier(dtensor_utils.get_current_dtensor_mesh(), 'export done')

  def load(self, model_path: str, **kwargs: Any):
    loaded = tf.saved_model.load(model_path, **kwargs)
    return loaded


def make_e2e_inference_fn(
    model_fn: Callable[..., Any], serving_config: ServingConfig
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
