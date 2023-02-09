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

"""Manage the exporting of a JAXModule."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from etils.epy.reraise_utils import maybe_reraise
from orbax.export.dtensor_utils import get_current_dtensor_mesh
from orbax.export.export_manager_base import ExportManagerBase
from orbax.export.jax_module import JaxModule
from orbax.export.serving_config import ServingConfig
import tensorflow as tf
from tensorflow.experimental import dtensor


class ExportManager(ExportManagerBase):
  """Exports a JAXModule with pre- and post-processors."""

  def __init__(self, module: JaxModule,
               serving_configs: Sequence[ServingConfig]):
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
    extra_trackables = []

    for sc in serving_configs:
      with maybe_reraise(f'Failed exporting signature_key={sc.signature_key} '):
        method = sc.get_infer_step(module.methods)
        concrete_fn = make_concrete_inference_fn(method, sc)
        if isinstance(sc.signature_key, str):
          keys = [sc.signature_key]
        else:
          keys = sc.signature_key
        for key in keys:
          if key in self._serving_signatures:
            raise ValueError(
                f'Duplicated key "{sc.signature_key}" in `serving_configs`.'
            )
          self._serving_signatures[key] = concrete_fn

        if sc.extra_trackable_resources is not None:
          extra_trackables.append(sc.extra_trackable_resources)

    self._module.extra_trackables = extra_trackables

  @property
  def tf_module(self) -> tf.Module:
    """Returns the tf.module maintained by the export manager."""
    return self._module

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    return self._serving_signatures

  def save(self,
           model_path: str,
           save_options: Optional[tf.saved_model.SaveOptions] = None):
    """Saves the JAX model to a Savemodel.

    Args:
      model_path: a directory in which to write the SavedModel.
      save_options: an optional tf.saved_model.SaveOptions for configuring save
        options.
    """
    save_options = save_options or tf.saved_model.SaveOptions()
    save_options.experimental_custom_gradients = (
        self._module.computation_module.with_gradient
    )
    tf.saved_model.save(
        self.tf_module,
        model_path,
        self.serving_signatures,
        options=save_options)

    if get_current_dtensor_mesh():
      # TODO(b/261191533): we can remove this once tf.saved_model.save is aware
      # of SPMD saving.
      dtensor.barrier(get_current_dtensor_mesh(), 'export done')

  def load(self, model_path: str, **kwargs: Any):
    loaded = tf.saved_model.load(model_path, **kwargs)
    return loaded


def make_concrete_inference_fn(
    model_fn: Callable[..., Any],
    serving_config: ServingConfig) -> Callable[..., Any]:
  """Creates an concrete end-to-end inference tf.function.

  Args:
    model_fn: a callable in TF context for the numeric computation.
    serving_config: a ServingConfig that defines the input sigature,
      pre-processor and post-processor of the inference function.

  Returns:
    A concrete tf.function for end-to-end inference.
  """
  input_signature = serving_config.input_signature
  if input_signature is None:
    if hasattr(serving_config.tf_preprocessor, 'input_signature'
              ) and serving_config.tf_preprocessor.input_signature is not None:
      input_signature = serving_config.tf_preprocessor.input_signature
    else:
      raise ValueError(
          (
              f'ServingConfig (key={serving_config.signature_key}) does not set'
              ' `input_signature`, and it cannot be inferred from'
              ' `tf_preprocessor.'
          ),
          ' Please set `input_signature` explictly.',
      )

  infer_step_func_map = serving_config.bind(model_fn, require_numpy=False)
  signature_key = serving_config.get_signature_keys()[0]
  inferece_tf_fn = tf.function(
      infer_step_func_map[signature_key], autograph=False, jit_compile=False)
  # TODO(b/239083475): tracing with an input signature is the most error-prone
  # step in export. We can additionally trace the pre-processor, core module and
  # the post-processor individually for better debuggability.
  return inferece_tf_fn.get_concrete_function(*input_signature)
