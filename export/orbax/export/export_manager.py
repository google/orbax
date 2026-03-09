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

"""Manage the exporting of a JAXModule."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, cast

from etils.epy import reraise_utils
from orbax.export import config
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_export
from orbax.export import serving_config as osc
from orbax.export import tensorflow_export
import tensorflow as tf

obx_export_config = config.config
maybe_reraise = reraise_utils.maybe_reraise


class ExportManager:
  """Exports a JAXModule with pre- and post-processors.

  This manager acts as a unified interface for exporting JAX modules. It
  handles the underlying serialization logic, dynamically routing to either
  Orbax-native export (`ObmExport`) or TensorFlow SavedModel export
  (`TensorFlowExport`) based on the configuration of the provided module.

  Example:
    Configure and export a JAX module using a specific serving configuration::

      import tensorflow as tf
      from orbax.export import ExportManager
      from orbax.export import serving_config

      # Assume `my_jax_module` is a fully initialized jax_module.JaxModule
      # Define how the model should handle incoming requests
      my_config = serving_config.ServingConfig(
          signature_key="serving_default",
          input_signature=[tf.TensorSpec(shape=(None, 32), dtype=tf.float32)],
      )

      # Initialize the manager
      export_mgr = ExportManager(
          module=my_jax_module,
          serving_configs=[my_config]
      )

      # Save the model to a directory
      export_mgr.save("/path/to/my/saved_model")
  """

  def __init__(
      self,
      module: jax_module.JaxModule | None,
      serving_configs: Sequence[osc.ServingConfig],
  ):
    """ExportManager constructor.

    Args:
      module: The `JaxModule` to be exported. Can be None in specific delayed
        initialization or native Orbax load scenarios.
      serving_configs: a sequence of which each element is a `ServingConfig`
        cooresponding to a serving signature of the exported SavedModel.
    """
    self._jax_module = module
    if (
        not self._jax_module
        or self._jax_module.export_version
        == constants.ExportModelType.ORBAX_MODEL
    ):
      self._serialization_functions = obm_export.ObmExport(
          self._jax_module, serving_configs
      )
    else:
      self._serialization_functions = tensorflow_export.TensorFlowExport(
          self._jax_module, serving_configs
      )

  @property
  def tf_module(self) -> tf.Module:
    """Returns the tf.module maintained by the export manager.

    Raises:
      TypeError: If the export version is `ExportModelType.ORBAX_MODEL` or if
        the module is not provided (as Orbax models do not use tf.Module).
    """
    if (
        not self._jax_module
        or self._jax_module.export_version
        == constants.ExportModelType.ORBAX_MODEL
    ):
      raise TypeError(
          'tf_module is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_export.TensorFlowExport,
        self._serialization_functions).tf_export_module()

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    return self._serialization_functions.serving_signatures

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
    self._serialization_functions.save(
        model_path=model_path,
        save_options=save_options,
        signature_overrides=signature_overrides,
    )

  def load(self, model_path: str, **kwargs: Any):
    """Loads the exported model from disk.

    Args:
      model_path: The directory from which to load the model.
      **kwargs: Additional keyword arguments passed to the underlying loader.

    Returns:
      The loaded model instance.
    """
    loaded = self._serialization_functions.load(model_path, **kwargs)
    return loaded
