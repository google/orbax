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
from typing import Any, Callable, Dict, Optional, cast

from etils.epy import reraise_utils
from orbax.export import config
from orbax.export import constants
from orbax.export import jax_module
from orbax.export import obm_export
from orbax.export import serving_config as osc
from orbax.export import tensorflow_export
from orbax.export.modules import obm_module
from orbax.export.modules import tensorflow_module
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
    self._version = version
    self._jax_module = module
    if self._version == constants.ExportModelType.ORBAX_MODEL:
      self.serialization_functions = obm_export.ObmExport(
          self._jax_module, serving_configs
      )
      obm_module_ = module.orbax_module()
      if not isinstance(obm_module_, obm_module.ObmModule):
        raise ValueError(
            'module.orbax_module() must return an `ObmModule`. '
            f'Got type: {type(obm_module_)}'
        )
      # TODO(bdwalker): Let `ObmExport.__init__() do this `build()` step.
      obm_module_.build(serving_configs)
    else:
      self.serialization_functions = tensorflow_export.TensorFlowExport(
          self._jax_module, serving_configs
      )

  @property
  def tf_module(self) -> tf.Module:
    """Returns the tf.module maintained by the export manager."""
    if self._version == constants.ExportModelType.ORBAX_MODEL:
      raise TypeError(
          'tf_module is not implemented for export version'
          ' ExportModelType.ORBAX_MODEL.'
      )
    return cast(
        tensorflow_module.TensorFlowModule,
        self._jax_module.export_module(),
    )

  @property
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""
    return self.serialization_functions.serving_signatures

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
    self.serialization_functions.save(
        model_path=model_path,
        save_options=save_options,
        signature_overrides=signature_overrides,
    )

  def load(self, model_path: str, **kwargs: Any):
    loaded = self.serialization_functions.load(model_path, **kwargs)
    return loaded
