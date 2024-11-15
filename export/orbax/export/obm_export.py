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

"""Export class that implements the save and load abstract class defined in Export Base for use with the Orbax Model export format."""

from typing import Any, cast

from absl import logging
from orbax.export import constants
from orbax.export import export_base
from orbax.export import jax_module as jax_module_lib
from orbax.export.modules import obm_module
import tensorflow as tf


class ObmExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using Orbax Model export."""

  def save(
      self,
      # TODO(b/363033166): Change this annotation once TF isolation is done.
      jax_module: tf.Module,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves a Jax model in the Orbax Model export format.

    Args:
      jax_module: The `JaxModule` to be exported.
      model_path: The path to save the model.
      **kwargs: Additional arguments to pass to the `save` method. Accepted
        arguments are `save_options` and `serving_signatures`.
    """

    # TODO(b/363033166): Remove this step once TF isolation is done.
    jax_module_: jax_module_lib.JaxModule = jax_module.computation_module

    if jax_module_.export_version() != constants.ExportModelType.ORBAX_MODEL:
      raise ValueError(
          "JaxModule is not of type ORBAX_MODEL. Please use the correct"
          " export_version. Expected ORBAX_MODEL, got"
          f" {jax_module_.export_version()}"
      )

  def load(self, model_path: str, **kwargs: Any):
    """Loads the model previously saved in the Orbax Model export format."""
    logging.info("Loading model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.load not implemented yet.")
