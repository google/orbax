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

from typing import Any, Union, cast

from absl import logging
from orbax.export import constants
from orbax.export import export_base
from orbax.export.modules import obm_module
from orbax.export.modules import orbax_module_base
import tensorflow as tf


class ObmExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using Orbax Model export."""

  def save(
      self,
      jax_module: Union[orbax_module_base.OrbaxModuleBase, tf.Module],
      model_path: str,
  ):
    """Saves a Jax model in the Orbax Model export format.

    Args:
      jax_module: The `JaxModule` to be exported.
      model_path: The path to save the model.
      save_options: The `SaveOptions` to use when exporting the model.
    """

    if jax_module.export_version() != constants.ExportModelType.ORBAX_MODEL:
      raise ValueError(
          "JaxModule is not of type ORBAX_MODEL. Please use the correct"
          " export_version. Expected ORBAX_MODEL, got"
          f" {jax_module.export_version()}"
      )

  def load(self, model_path: str, **kwargs: Any):
    """Loads the model previously saved in the Orbax Model export format."""
    logging.info("Loading model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.load not implemented yet.")
