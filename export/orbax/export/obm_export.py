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
from typing import Any

from absl import logging
from orbax.export import export_base
import tensorflow as tf


class ObmExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using Orbax Model export."""

  def save(
      self,
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

     # TODO(b/363061755): Implment the ObmExport.save method.
    logging.info("Exporting model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.save not implemented yet.")

  def load(self, model_path: str, **kwargs: Any):
    """Loads the model previously saved in the Orbax Model export format."""
    # TODO(b/363061755): Implment the ObmExport.load method.
    logging.info("Loading model using Orbax Export Model.")
    raise NotImplementedError("ObmExport.load not implemented yet.")
