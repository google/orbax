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

"""Export class that implements the save and load abstract class defined in Export Base for use with the TensorFlow SavedModel export format."""

from typing import Any
from absl import logging
from orbax.export import export_base
import tensorflow as tf


class TensorFlowExport(export_base.ExportBase):
  """Defines the save and load methods for exporting a model using TensorFlow SavedModel."""

  def save(
      self,
      jax_module: tf.Module,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves the model.

    Args:
      jax_module: The `JaxModule` to be exported.
      model_path: The path to save the model.
      **kwargs: Additional arguments to pass to the `save` method. Accepted
        arguments are `save_options` and `serving_signatures`.
    """

    logging.info('Exporting model using TensorFlow SavedModel.')
    save_options = (
        kwargs['save_options']
        if 'save_options' in kwargs and kwargs['save_options'] is not None
        else tf.saved_model.SaveOptions()
    )

    save_options.experimental_custom_gradients = (
        jax_module.computation_module.with_gradient
    )

    serving_signatures = (
        kwargs['serving_signatures'] if 'serving_signatures' in kwargs else {}
    )

    tf.saved_model.save(
        jax_module,
        model_path,
        serving_signatures,
        options=save_options,
    )

  def load(self, model_path: str, **kwargs: Any) -> Any:
    """Loads the model previously saved in the TensorFlow SavedModel format."""
    logging.info('Loading model using TensorFlow SavedModel.')
    return tf.saved_model.load(model_path, **kwargs)
