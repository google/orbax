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

"""Abstract base class for different export classes."""

import abc
from typing import Any

import tensorflow as tf


class ExportBase(abc.ABC):
  """Abstract base class for different export classes."""

  # TODO: b/363033166 - Remove dependencies on TF in the base class.
  @abc.abstractmethod
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

  @abc.abstractmethod
  def load(self, model_path: str, **kwargs: Any) -> Any:
    """Loads the model.

    Args:
      model_path: The path to load the model.
      **kwargs: Additional arguments to pass to the `load` method.

    Returns:
      The loaded model in the format chosen during the export.  If
      TF_SAVED_MODEL is chosen, then a loaded SavedModel is returned. If
      ORBAX_MODEL is chosen, the return type will be in the OrbaxModel format.
    """
