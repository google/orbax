# Copyright 2025 The Orbax Authors.
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
from typing import Any, Callable, Mapping, Sequence
from orbax.export import jax_module
from orbax.export import serving_config as osc


class ExportBase(abc.ABC):
  """Abstract base class for different export classes."""

  def __init__(
      self,
      module: jax_module.JaxModule,
      serving_configs: Sequence[osc.ServingConfig],
  ):
    self._module = module
    self._serving_configs = serving_configs

  @abc.abstractmethod
  def serving_signatures(self) -> Mapping[str, Callable[..., Any]]:
    """Returns a map of signature keys to serving functions."""

  @abc.abstractmethod
  def save(
      self,
      model_path: str,
      **kwargs: Any,
  ):
    """Saves the model.

    Args:
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
