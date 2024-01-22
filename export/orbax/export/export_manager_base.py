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

"""ExportManager public API interface."""

import abc


class ExportManagerBase(abc.ABC):
  """Define the base class API which manages the exporting of a model."""

  @abc.abstractmethod
  def save(self, model_path: str, **kwargs):
    """Saves the JAX model to a Savemodel."""

  @abc.abstractmethod
  def load(self, model_path: str, **kwargs):
    """Load the model from a Savemodel path."""
