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

"""Provides definitions for AggregateHandler and implementations."""

import abc

from etils import epath
import jax

PyTreeDef = jax.tree_util.PyTreeDef


class AggregateHandler(abc.ABC):
  """Interface for reading and writing a PyTree using a specific format."""

  @abc.abstractmethod
  async def serialize(self, directory: epath.Path, item: PyTreeDef):
    """Serializes and writes `item` to a given `directory`.

    The function is compatible with a multihost setting, but does not include
    extra logic to ensure atomicity.

    Args:
      directory: the folder to which the item should be written.
      item: a PyTree.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, directory: epath.Path) -> PyTreeDef:
    """Reads and deserializes a PyTree from the given directory."""
    pass
