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
import flax
import jax
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import utils

PyTreeDef = jax.tree_util.PyTreeDef


class AggregateHandler(abc.ABC):
  """Interface for reading and writing a PyTree using a specific format."""

  @abc.abstractmethod
  async def serialize(self, path: epath.Path, item: PyTreeDef):
    """Serializes and writes `item` to a given `path`.

    The function is compatible with a multihost setting, but does not include
    extra logic to ensure atomicity.

    Args:
      path: the folder to which the item should be written.
      item: a PyTree.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, path: epath.Path) -> PyTreeDef:
    """Reads and deserializes a PyTree from the given directory."""
    pass


class MsgpackHandler(AggregateHandler):
  """An implementation of AggregateHandler that uses msgpack to store the tree.
  """

  async def serialize(self, path: epath.Path, item: PyTreeDef):
    """See superclass documentation."""
    if jax.process_index() == 0:
      state_dict = flax.serialization.to_state_dict(item)
      msgpack = msgpack_utils.msgpack_serialize(state_dict)
      await utils.async_write_bytes(path, msgpack)

  def deserialize(self, path: epath.Path) -> PyTreeDef:
    """See superclass documentation."""
    if path.exists():
      msgpack = path.read_bytes()
      restored = utils.msgpack_restore(msgpack)
    else:
      raise FileNotFoundError(f'Checkpoint does not exist at {path}.')
    return restored


_AGGREGATE_HANDLER = None
_DEFAULT_AGGREGATE_HANDLER = MsgpackHandler()


def set_default_aggregate_handler(handler: AggregateHandler):
  """Sets the default AggregateHandler for use by PyTreeCheckpointHandler.

  Args:
    handler: an AggregateHandler implementation.
  """
  global _AGGREGATE_HANDLER
  _AGGREGATE_HANDLER = handler


def get_aggregate_handler() -> AggregateHandler:
  return _AGGREGATE_HANDLER or _DEFAULT_AGGREGATE_HANDLER
