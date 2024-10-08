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

"""Checkpoint Metadata handler for Pathways."""

from typing import Optional, Any

from etils import epath
from jax import tree_util
import jax.numpy as jnp
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler


PyTree = base_pytree_checkpoint_handler.PyTree
BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)
BasePyTreeSaveArgs = base_pytree_checkpoint_handler.BasePyTreeSaveArgs
BasePyTreeRestoreArgs = base_pytree_checkpoint_handler.BasePyTreeRestoreArgs


class PathwaysPyTreeCheckpointHandler(BasePyTreeCheckpointHandler):
  """PathwaysPyTreeCheckpointHandler is used to save and restore metadata for Pathways."""

  def _encode_dict(self, data: Any) -> Any:
    """Encodes an argument as a Pytree using JAX.

    Encodes values as unicode characters.

    Args:
      data: A dictionary or Pytree where leaves are strings,
            numbers, or lists/tuples of numbers.

    Returns:
      An encoded Pytree with JAX arrays as leaves.
    """

    def _encode_leaf(value):
      if isinstance(value, str):
        return jnp.array(list(value.encode('utf-8')), dtype=jnp.uint8)
      elif isinstance(value, (int, float)):
        return jnp.array([value])  # Convert single numbers to an array
      elif isinstance(value, (list, tuple)):
        return jnp.array(value)
      else:
        raise ValueError(f'Unsupported value type: {type(value)}')

    return tree_util.tree_map(_encode_leaf, data)

  def _decode_pytree(self, encoded_data: PyTree) -> PyTree:
    """Decodes a Pytree encoded by this class.

    Args:
      encoded_data: An encoded Pytree with JAX arrays as leaves.

    Returns:
      A decoded Pytree.
    """

    def decode_leaf(value):
      if value.dtype == jnp.uint8:
        return value.tobytes().decode('utf-8')
      else:
        # Convert back to Python list, handling single-element arrays
        decoded_value = value.tolist()
        if isinstance(decoded_value, list) and len(decoded_value) == 1:
          return decoded_value[0]
        return decoded_value

    return tree_util.tree_map(decode_leaf, encoded_data)

  def save(self, directory: epath.Path, args: dict[str, Any]) -> None:
    """Saves the given item to the provided directory."""
    encoded_args = BasePyTreeSaveArgs(item=self._encode_dict(args))
    super().save(directory, encoded_args)

  def restore(
      self,
      directory: epath.Path,
      args: Optional[BasePyTreeRestoreArgs] = None,
  ) -> PyTree:
    """Restores the given item from the provided directory."""
    encoded_pytree = super().restore(directory, args)
    return self._decode_pytree(encoded_pytree.item)

