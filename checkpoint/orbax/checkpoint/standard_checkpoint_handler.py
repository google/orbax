# Copyright 2023 The Orbax Authors.
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

"""StandardCheckpointHandler class."""

import dataclasses
from typing import Any, List, Optional

from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import future
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import utils


PyTree = Any
CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler


class StandardCheckpointHandler(
    pytree_checkpoint_handler.PyTreeCheckpointHandler
):
  """A CheckpointHandler implementation for any PyTree structure.

  See JAX documentation for more information on what constitutes a "PyTree".
  This handler is capable of saving and restoring PyTrees with leaves of type
  Python scalar, np.ndarray, and jax.Array

  As with all `CheckpointHandler` subclasses, `StandardCheckpointHandler` should
  only be used in conjunction with a `Checkpointer` (or subclass). By itself,
  the `CheckpointHandler` is non-atomic.

  Example::

    ckptr = Checkpointer(StandardCheckpointHandler())
    # OR
    ckptr = StandardCheckpointer()

  If you find that your use case is not covered by `StandardCheckpointHandler`,
  consider using the parent class directly, or explore a custom implementation
  of `CheckpointHandler`.
  """

  def __init__(
      self,
      concurrent_gb: int = 96,
      use_ocdbt: bool = True,
      write_tree_metadata: bool = True,
  ):
    """Creates StandardCheckpointHandler.

    Args:
      concurrent_gb: max concurrent GB that are allowed to be read. Can help to
        reduce the possibility of OOM's when large checkpoints are restored.
      use_ocdbt: enables Tensorstore OCDBT driver. This option allows using a
        different checkpoint format which is faster to read and write, as well
        as more space efficient.
      write_tree_metadata: Writes tree metadata in JSON format. The tree
        metadata is used to enable a checkpoint which is fully self-describing.
    """
    super().__init__(
        concurrent_gb=concurrent_gb,
        use_ocdbt=use_ocdbt,
        write_tree_metadata=write_tree_metadata,
    )
    self._supported_types = checkpoint_utils.STANDARD_ARRAY_TYPES

  def _validate_save_state(
      self, state: PyTree, save_args: Optional[PyTree] = None
  ):
    if state is None:
      raise ValueError('Must provide state to save.')
    if save_args is None:
      save_args = jax.tree_util.tree_map(lambda x: None, state)

    def _check_input(k, x, arg):
      if arg is not None:
        if arg.aggregate:
          raise ValueError(f'Unsupported option `aggregate` for key: {k}.')
      if not isinstance(x, self._supported_types):
        k = utils.tuple_path_from_keypath(k)
        raise ValueError(f'Unsupported type: {type(x)} for key: {k}.')

    jax.tree_util.tree_map_with_path(_check_input, state, save_args)

  def _validate_restore_state(self, state: PyTree):
    def _check_input(k, x):
      if not isinstance(x, self._supported_types) and not isinstance(
          x, jax.ShapeDtypeStruct
      ):
        k = utils.tuple_path_from_keypath(k)
        raise ValueError(f'Unsupported type: {type(x)} for key: {k}.')

    jax.tree_util.tree_map_with_path(_check_input, state)

  async def async_save(
      self,
      directory: epath.Path,
      state: PyTree,
      save_args: Optional[PyTree] = None,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree. See superclass documentation."""
    self._validate_save_state(state, save_args=save_args)
    return await super().async_save(directory, state, save_args=save_args)

  def restore(
      self,
      directory: epath.Path,
      state: Optional[PyTree] = None,
  ) -> PyTree:  # pytype: disable=signature-mismatch
    """Restores a PyTree.

    Example::

      ckptr = StandardCheckpointer()
      state = {
          'layer0': {
              'w': jax.Array(...),
              'b': np.ndarray(...),
          },
      }
      ckptr.save(dir, state)

      target = {
          'layer0': {
              'w': jax.ShapeDtypeStruct(...),
              'b': jax.Array(...),
          },
      }
      ckptr.restore(dir, target)

    Args:
      directory: path from which to restore.
      state: target PyTree. Currently non-optional. Values may be either real
        array or scalar values, or they may be jax.ShapeDtypeStruct. If real
        values are provided, that value will be restored as the given type, with
        the given properties. If jax.ShapeDtypeStruct is provided, the value
        will be restored as np.ndarray, unless `sharding` is specified. If
        `state` is a custom PyTree class, the tree will be restored with the
        same structure as provided. If not provided, restores as a serialized
        nested dict representation of the custom class.

    Returns:
      a restore PyTree.
    """
    if state:
      self._validate_restore_state(state)
      restore_args = checkpoint_utils.construct_restore_args(state)
    else:
      restore_args = checkpoint_utils.construct_restore_args(
          self.metadata(directory)
      )
    return super().restore(
        directory,
        item=state,
        restore_args=restore_args,
    )


@register_with_handler(StandardCheckpointHandler)
@dataclasses.dataclass
class StandardSaveArgs(CheckpointArgs):
  """Parameters for saving a standard PyTree.

  Attributes:
    state (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `state`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
  """

  state: PyTree
  save_args: Optional[PyTree] = None


@register_with_handler(StandardCheckpointHandler)
@dataclasses.dataclass
class StandardRestoreArgs(CheckpointArgs):
  """Parameters for restoring a standard PyTree.

  If you require more flexibility, see `PyTreeRestore`.

  Attributes (all optional):
    state: target PyTree. Currently non-optional. Values may be either real
        array or scalar values, or they may be jax.ShapeDtypeStruct. If real
        values are provided, that value will be restored as the given type, with
        the given properties. If jax.ShapeDtypeStruct is provided, the value
        will be restored as np.ndarray, unless `sharding` is specified. If
        `state` is a custom PyTree class, the tree will be restored with the
        same structure as provided. If not provided, restores as a serialized
        nested dict representation of the custom class.
  """

  state: Optional[PyTree] = None
