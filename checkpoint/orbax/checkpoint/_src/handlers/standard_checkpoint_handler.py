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

"""StandardCheckpointHandler class."""

import asyncio
import dataclasses
from typing import Any, List, Optional

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import future
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import tree as tree_utils
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler


PyTree = Any
CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler


class StandardCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
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
      *,
      save_concurrent_gb: int = 96,
      restore_concurrent_gb: int = 96,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
  ):
    """Creates StandardCheckpointHandler.

    Args:
      save_concurrent_gb: max concurrent GB that are allowed to be saved. Can
        help to reduce the possibility of OOM's when large checkpoints are
        saved.
      restore_concurrent_gb: max concurrent GB that are allowed to be restored.
        Can help to reduce the possibility of OOM's when large checkpoints are
        restored.
      multiprocessing_options: See orbax.checkpoint.options.
    """
    self._supported_types = checkpoint_utils.STANDARD_ARRAY_TYPES
    self._impl = pytree_checkpoint_handler.PyTreeCheckpointHandler(
        save_concurrent_gb=save_concurrent_gb,
        restore_concurrent_gb=restore_concurrent_gb,
        multiprocessing_options=multiprocessing_options,
    )

  def _validate_save_state(
      self, item: PyTree, save_args: Optional[PyTree] = None
  ):
    if item is None:
      raise ValueError('Must provide item to save.')
    if save_args is None:
      save_args = jax.tree.map(lambda x: None, item)

    def _check_input(k, x, arg):
      if arg is not None:
        if arg.aggregate:
          raise ValueError(f'Unsupported option `aggregate` for key: {k}.')
      if not isinstance(x, self._supported_types):
        k = tree_utils.tuple_path_from_keypath(k)
        raise ValueError(f'Unsupported type: {type(x)} for key: {k}.')

    jax.tree_util.tree_map_with_path(_check_input, item, save_args)

  def _validate_restore_state(self, item: PyTree):
    def _check_input(k, x):
      if not isinstance(x, self._supported_types) and not isinstance(
          x, jax.ShapeDtypeStruct
      ):
        k = tree_utils.tuple_path_from_keypath(k)
        raise ValueError(f'Unsupported type: {type(x)} for key: {k}.')

    jax.tree_util.tree_map_with_path(_check_input, item)

  async def async_save(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      save_args: Optional[PyTree] = None,
      args: Optional['StandardSaveArgs'] = None,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree of array-like objects. See PyTreeCheckpointHandler."""
    if isinstance(item, CheckpointArgs):
      raise ValueError(
          'Make sure to specify kwarg name `args=` when providing'
          ' `StandardSaveArgs`.'
      )
    if args is not None:
      item = args.item
      save_args = args.save_args

    self._validate_save_state(item, save_args=save_args)
    return await self._impl.async_save(
        directory,
        args=pytree_checkpoint_handler.PyTreeSaveArgs(
            item=item, save_args=save_args
        ),
    )

  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves the provided item synchronously."""

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio.run(async_save(directory, *args, **kwargs))

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      args: Optional['StandardRestoreArgs'] = None,
  ) -> PyTree:
    """Restores a PyTree. See PyTreeCheckpointHandler.

    Example::

      ckptr = StandardCheckpointer()
      item = {
          'layer0': {
              'w': jax.Array(...),
              'b': np.ndarray(...),
          },
      }
      ckptr.save(dir, StandardSaveArgs(item))

      target = {
          'layer0': {
              'w': jax.ShapeDtypeStruct(...),
              'b': jax.Array(...),
          },
      }
      ckptr.restore(dir, StandardRestoreArgs(target))

    Args:
      directory: path from which to restore.
      item: Deprecated, use `args`.
      args: `StandardRestoreArgs` (see below).

    Returns:
      a restored PyTree.
    """
    if isinstance(item, CheckpointArgs):
      raise ValueError(
          'Make sure to specify kwarg name `args=` when providing'
          ' `StandardRestoreArgs`.'
      )
    if not args:
      args = StandardRestoreArgs(item=item)
    if args.item is not None:
      self._validate_restore_state(args.item)
      restore_args = checkpoint_utils.construct_restore_args(args.item)
    else:
      logging.warning(
          '`StandardCheckpointHandler` expects a target tree to be provided for'
          ' restore. Not doing so is generally UNSAFE unless you know the'
          ' present topology to be the same one as the checkpoint was saved'
          ' under.'
      )
      restore_args = checkpoint_utils.construct_restore_args(
          self.metadata(directory)
      )
    return self._impl.restore(
        directory,
        args=pytree_checkpoint_handler.PyTreeRestoreArgs(
            item=args.item, restore_args=restore_args
        ),
    )

  def metadata(self, directory: epath.Path) -> PyTree:
    """Returns metadata about the saved item."""
    return self._impl.metadata(directory)

  def finalize(self, directory: epath.Path) -> None:
    self._impl.finalize(directory)

  def close(self):
    self._impl.close()


@register_with_handler(StandardCheckpointHandler, for_save=True)
@dataclasses.dataclass
class StandardSaveArgs(CheckpointArgs):
  """Parameters for saving a standard PyTree.

  Also see `PyTreeSave` for additional options.

  Attributes:
    item (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `item`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
  """

  item: PyTree
  save_args: Optional[PyTree] = None


@register_with_handler(StandardCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class StandardRestoreArgs(CheckpointArgs):
  """Parameters for restoring a standard PyTree.

  Also see `PyTreeRestore` for additional options.

  Attributes (all optional):
    item: target PyTree. Currently non-optional. Values may be either real
        array or scalar values, or they may be jax.ShapeDtypeStruct. If real
        values are provided, that value will be restored as the given type, with
        the given properties. If jax.ShapeDtypeStruct is provided, the value
        will be restored as np.ndarray, unless `sharding` is specified. If
        `item` is a custom PyTree class, the tree will be restored with the
        same structure as provided. If not provided, restores as a serialized
        nested dict representation of the custom class.
  """

  item: Optional[PyTree] = None
