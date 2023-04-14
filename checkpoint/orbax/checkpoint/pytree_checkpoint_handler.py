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

"""PyTreeCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

import asyncio
import dataclasses
import functools
import re
from typing import Any, Callable, List, Optional, Tuple, Union

from etils import epath
import jax
from jax.experimental.gda_serialization import serialization
import numpy as np
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import lazy_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.future import Future
import tensorstore as ts


PyTree = Any
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
SaveArgs = type_handlers.SaveArgs
ParamInfo = type_handlers.ParamInfo
TypeHandler = type_handlers.TypeHandler
AggregateHandler = aggregate_handlers.AggregateHandler
MsgpackHandler = aggregate_handlers.MsgpackHandler
TransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]
Transform = transform_utils.Transform
LazyValue = lazy_utils.LazyValue

_TYPE_METADATA_FILE = 'type_metadata'
_CHECKPOINT_FILE = 'checkpoint'


async def _create_param_save_dir(param_info: ParamInfo, args: SaveArgs):
  # Directory will be unused.
  path = param_info.path
  if path is None or args.aggregate:
    return
  if jax.process_index() == 0:
    await utils.async_makedirs(path)


def _maybe_set_default_save_args(value, args):
  # If already set, return.
  if isinstance(args, SaveArgs):
    return args
  aggregate = not type_handlers.has_type_handler(type(value))
  return SaveArgs(aggregate=aggregate)


def _maybe_set_default_restore_args(args):
  if isinstance(args, RestoreArgs):
    return args
  return RestoreArgs(restore_type=None)


def _try_array_cast(arr, dtype):
  if dtype is not None:
    if utils.is_scalar(arr):
      arr = np.asarray(arr).astype(dtype).item()
    else:
      if hasattr(arr, 'astype'):
        arr = arr.astype(dtype)
  return arr


def _get_param_names(item: PyTree) -> PyTree:
  """Gets parameter names for PyTree elements."""
  def _param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
    return '.'.join([str(utils.get_key_name(k)) for k in keypath])

  return jax.tree_util.tree_map_with_path(
      lambda kp, _: _param_name_from_keypath(kp),
      item,
      is_leaf=utils.is_empty_or_leaf,
  )


def _get_param_infos_from_structure(directory: epath.Path,
                                    structure: PyTree) -> PyTree:
  """Construct ParamInfos based on a PyTree."""
  names = _get_param_names(structure)
  is_ocdbt_checkpoint = type_handlers.is_ocdbt_checkpoint(directory)

  def _get_param_info(name: str, leaf: Any) -> ParamInfo:
    if utils.leaf_is_placeholder(leaf):
      # Leaf is a param name.
      path = directory / utils.name_from_leaf_placeholder(leaf)
    # The following is kept for backwards compatibility.
    elif isinstance(leaf, ts.Spec):
      tspec = leaf.to_json()  # pytype: disable=attribute-error
      # Skip '.', since we need special regex handling for this char.
      pattern = r'\.' + utils.TMP_DIR_SUFFIX[1:] + r'\d+'
      path = re.sub(pattern, '', tspec['kvstore']['path'])
    elif utils.is_supported_empty_aggregation_type(leaf):
      return leaf  # Empty node, ParamInfo should not be returned.
    elif utils.is_supported_aggregation_type(leaf):
      # Value already restored, do not need ts.Spec.
      path = None
    else:
      raise ValueError(f'Unsupported type: {type(leaf)}')
    return ParamInfo(
        name=name,
        path=path,
        aggregate=(path is None),
        is_ocdbt_checkpoint=is_ocdbt_checkpoint,
    )

  return jax.tree_util.tree_map(_get_param_info, names, structure)


def _get_tree_for_aggregation(param_infos, save_args, item):
  """Get tree for aggregated checkpoint."""

  def _get_leaf_for_aggregation(param_info, arg, value):
    if arg.aggregate:  # Param was aggregated, return value after cast.
      if not utils.is_supported_aggregation_type(value):
        value = None
      return _try_array_cast(value, arg.dtype)
    else:  # Placeholder string for non-aggregated value.
      return utils.leaf_placeholder(param_info.name)

  return jax.tree_util.tree_map(
      _get_leaf_for_aggregation, param_infos, save_args, item
  )


def _transform_structure(
    item: PyTree,
    restored: PyTree,
    transforms: Optional[PyTree],
    transforms_default_to_original: bool,
) -> PyTree:
  """Optionally transforms the restored PyTree to the structure of `item`.

  Args:
    item: a PyTree representing the result structure ("new tree structure").
    restored: a PyTree representing the original tree structure. Note: this is a
      tree of LazyValues.
    transforms: provides instructions on how to transform the input trees. See
      transform_utils.
    transforms_default_to_original: See transform_utils.

  Returns:
    A transformed PyTree.
  """
  if item is None:
    if transforms is not None:
      msg = ('If providing `transforms`, must provide `item` matching structure'
             ' of expected result.')
      raise ValueError(msg)
    item = restored
  else:
    if transforms is None:
      item = utils.deserialize_tree(item, restored)
    else:
      transforms = _construct_lazy_transform_wrappers(transforms)
      item = transform_utils.apply_transformations(
          restored, transforms, item, transforms_default_to_original)
  return item


def _construct_lazy_transform_wrappers(transforms: PyTree) -> PyTree:
  """Constructs wrapper functions for user-provided to handle `LazyValue`.

  User-provided value-based transformation functions are written in terms of
  real values, not `LazyValue`. However, we want to be able to load all
  parameters as `LazyValue`, so as to avoid materializing unneeded parameters
  until necessary. As such, we construct wrapper functions which accept
  `LazyValue` and materialize the value before providing it to the user-defined
  function.

  Args:
    transforms: User-provided tree of Transform objects.

  Returns:
    A tree of Transform objects which is roughly the same as the input, but
    where all instances of `value_fn` or `multi_value_fn` are wrapped to accept
    `LazyValue` as input and return `LazyValue` as output.
  """

  def _maybe_wrap_transform(transform: Transform):
    async def _lazy_value_fn(lazy_value: LazyValue, args: RestoreArgs) -> Any:
      # `lazy_value` is the value over which the function is performed.
      # `args` is `RestoreArgs`, which is used to materialize the value.
      value = await lazy_value.get_async(args=args)
      return transform.value_fn(value)

    async def _lazy_multi_value_fn(
        transform_key: str, tree: PyTree, args: RestoreArgs
    ) -> Any:
      # `original_tree` consists of `LazyValue`s.
      # `args` is unused, since relevant restoration args come from
      # multi_value_fn_input_args.
      del args
      if not transform.multi_value_fn_input_args:
        raise ValueError(
            '`multi_value_fn` was specified, but `multi_value_fn_input_args`'
            ' were not. The latter must be specified to identify inputs for the'
            ' function.'
        )

      async def _materialize_lazy_tree_value(
          keypath: Tuple[str], lazy_value: LazyValue
      ) -> Union[Any, LazyValue]:
        key = '/'.join([str(utils.get_key_name(k)) for k in keypath])
        for (
            input_key,
            restore_args,
        ) in transform.multi_value_fn_input_args.items():
          if re.fullmatch(input_key, key):
            return await lazy_value.get_async(args=restore_args)
        return lazy_value  # Should not be used, so do not materialize.

      futures, treedef = jax.tree_util.tree_flatten(
          jax.tree_util.tree_map_with_path(_materialize_lazy_tree_value, tree)
      )
      flat_tree = await asyncio.gather(*futures)
      return transform.multi_value_fn(
          transform_key, jax.tree_util.tree_unflatten(treedef, flat_tree)
      )

    def _wrap_as_lazy_value(func, *args, **kwargs):
      return LazyValue(functools.partial(func, *args, **kwargs))

    # Only needed values are carried over to the wrapped Transform.
    if transform.value_fn is not None:
      return Transform(
          original_key=transform.original_key,
          value_fn=functools.partial(_wrap_as_lazy_value, _lazy_value_fn),
      )
    elif transform.multi_value_fn is not None:
      return Transform(
          multi_value_fn=functools.partial(
              _wrap_as_lazy_value, _lazy_multi_value_fn
          )
      )
    else:
      return transform

  return jax.tree_util.tree_map(_maybe_wrap_transform, transforms)


class PyTreeCheckpointHandler(AsyncCheckpointHandler):
  """A CheckpointHandler implementation for any PyTree structure.

  The PyTree is assumed to be a nested dictionary with array values represented
  as array-like objects (see type_handlers for supported objects). If not
  `jax.Array`, arrays are expected to be fully replicated.
  """

  def __init__(
      self,
      aggregate_filename: Optional[str] = None,
      concurrent_gb: int = 96,
      use_ocdbt: bool = False,
  ):
    """Creates PyTreeCheckpointHandler.

    Args:
      aggregate_filename: name that the aggregated checkpoint should be saved
        as.
      concurrent_gb: max concurrent GB that are allowed to be read.
      use_ocdbt: enables Tensorstore OCDBT driver.
    """
    self._aggregate_handler = aggregate_handlers.get_aggregate_handler()
    if aggregate_filename is None:
      aggregate_filename = _CHECKPOINT_FILE
    self._aggregate_filename = aggregate_filename
    self._concurrent_gb = concurrent_gb
    self._use_ocdbt = use_ocdbt

  def _get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    return _get_param_names(item)

  def _get_param_infos(self, item: PyTree, directory: epath.Path,
                       save_args: PyTree) -> PyTree:
    """Returns parameter information for elements in `item`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.
      save_args: PyTree matching item containing SaveArgs.

    Returns:
      A PyTree matching `item` of ParamInfo.
    """
    if not item:
      raise ValueError('Found empty item')
    names = self._get_param_names(item)

    def _param_info(name, args):
      return ParamInfo(
          name=name, path=(directory / name), aggregate=args.aggregate)

    return jax.tree_util.tree_map(_param_info, names, save_args)

  async def _write_aggregate_file(self, directory: epath.Path, item: PyTree,
                                  param_infos: PyTree, save_args: PyTree):
    ser_item = _get_tree_for_aggregation(param_infos, save_args, item)
    await self._aggregate_handler.serialize(
        directory / self._aggregate_filename, ser_item)

  async def async_save(
      self,
      directory: epath.Path,
      item: PyTree,
      save_args: Optional[PyTree] = None) -> Optional[List[Future]]:
    """Saves a PyTree from a given training step.

    This operation is compatible with a multi-host, multi-device setting. Tree
    leaf values must be supported by type_handlers. Standard supported types
    include scalars, np.ndarray, jax.Array, string.

    After saving, all files will be located in directory/

    Saves an additional file to directory/checkpoint on host 0 which
    contains the serialized structure of `item`, along with any parameters that
    request aggregation.

    Args:
      directory: save location directory.
      item: a PyTree to be saved.
      save_args: a PyTree matching `item` which consists of SaveArgs objects as
        values.

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    item = await lazy_utils.maybe_get_tree_async(item)

    # Because of empty states, the user-provided args may not contain
    # all necessary arguments. These should be filled in with default args.
    save_args = jax.tree_util.tree_map(
        _maybe_set_default_save_args,
        item,
        item if save_args is None else save_args,
        is_leaf=utils.is_empty_or_leaf,
    )
    param_infos = self._get_param_infos(item, directory, save_args)
    if not self._use_ocdbt:
      # Create directories in parallel.
      await asyncio.gather(
          *jax.tree_util.tree_flatten(
              jax.tree_util.tree_map(
                  _create_param_save_dir, param_infos, save_args
              )
          )[0]
      )
      utils.sync_global_devices(
          'PyTreeCheckpointHandler:create_param_save_dirs'
      )

    async def serialize(value, info, args):
      if args.aggregate:
        return  # skip serialize now, include in aggregated file
      handler = type_handlers.get_type_handler(type(value))
      return await handler.serialize(value, info, args)

    future_tree = jax.tree_util.tree_map(
        serialize, item, param_infos, save_args, is_leaf=utils.is_empty_or_leaf
    )
    copy_futures, _ = jax.tree_util.tree_flatten(future_tree)
    assert isinstance(copy_futures, list)
    # Await copy futures.
    commit_futures = await asyncio.gather(*copy_futures)
    commit_futures, _ = jax.tree_util.tree_flatten(commit_futures)
    await self._write_aggregate_file(directory, item, param_infos, save_args)
    return commit_futures

  def save(self, directory: epath.Path, item: Any, *args, **kwargs):
    """Saves the provided item.

    Blocks until both copy and commit complete.

    See async_save.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for future in commit_futures:
          future.result()  # Block on result.

    asyncio.run(async_save(directory, item, *args, **kwargs))
    utils.sync_global_devices('PyTreeCheckpointHandler:save')

  def _maybe_deserialize(self, param_info: ParamInfo, value: Any) -> LazyValue:
    """Deserializes using handler or returns already restored value.

    If the ParamInfo indicates that the parameter was aggregated, then it must
    have already been restored. In this case, we simply perform a cast and
    convert to LazyArray if requested.

    Otherwise, we deserialize using an appropriate TypeHandler, converting to
    LazyArray and casting if requested.

    Args:
      param_info: ParamInfo
      value: a tree value which may have already been restored. Not relevant if
        info.aggregate is False.

    Returns:
      A LazyValue.
    """

    async def _maybe_cast(val: Any, args: RestoreArgs) -> Any:
      return _try_array_cast(val, args.dtype)

    async def _deserialize(info: ParamInfo, args: RestoreArgs) -> Any:
      handler = type_handlers.get_type_handler(args.restore_type)
      return await handler.deserialize(info, args)

    if param_info.aggregate:  # Already restored from AggregateHandler.
      get_fn = functools.partial(_maybe_cast, val=value)
    else:
      get_fn = functools.partial(_deserialize, info=param_info)

    return LazyValue(get_fn)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      restore_args: Optional[PyTree] = None,
      transforms: Optional[PyTree] = None,
      transforms_default_to_original: bool = True,
      transform_fn: Optional[TransformFn] = None,
  ) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given step.

    Optional arguments meshes and mesh_axes define how each array in the
    restored tree should be partitioned. For more information, see below and see
    pjit documentation.

    Args:
      directory: save location directory.
      item: provides the structure for the restored item. If not provided, will
        infer the structure from the saved checkpoint. Transformations will not
        be run.
      restore_args: optional object containing additional arguments for
        restoration. It should be a PyTree matching the structure of `item`, and
        should contain a RestoreArgs object for every value. If `item` is not
        provided, should match the structure of the checkpoint.
      transforms: a PyTree of transformations that should be applied to the
        saved item in order to obtain a final structure. See `transform_utils`
        for further information.
      transforms_default_to_original: See transform_utils.apply_transformations.
      transform_fn: A function which accepts the `item` argument, a PyTree
        checkpoint structure and a PyTree of ParamInfos based on the checkpoint.
        Returns a transformed PyTree matching the desired return tree structure,
        and a matching ParamInfo tree.

    Returns:
      A PyTree matching the structure of `item`. If `lazy` restoration is
      enabled, leaves will be returned as `LazyValue`.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `multi_value_fn`.
    """
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}')

    structure = self.structure(directory)
    param_infos = _get_param_infos_from_structure(directory, structure)

    if transform_fn is not None and transforms is not None:
      raise ValueError('Cannot provide both `transforms` and `transform_fn`.')
    if transform_fn is not None:
      structure, param_infos = transform_fn(item, structure, param_infos)

    async def _create_byte_limiter():
      # Wrap creation in async function to avoid issues on python<=3.9.
      concurrent_bytes = self._concurrent_gb * 10**9
      # Construction must take place here so that it is within the same async
      # method, to prevent errors resulting from different event loops, and
      # cannot be created below this level because there must be a single object
      # for the entire restore call.
      return serialization._LimitInFlightBytes(concurrent_bytes)  # pylint: disable=protected-access

    byte_limiter = asyncio.run(_create_byte_limiter())
    param_infos = jax.tree_util.tree_map(
        functools.partial(dataclasses.replace, byte_limiter=byte_limiter),
        param_infos,
    )
    lazy_restored_item = jax.tree_util.tree_map(
        self._maybe_deserialize,
        param_infos,
        structure,
    )

    if not transform_fn:
      lazy_restored_item = _transform_structure(
          item, lazy_restored_item, transforms, transforms_default_to_original
      )

    if restore_args is None:
      restore_args = jax.tree_util.tree_map(
          lambda x: RestoreArgs(),
          item or structure,
      )

    def _maybe_get_materialization_function(
        value: Union[LazyValue, Any], args: RestoreArgs
    ) -> Any:
      # Depending on the value of args.lazy, we either return a function that
      # allows materializing the value, or return a function that returns
      # another LazyValue, after passing in RestoreArgs.
      if args.lazy:
        if isinstance(value, LazyValue):
          async_get_fn = functools.partial(value.get_async, args=args)
        else:
          async_get_fn = lazy_utils.identity(value)
        return lazy_utils.identity(LazyValue(async_get_fn))()
      else:
        return lazy_utils.maybe_get_async(value, args=args)

    async def _restore():
      # Provide RestoreArgs now, since it was previously deferred.
      flat, item_structure = jax.tree_util.tree_flatten(
          jax.tree_util.tree_map(
              _maybe_get_materialization_function,
              lazy_restored_item,
              restore_args,
          )
      )
      flat = await asyncio.gather(*flat)
      return jax.tree_util.tree_unflatten(item_structure, flat)

    restored_item = asyncio.run(_restore())
    utils.sync_global_devices('PyTreeCheckpointHandler:restore')
    return restored_item

  def structure(self, directory: epath.Path) -> PyTree:
    """Restores the saved PyTree structure without regard for its leaf values.

    Args:
      directory: the directory to restore from.

    Returns:
      The structure of the checkpointed PyTree. Leaves may be of any type.

    Raises:
      FileNotFoundError: if the checkpoint is not found.
    """
    checkpoint_path = directory / self._aggregate_filename
    if checkpoint_path.exists():
      return self._aggregate_handler.deserialize(checkpoint_path)
    else:
      raise FileNotFoundError(
          f'Checkpoint not found: {checkpoint_path}.'
      )
