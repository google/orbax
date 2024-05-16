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

"""PyTreeCheckpointHandler class.

Implementation of `CheckpointHandler` interface dealing with JAX PyTrees. Much
of the underlying reading/writing logic for individual leaf types can be
customized, and is delegated to the `TypeHandler` class.
"""

import asyncio
import dataclasses
import json
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import base_pytree_checkpoint_handler
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import future
from orbax.checkpoint import transform_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
import tensorstore as ts


PyTree = Any
TupleKey = Tuple[str, ...]
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
SaveArgs = type_handlers.SaveArgs
ParamInfo = type_handlers.ParamInfo
TypeHandler = type_handlers.TypeHandler
TypeHandlerRegistry = type_handlers.TypeHandlerRegistry
AggregateHandler = aggregate_handlers.AggregateHandler
MsgpackHandler = aggregate_handlers.MsgpackHandler
LegacyTransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]
Transform = transform_utils.Transform
RestoreTransform = transform_utils.RestoreTransform
CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler

BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)
BasePyTreeSaveArgs = base_pytree_checkpoint_handler.BasePyTreeSaveArgs
BasePyTreeRestoreArgs = base_pytree_checkpoint_handler.BasePyTreeRestoreArgs
get_byte_limiter = base_pytree_checkpoint_handler.get_byte_limiter
LimitInFlightBytes = base_pytree_checkpoint_handler.LimitInFlightBytes
_InternalValueMetadata = base_pytree_checkpoint_handler._InternalValueMetadata  # pylint: disable=protected-access
get_param_names = base_pytree_checkpoint_handler.get_param_names

_CHECKPOINT_FILE = 'checkpoint'
_METADATA_FILE = base_pytree_checkpoint_handler.METADATA_FILE


def _keystr(key: Tuple[Any, ...]) -> str:
  return '/'.join(key)


def _find_matching_input_args(
    input_key: TupleKey,
    flat_item: Dict[TupleKey, Any],
    flat_transforms: Dict[TupleKey, Transform],
    flat_restore_args: Dict[TupleKey, RestoreArgs],
) -> Optional[RestoreArgs]:
  """Given an input_key, tries to find matching RestoreArgs for the input.

  Args:
    input_key: A key in the input tree.
    flat_item: The flattened, user-provided item.
    flat_transforms: Flattened transformations dict.
    flat_restore_args: Flattened tree of RestoreArgs, relative to item.

  Returns:
    RestoreArgs that match the given input_key, according to the
    transformations, or None if no match is found.
  """
  for transform_key, transform in flat_transforms.items():
    if transform.multi_value_fn is not None:
      if not isinstance(transform, RestoreTransform):
        raise ValueError(
            'Must use RestoreTransform in order to use multi_value_fn'
            ' during restore.'
        )
      if transform.multi_value_fn_input_args is None:
        raise ValueError(
            '`multi_value_fn` was specified, but'
            ' `multi_value_fn_input_args` were not. The latter must be'
            ' specified to identify inputs for the function.'
        )
      for (
          input_key_regex,
          input_args,
      ) in transform.multi_value_fn_input_args.items():
        if re.fullmatch(input_key_regex, _keystr(input_key)):
          return input_args
    elif not transform.use_fallback:
      # The following is done to reverse-engineer the regex for the key in
      # the original tree.
      for output_key in flat_item:
        match = re.fullmatch(_keystr(transform_key), _keystr(output_key))
        if match:
          if transform.original_key is None:
            # If transform.original_key is not specified, this transform
            # does not rename the original key. We can reuse the key from
            # the item.
            input_key_pattern = _keystr(output_key)
          else:
            input_key_pattern = match.expand(transform.original_key)
          if input_key_pattern == _keystr(input_key):
            return flat_restore_args[output_key]
  return None


def _has_use_fallback_transform(
    input_key: TupleKey, flat_transforms: Dict[TupleKey, Transform]
) -> bool:
  result = False
  for transform_key, transform in flat_transforms.items():
    match = re.fullmatch(_keystr(transform_key), _keystr(input_key))
    if match and transform.use_fallback:
      result = True
  return result


def _get_restore_parameters(
    directory: epath.Path,
    item: Optional[PyTree],
    structure: PyTree,
    transforms: Optional[PyTree],
    restore_args: Optional[PyTree],
    byte_limiter: Optional[LimitInFlightBytes] = None,
    transforms_default_to_original: bool = True,
    use_zarr3: bool = False,
) -> Tuple[PyTree, PyTree]:
  """Construct parameters needed for restoration.

  If transforms are not provided, the method is pretty simple: param_infos are
  constructed from the structure of the original checkpoint, and restore_args
  are serialized to a tree structure compatible with param_infos and structure.

  If transforms are provided, things become more complicated because we must
  determine exactly which parameters the user desires to restore, and construct
  param_infos and restore_args for these, while discarding unneeded parameters.
  In essence, the process can be thought of as reversing the transformations.
  This happens differently for different types of transforms.
  1. Renamed key: Identify the original key name (in the checkpoint) and carry
    over the provided restore args for the parameter.
  2. multi_value_fn: Users are required to specify multi_value_fn_input_args.
    Any keys named here must be loaded, and their restore args are also given
    here.
  3. Unspecified key: A key which is unspecified in the transforms but present
    in the `item` is a key that is carried over from the checkpoint unchanged.
  4. Fallback key: This is a key that is present in the `item` but not in the
    original checkpoint. It does not need to be restored.
  5. Keys present in the original checkpoint but not in the `item`/`transforms`
    are implicitly ignored, and not restored.

  Args:
    directory: Checkpoint directory.
    item: Optional reference item.
    structure: The structure of the original checkpoint.
    transforms: User-provided transformations. If None, they were not provided.
      Has the structure of the desired output tree.
    restore_args: User-provided restoration arguments. If None, they were not
      provided. Otherwise, the tree has the same structure as the desired output
      tree.
    byte_limiter: A LimitInFlightBytes object.
    transforms_default_to_original: See transform_utils.apply_transformations.
    use_zarr3: If True, use Zarr ver3 otherwise Zarr ver2

  Returns:
    Tuple of param_infos, and restore_args.
  """
  flat_structure = utils.to_flat_dict(structure, keep_empty_nodes=True)
  if restore_args is None:
    restore_args = jax.tree_util.tree_map(lambda x: RestoreArgs(), structure)
  flat_restore_args = utils.to_flat_dict(restore_args, keep_empty_nodes=True)
  flat_param_infos = {}
  flat_input_restore_args = {}
  is_ocdbt_checkpoint = type_handlers.is_ocdbt_checkpoint(directory)

  def _get_param_info(
      nested_name: Tuple[str, ...],
      meta: _InternalValueMetadata,
  ) -> Union[ParamInfo, Any]:
    if utils.is_supported_empty_aggregation_type(meta):
      # Empty node, ParamInfo should not be returned.
      return meta
    name = '.'.join(nested_name)
    return ParamInfo(
        name=name,
        path=directory / name,
        parent_dir=directory,
        skip_deserialize=meta.skip_deserialize,
        is_ocdbt_checkpoint=is_ocdbt_checkpoint,
        byte_limiter=byte_limiter,
        use_zarr3=use_zarr3,
    )

  if transforms is None:
    for key, meta in flat_structure.items():
      flat_param_infos[key] = _get_param_info(key, meta)
    restore_args = utils.serialize_tree(restore_args, keep_empty_nodes=True)
  else:
    if item is None:
      raise ValueError(
          'If providing `transforms`, must provide `item` matching structure'
          ' of expected result.'
      )
    flat_item = utils.to_flat_dict(item, keep_empty_nodes=True)
    flat_transforms = utils.to_flat_dict(transforms)

    for input_key, meta in flat_structure.items():
      maybe_input_args = _find_matching_input_args(
          input_key, flat_item, flat_transforms, flat_restore_args
      )
      if maybe_input_args:
        flat_param_infos[input_key] = _get_param_info(input_key, meta)
        flat_input_restore_args[input_key] = maybe_input_args
      elif input_key in flat_item and input_key in flat_structure:
        # Key is present in both input and output.
        if _has_use_fallback_transform(input_key, flat_transforms):
          # Indicates that a `use_fallback` transformation was specified.
          if transforms_default_to_original:
            # Specified `use_fallback`, but key was also present in the
            # checkpoint. This means we should skip loading, since it will be
            # overridden with a new value.
            flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
            flat_input_restore_args[input_key] = RestoreArgs()
          else:
            # Specified `use_fallback`, but `transforms_default_to_original`
            # is False. This means we draw the value from the user-provided
            # `item`.
            flat_param_infos[input_key] = _get_param_info(input_key, meta)
            flat_input_restore_args[input_key] = flat_restore_args[input_key]
        else:
          # Transform not specified.
          if transforms_default_to_original:
            # Key/value is carried over from the original unchanged.
            flat_param_infos[input_key] = _get_param_info(input_key, meta)
            flat_input_restore_args[input_key] = flat_restore_args[input_key]
          else:
            # Take the value from the user-provided `item`, ignoring any value
            # in the checkpoint.
            flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
            flat_input_restore_args[input_key] = RestoreArgs()
      else:
        # No match, restoration not required since it will be dropped from the
        # output.
        flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
        flat_input_restore_args[input_key] = RestoreArgs()

    restore_args = utils.from_flat_dict(
        flat_input_restore_args, target=structure
    )

  return (
      utils.from_flat_dict(flat_param_infos, target=structure),
      restore_args,
  )


def _multi_value_fns_with_args(
    transforms: PyTree, restore_args: PyTree
) -> PyTree:
  """Constructs a wrapper for multi_value_fn including RestoreArgs."""
  flat_restore_args = utils.to_flat_dict(restore_args, sep='/')

  def _maybe_wrap_transform(transform: Transform):
    def _multi_value_fn_with_args(transform_key: str, tree: PyTree) -> Any:
      nonlocal transform
      transform = typing.cast(RestoreTransform, transform)
      return transform.multi_value_fn(
          transform_key, tree, flat_restore_args[transform_key]
      )

    if transform.multi_value_fn is not None:
      return Transform(multi_value_fn=_multi_value_fn_with_args)
    else:
      return transform

  return jax.tree_util.tree_map(_maybe_wrap_transform, transforms)


def _transform_checkpoint(
    item: PyTree,
    restored: PyTree,
    restore_args: Optional[PyTree],
    transforms: Optional[PyTree],
    transforms_default_to_original: bool,
) -> PyTree:
  """Optionally transforms the restored PyTree to the structure of `item`.

  Args:
    item: a PyTree representing the result structure ("new tree structure").
    restored: a PyTree representing the original tree structure.
    restore_args: tree of RestoreArgs, with the same structure as `item`.
    transforms: provides instructions on how to transform the input trees. See
      transform_utils.
    transforms_default_to_original: See transform_utils.

  Returns:
    A transformed PyTree.
  """
  if item is None:
    if transforms is not None:
      msg = (
          'If providing `transforms`, must provide `item` matching structure'
          ' of expected result.'
      )
      raise ValueError(msg)
    item = restored
  else:
    if transforms is None:
      item = utils.deserialize_tree(restored, item)
    else:
      if restore_args is None:
        raise ValueError(
            'If providing `transforms`, must provide `restore_args` matching'
            ' structure of expected result.'
        )
      transforms = _multi_value_fns_with_args(transforms, restore_args)
      item = transform_utils.apply_transformations(
          restored, transforms, item, transforms_default_to_original
      )
  return item


def _get_impl_save_args(
    item: Optional[PyTree] = None,
    save_args: Optional['PyTreeSaveArgs'] = None,
    args: Optional['PyTreeSaveArgs'] = None,
) -> BasePyTreeSaveArgs:
  """Construct BasePyTreeSaveArgs."""
  if isinstance(item, CheckpointArgs):
    raise ValueError(
        'Make sure to specify kwarg name `args=` when providing'
        ' `PyTreeSaveArgs`.'
    )
  if args is None:
    args = PyTreeSaveArgs(
        item=item,
        save_args=save_args,
    )
  return BasePyTreeSaveArgs(
      item=args.item,
      save_args=args.save_args,
      ocdbt_target_data_file_size=args.ocdbt_target_data_file_size,
  )


class PyTreeCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """A CheckpointHandler implementation for any PyTree structure.

  See JAX documentation for more information on what consistutes a "PyTree".
  This handler is capable of saving and restoring any leaf object for which a
  `TypeHandler` (see documentation) is registered. By default, `TypeHandler`s
  for standard types like `np.ndarray`, `jax.Array`, Python scalars, and others
  are registered.

  As with all `CheckpointHandler` subclasses, `PyTreeCheckpointHandler` should
  only be used in conjunction with a `Checkpointer` (or subclass). By itself,
  the `CheckpointHandler` is non-atomic.

  Example::

    ckptr = Checkpointer(PyTreeCheckpointHandler())
  """

  def __init__(
      self,
      aggregate_filename: Optional[str] = None,
      concurrent_gb: int = 96,
      use_ocdbt: bool = True,
      use_zarr3: bool = False,
      primary_host: Optional[int] = 0,
      type_handler_registry: TypeHandlerRegistry = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
      handler_impl: Optional[BasePyTreeCheckpointHandler] = None,
  ):
    """Creates PyTreeCheckpointHandler.

    Args:
      aggregate_filename: name that the aggregated checkpoint should be saved
        as.
      concurrent_gb: max concurrent GB that are allowed to be read. Can help to
        reduce the possibility of OOM's when large checkpoints are restored.
      use_ocdbt: enables Tensorstore OCDBT driver. This option allows using a
        different checkpoint format which is faster to read and write, as well
        as more space efficient.
      use_zarr3: If True, use Zarr ver3 otherwise Zarr ver2
      primary_host: the host id of the primary host.  Default to 0.  If it's set
        to None, then all hosts will be considered as primary.  It's useful in
        the case that all hosts are only working with local storage.
      type_handler_registry: a type_handlers.TypeHandlerRegistry. If not
        specified, the global type handler registry will be used.
      handler_impl: Allows overriding the internal implementation.
    """
    self._aggregate_handler = MsgpackHandler(primary_host=primary_host)
    if aggregate_filename is None:
      aggregate_filename = _CHECKPOINT_FILE
    self._aggregate_filename = aggregate_filename
    self._concurrent_gb = concurrent_gb
    self._use_ocdbt = use_ocdbt
    self._use_zarr3 = use_zarr3
    self._primary_host = primary_host
    self._type_handler_registry = type_handler_registry

    self._handler_impl = handler_impl or BasePyTreeCheckpointHandler(
        aggregate_filename=aggregate_filename,
        concurrent_gb=concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        primary_host=primary_host,
        type_handler_registry=type_handler_registry,
    )

  async def async_save(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      save_args: Optional['PyTreeSaveArgs'] = None,
      args: Optional['PyTreeSaveArgs'] = None,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree to a given directory.

    This operation is compatible with a multi-host, multi-device setting. Tree
    leaf values must be supported by the type_handler_registry given in the
    constructor. Standard supported types include Python scalars, `np.ndarray`,
    `jax.Array`, and strings.

    After saving, all files will be located in "directory/". The exact files
    that are saved depend on the specific combination of options, including
    `use_ocdbt`. A JSON metadata file will be present to store the
    tree structure.
    In addition, a msgpack file may be present, allowing users to store
    aggregated values (see below).

    Example usage::

      ckptr = Checkpointer(PyTreeCheckpointHandler())
      item = {
          'layer0': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
          'layer1': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
      }
      # Note: save_args may be None if no customization is desired for saved
      # parameters.
      # In this case, we "aggregate" small parameters into a single file to
      # allow for greater file read/write efficiency (and potentially less)
      # wasted space). With OCDBT format active, this parameter is obsolete.
      save_args =
        jax.tree_util.tree_map(
            lambda x: SaveArgs(aggregate=x.size < some_size), item)
      # Eventually calls through to `async_save`.
      ckptr.save(path, item, save_args)

    Args:
      directory: save location directory.
      item: Deprecated, use `args.
      save_args: Deprecated, use `args`.
      args: `PyTreeSaveArgs` (see below).

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    args = _get_impl_save_args(item, save_args, args)
    return await self._handler_impl.async_save(directory, args=args)

  def save(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      save_args: Optional['PyTreeSaveArgs'] = None,
      args: Optional['PyTreeSaveArgs'] = None,
  ):
    """Saves the provided item. See async_save."""
    args = _get_impl_save_args(item, save_args, args)
    self._handler_impl.save(directory, args=args)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      restore_args: Optional[PyTree] = None,
      transforms: Optional[PyTree] = None,
      transforms_default_to_original: bool = True,
      legacy_transform_fn: Optional[LegacyTransformFn] = None,
      args: Optional['PyTreeRestoreArgs'] = None,
  ) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given path.

    In the most basic case, only `directory` is required. The tree will be
    restored exactly as saved, and all leaves will be restored as the correct
    types (assuming the tree metadata is present).

    However, `restore_args` is often required as well. This PyTree gives a
    `RestoreArgs` object (or subclass) for every leaf in the tree. Many types,
    such as string or `np.ndarray` do not require any special options for
    restoration. When restoring an individual leaf as `jax.Array`, however,
    some properties may be required.

    One example is `sharding`, which defines how a `jax.Array` in the restored
    tree should be partitioned. `mesh` and `mesh_axes` can also be used to
    specify `sharding`, but `sharding` is the preferred way of specifying this
    partition since `mesh` and `mesh_axes` only constructs
    `jax.sharding.NamedSharding`. For more information, see `ArrayTypeHandler`
    documentation and JAX sharding documentation.

    Example::

      ckptr = Checkpointer(PyTreeCheckpointHandler())
      restore_args = {
          'layer0': {
              'w': RestoreArgs(),
              'b': RestoreArgs(),
          },
          'layer1': {
              'w': ArrayRestoreArgs(
                  # Restores as jax.Array, regardless of how it was saved.
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  # Warning: may truncate or pad!
                  global_shape=(x, y),
                ),
              'b': ArrayRestoreArgs(
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  global_shape=(x, y),
                ),
          },
      }
      ckptr.restore(path, restore_args=restore_args)

    Providing `item` is typically only necessary when restoring a custom PyTree
    class (or when using transformations). In this case, the restored object
    will take on the same structure as `item`.

    Example::

      @flax.struct.dataclass
      class TrainState:
        layer0: dict[str, jax.Array]
        layer1: dict[str, jax.Array]

      ckptr = Checkpointer(PyTreeCheckpointHandler())
      train_state = TrainState(
          layer0={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
          layer1={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
      )
      restore_args = jax.tree_util.tree_map(_make_restore_args, train_state)
      ckptr.restore(path, item=train_state, restore_args=restore_args)
      # restored tree is of type `TrainState`.

    Args:
      directory: saved checkpoint location directory.
      item: Deprecated, use `args`.
      restore_args: Deprecated, use `args`.
      transforms: Deprecated, use `args`.
      transforms_default_to_original: See transform_utils.apply_transformations.
      legacy_transform_fn: Deprecated, use `args`.
      args: `PyTreeRestoreArgs` (see below).

    Returns:
      A PyTree matching the structure of `item`.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `multi_value_fn`.
    """
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}.'
      )
    if isinstance(item, CheckpointArgs):
      raise ValueError(
          'Make sure to specify kwarg name `args=` when providing'
          ' `PyTreeRestoreArgs`.'
      )
    if args is None:
      args = PyTreeRestoreArgs(
          item,
          restore_args,
          transforms,
          transforms_default_to_original,
          legacy_transform_fn,
      )
    item = args.item
    restore_args = args.restore_args
    transforms = args.transforms
    transforms_default_to_original = args.transforms_default_to_original
    legacy_transform_fn = args.legacy_transform_fn

    # Delegate to `TreeCheckpointHandler` as long as transformation options are
    # not specified and metadata file exists.
    if (
        (directory / _METADATA_FILE).exists()
        and transforms is None
        and legacy_transform_fn is None
    ):
      args = BasePyTreeRestoreArgs(
          item,
          restore_args=restore_args,
      )
      return self._handler_impl.restore(directory, args=args)

    logging.debug('directory=%s, restore_args=%s', directory, restore_args)
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}'
      )
    byte_limiter = get_byte_limiter(self._concurrent_gb)
    structure, use_zarr3_metadata = self._handler_impl._get_internal_metadata(  # pylint: disable=protected-access
        directory
    )
    # `checkpoint_restore_args` has a structure relative to the checkpoint,
    # while `restore_args` remains structured relative to the output.
    param_infos, checkpoint_restore_args = _get_restore_parameters(
        directory,
        item,
        structure,
        transforms,
        restore_args,
        byte_limiter=byte_limiter,
        transforms_default_to_original=transforms_default_to_original,
        use_zarr3=use_zarr3_metadata
        if use_zarr3_metadata is not None
        else self._use_zarr3,
    )

    if legacy_transform_fn is not None and transforms is not None:
      raise ValueError(
          'Cannot provide both `transforms` and `legacy_transform_fn`.'
      )
    if legacy_transform_fn is not None:
      structure, param_infos = legacy_transform_fn(item, structure, param_infos)
      if restore_args is None:
        restore_args = jax.tree_util.tree_map(lambda x: RestoreArgs(), item)
      checkpoint_restore_args = restore_args

    def _maybe_set_default_restore_types(
        meta: _InternalValueMetadata, arg: RestoreArgs
    ):
      if not meta.skip_deserialize and meta.restore_type is None:
        return dataclasses.replace(
            meta, restore_type=type_handlers.default_restore_type(arg)
        )
      return meta

    # If metadata file was missing in the checkpoint, we need to decide
    # restore_type based on RestoreArgs.
    structure = jax.tree_util.tree_map(
        _maybe_set_default_restore_types, structure, checkpoint_restore_args
    )

    restored_item = asyncio.run(
        self._handler_impl._maybe_deserialize(  # pylint: disable=protected-access
            structure, param_infos, checkpoint_restore_args
        )
    )

    if not legacy_transform_fn:
      restored_item = _transform_checkpoint(
          item,
          restored_item,
          restore_args,
          transforms,
          transforms_default_to_original,
      )

    if logging.level_debug():
      logging.debug('param_infos: %s', param_infos)
      logging.debug('checkpoint_restore_args: %s', checkpoint_restore_args)
      logging.debug(
          'restored_item: %s', jax.tree_util.tree_structure(restored_item)
      )
      logging.debug(
          'ts_metrics: %s',
          json.dumps(ts.experimental_collect_matching_metrics('/tensorstore/')),
      )

    return restored_item

  def metadata(self, directory: epath.Path) -> Optional[PyTree]:
    """Returns tree metadata.

    The result will be a PyTree matching the structure of the saved checkpoint.
    Note that if the item saved was a custom class, the restored metadata will
    be returned as a nested dictionary representation.

    Example::

      {
        'layer0': {
            'w': ArrayMetadata(dtype=jnp.float32, shape=(8, 8), shards=(1, 2)),
            'b': ArrayMetadata(dtype=jnp.float32, shape=(8,), shards=(1,)),
        },
        'step': ScalarMetadata(dtype=jnp.int64),
      }

    If the required metadata file is not present, this method will raise an
    error.

    Args:
      directory: checkpoint location.

    Returns:
      tree containing metadata.
    """
    return self._handler_impl.metadata(directory)

  def finalize(self, directory: epath.Path) -> None:
    """Finalization step.

    Called automatically by the Checkpointer/AsyncCheckpointer just before the
    checkpoint is considered "finalized" in the sense of ensuring atomicity. See
    documentation for `type_handlers.merge_ocdbt_per_process_files`.

    Args:
      directory: Path where the checkpoint is located.
    """
    self._handler_impl.finalize(directory)

  def close(self):
    """Closes the handler. Called automatically by Checkpointer."""
    self._handler_impl.close()


@register_with_handler(PyTreeCheckpointHandler, for_save=True)
@dataclasses.dataclass
class PyTreeSaveArgs(CheckpointArgs):
  """Parameters for saving a PyTree.

  Attributes:
    item (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `item`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
    ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
      OCDBT data file.  It only applies when OCDBT is enabled and Zarr3 must be
      turned on.  If left unspecified, default size is 2GB.  A value of 0
      indicates no maximum file size limit.  For best results, ensure
      chunk_byte_size is smaller than this value.  For more details, refer to
      https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
  """

  item: PyTree
  save_args: Optional[PyTree] = None
  ocdbt_target_data_file_size: Optional[int] = None


@register_with_handler(PyTreeCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class PyTreeRestoreArgs(CheckpointArgs):
  """Parameters for restoring a PyTree.

  Attributes (all optional):
    item: provides the tree structure for the restored item. If not provided,
      will infer the structure from the saved checkpoint. Transformations will
      not be run in this case. Necessary particularly in the case where the
      caller needs to restore the tree as a custom object.
    restore_args: optional object containing additional arguments for
      restoration. It should be a PyTree matching the structure of `item`, or
      if `item` is not provided, then it should match the structure of the
      checkpoint. Each value in the tree should be a `RestoreArgs` object (OR
      a subclass of `RestoreArgs`). Importantly, note that when restoring a
      leaf as a certain type, a specific subclass of `RestoreArgs` may be
      required. `RestoreArgs` also provides the option to customize the
      restore type of an individual leaf.
    transforms: a PyTree of transformations that should be applied to the
      saved tree in order to obtain a final structure. The `transforms` tree
      structure should conceptually match that of `item`, but the use of
      regexes and implicit keys means that it does not need to match
      completely. See `transform_utils` for further information.
    transforms_default_to_original: See transform_utils.apply_transformations.
    legacy_transform_fn: WARNING: NOT GENERALLY SUPPORTED. A function which
      accepts the `item` argument, a PyTree checkpoint structure and a PyTree
      of ParamInfos based on the checkpoint. Returns a transformed PyTree
      matching the desired return tree structure, and a matching ParamInfo
      tree.
  """

  item: Optional[PyTree] = None
  restore_args: Optional[PyTree] = None
  transforms: Optional[PyTree] = None
  transforms_default_to_original: bool = True
  legacy_transform_fn: Optional[LegacyTransformFn] = None
