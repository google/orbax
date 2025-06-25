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

"""Compatibility wrapper to help leaf handlers to work as V0 type_handlers."""

import copy
from typing import Generic, Sequence, Tuple, cast
from absl import logging
from etils import epath
import jax
from jax import tree_util as jtu
import numpy as np
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import type_handlers as type_handlers_v0
from orbax.checkpoint._src.serialization import types as types_v0
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


class _PathAwaitingCreation(path_types.PathAwaitingCreation):
  """Implementation of `PathAwaitingCreation` that awaits contracted signals."""

  def __init__(self, path: path_types.Path, operation_id: str):
    self._path = path
    self._operation_id = operation_id

  def __truediv__(
      self, other: path_types.PathAwaitingCreation | path_types.PathLike
  ) -> path_types.PathAwaitingCreation:
    if isinstance(other, path_types.PathAwaitingCreation):
      other = other.path
    return _PathAwaitingCreation(self._path / other, self._operation_id)

  async def await_creation(self) -> path_types.Path:
    await synchronization.await_contracted_signals(self._operation_id)
    return self._path

  @property
  def path(self) -> path_types.Path:
    return self._path


def _keypath_from_param_name(param_name: str) -> tree_types.PyTreeKeyPath:
  """Converts a param name to a PyTreeKeyPath.

  This is based on reversing of the name construction from tree/utils.py's
  param_name_from_keypath.

  Args:
    param_name: A string representing the parameter name.

  Returns:
    A PyTreeKeyPath representing the parameter name.
  """
  return tuple([jtu.GetAttrKey(s) for s in param_name.split('.')])


def _construct_serialization_param(
    value: types.Leaf,
    info: types_v0.ParamInfo,
) -> types.SerializationParam[types.Leaf]:
  return types.SerializationParam(
      keypath=_keypath_from_param_name(info.name),
      value=value,
  )


def _construct_serialization_context(
    info: types_v0.ParamInfo,
) -> types.SerializationContext:
  return types.SerializationContext(
      # TODO(dnlng): should use actual wait
      parent_dir=_PathAwaitingCreation(
          info.parent_dir, context_lib.get_context().operation_id()
      ),
      ts_context=info.ts_context,
      byte_limiter=info.byte_limiter,
  )


def _construct_deserialization_param(
    info: types_v0.ParamInfo,
    restore_args: types_v0.RestoreArgs,
) -> types.DeserializationParam[
    array_leaf_handler.AbstractArray
    | numpy_leaf_handler.AbstractNumpy
    | scalar_leaf_handler.AbstractScalar
    | None
]:
  """Constructs a DeserializationParam from a ParamInfo and RestoreArg."""

  logging.vlog(1, 'compatibility.py: restore_args: %s', restore_args)

  value = None

  if restore_args.restore_type == np.ndarray:
    # Numpy type
    value = numpy_leaf_handler.NumpyShapeDtype(
        dtype=restore_args.dtype,
        shape=None,
    )
  elif restore_args.restore_type in (scalar_leaf_handler.Scalar.__args__):
    # JAX Array type, construct value as jax.ShapeDtypeStruct.
    value = scalar_leaf_handler.AbstractScalar(
        dtype=restore_args.restore_type,
    )
  elif isinstance(restore_args, type_handlers_v0.ArrayRestoreArgs):
    # JAX Array type, construct value as jax.ShapeDtypeStruct.
    arg = cast(type_handlers_v0.ArrayRestoreArgs, restore_args)

    logging.info(
        'name: %s, ArrayRestoreArgs: %s, write_shape: %s',
        info.name,
        arg,
        info.write_shape,
    )

    if arg.mesh is not None and arg.mesh_axes is not None:
      sharding = jax.sharding.NamedSharding(arg.mesh, arg.mesh_axes)
    elif arg.sharding is not None:
      if isinstance(arg.sharding, sharding_metadata.ShardingMetadata):
        sharding = arg.sharding.to_jax_sharding()
      else:
        sharding = arg.sharding
    else:
      sharding = None
    value = jax.ShapeDtypeStruct(arg.shape, arg.dtype, sharding=sharding)
  elif info.write_shape is not None:
    # TODO(dnlng): this is needed due to write_shape is passed into the
    # metadata() call, and then returned metadata will include this write_shape
    # in it. We should refractor how this write_shape is populated from the v0
    # so that the compatibility wrapper doesn't only work specific for
    # ArrayLeafHandler.
    value = array_leaf_handler.ArrayMetadata(
        shape=None,
        dtype=None,
        sharding_metadata=None,
        storage_metadata=value_metadata.StorageMetadata(
            chunk_shape=None,
            write_shape=info.write_shape,
        ),
    )

  return types.DeserializationParam(
      keypath=_keypath_from_param_name(info.name),
      value=value,
  )


def _construct_deserialization_context(
    info: types_v0.ParamInfo,
) -> types.DeserializationContext:
  return types.DeserializationContext(
      parent_dir=info.parent_dir,
      ocdbt_checkpoint=info.is_ocdbt_checkpoint,
      zarr3_checkpoint=info.use_zarr3,
      ts_context=info.ts_context,
      byte_limiter=info.byte_limiter,
  )


def _validate_serialization_infos(
    infos: Sequence[types_v0.ParamInfo],
) -> None:
  """Validates that all infos share the same properties."""
  info0 = infos[0]
  for info in infos[1:]:
    if (
        (info0.parent_dir != info.parent_dir)
        or (info0.ts_context != info.ts_context)
        or (info0.byte_limiter != info.byte_limiter)
    ):
      raise ValueError(
          'All infos must have the same parent_dir, ts_context, and'
          ' byte_limiter.'
      )

  # TODO(dnlng): Add validation for ocdbt & zarr3.


def _validate_deserialization_infos(
    infos: Sequence[types_v0.ParamInfo],
) -> None:
  """Validates that all infos share the same properties."""
  info0 = infos[0]
  for info in infos:
    if (
        info0.parent_dir != info.parent_dir
        or info0.is_ocdbt_checkpoint != info.is_ocdbt_checkpoint
        or info0.use_zarr3 != info.use_zarr3
        or info0.ts_context != info.ts_context
        or info0.byte_limiter != info.byte_limiter
    ):
      raise ValueError(
          'All infos must have the same parent_dir, is_ocdbt_checkpoint,'
          ' use_zarr3, ts_context, and byte_limiter.'
      )


def _convert_v1_metadata_to_v0(
    name: str,
    directory: epath.Path | None,
    metadata: array_leaf_handler.AbstractArray,
) -> value_metadata.Metadata:
  """Converts a v1 metadata to a v0 metadata."""
  if isinstance(metadata, array_leaf_handler.ArrayMetadata):
    metadata = cast(array_leaf_handler.ArrayMetadata, metadata)
    ret = value_metadata.ArrayMetadata(
        name=name,
        directory=directory,
        sharding=metadata.sharding_metadata,
        shape=metadata.shape,
        dtype=metadata.dtype,
        storage=metadata.storage_metadata,
    )
    logging.vlog(1, 'ArrayMetadata: %s', ret)
    return ret
  elif isinstance(metadata, numpy_leaf_handler.NumpyMetadata):
    metadata = cast(numpy_leaf_handler.NumpyMetadata, metadata)
    ret = value_metadata.ArrayMetadata(
        name=name,
        directory=directory,
        sharding=None,
        shape=metadata.shape,
        dtype=metadata.dtype,
        storage=metadata.storage_metadata,
    )
    logging.vlog(1, 'NumpyMetadata: %s', ret)
    return ret
  elif isinstance(metadata, scalar_leaf_handler.AbstractScalar):
    ret = value_metadata.ScalarMetadata(
        name=name,
        directory=directory,
        dtype=metadata.dtype,
    )
    logging.vlog(1, 'ScalarMetadata: %s', ret)
    return ret
  else:
    logging.warning(
        'Unsupported metadata type: %s. Returning value_metadata.Metadata'
        ' name=%s.',
        type(metadata),
        name,
    )
    return value_metadata.Metadata(
        name=name,
        directory=directory,
    )


class CompatibleTypeHandler(
    types_v0.TypeHandler, Generic[types.Leaf, types.AbstractLeaf]
):
  """Compatibility wrapper to help leaf handlers to work as V0 type_handlers."""

  def __init__(
      self,
      leaf_handler: types.LeafHandler[types.Leaf, types.AbstractLeaf],
      typestr: str,
  ):
    self._leaf_handler = leaf_handler
    self._typestr = typestr

  def typestr(self) -> str:
    return self._typestr

  async def serialize(
      self,
      values: Sequence[types.Leaf],
      infos: Sequence[types_v0.ParamInfo],
      args: Sequence[types_v0.SaveArgs] | None = None,
  ) -> Sequence[future.Future]:
    _validate_serialization_infos(infos)

    params = []
    info0 = infos[0]
    for info, value in zip(infos, values):
      logging.vlog(1, 'info: %s', info)
      params.append(_construct_serialization_param(value, info))
    serialization_context = _construct_serialization_context(info0)
    serialization_task = await self._leaf_handler.serialize(
        params, serialization_context
    )

    async def _background_serialize():
      await serialization_task

    operation_id = context_lib.get_context().operation_id()

    return [
        future.CommitFuture(
            coro=_background_serialize(),
            operation_id=operation_id,
        )
    ]

  async def deserialize(
      self,
      infos: Sequence[types_v0.ParamInfo],
      args: Sequence[types_v0.RestoreArgs] | None = None,
  ) -> Sequence[types.Leaf]:
    _validate_deserialization_infos(infos)

    params = []
    if args is None:
      args = [types_v0.RestoreArgs()] * len(infos)

    for info, restore_arg in zip(infos, args):
      if logging.vlog_is_on(1):
        logging.vlog(1, 'info: %s', info)
        logging.vlog(1, 'restore_arg: %s', restore_arg)

      # TODO(dnlng): need to allow passing in _construct_deserialization_param
      # for different leaf handlers.
      params.append(_construct_deserialization_param(info, restore_arg))

    info0 = infos[0]
    deserialization_context = _construct_deserialization_context(info0)
    task = await self._leaf_handler.deserialize(params, deserialization_context)
    return await task

  async def metadata(
      self, infos: Sequence[types_v0.ParamInfo]
  ) -> Sequence[value_metadata.Metadata]:
    """Constructs object metadata from a stored parameter location.

    Args:
      infos: sequence of ParamInfo

    Returns:
      Sequence of Metadata for each provided ParamInfo.
    """
    _validate_deserialization_infos(infos)
    args = [types_v0.RestoreArgs()] * len(infos)
    params = []

    for info, restore_arg in zip(infos, args):
      logging.vlog(1, 'info: %s', info)
      params.append(_construct_deserialization_param(info, restore_arg))

    info0 = infos[0]
    deserialization_context = _construct_deserialization_context(info0)

    metadatas = await self._leaf_handler.metadata(
        params, deserialization_context
    )

    ret = []
    for info, metadata in zip(infos, metadatas):
      ret.append(_convert_v1_metadata_to_v0(info.name, info.path, metadata))
    return ret

  def memory_size(
      self, values: Sequence[types.Leaf]
  ) -> Sequence[Tuple[int, int]]:
    # this only works for leaf handler that based on V0 TypeHandlers and stored
    # it in self._leaf_handler._handler_impl.
    if hasattr(self._leaf_handler, '_handler_impl'):
      v0_handler = self._leaf_handler._handler_impl  # pylint: disable=protected-access

      return v0_handler.memory_size(values)

    raise NotImplementedError(
        'Cannot resolve memory_size for this v1 leaf handler, '
        f' {self._leaf_handler!r}.'
    )

  @property
  def _array_metadata_store(self):
    # as the array_metadata_store.resolve_array_metadata_store read the metadata
    # store directly from _array_metadata_store, this is to provide the same
    # interface.  Currently, this only works for LeafHandler that based on
    # V0 ArrayHandler.
    if hasattr(self._leaf_handler, '_handler_impl') and hasattr(
        self._leaf_handler._handler_impl, '_array_metadata_store'  # pylint: disable=protected-access
    ):
      return self._leaf_handler._handler_impl._array_metadata_store  # pylint: disable=protected-access
    else:
      logging.warning(
          'Cannot resolve _array_metadata_store for this v1 leaf handler: %r',
          self._leaf_handler,
      )
      return None


def get_compatible_type_handler_registry(
    context: context_lib.Context | None = None,
    type_handler_registry: type_handlers_v0.TypeHandlerRegistry | None = None,
) -> type_handlers_v0.TypeHandlerRegistry:
  """Returns a V0 type handler registry that using v1 leaf handlers.

  This is a helper function to setup a v0 type handler registry that will be
  registered all existing v1 leaf handlers.

  Args:
    context: The context to that will be passed to create leaf handlers.
    type_handler_registry: The v0 type handler registry to use. If not provided,
      a new registry will be created.

  Returns:
    The v0 type handler registry that using standard v1 leaf handlers.
  """
  if type_handler_registry is None:
    type_handler_registry = copy.deepcopy(
        type_handlers_v0.GLOBAL_TYPE_HANDLER_REGISTRY
    )

  type_handler_registry.add(
      jax.Array,
      CompatibleTypeHandler(
          array_leaf_handler.ArrayLeafHandler(context=context),
          typestr=type_handlers_v0.JAX_ARRAY_TYPE_STR,
      ),
      override=True,
      ignore_warnings=True,
  )
  type_handler_registry.add(
      np.ndarray,
      CompatibleTypeHandler(
          numpy_leaf_handler.NumpyLeafHandler(context=context),
          typestr='np.ndarray',
      ),
      override=True,
      ignore_warnings=True,
  )

  for scalar_type in (int, float, bytes, np.number):
    type_handler_registry.add(
        scalar_type,
        CompatibleTypeHandler(
            scalar_leaf_handler.ScalarLeafHandler(context=context),
            typestr='scalar',
        ),
        override=True,
        ignore_warnings=True,
    )
  return type_handler_registry
