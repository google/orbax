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

import dataclasses
from typing import Any, Generic, Sequence, Tuple, Type, cast, get_args

from absl import logging
from etils import epath
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
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
from orbax.checkpoint.experimental.v1._src.serialization import string_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


@dataclasses.dataclass
class V0RestoreArgs(types_v0.RestoreArgs):
  abstract_leaf: Type[Any] | None = None


@dataclasses.dataclass
class V0Metadata(value_metadata.Metadata):
  v1_metadata: Any | None = None


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
    | string_leaf_handler.AbstractString
    | None
]:
  """Constructs a DeserializationParam from a ParamInfo and RestoreArg."""

  logging.vlog(1, 'compatibility.py: restore_args: %s', restore_args)

  if restore_args.restore_type == np.ndarray:
    # Numpy type
    value = numpy_leaf_handler.NumpyShapeDtype(
        dtype=restore_args.dtype,
        shape=None,
    )
  elif isinstance(restore_args.restore_type, type) and issubclass(
      restore_args.restore_type, get_args(scalar_leaf_handler.Scalar)
  ):
    # Scalar type
    logging.vlog(1, 'Scalar restore_type set to: %s', restore_args.restore_type)
    value = restore_args.restore_type
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

    if sharding is None:
      # it's a numpy type
      value = numpy_leaf_handler.NumpyShapeDtype(
          dtype=arg.dtype,
          shape=arg.shape,
      )
    else:
      # it's a jax.Array type
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
    logging.vlog(1, 'ArrayMetadata as param.value: %s', value)
  elif (
      restore_args.restore_type in (None, int, float)
      or isinstance(restore_args.restore_type, (np.dtype, jnp.dtype))
      or issubclass(restore_args.restore_type, np.number)
  ):
    # scalar type
    value = restore_args.restore_type
  elif issubclass(restore_args.restore_type, str):
    # string type
    value = str
  else:
    raise ValueError(
        'Unsupported restore_args: %s. Cannot construct DeserializationParam.'
        % restore_args
    )

  logging.vlog(1, 'deserialization_param.value: %r', value)

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
  """Wrap V1 metadata into V0Metadata."""
  return V0Metadata(
      name=name,
      directory=directory,
      v1_metadata=metadata,
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
      args = [V0RestoreArgs()] * len(infos)

    for info, restore_arg in zip(infos, args):
      if isinstance(restore_arg, V0RestoreArgs):
        v0_restore_arg = cast(V0RestoreArgs, restore_arg)
        abstract_leaf = v0_restore_arg.abstract_leaf
      else:
        abstract_leaf = None

      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            'deserialize: restore_arg: %s, info: %s, abstract_leaf: %s',
            restore_arg,
            info,
            abstract_leaf,
        )

      params.append(
          types.DeserializationParam(
              keypath=_keypath_from_param_name(info.name),
              value=abstract_leaf,
          )
      )

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


def get_v0_type_handler_registry(
    leaf_handler_registry: types.LeafHandlerRegistry,
    context: context_lib.Context | None = None,
):
  """Returns a v0 type handler registry based on the `leaf_handler_registry`.

  Args:
    leaf_handler_registry: The LeafHandlerRegistry to be used to create a v0
      type handler registry.
    context: The Context to be used to default construct the LeafHandlers.
  """

  def _get_typestr(leaf_type: Any) -> str:
    if leaf_type == jax.Array:
      return type_handlers_v0.JAX_ARRAY_TYPE_STR
    elif leaf_type == np.ndarray:
      return 'np.ndarray'
    elif leaf_type in (int, float, bytes, np.number):
      return 'scalar'
    elif leaf_type == str:
      return 'string'
    else:
      return f'{leaf_type!r}'

  # register standardard v1 leaf handlers to the v0 type handler registry.
  handlers = []
  for leaf_type, _, leaf_handler_type in leaf_handler_registry.get_all():
    try:
      leaf_handler = leaf_handler_type(context=context)  # pytype: disable=wrong-keyword-args
    except TypeError as e:
      raise ValueError(
          f'Failed to default construct LeafHandler[{leaf_type}].  All'
          ' LeafHandler types must be able to be constructed with a context.'
      ) from e
    handlers.append((
        leaf_type,
        CompatibleTypeHandler(
            leaf_handler,
            typestr=_get_typestr(leaf_type),
        ),
    ))
  return type_handlers_v0.create_type_handler_registry(*handlers)
