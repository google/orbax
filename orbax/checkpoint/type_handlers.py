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

"""Provides utils for PytreeCheckpointHandler."""

import abc
import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from etils import epath
import jax
from jax.experimental.gda_serialization import serialization
from jax.experimental.gda_serialization.serialization import get_tensorstore_spec
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from orbax.checkpoint.future import Future
import tensorstore as ts

PyTreeDef = jax.tree_util.PyTreeDef
Scalar = Union[int, float, np.number]


@dataclasses.dataclass
class ParamInfo:
  """Information describing a parameter in a PyTree.

  Note that ParamInfo is distinct from SaveArgs and RestoreArgs in that in
  represents information not provided by a user, and should be computed
  internally.

  name: name of the parameter.
  path: A path providing a location where file(s) should be saved. The path is
    assumed to be a directory.
  aggregate: whether the parameter should be / was aggregated.
  """
  name: Optional[str] = None
  path: Optional[epath.Path] = None
  aggregate: Optional[bool] = None
  byte_limiter: Optional[serialization._LimitInFlightBytes] = None  # pylint: disable=protected-access


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  aggregate: if true, saves the given parameter in an aggregated tree format
    rather than individually. See AggregateHandler.
  dtype: if provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  aggregate: bool = False
  dtype: Optional[jnp.dtype] = None


@dataclasses.dataclass
class RestoreArgs:
  """Extra arguments that can be provided for restoration.

  lazy: if True, restores using LazyArray. The actual read operation will not be
    performed until `get` is called for the restored LazyArray.
  restore_type: Specifies the object type of the restored parameter. The type
    must have a corresponding TypeHandler for restoration. Ignored if the
    parameter is restored from an aggregated checkpoint file.
  dtype: if provided, casts the parameter to the given dtype after restoring.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """
  lazy: bool = False
  # TODO(b/253238305) Consider deprecating this in favor of saving type
  # information in checkpoint metadata.
  restore_type: Any = np.ndarray
  dtype: Optional[jnp.dtype] = None


class TypeHandler(abc.ABC):
  """Interface for reading and writing a PyTree leaf."""

  @abc.abstractmethod
  async def serialize(
      self,
      value: Any,
      info: ParamInfo,
      args: Optional[SaveArgs] = None) -> List[Future]:
    """Writes the parameter to a storage location.

    This method is responsible for copying the parameter from a remote device in
    a synchronous fashion (if applicable). It should then return a list of
    futures which can be later awaited to complete the final commit operation
    to a storage location.

    The function can be used in a multihost setting, but should not implement
    extra logic to ensure atomicity.

    Args:
      value: the parameter to save.
      info: contains relevant information for serialization.
      args: additional arguments for serialization, provided by the user.

    Returns:
      List of commit futures which can be awaited to complete the save
      operation.
    """
    pass

  @abc.abstractmethod
  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> Any:
    """Reads the parameter from a storage location.

    Args:
      info: parameter information.
      args: user-provided restoration information.

    Returns:
      The deserialized parameter.
    """
    pass


def _get_cast_tspec_serialize(tspec, value, args):
  """Creates a Tensorstore spec for casting a param during serialize."""
  tspec = {
      'base': tspec,
      'driver': 'cast',
  }
  # Origin dtype.
  tspec['dtype'] = jnp.dtype(value.dtype).name
  # Destination dtype.
  if args.dtype is None:
    tspec['base']['dtype'] = jnp.dtype(value.dtype).name
  else:
    tspec['base']['dtype'] = jnp.dtype(args.dtype).name
  return tspec


def _get_cast_tspec_deserialize(tspec, args):
  """Creates a Tensorstore spec for casting a param during deserialize."""
  if args.dtype is not None:
    tspec = {
        'base': tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(args.dtype).name,
    }
  return tspec


class NumpyHandler(TypeHandler):
  """Provides an implementation of TypeHandler for replicated numpy arrays."""

  def __init__(self, metadata_key: Optional[str] = None):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
    """
    self._metadata_key = metadata_key

  def _get_json_tspec(self,
                      info: ParamInfo,
                      value: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec: Dict[str, Any] = get_tensorstore_spec(path)
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    tspec['metadata'] = {
        'compressor': {
            'id': 'gzip'
        },
    }
    if value is not None:
      tspec['metadata']['shape'] = value.shape
      tspec['metadata']['chunks'] = value.shape
    return tspec

  async def serialize(self,
                      value: np.ndarray,
                      info: ParamInfo,
                      args: Optional[SaveArgs] = None) -> List[Future]:
    """Uses Tensorstore to serialize a numpy array."""
    args = args or SaveArgs()
    tspec = self._get_json_tspec(info, value)
    tspec = _get_cast_tspec_serialize(tspec, value, args)
    t = await ts.open(
        ts.Spec(tspec),
        create=True,
        open=True,
        context=ts.Context({'file_io_concurrency': {
            'limit': 128
        }}))
    write_future = t.write(value)
    await write_future.copy
    return [write_future.commit]

  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> np.ndarray:
    """Deserializes the array using Tensorstore."""
    args = args or RestoreArgs()
    tspec = self._get_json_tspec(info)
    tspec = _get_cast_tspec_deserialize(tspec, args)
    t = await ts.open(ts.Spec(tspec), open=True)
    return await t.read()


class ScalarHandler(NumpyHandler):
  """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.).
  """

  async def serialize(self,
                      value: Scalar,
                      info: ParamInfo,
                      args: Optional[SaveArgs] = None) -> List[Future]:
    """See superclass documentation."""
    value = np.asarray(value)
    return await super().serialize(value, info, args)

  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> np.ndarray:
    """See superclass documentation."""
    result = await super().deserialize(info, args)
    if result.ndim != 0:
      raise ValueError('Restored result is not a scalar.')
    return result.item()


@dataclasses.dataclass
class ArrayRestoreArgs(RestoreArgs):
  """Arguments used when restoring with ArrayHandler.

  mesh: the device mesh that the array should be restored as. Cannot be None.
  mesh_axes: the mesh_axes that the array should be restored as. Cannot be None.
  global_shapes: the global shape that the array should be restored into. If not
    provided, the shape will be restored as written.
  """
  restore_type: Any = jax.Array
  mesh: Optional[Mesh] = None
  mesh_axes: Optional[jax.sharding.PartitionSpec] = None
  global_shape: Optional[Tuple[int]] = None


class ArrayHandler(TypeHandler):
  """An implementation of TypeHandler for jax.Array."""

  def __init__(
      self, metadata_key: Optional[str] = None
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
    """
    self._metadata_key = metadata_key

  def _get_json_tspec(self,
                      info: ParamInfo,
                      value: Optional[Any] = None) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec: Dict[str, Any] = get_tensorstore_spec(path)
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    if value is not None:
      tspec['metadata'] = serialization._get_metadata(value)  # pylint: disable=protected-access
      del tspec['metadata']['dtype']
    return tspec

  async def serialize(
      self, value: jax.Array, info: ParamInfo, args: Optional[SaveArgs] = None
  ) -> List[Future]:
    """See superclass documentation."""
    args = args or SaveArgs()
    tspec = self._get_json_tspec(info, value)
    tspec = _get_cast_tspec_serialize(tspec, value, args)
    commit_futures = []
    await serialization.async_serialize(
        value, tspec, commit_future=commit_futures)
    return commit_futures

  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> Any:
    """See superclass documentation.

    Args:
      info: ParamInfo.
      args: must be of type `ArrayRestoreArgs`.

    Returns:
      The deserialized parameter.

    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.mesh` or `args.mesh_axes` are not provided.
    """
    if args is None:
      raise ValueError('Must provide ArrayRestoreArgs to restore as jax.Array.')
    args = cast(ArrayRestoreArgs, args)
    if args.mesh is None or args.mesh_axes is None:
      raise ValueError(
          'Sharding of jax.Array cannot be None. Provide `mesh`'
          ' and `mesh_axes`.'
      )
    tspec = self._get_json_tspec(info)
    tspec = _get_cast_tspec_deserialize(tspec, args)
    s = jax.sharding.NamedSharding(args.mesh, args.mesh_axes)
    return await serialization.async_deserialize(
        s,
        tspec,
        global_shape=args.global_shape,
        byte_limiter=info.byte_limiter,
    )


class StringHandler(TypeHandler):
  """TypeHandler for strings that enforces aggregation."""

  async def serialize(self,
                      value: str,
                      info: ParamInfo,
                      args: Optional[SaveArgs] = None) -> List[Future]:
    """See superclass documentation."""
    args = args or SaveArgs()
    if not args.aggregate:
      raise ValueError('Non-aggregated string serialization is not supported.')
    return []

  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> str:
    """See superclass documentation."""
    raise NotImplementedError


_TYPE_REGISTRY = [
    (lambda ty: issubclass(ty, int), ScalarHandler()),
    (lambda ty: issubclass(ty, float), ScalarHandler()),
    (lambda ty: issubclass(ty, np.number), ScalarHandler()),
    (lambda ty: issubclass(ty, np.ndarray), NumpyHandler()),
    (lambda ty: issubclass(ty, jax.Array) and jax.config.jax_array,
     ArrayHandler()),
    (lambda ty: issubclass(ty, str), StringHandler()),
]


def register_type_handler(ty: Any,
                          handler: TypeHandler,
                          func: Optional[Callable[[Any], bool]] = None,
                          override: bool = False):
  """Registers a type for serialization/deserialization with a given handler.

  Note that it is possible for a type to match multiple different entries in
  the registry, each with a different handler. In this case, only the first
  match is used.

  Args:
    ty: A type to register.
    handler: a TypeHandler capable of reading and writing parameters of type
      `ty`.
    func: A function that accepts a type and returns True if the type should be
      handled by the provided TypeHandler. If not specified, defaults to
      `lambda t: issubclass(t, ty)`.
    override: if True, will override an existing mapping of type to handler.

  Raises:
    ValueError if a type is already registered and override is False.
  """
  if func is None:
    func = lambda t: issubclass(t, ty)

  existing_handler_idx = None
  for i, (f, _) in enumerate(_TYPE_REGISTRY):
    if f(ty):
      existing_handler_idx = i
      # Ignore the possibility for subsequent matches, as these will not be used
      # anyway.
      break

  if existing_handler_idx is None:
    _TYPE_REGISTRY.append((func, handler))
  elif override:
    _TYPE_REGISTRY[existing_handler_idx] = (func, handler)
  else:
    raise ValueError(f'A TypeHandler for "{ty}" is already registered.')


def get_type_handler(ty: Any) -> TypeHandler:
  """Returns the handler registered for a given type, if available.

  Args:
    ty: an object type.

  Returns:
    The TypeHandler that is registered for the given type.

  Raises:
    ValueError if the given type has no registered handler.
  """
  for func, handler in _TYPE_REGISTRY:
    if func(ty):
      return handler
  raise ValueError(f'Unknown type: "{ty}". Must register a TypeHandler.')
