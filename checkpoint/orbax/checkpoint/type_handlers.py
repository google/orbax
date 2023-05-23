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

"""Provides utils for PytreeCheckpointHandler."""

import abc
import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint.future import Future
import tensorstore as ts


Scalar = Union[int, float, np.number]
_OCDBT_MANIFEST_FILE = 'manifest.ocdbt'
_COORDINATOR_SETUP_TIMEOUT_SECS = 30


def _get_coordinator_address_without_port(
    coordinator_address: Optional[str],
) -> str:
  """Returns JAX coordinator address stripped of port number."""
  if not coordinator_address:
    raise ValueError('Coordinator address not set.')
  return coordinator_address.split(':')[0]


def create_coordinator_server_and_context() -> (
    Tuple[ts.Context, Optional[ts.ocdbt.DistributedCoordinatorServer]]
):
  """Creates OCDBT coordinator and Tensorstore context.

  This function must be called at the start of the program across all processes
  in order to initialize the OCDBT coordinator service. The coordinator object
  should be kept alive for the life of the program, while the returned
  ts.Context should be provided when saving or restoring using Tensorstore.

  Example usage:
  ```
  ocdbt_context, coordinator_server = (
      orbax.checkpoint.type_handlers.create_coordinator_server_and_context()
  )
  orbax.checkpoint.type_handlers.register_standard_handlers_with_options(
      use_ocdbt=True, ts_context=ocdbt_context
  )
  orbax.checkpoint.utils.sync_global_devices('init_ocdbt_server')
  ```

  Returns:
    Tuple of ts.Context and OCDBT coordinator server object.
  """
  jax_global_state = jax._src.distributed.global_state  # pylint: disable=protected-access
  ocdbt_address = _get_coordinator_address_without_port(
      jax_global_state.coordinator_address
  )

  coordinator_server = None
  if jax_global_state.process_id == 0:
    bind_address = f'{ocdbt_address}:0'
    logging.info('Starting DistributedCoordinatorServer at: %s', bind_address)
    coordinator_server = ts.ocdbt.DistributedCoordinatorServer({
        'bind_addresses': [bind_address],
    })
    jax_global_state.client.key_value_set(
        'ocdbt_coordinator', f'{ocdbt_address}:{coordinator_server.port}'
    )

  ocdbt_address = jax_global_state.client.blocking_key_value_get(
      'ocdbt_coordinator', _COORDINATOR_SETUP_TIMEOUT_SECS * 1000
  )
  ts_context = {
      'ocdbt_coordinator': {
          'address': ocdbt_address,
      },
      # Provide cache pool for B-tree nodes to avoid repeated reads.
      'cache_pool#ocdbt': {'total_bytes_limit': 100000000},
  }
  return (
      ts.Context(ts_context, parent=serialization.TS_CONTEXT),
      coordinator_server,
  )


async def _assert_parameter_files_exist(
    param_dir: epath.Path, metadata_key: Optional[str]
):
  """Checks for existence of parameter subdir and .zarray file."""
  exists = await utils.async_exists(param_dir)
  if not exists:
    raise FileNotFoundError(
        f'Individual parameter subdirectory not found at path: {param_dir}.'
    )
  if metadata_key is None:
    metadata_key = '.zarray'
  metadata_path = param_dir / metadata_key
  exists = await utils.async_exists(metadata_path)
  if not exists:
    raise FileNotFoundError(
        f'File not found: {metadata_path}. In many cases, this results from'
        ' copying a checkpoint without using the `-a` flag.'
    )


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
  byte_limiter: object to limit the number of bytes that can be read in
    parallel.
  is_ocdbt_checkpoint: indicates whether the checkpoint path uses OCDBT format
    or not. Only used for restoration.
  """
  name: Optional[str] = None
  path: Optional[epath.Path] = None
  aggregate: Optional[bool] = None
  byte_limiter: Optional[serialization._LimitInFlightBytes] = None  # pylint: disable=protected-access
  is_ocdbt_checkpoint: Optional[bool] = None


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


def is_ocdbt_checkpoint(path: epath.Path) -> bool:
  """Determines whether a checkpoint uses OCDBT format."""
  return (path / _OCDBT_MANIFEST_FILE).exists()


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


def _add_base_tspec_ocdbt_options(tspec: Dict[str, Any]) -> Dict[str, Any]:
  tspec.update({'recheck_cached_data': False, 'recheck_cached_metadata': False})
  tspec['kvstore'].update({  # Enable read coalescing.
      'experimental_read_coalescing_threshold_bytes': 1000000,
      # References the cache specified in ts.Context.
      'cache_pool': 'cache_pool#ocdbt',
  })
  return tspec


def _add_write_tspec_ocdbt_options(tspec: Dict[str, Any]) -> Dict[str, Any]:
  tspec['kvstore']['config'] = {
      # Store .zarray metadata inline but not large chunks.
      'max_inline_value_bytes': 1024,
      # Large value allows a single root node to support faster traversal.
      'max_decoded_node_bytes': 100000000,
  }
  return tspec


class NumpyHandler(TypeHandler):
  """Provides an implementation of TypeHandler for replicated numpy arrays."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
      use_ocdbt: bool = False,
      ts_context: Optional[ts.Context] = None,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
      use_ocdbt: enables Tensorstore OCDBT driver.
      ts_context: Tensorstore context.
    """
    self._metadata_key = metadata_key
    self._use_ocdbt = use_ocdbt
    if self._use_ocdbt and not ts_context:
      raise ValueError(
          'Must provide a ts.Context if use_ocdbt is True. Ensure that the'
          ' context contains a coordinator address.'
      )
    self._ts_context = ts_context or ts.Context(
        {'file_io_concurrency': {'limit': 128}}
    )

  def _get_json_tspec(
      self,
      info: ParamInfo,
      use_ocdbt: bool = False,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec: Dict[str, Any] = get_tensorstore_spec(path, ocdbt=use_ocdbt)
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    if use_ocdbt:
      tspec = _add_base_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_write(
      self, info: ParamInfo, value: Any, use_ocdbt: bool = False
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for writing."""
    tspec = self._get_json_tspec(info, use_ocdbt=use_ocdbt)
    tspec['metadata'] = {
        'compressor': {
            'id': 'zstd'
        },
    }
    if value is not None:
      tspec['metadata']['shape'] = value.shape
      tspec['metadata']['chunks'] = value.shape
    if use_ocdbt:
      tspec = _add_write_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_read(
      self, info: ParamInfo, use_ocdbt: bool = False
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for reading."""
    return self._get_json_tspec(info, use_ocdbt=use_ocdbt)

  async def serialize(self,
                      value: np.ndarray,
                      info: ParamInfo,
                      args: Optional[SaveArgs] = None) -> List[Future]:
    """Uses Tensorstore to serialize a numpy array."""
    args = args or SaveArgs()
    tspec = self._get_json_tspec_write(info, value, use_ocdbt=self._use_ocdbt)
    tspec = _get_cast_tspec_serialize(tspec, value, args)
    t = await ts.open(
        ts.Spec(tspec), create=True, open=True, context=self._ts_context
    )
    write_future = t.write(value)
    await write_future.copy
    return [write_future.commit]

  async def deserialize(self,
                        info: ParamInfo,
                        args: Optional[RestoreArgs] = None) -> np.ndarray:
    """Deserializes the array using Tensorstore."""
    args = args or RestoreArgs()
    if not info.is_ocdbt_checkpoint:
      await _assert_parameter_files_exist(info.path, self._metadata_key)
    # Using OCDBT, but existing checkpoint may be stored in old format.
    use_ocdbt = self._use_ocdbt and info.is_ocdbt_checkpoint
    tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
    tspec = _get_cast_tspec_deserialize(tspec, args)
    t = await ts.open(ts.Spec(tspec), open=True, context=self._ts_context)
    return await t.read()


class ScalarHandler(NumpyHandler):
  """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.).
  """

  async def serialize(self,
                      value: Scalar,  # pytype: disable=signature-mismatch  # numpy-scalars
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
  sharding: jax.sharding.Sharding object which takes precedence over mesh and
    mesh_axes if provided. Otherwise, mesh and mesh_axes will be used to
    construct a NamedSharding object.
  global_shapes: the global shape that the array should be restored into. If not
    provided, the shape will be restored as written. Presently, arbitrary shape
    transformations are not supported (for example, reshaping to different
    dimensions). Padding and truncating are supported. When the global_shape is
    greater than that of the saved array, 0's will be appended. If the
    global_shape is shorter than that of the saved array, excess elements will
    be dropped from the end of the array.
  """
  restore_type: Any = jax.Array
  mesh: Optional[Mesh] = None
  mesh_axes: Optional[jax.sharding.PartitionSpec] = None
  sharding: Optional[jax.sharding.Sharding] = None
  global_shape: Optional[Tuple[int]] = None


class ArrayHandler(TypeHandler):
  """An implementation of TypeHandler for jax.Array."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
      use_ocdbt: bool = False,
      ts_context: Optional[ts.Context] = None,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
      use_ocdbt: allows using Tensorstore OCDBT driver.
      ts_context: Tensorstore context.
    """
    self._metadata_key = metadata_key
    self._use_ocdbt = use_ocdbt
    if self._use_ocdbt and not ts_context:
      raise ValueError(
          'Must provide a ts.Context if use_ocdbt is True. Ensure that the'
          ' context contains a coordinator address.'
      )
    self._ts_context = ts_context or ts.Context(
        {'file_io_concurrency': {'limit': 128}}
    )

  def _get_json_tspec(
      self,
      info: ParamInfo,
      use_ocdbt: bool = False,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec: Dict[str, Any] = get_tensorstore_spec(path, ocdbt=use_ocdbt)
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    if use_ocdbt:
      tspec = _add_base_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_write(
      self, info: ParamInfo, value: Any, use_ocdbt: bool = False
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for writing."""
    tspec = self._get_json_tspec(info, use_ocdbt=use_ocdbt)
    tspec['metadata'] = serialization._get_metadata(value)  # pylint: disable=protected-access
    del tspec['metadata']['dtype']
    if use_ocdbt:
      tspec = _add_write_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_read(
      self, info: ParamInfo, use_ocdbt: bool = False
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for reading."""
    return self._get_json_tspec(info, use_ocdbt=use_ocdbt)

  async def serialize(
      self, value: jax.Array, info: ParamInfo, args: Optional[SaveArgs] = None
  ) -> List[Future]:
    """See superclass documentation."""
    if (
        isinstance(value, jax.Array)
        and jax.process_count() > 1
        and value.is_fully_addressable
    ):
      raise ValueError(
          'Cannot serialize host local arrays. Arrays like this are typically'
          ' obtained using pmap. Consider using'
          ' fully_replicated_host_local_array_to_global_array in'
          ' orbax/checkpoint/utils.py to convert your arrays into serializable'
          ' objects.'
      )
    args = args or SaveArgs()
    tspec = self._get_json_tspec_write(info, value, use_ocdbt=self._use_ocdbt)
    tspec = _get_cast_tspec_serialize(tspec, value, args)
    commit_futures = []
    await serialization.async_serialize(
        value, tspec, commit_future=commit_futures, context=self._ts_context
    )
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
    if args.sharding is None and (args.mesh is None or args.mesh_axes is None):
      raise ValueError(
          'Sharding of jax.Array cannot be None. Provide `mesh`'
          ' and `mesh_axes` OR `sharding`.'
      )
    if args.sharding is None:
      sharding = jax.sharding.NamedSharding(args.mesh, args.mesh_axes)
    else:
      sharding = args.sharding
    if not info.is_ocdbt_checkpoint:
      await _assert_parameter_files_exist(info.path, self._metadata_key)
    # Using OCDBT, but existing checkpoint may be stored in old format.
    use_ocdbt = self._use_ocdbt and info.is_ocdbt_checkpoint
    tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
    tspec = _get_cast_tspec_deserialize(tspec, args)
    return await serialization.async_deserialize(
        sharding,
        tspec,
        global_shape=args.global_shape,
        byte_limiter=info.byte_limiter,
        context=self._ts_context,
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
    (lambda ty: issubclass(ty, bytes), ScalarHandler()),
    (lambda ty: issubclass(ty, np.number), ScalarHandler()),
    (lambda ty: issubclass(ty, np.ndarray), NumpyHandler()),
    (lambda ty: issubclass(ty, jax.Array), ArrayHandler()),
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


def has_type_handler(ty: Any) -> bool:
  try:
    get_type_handler(ty)
    return True
  except ValueError:
    return False


def register_standard_handlers_with_options(**kwargs):
  """Re-registers a select set of handlers with the given options."""
  register_type_handler(int, ScalarHandler(**kwargs), override=True)
  register_type_handler(float, ScalarHandler(**kwargs), override=True)
  register_type_handler(
      np.number,
      ScalarHandler(**kwargs),
      override=True,
  )
  register_type_handler(
      np.ndarray,
      NumpyHandler(**kwargs),
      override=True,
  )
  register_type_handler(
      jax.Array,
      ArrayHandler(**kwargs),
      override=True,
  )
