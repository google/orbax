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
import asyncio
import base64
import copy
import dataclasses
import enum
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast
import warnings

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint import value_metadata
from orbax.checkpoint.future import Future
import tensorstore as ts


Scalar = Union[int, float, np.number]
Metadata = value_metadata.Metadata
NamedSharding = jax.sharding.NamedSharding
ScalarMetadata = value_metadata.ScalarMetadata
ArrayMetadata = value_metadata.ArrayMetadata
StringMetadata = value_metadata.StringMetadata
_MESH_AXES = 'axis_names'
_MESH_SHAPE = 'shape'
_NAMED_SHARDING = 'NamedSharding'
_OCDBT_MANIFEST_FILE = 'manifest.ocdbt'
_COORDINATOR_SETUP_TIMEOUT_SECS = 300
_OCDBT_TS_CONTEXT = None
_OCDBT_COORDINATOR_SERVER = None
_PARTITION_SPEC = 'partition_spec'
_SHARDING = '_sharding'
_SHARDING_TYPE = 'sharding_type'

RESTORE_TYPE_NONE = 'None'
RESTORE_TYPE_DICT = 'Dict'
RESTORE_TYPE_LIST = 'List'


class ShardingTypes(enum.Enum):
  NAMED_SHARDING = 'NamedSharding'
  SINGLE_DEVICE_SHARDING = 'SingleDeviceSharding'
  POSITIONAL_SHARDING = 'PositionalSharding'
  GSPMD_SHARDING = 'GSPMDSharding'


def _get_coordinator_address_without_port(coordinator_address: str) -> str:
  """Returns JAX coordinator address stripped of port number."""
  return coordinator_address.split(':')[0]


def _enable_ocdbt_for_handlers():
  # TODO(b/293331479) remove this once OCDBT is enabled by default
  global _TYPESTR_REGISTRY
  if _OCDBT_TS_CONTEXT is not None:
    for _, handler in _TYPE_REGISTRY:
      if hasattr(handler, 'enable_ocdbt') and callable(handler.enable_ocdbt):
        handler.enable_ocdbt(_OCDBT_TS_CONTEXT)
    _TYPESTR_REGISTRY = _make_typestr_registry(_TYPE_REGISTRY)


def _serialize_sharding(sharding: jax.sharding.Sharding) -> str:
  """Serializes `jax.sharding.Sharding` into a json string."""

  if isinstance(sharding, NamedSharding):
    sharding_data = {}

    sharding_data[_SHARDING_TYPE] = ShardingTypes.NAMED_SHARDING.value
    sharding_data[_MESH_SHAPE] = list(sharding.mesh.shape.values())
    sharding_data[_MESH_AXES] = sharding.mesh.axis_names
    sharding_data[_PARTITION_SPEC] = sharding.spec

    serialized_string = json.dumps(sharding_data)
    return serialized_string

  elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
    warnings.warn(
        'Serialization for `jax.sharding.SingleDeviceSharding` has not been'
        ' implemented.'
    )

  elif isinstance(sharding, jax.sharding.PositionalSharding):
    warnings.warn(
        'Serialization for `jax.sharding.PositionalSharding` has not been'
        ' implemented.'
    )

  elif isinstance(sharding, jax.sharding.GSPMDSharding):
    warnings.warn(
        'Serialization for `jax.sharding.PositionalSharding` has not been'
        ' implemented.'
    )

  else:
    warnings.warn(f'Sharding type {type(sharding)} is not supported.')

  return ''


def _deserialize_sharding_from_json_string(
    sharding_string: str,
) -> jax.sharding.Sharding:
  """Deserializes a json string to `jax.sharding.Sharding`."""

  deserialized_dict = json.loads(sharding_string)

  if deserialized_dict[_SHARDING_TYPE] == ShardingTypes.NAMED_SHARDING.value:
    shape = deserialized_dict[_MESH_SHAPE]
    axis_names = list(deserialized_dict[_MESH_AXES])
    partition_spec = tuple(deserialized_dict[_PARTITION_SPEC])

    sharding = NamedSharding(
        jax.sharding.Mesh(
            np.array(jax.devices()).reshape(shape), axis_names=axis_names
        ),
        jax.sharding.PartitionSpec(*partition_spec),
    )
    return sharding

  else:
    raise NotImplementedError(
        'Sharding types other than `jax.sharding.NamedSharding` have not been '
        'implemented.'
    )


def create_coordinator_server_and_context() -> Tuple[None, None]:
  # TODO(b/293331479) remove this once OCDBT is enabled by default
  warnings.warn('This function has been deprecated.  Do not use.')
  return (None, None)


def start_coordinator_server_and_create_context() -> None:
  """Start a OCDBT coordinator and create a Tensorstore context.

  This function is only for Orbax internal use.

  The following function starts a coordinator_server and update type handlers
  with enable_ocdbt() defined.

  The context and server will be stored as global variables in _OCDBT_TS_CONTEXT
  and _OCDBT_COORDINATOR_SERVER.  They will be preserved for the life of the
  program.  Succeeding calls to this function will not try to start the
  coordinator server again.

  For testing purpose, if one needs to restart the coordinator server, set
  _OCDBT_TS_CONTEXT and _OCDBT_COORDINATOR_SERVER to None and call this function
  again.

  Returns:
    None
  """
  global _OCDBT_TS_CONTEXT, _OCDBT_COORDINATOR_SERVER

  if _OCDBT_TS_CONTEXT is not None:
    # OCDBT ts_context is already set, return
    return

  ts_context = {
      # Provide cache pool for B-tree nodes to avoid repeated reads.
      # 100MB limit.
      'cache_pool#ocdbt': {'total_bytes_limit': 100000000},
  }

  jax_global_state = jax._src.distributed.global_state  # pylint: disable=protected-access
  if (
      jax_global_state.coordinator_address
      and jax_global_state.num_processes > 1
  ):
    ocdbt_address = _get_coordinator_address_without_port(
        jax_global_state.coordinator_address
    )

    if jax_global_state.process_id == 0:
      bind_address = f'{ocdbt_address}:0'
      _OCDBT_COORDINATOR_SERVER = ts.ocdbt.DistributedCoordinatorServer(
          {
              'bind_addresses': [bind_address],
          }
      )
      ocdbt_coordinator = f'{ocdbt_address}:{_OCDBT_COORDINATOR_SERVER.port}'
      logging.info(
          'Started OCDBT DistributedCoordinatorServer at: %s', ocdbt_coordinator
      )
      jax_global_state.client.key_value_set(
          'ocdbt_coordinator', ocdbt_coordinator
      )

    ocdbt_address = jax_global_state.client.blocking_key_value_get(
        'ocdbt_coordinator', _COORDINATOR_SETUP_TIMEOUT_SECS * 1000
    )

    # add ocdbt_coordinator spec into ts_context
    ts_context['ocdbt_coordinator'] = {
        'address': ocdbt_address,
    }

  _OCDBT_TS_CONTEXT = ts.Context(ts_context, parent=serialization.TS_CONTEXT)
  _enable_ocdbt_for_handlers()
  logging.info('OCDBT is initialized successfully.')


def _use_ocdbt_for_restore(
    maybe_use_ocdbt: bool, checkpoint_is_ocdbt: bool
) -> bool:
  """Determines whether the checkpoint should be restored using OCDBT."""
  if not maybe_use_ocdbt and checkpoint_is_ocdbt:
    raise ValueError(
        'TypeHandler is not configured to allow OCDBT restoration, but found'
        ' OCDBT checkpoint.'
    )
  return maybe_use_ocdbt and checkpoint_is_ocdbt


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


def get_empty_value_typestr(value: Any) -> str:
  if not utils.is_supported_empty_aggregation_type(value):
    raise ValueError(f'{value} is not a supported empty aggregation type.')
  if isinstance(value, list):
    return RESTORE_TYPE_LIST
  elif isinstance(value, dict):
    return RESTORE_TYPE_DICT
  elif isinstance(value, type(None)):
    return RESTORE_TYPE_NONE
  else:
    raise ValueError(f'Unrecognized empty type: {value}.')


def is_empty_typestr(typestr: str) -> bool:
  return (
      typestr == RESTORE_TYPE_LIST
      or typestr == RESTORE_TYPE_DICT
      or typestr == RESTORE_TYPE_NONE
  )


def get_empty_value_from_typestr(typestr: str) -> Any:
  if typestr == RESTORE_TYPE_LIST:
    return []
  elif typestr == RESTORE_TYPE_DICT:
    return {}
  elif typestr == RESTORE_TYPE_NONE:
    return None
  else:
    raise ValueError(f'Unrecognized typestr: {typestr}.')


@dataclasses.dataclass
class ParamInfo:
  """Information describing a parameter in a PyTree.

  Note that ParamInfo is distinct from SaveArgs and RestoreArgs in that in
  represents information not provided by a user, and should be computed
  internally.

  name:
    Name of the parameter.
  path:
    A path providing a location where file(s) should be saved. The path is
    assumed to be a directory.
  parent_dir:
    A path providing location where all files under the same checkpoint should
    be saved under. All `ParamInfo` provided to a given TypeHandler should have
    the same `parent_dir`. The parent_dir is assumed to be a directory.
  skip_deserialize:
    If specified, skips deserialization of the given parameter using the
    TypeHandler. This may be for multiple different reasons, including that the
    parameter may have been aggregated, or it will be unneeded after
    transformations. Note: this parameter is handled by PyTreeCheckpointHandler,
    so it is unnecessary for TypeHandler implementations to deal with it.
  byte_limiter:
    Object to limit the number of bytes that can be read in
    parallel.
  is_ocdbt_checkpoint:
    Indicates whether the checkpoint path uses OCDBT format
    or not. Only used for restoration.
  """

  name: Optional[str] = None
  path: Optional[epath.Path] = None
  parent_dir: Optional[epath.Path] = None
  skip_deserialize: Optional[bool] = None
  byte_limiter: Optional[serialization._LimitInFlightBytes] = (
      None  # pylint: disable=protected-access
  )
  is_ocdbt_checkpoint: Optional[bool] = None


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  aggregate:
    If true, saves the given parameter in an aggregated tree format
    rather than individually. See AggregateHandler.
  dtype:
    If provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """

  aggregate: bool = False
  dtype: Optional[jnp.dtype] = None


@dataclasses.dataclass
class RestoreArgs:
  """Extra arguments that can be provided for restoration.

  restore_type:
    Specifies the object type of the restored parameter. The type
    must have a corresponding TypeHandler for restoration. Ignored if the
    parameter is restored from an aggregated checkpoint file.
  dtype:
    If provided, casts the parameter to the given dtype after restoring.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  """

  restore_type: Optional[Any] = None
  dtype: Optional[jnp.dtype] = None


class TypeHandler(abc.ABC):
  """Interface for reading and writing a PyTree leaf."""

  @abc.abstractmethod
  def typestr(self) -> str:
    """A string representation of the type.

    Cannot conflict with other types.

    Returns:
      The type as a string.
    """
    pass

  @abc.abstractmethod
  async def metadata(self, infos: Sequence[ParamInfo]) -> Sequence[Metadata]:
    """Constructs object metadata from a stored parameter location.

    Args:
      infos: sequence of ParamInfo

    Returns:
      Sequence of Metadata for each provided ParamInfo.
    """
    pass

  @abc.abstractmethod
  async def serialize(
      self,
      values: Sequence[Any],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[SaveArgs]] = None,
  ) -> Sequence[Future]:
    """Writes the parameter to a storage location.

    This method is responsible for copying the parameter from a remote device in
    a synchronous fashion (if applicable). It should then return a list of
    futures which can be later awaited to complete the final commit operation
    to a storage location.

    The function can be used in a multihost setting, but should not implement
    extra logic to ensure atomicity.

    Args:
      values: a sequence of parameters to save.
      infos: a sequence of ParamInfo containing relevant information for
        serialization of each value.
      args: a sequnece of additional arguments for serialization, provided by
        the user.

    Returns:
      Sequence of commit futures which can be awaited to complete the save
      operation.
    """
    pass

  @abc.abstractmethod
  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[Any]:
    """Reads the parameter from a storage location.

    Args:
      infos: Sequnece of ParamInfo for deserialization.
      args: Sequence of user-provided restoration information.

    Returns:
      The deserialized parameters.
    """
    pass


def check_input_arguments(*args):
  l = None
  for arg in args:
    if l == 0:
      raise ValueError('Cannot pass TypeHandler input of length 0.')
    if l is None:
      l = len(arg)
    elif len(arg) != l:
      raise ValueError('Found input args with mismatched lengths.')


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
  """Add base Tensorstore config parameters for OCDBT."""
  tspec.update({'recheck_cached_data': False, 'recheck_cached_metadata': False})
  tspec['kvstore'].update({
      # Enable read coalescing.  This feature merges adjacent read_ops into
      # one, which could reduce I/O ops by a factor of 10. This is especially
      # beneficial for unstacked models.
      'experimental_read_coalescing_threshold_bytes': 1000000,
      'experimental_read_coalescing_merged_bytes': 500000000000,
      'experimental_read_coalescing_interval': '1ms',
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


def _array_metadata_from_tensorstore(
    t: Any, sharding: Optional[jax.sharding.Sharding] = None
) -> ArrayMetadata:
  return ArrayMetadata(
      shape=t.shape,
      dtype=jnp.dtype(t.dtype.name),
      sharding=sharding,
  )


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
    self._ts_context = ts_context or serialization.TS_CONTEXT

  def enable_ocdbt(self, ts_context: ts.Context) -> None:
    self._use_ocdbt = True
    self._ts_context = ts_context

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
      self, info: ParamInfo, value: np.ndarray, use_ocdbt: bool = False
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for writing."""
    tspec = self._get_json_tspec(info, use_ocdbt=use_ocdbt)
    tspec['metadata'] = {
        'compressor': {'id': 'zstd'},
    }
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

  def typestr(self) -> str:
    return 'np.ndarray'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[ArrayMetadata]:
    open_ops = []
    for info in infos:
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = _use_ocdbt_for_restore(
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=self._ts_context)
      )

    tensorstores = await asyncio.gather(*open_ops)
    return [
        _array_metadata_from_tensorstore(t, sharding=None)
        for (t) in tensorstores
    ]

  async def serialize(
      self,
      values: Sequence[np.ndarray],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[SaveArgs]] = None,
  ) -> Sequence[Future]:
    """Uses Tensorstore to serialize a numpy array."""
    args = args or [SaveArgs()] * len(values)
    check_input_arguments(values, infos, args)
    copy_ops = []
    futures = []
    for value, info, arg in zip(values, infos, args):
      tspec = self._get_json_tspec_write(info, value, use_ocdbt=self._use_ocdbt)
      tspec = _get_cast_tspec_serialize(tspec, value, arg)
      if jax.process_index() == 0:
        # Open once to create metadata and allow the operation to happen
        # asynchronously.
        open_future = ts.open(
            ts.Spec(tspec), create=True, open=True, context=self._ts_context
        )
        # Open again (no disk I/O) to get the write location.
        t = await ts.open(
            ts.Spec(tspec),
            open=True,
            assume_metadata=True,
            context=self._ts_context,
        )
        write_future = t.write(value)
        copy_ops += [write_future.copy]
        futures += [open_future, write_future.commit]
    await asyncio.gather(*copy_ops)
    return futures

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[np.ndarray]:
    """Deserializes the array using Tensorstore."""
    args = args or [RestoreArgs()] * len(infos)
    check_input_arguments(infos, args)
    open_futures = []
    for info, arg in zip(infos, args):
      if not info.is_ocdbt_checkpoint:
        await _assert_parameter_files_exist(info.path, self._metadata_key)
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = _use_ocdbt_for_restore(
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = _get_cast_tspec_deserialize(tspec, arg)
      open_futures += [
          ts.open(ts.Spec(tspec), open=True, context=self._ts_context)
      ]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [t.read() for t in tensorstores]
    return await asyncio.gather(*read_ops)


class ScalarHandler(NumpyHandler):
  """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.)."""

  def typestr(self) -> str:
    return 'scalar'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[ScalarMetadata]:
    metadatas = await super().metadata(infos)
    return [ScalarMetadata(dtype=m.dtype) for m in metadatas]

  async def serialize(
      self,
      values: Sequence[Scalar],  # pytype: disable=signature-mismatch
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[SaveArgs]] = None,
  ) -> Sequence[Future]:
    """See superclass documentation."""
    values = [np.asarray(v) for v in values]
    return await super().serialize(values, infos, args)

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[Scalar]:  # pytype: disable=signature-mismatch
    """See superclass documentation."""
    results = await super().deserialize(infos, args)
    for r in results:
      if r.ndim != 0:
        raise ValueError('Restored result is not a scalar.')
    return [r.item() for r in results]


@dataclasses.dataclass
class ArrayRestoreArgs(RestoreArgs):
  """Arguments used when restoring with ArrayHandler.

  mesh:
    The device mesh that the array should be restored as. Cannot be None.
  mesh_axes:
    The mesh_axes that the array should be restored as. Cannot be None.
  sharding:
    jax.sharding.Sharding object which takes precedence over mesh and
    mesh_axes if provided. Otherwise, mesh and mesh_axes will be used to
    construct a NamedSharding object.
  global_shape:
    The global shape that the array should be restored into. If not
    provided, the shape will be restored as written. Presently, arbitrary shape
    transformations are not supported (for example, reshaping to different
    dimensions). Padding and truncating are supported. When the global_shape is
    greater than that of the saved array, 0's will be appended. If the
    global_shape is shorter than that of the saved array, excess elements will
    be dropped from the end of the array.
  """

  mesh: Optional[Mesh] = None
  mesh_axes: Optional[jax.sharding.PartitionSpec] = None
  sharding: Optional[jax.sharding.Sharding] = None
  global_shape: Optional[Tuple[int, ...]] = None


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
    self._ts_context = ts_context or serialization.TS_CONTEXT

  def enable_ocdbt(self, ts_context: ts.Context) -> None:
    self._use_ocdbt = True
    self._ts_context = ts_context

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

  def typestr(self) -> str:
    return 'jax.Array'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[ArrayMetadata]:
    open_ops = []
    sharding_open_ops = []
    shardings = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING
    sharding_file_exists = await utils.async_exists(sharding_file_path)
    for info in infos:
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = _use_ocdbt_for_restore(
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=self._ts_context)
      )
      sharding_info = copy.deepcopy(info)
      sharding_info = dataclasses.replace(
          sharding_info, path=sharding_file_path
      )
      tspec_sharding = self._get_json_tspec_read(sharding_info, use_ocdbt=False)
      sharding_op = None
      if sharding_info.name:
        tspec_sharding = {
            'driver': 'json',
            'kvstore': tspec_sharding['kvstore'],
            'json_pointer': '/' + base64.urlsafe_b64encode(
                sharding_info.name.encode()
            ).decode('utf-8'),
        }
        if sharding_file_exists:
          sharding_op = ts.open(
              tspec_sharding, open=True, read=True, context=self._ts_context
          )
      sharding_open_ops.append(sharding_op)
    tensorstores = await asyncio.gather(*open_ops)
    if sharding_file_exists:
      sharding_tensorstores = await asyncio.gather(*sharding_open_ops)
      for sharding_tensorstore in sharding_tensorstores:
        if sharding_tensorstore:
          sharding_string = await sharding_tensorstore.read()
          if not sharding_string.item():
            shardings.append(None)
            continue
          deserialized = _deserialize_sharding_from_json_string(
              sharding_string.item()
          )
          shardings.append(deserialized or None)
        else:
          shardings.append(None)
    else:
      shardings = [None] * len(tensorstores)
    return [
        _array_metadata_from_tensorstore(t, sharding)
        for (t, sharding) in zip(tensorstores, shardings)
    ]

  async def serialize(
      self,
      values: Sequence[jax.Array],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[SaveArgs]] = None,
  ) -> Sequence[Future]:
    """See superclass documentation."""
    for v in values:
      if (
          isinstance(v, jax.Array)
          and jax.process_count() > 1
          and v.is_fully_addressable
      ):
        raise ValueError(
            'Cannot serialize host local arrays. Arrays like this are typically'
            ' obtained using pmap. Consider using'
            ' fully_replicated_host_local_array_to_global_array in'
            ' orbax/checkpoint/utils.py to convert your arrays into'
            ' serializable objects.'
        )
    args = args or [SaveArgs()] * len(values)
    check_input_arguments(values, infos, args)
    synchronous_ops = []
    futures = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING
    for value, info, arg in zip(values, infos, args):
      tspec = self._get_json_tspec_write(info, value, use_ocdbt=self._use_ocdbt)
      tspec = _get_cast_tspec_serialize(tspec, value, arg)
      synchronous_ops += [
          serialization.async_serialize(
              value,
              tspec,
              commit_future=futures,
              context=self._ts_context,
          )
      ]
      if value.sharding is not None:
        sharding_info = info
        sharding_info.path = epath.Path(sharding_file_path)
        tspec_sharding = self._get_json_tspec_write(
            sharding_info, value, use_ocdbt=False
        )
        tspec_sharding = {
            'driver': 'json',
            'kvstore': tspec_sharding['kvstore'],
            'json_pointer': '/' + base64.urlsafe_b64encode(
                sharding_info.name.encode()
            ).decode('utf-8'),
        }
        if jax.process_index() == 0:
          open_future = ts.open(
              tspec_sharding, open=True, context=self._ts_context
          )
          t = await ts.open(
              tspec_sharding,
              open=True,
              assume_metadata=True,
              context=self._ts_context,
          )
          serialized_sharding = _serialize_sharding(value.sharding)
          if serialized_sharding is not None:
            write_future = t.write(serialized_sharding)
            synchronous_ops += [write_future.copy]
            futures += [open_future, write_future]
    await asyncio.gather(*synchronous_ops)
    return futures

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[jax.Array]:
    """See superclass documentation.

    Args:
      infos: ParamInfo.
      args: must be of type `ArrayRestoreArgs`.

    Returns:
      The deserialized parameter.

    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.sharding` is not provided or `args.mesh` and
      `args.mesh_axes` are not provided.
    """
    if args is None:
      raise ValueError('Must provide ArrayRestoreArgs to restore as jax.Array.')
    check_input_arguments(infos, args)
    deserialize_ops = []
    if infos[0].parent_dir is None:
      raise ValueError('parent_dir cannot be None')
    sharding_file_path = infos[0].parent_dir / _SHARDING
    sharding_file_exists = await utils.async_exists(sharding_file_path)
    for info, arg in zip(infos, args):
      arg = cast(ArrayRestoreArgs, arg)
      if (
          isinstance(arg, ArrayRestoreArgs)
          and arg.mesh is not None
          and arg.mesh_axes is not None
      ):
        sharding = NamedSharding(arg.mesh, arg.mesh_axes)
      elif isinstance(arg, ArrayRestoreArgs) and arg.sharding is not None:
        sharding = arg.sharding
      elif sharding_file_exists:
        warnings.warn(
            "Couldn't find sharding info under RestoreArgs. Populating sharding"
            ' info from sharding file. Please note restoration time will be'
            ' slightly increased due to reading from file instead of directly'
            ' from RestoreArgs.'
        )
        sharding_info = copy.deepcopy(info)
        sharding_info = dataclasses.replace(
            sharding_info, path=sharding_file_path
        )
        tspec_sharding = self._get_json_tspec_read(
            sharding_info, use_ocdbt=False
        )
        sharding = None
        if sharding_info.name:
          tspec_sharding = {
              'driver': 'json',
              'kvstore': tspec_sharding['kvstore'],
              'json_pointer': '/' + base64.urlsafe_b64encode(
                  sharding_info.name.encode()
              ).decode('utf-8'),
          }
          t = await ts.open(
              tspec_sharding, context=self._ts_context, open=True, read=True
          )
          serialized_string = await t.read()
          if serialized_string:
            sharding = (
                _deserialize_sharding_from_json_string(serialized_string.item())
                or None
            )
        else:
          raise ValueError('Unable to deserialize sharding.')
      else:
        raise ValueError(
            'Sharding of jax.Array cannot be None. Provide `mesh`'
            ' and `mesh_axes` OR `sharding`'
        )
      if not info.is_ocdbt_checkpoint:
        await _assert_parameter_files_exist(info.path, self._metadata_key)
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = _use_ocdbt_for_restore(
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = _get_cast_tspec_deserialize(tspec, arg)
      deserialize_ops += [
          serialization.async_deserialize(
              sharding,
              tspec,
              global_shape=arg.global_shape
              if hasattr(arg, 'global_shape')
              else None,
              byte_limiter=info.byte_limiter,
              context=self._ts_context,
          )
      ]
    return await asyncio.gather(*deserialize_ops)


class StringHandler(TypeHandler):
  """TypeHandler for strings."""

  def __init__(
      self,
      filename: Optional[str] = None,
  ):
    self._filename = filename or '_strings.json'
    self._ts_context = serialization.TS_CONTEXT

  def _get_json_tspec(
      self,
      info: ParamInfo,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec: Dict[str, Any] = get_tensorstore_spec(path, ocdbt=False)
    tspec = {
        'driver': 'json',
        'kvstore': tspec['kvstore'],
        'json_pointer': '/' + info.name,
    }
    return tspec

  def typestr(self) -> str:
    return 'string'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[StringMetadata]:
    return [StringMetadata()] * len(infos)

  async def _convert_to_string(self, tensorstore):
    result = await tensorstore.read()
    return str(result)

  async def serialize(
      self,
      values: Sequence[str],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[SaveArgs]] = None,
  ) -> Sequence[Future]:
    """See superclass documentation."""
    del args
    check_input_arguments(values, infos)
    synchronous_ops = []
    futures = []
    directory = epath.Path(infos[0].path).parent
    for (
        info,
        value,
    ) in zip(infos, values):
      info.path = epath.Path(directory / self._filename)
      tspec = self._get_json_tspec(info)
      if jax.process_index() == 0:
        open_future = ts.open(tspec, open=True, context=self._ts_context)
        t = await ts.open(
            tspec,
            open=True,
            assume_metadata=True,
            context=self._ts_context,
        )
        write_future = t.write(value)
        synchronous_ops += [write_future.copy]
        futures += [open_future, write_future]
    await asyncio.gather(*synchronous_ops)
    return futures

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[RestoreArgs]] = None,
  ) -> Sequence[Optional[str]]:
    """See superclass documentation."""
    del args
    check_input_arguments(infos)
    directory = epath.Path(infos[0].path).parent
    open_futures = []

    for info in infos:
      info.path = epath.Path(directory / self._filename)
      tspec = self._get_json_tspec(info)
      open_future = ts.open(
          tspec, open=True, read=True, context=self._ts_context
      )
      open_futures += [open_future]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [self._convert_to_string(t) for t in tensorstores]
    return await asyncio.gather(*read_ops)

_TYPE_REGISTRY = [
    (lambda ty: issubclass(ty, int), ScalarHandler()),
    (lambda ty: issubclass(ty, float), ScalarHandler()),
    (lambda ty: issubclass(ty, bytes), ScalarHandler()),
    (lambda ty: issubclass(ty, np.number), ScalarHandler()),
    (lambda ty: issubclass(ty, np.ndarray), NumpyHandler()),
    (lambda ty: issubclass(ty, jax.Array), ArrayHandler()),
    (lambda ty: issubclass(ty, str), StringHandler()),
]


def _make_typestr_registry(type_registry: Any) -> Dict[str, TypeHandler]:
  return {h.typestr(): h for _, h in type_registry}


_TYPESTR_REGISTRY = _make_typestr_registry(_TYPE_REGISTRY)


def register_type_handler(
    ty: Any,
    handler: TypeHandler,
    func: Optional[Callable[[Any], bool]] = None,
    override: bool = False,
):
  """Registers a type for serialization/deserialization with a given handler.

  Note that it is possible for a type to match multiple different entries in
  the registry, each with a different handler. In this case, only the first
  match is used.

  Args:
    ty: A type to register.
    handler: a TypeHandler capable of reading and writing parameters of type
      `ty`.
    func: A function that accepts a type and returns True if the type should be
      handled by the provided TypeHandler. If this parameter is not specified,
      defaults to `lambda t: issubclass(t, ty)`.
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
    if handler.typestr() in _TYPESTR_REGISTRY:
      raise ValueError(
          f'Type "{ty}" has a `typestr` ("{handler.typestr()}") which collides'
          ' with that of an existing TypeHandler.'
      )
    _TYPE_REGISTRY.append((func, handler))
    _TYPESTR_REGISTRY[handler.typestr()] = handler
  elif override:
    _TYPE_REGISTRY[existing_handler_idx] = (func, handler)
    _TYPESTR_REGISTRY[handler.typestr()] = handler
  else:
    raise ValueError(f'A TypeHandler for "{ty}" is already registered.')


def get_type_handler(ty: Any) -> TypeHandler:
  """Returns the handler registered for a given type, if available.

  Args:
    ty: an object type (or string representation of the type.)

  Returns:
    The TypeHandler that is registered for the given type.

  Raises:
    ValueError if the given type has no registered handler.
  """
  if isinstance(ty, str):
    if ty in _TYPESTR_REGISTRY:
      return _TYPESTR_REGISTRY[ty]
  else:
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


# TODO(b/253238305) Deprecate when all checkpoints have saved types.
def default_restore_type(args: RestoreArgs) -> Any:
  if isinstance(args, ArrayRestoreArgs):
    return jax.Array
  elif isinstance(args, RestoreArgs):
    return np.ndarray
  else:
    raise ValueError(f'Unsupported restore_args type: {type(args)}')
