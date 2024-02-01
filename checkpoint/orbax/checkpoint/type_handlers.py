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

"""Provides utils for PytreeCheckpointHandler."""

import abc
import asyncio
import base64
import dataclasses
import enum
import json
import os
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast
import warnings

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization import serialization
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import utils
from orbax.checkpoint import value_metadata
from orbax.checkpoint.future import Future
import tensorstore as ts


Scalar = Union[int, float, np.number]
Metadata = value_metadata.Metadata
NamedSharding = jax.sharding.NamedSharding
SingleDeviceSharding = jax.sharding.SingleDeviceSharding
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
_DEFAULT_OCDBT_TS_CONTEXT = ts.Context(
    {
        # Provide cache pool for B-tree nodes to avoid repeated reads.
        # 100MB limit.
        'cache_pool#ocdbt': {'total_bytes_limit': 100000000},
    },
    parent=serialization.TS_CONTEXT,
)

_PARTITION_SPEC = 'partition_spec'
_SHARDING = '_sharding'
_SHARDING_TYPE = 'sharding_type'
_DEVICE_STR = 'device_str'

RESTORE_TYPE_NONE = 'None'
RESTORE_TYPE_DICT = 'Dict'
RESTORE_TYPE_LIST = 'List'

_DEFAULT_DRIVER = 'file'
_PROCESS_SUBDIR_PREFIX = 'ocdbt.process_'
_OCDBT_PROCESS_ID_RE = r'[A-Za-z0-9]+'

ZARR_VER2 = 'zarr'
ZARR_VER3 = 'zarr3'


class ShardingTypes(enum.Enum):
  NAMED_SHARDING = 'NamedSharding'
  SINGLE_DEVICE_SHARDING = 'SingleDeviceSharding'
  POSITIONAL_SHARDING = 'PositionalSharding'
  GSPMD_SHARDING = 'GSPMDSharding'


def _serialize_sharding(sharding: jax.sharding.Sharding) -> str:
  """Serializes `jax.sharding.Sharding` into a json string."""

  sharding_data = {}
  if isinstance(sharding, NamedSharding):
    sharding_data[_SHARDING_TYPE] = ShardingTypes.NAMED_SHARDING.value
    sharding_data[_MESH_SHAPE] = list(sharding.mesh.shape.values())
    sharding_data[_MESH_AXES] = sharding.mesh.axis_names
    sharding_data[_PARTITION_SPEC] = sharding.spec

  elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
    sharding_data[_SHARDING_TYPE] = ShardingTypes.SINGLE_DEVICE_SHARDING.value
    # There is no serialization method for Device.
    # Get the Device object and then retreive the str representation
    sharding_data[_DEVICE_STR] = str(next(iter(sharding.device_set)))

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

  if _SHARDING_TYPE in sharding_data:
    serialized_string = json.dumps(sharding_data)
    return serialized_string
  else:
    return ''


def _deserialize_sharding_from_json_string(
    sharding_string: str,
) -> Optional[jax.sharding.Sharding]:
  """Deserializes a json string to `jax.sharding.Sharding`."""

  deserialized_dict = json.loads(sharding_string)

  if deserialized_dict[_SHARDING_TYPE] == ShardingTypes.NAMED_SHARDING.value:
    shape = deserialized_dict[_MESH_SHAPE]
    axis_names = list(deserialized_dict[_MESH_AXES])
    partition_spec = tuple(deserialized_dict[_PARTITION_SPEC])

    if len(jax.devices()) != np.prod(shape):
      return None
    else:
      return NamedSharding(
          jax.sharding.Mesh(
              np.array(jax.devices()).reshape(shape), axis_names=axis_names
          ),
          jax.sharding.PartitionSpec(*partition_spec),
      )

  elif (
      deserialized_dict[_SHARDING_TYPE]
      == ShardingTypes.SINGLE_DEVICE_SHARDING.value
  ):
    # Initialize a 'cached' Dict[str, Device] to help look up Devices by
    # their str representation.
    # Cache tip: See Function Attributes https://peps.python.org/pep-0232/.
    if not hasattr(_deserialize_sharding_from_json_string, 'device_map'):
      _deserialize_sharding_from_json_string.device_map = {
          str(device): device for device in jax.local_devices()
      }
    device_str = deserialized_dict[_DEVICE_STR]
    if device := _deserialize_sharding_from_json_string.device_map.get(
        device_str, None
    ):
      return SingleDeviceSharding(device)

    raise ValueError(
        f'{ShardingTypes.SINGLE_DEVICE_SHARDING.value} with'
        f' Device={device_str} was not found in jax.local_devices().'
    )

  else:
    raise NotImplementedError(
        'Sharding types other than `jax.sharding.NamedSharding` have not been '
        'implemented.'
    )


def _get_coordinator_address_without_port(coordinator_address: str) -> str:
  """Returns JAX coordinator address stripped of port number."""
  return coordinator_address.split(':')[0]


def create_coordinator_server_and_context() -> Tuple[None, None]:
  # TODO(b/293331479) remove this once OCDBT is enabled by default.
  warnings.warn('This function has been deprecated. Do not use.')
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
      _OCDBT_COORDINATOR_SERVER = ts.ocdbt.DistributedCoordinatorServer({
          'bind_addresses': [bind_address],
      })
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
  logging.info('OCDBT is initialized successfully.')


async def _assert_parameter_files_exist(
    param_dir: epath.Path, metadata_key: Optional[str], use_zarr3: bool = False
):
  """Checks for existence of parameter subdir and .zarray file."""
  exists = await utils.async_exists(param_dir)
  if not exists:
    raise FileNotFoundError(
        f'Individual parameter subdirectory not found at path: {param_dir}.'
    )
  if metadata_key is None:
    metadata_key = 'zarr.json' if use_zarr3 else '.zarray'
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


class LimitInFlightBytes(serialization._LimitInFlightBytes):  # pylint: disable=protected-access
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def wait_for_bytes_sync(self, requested_bytes):
    asyncio.run(self.wait_for_bytes(requested_bytes))

  def release_bytes_sync(self, requested_bytes):
    asyncio.run(self.release_bytes(requested_bytes))


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
  use_zarr3:
    If True, use Zarr ver3 otherwise ver2.
  """

  name: Optional[str] = None
  path: Optional[epath.Path] = None
  parent_dir: Optional[epath.Path] = None
  skip_deserialize: Optional[bool] = None
  byte_limiter: Optional[serialization._LimitInFlightBytes] = None  # pylint: disable=protected-access
  is_ocdbt_checkpoint: Optional[bool] = None
  ocdbt_merge: Optional[bool] = None
  use_zarr3: Optional[bool] = False


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  aggregate:
    Deprecated, please use custom TypeHandler
    (https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler) or
    contact Orbax team to migrate before May 1st, 2024. If true, saves the given
    parameter in an aggregated tree format rather than individually. See
    AggregateHandler.
  dtype:
    If provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  write_chunk_shape:
    This only applies to Zarr version 3.  This specifies the shape of a shard
    used in writing.  The default(None) is set to equal to the array shard size,
    so there are equal number of write chunks and shards. The write_chunk_shape
    needs to be a divisor of the array shape.
  read_chunk_shape:
    This only applies to Zarr version 3.  This specifies the chunk sizes within
    a write chunk. Default is set to equal to the write_chunk_shape. The
    read_chunk_shape is reqired to be a divisor of the write_chunk_shape.
  chunk_byte_size:
    This is an experimental feature that automatically chooses the largest chunk
    shape possible, while keeping the chunk byte size less than or equal to the
    specified chunk_byte_size. Both the write_chunk_shape and read_chunk_shape
    are automatically set to the chosen shape. This uses a greedy algorithm that
    prioritizes splitting the largest dimensions first. In order to enable this
    feature, both write_chunk_shape and read_chunk_shape must be set to None.
  """

  aggregate: bool = False
  dtype: Optional[jnp.dtype] = None
  write_chunk_shape: Optional[tuple[int, ...]] = None
  read_chunk_shape: Optional[tuple[int, ...]] = None
  chunk_byte_size: Optional[int] = None

  def __post_init__(self):
    if self.aggregate:
      logging.warning(
          'SaveArgs.aggregate is deprecated, please use custom TypeHandler'
          ' (https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler)'
          ' or contact Orbax team to migrate before May 1st, 2024.'
      )


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


def _choose_chunk_shape(
    global_shape: Sequence[int],
    write_shape: Sequence[int],
    dtype: Union[jnp.dtype, np.dtype],
    target_byte_size: int,
) -> Sequence[int]:
  """Chooses a chunk shape that divides the `write_shape`.

  The chunk shape is chosen such that the resulting byte size is less than
  or equal to `target_byte_size`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest and sharded
  dimensions first.

  Args:
    global_shape: the global shape of the array
    write_shape: the local shape being written
    dtype: the dtype of the array
    target_byte_size: Desired chunk byte size.  Must be >= dtype.itemsize.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """
  assert len(global_shape) == len(write_shape)
  if target_byte_size < 52428800:  # 50MB
    logging.warning(
        'Setting the target_byte_size too small could reduce performance.'
    )

  sharded_dimensions = np.array(global_shape) != np.array(write_shape)
  dtype_size = dtype.itemsize
  target_elements = target_byte_size // dtype_size

  rank = len(write_shape)

  # `dim_factors[i]` is the list of divisors of `write_shape[i]`
  dim_factors = [
      [i for i in range(1, size + 1) if size % i == 0] for size in write_shape
  ]

  # The current chunk shape is:
  # [dim_factors[i][-1] for i in range(rank)]

  def get_total_elements():
    """Returns the number of elements in the current chunk shape."""
    total_elements = 1
    for i in range(rank):
      total_elements *= dim_factors[i][-1]
    return total_elements

  # Reduce the current chunk shape until the desired number of elements is
  # reached.
  while get_total_elements() > target_elements:
    # Greedily reduce the largest dimension.  This is not guaranteed to bring us
    # the closest to `target_elements`, but is simple to implement and should
    # work well enough.
    dim_to_reduce = -1
    dim_to_reduce_size = 1
    for i in range(rank):
      size = dim_factors[i][-1]
      if sharded_dimensions[i] and size > dim_to_reduce_size:
        dim_to_reduce_size = size
        dim_to_reduce = i

    if dim_to_reduce_size > 1:
      dim_factors[dim_to_reduce].pop()
    else:
      # need to start spliting on unsharded dimension
      sharded_dimensions = np.ones(len(write_shape))

  chosen_shape = [dim_factors[i][-1] for i in range(rank)]

  logging.debug(
      'global_shape=%s, write_shape=%s, dtype=%s, target_byte_size=%d,'
      ' chosen_shape=%s',
      global_shape,
      write_shape,
      dtype,
      target_byte_size,
      chosen_shape,
  )

  return chosen_shape


def _validate_divisible_shapes(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> bool:
  """Return True if shape2 is a divisor of shape1 otherwise False."""
  try:
    return not np.mod(shape1, shape2).any()
  except ValueError:
    # eg. imcompatible shape
    return False


def _build_ts_zarr_shard_and_chunk_metadata(
    global_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    use_zarr3: bool,
    dtype: Union[jnp.dtype, np.dtype],
    write_chunk_shape: Optional[tuple[int, ...]] = None,
    read_chunk_shape: Optional[tuple[int, ...]] = None,
    chunk_byte_size: Optional[int] = None,
) -> Dict[Any, Any]:
  """This function returns the TS metadata for write spec."""
  metadata = {}

  if not use_zarr3:
    # Zarr ver2
    metadata['chunks'] = np.array(np.maximum(1, shard_shape))
    metadata['compressor'] = {'id': 'zstd'}
  else:
    # Zarr ver3
    # Shard configs{

    # choose write_chunk_shape and read_chunk_shape that result a chunk byte
    # size equal or less than the `chunk_byte_size`
    if chunk_byte_size:
      if write_chunk_shape is None and read_chunk_shape is None:
        if chunk_byte_size < dtype.itemsize:
          raise ValueError(
              f'chunk_byte_size={chunk_byte_size} must be >= {dtype.itemsize}'
          )
        write_chunk_shape = read_chunk_shape = _choose_chunk_shape(
            global_shape, shard_shape, dtype, chunk_byte_size
        )
      else:
        logging.warning(
            '`chunk_byte_size` is ignored because `write_chunk_shape` or'
            ' `read_chunk_shape` is not None.'
        )

    # If `write_chunk_shape` is not specified, make it the same as Jax sharding
    if write_chunk_shape:
      if not _validate_divisible_shapes(shard_shape, write_chunk_shape):
        raise ValueError(
            f'write_chunk_shape={write_chunk_shape} is not a divisor of'
            f' shard_shape={shard_shape}'
        )
      write_shape = write_chunk_shape
    else:
      write_shape = shard_shape

    metadata['chunk_grid'] = {
        'name': 'regular',
        'configuration': {
            'chunk_shape': write_shape,
        },
    }

    # Sub-chunk configs
    # If `read_chunk_shape` is not specified, make it the same as
    # `write_shape`
    if read_chunk_shape:
      if not _validate_divisible_shapes(write_shape, read_chunk_shape):
        raise ValueError(
            f'read_chunk_shape={read_chunk_shape} is not a divisor of'
            f' write_chunk_shape={write_shape}'
        )
      read_shape = read_chunk_shape
    else:
      read_shape = shard_shape

    metadata['codecs'] = [
        {
            'name': 'sharding_indexed',
            'configuration': {
                'chunk_shape': read_shape,
                'codecs': [
                    {'name': 'bytes', 'configuration': {'endian': 'little'}},
                    {'name': 'zstd'},
                ],
                'index_codecs': [
                    {'name': 'bytes', 'configuration': {'endian': 'little'}},
                    {'name': 'crc32c'},
                ],
                'index_location': 'end',
            },
        },
    ]

  return metadata


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

  def finalize(self, directory: epath.Path):
    """Performs any logic to finalize parameter files written by this class.

    By default, does nothing.

    Args:
      directory: A path to the location of the checkpoint. This corresponds to
        `param_info.parent_dir`.
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


def merge_ocdbt_per_process_files(directory: epath.Path):
  """Merges OCDBT files written to per-process subdirectories.

  With Tensorstore's OCDBT format, arrays are initially written to per-process
  subdirectories, depending on which host is doing the writing. This function
  can be called to merge the per-process files into a global key-value store.

  The original per-process subdirectories are not and should not be deleted -
  the global kvstore continues to reference them.

  Args:
    directory: checkpoint location.
  """
  ts_context = _DEFAULT_OCDBT_TS_CONTEXT

  open_ops = []

  parent_tspec = get_tensorstore_spec(os.fspath(directory), use_ocdbt=True)
  parent_tspec = parent_tspec['kvstore']
  open_ops.append(
      ts.KvStore.open(
          ts.KvStore.Spec(parent_tspec),
          context=ts_context,
      )
  )

  for process_dir in directory.glob(f'{_PROCESS_SUBDIR_PREFIX}*'):
    process_id = process_dir.name.split('_')[-1]
    child_tspec = get_tensorstore_spec(
        os.fspath(directory), use_ocdbt=True, process_id=process_id
    )
    child_tspec = child_tspec['kvstore']
    open_ops.append(
        ts.KvStore.open(
            ts.KvStore.Spec(child_tspec),
            context=ts_context,
        )
    )

  async def open_and_copy():
    opened = await asyncio.gather(*open_ops)
    parent, children = opened[0], opened[1:]
    copy_ops = []
    for child in children:
      copy_ops.append(child.experimental_copy_range_to(parent))
    await asyncio.gather(*copy_ops)

  asyncio.run(open_and_copy())


def _get_kvstore_for_gcs(ckpt_path: str) -> Dict[str, Any]:
  m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError(
        'The ckpt_path should contain the bucket name and the '
        f'file path inside the bucket. Got: {ckpt_path}'
    )
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': gcs_bucket, 'path': path_without_bucket}


def _get_metadata(
    arr,
    use_zarr3,
    write_chunk_shape,
    read_chunk_shape,
    chunk_byte_size,
):
  """build metadata for a Tensorstore array."""
  if arr.dtype == jnp.bfloat16:
    # Tensorstore uses 'bfloat16', not '<V2'.
    dtype = 'bfloat16'
  else:
    dtype = np.dtype(arr.dtype).str
  metadata = {
      'shape': arr.shape,
      'dtype': dtype,
  }
  local_shape = arr.addressable_data(0).shape
  metadata.update(
      _build_ts_zarr_shard_and_chunk_metadata(
          global_shape=arr.shape,
          shard_shape=local_shape,
          dtype=arr.dtype,
          use_zarr3=use_zarr3,
          write_chunk_shape=write_chunk_shape,
          read_chunk_shape=read_chunk_shape,
          chunk_byte_size=chunk_byte_size,
      )
  )
  return metadata


def get_tensorstore_spec(
    directory: str,
    name: Optional[str] = None,
    use_ocdbt: bool = True,
    process_id: Optional[Union[int, str]] = None,
    use_zarr3: Optional[bool] = False,
) -> Dict[str, Any]:
  """Constructs a Tensorstore spec.

  Args:
    directory: Parent directory where the parameter will be written.
    name: Name of the parameter.
    use_ocdbt: Whether to use OCDBT to write the array.
    process_id: If provided, will write to a sub-directory named
      `ocdbt.process_<process_id>`. If a string, must conform to [A-Za-z0-9]+
      pattern.
    use_zarr3: If True, use ZARR_VER3 driver, otherwise, use ZARR_VER2 driver.

  Returns:
    A ts.Spec in dictionary form.
  """
  default_driver = serialization._DEFAULT_DRIVER  # pylint: disable=protected-access
  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  directory = os.path.normpath(directory).replace('gs:/', 'gs://')
  is_gcs_path = directory.startswith('gs://')
  spec = {'driver': ZARR_VER3 if use_zarr3 else ZARR_VER2, 'kvstore': {}}

  if use_ocdbt:
    if not is_gcs_path and not os.path.isabs(directory):
      raise ValueError(f'Checkpoint path should be absolute. Got {directory}')
    base_path = directory if is_gcs_path else f'{default_driver}://{directory}'
    if process_id is not None:
      process_id = str(process_id)
      assert re.fullmatch(_OCDBT_PROCESS_ID_RE, process_id) is not None, (
          f'process_id must conform to {_OCDBT_PROCESS_ID_RE} pattern'
          f', got {process_id}'
      )
      base_path = os.path.join(
          base_path, f'{_PROCESS_SUBDIR_PREFIX}{process_id}'
      )
    spec['kvstore'] = {
        'driver': 'ocdbt',
        'base': base_path,
    }
    if name is not None:
      spec['kvstore']['path'] = name
    spec.update(
        {'recheck_cached_data': False, 'recheck_cached_metadata': False}
    )
    spec['kvstore'].update({
        # Enable read coalescing.  This feature merges adjacent read_ops into
        # one, which could reduce I/O ops by a factor of 10. This is especially
        # beneficial for unstacked models.
        'experimental_read_coalescing_threshold_bytes': 1000000,
        'experimental_read_coalescing_merged_bytes': 500000000000,
        'experimental_read_coalescing_interval': '1ms',
        # References the cache specified in ts.Context.
        'cache_pool': 'cache_pool#ocdbt',
    })
  else:
    if name is None:
      ckpt_path = directory
    else:
      ckpt_path = os.path.join(directory, name)
    if is_gcs_path:
      spec['kvstore'] = _get_kvstore_for_gcs(ckpt_path)
    else:
      spec['kvstore'] = {'driver': default_driver, 'path': ckpt_path}

  return spec


def get_process_index_for_subdir(
    use_ocdbt: bool,
    ocdbt_merge: bool,
) -> Optional[int]:
  """If OCDBT + merge feature is in use, returns a process index."""
  if use_ocdbt and ocdbt_merge:
    return jax.process_index()
  else:
    return None


def get_ts_context(use_ocdbt: bool, ocdbt_merge: bool = True) -> ts.Context:
  """Returns a shared global TensorStore Context instance to use."""
  if not use_ocdbt:
    return serialization.TS_CONTEXT

  if ocdbt_merge:
    return _DEFAULT_OCDBT_TS_CONTEXT
  if _OCDBT_TS_CONTEXT is None:
    raise ValueError(
        'Coordinator-based TensorStore context should be configured'
        ' if OCDBT merging is not enabled.'
    )
  return _OCDBT_TS_CONTEXT


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


def _add_write_tspec_ocdbt_options(tspec: Dict[str, Any]) -> Dict[str, Any]:
  tspec['kvstore']['config'] = {
      # Store .zarray metadata inline but not large chunks.
      'max_inline_value_bytes': 1024,
      # Large value allows a single root node to support faster traversal.
      'max_decoded_node_bytes': 100000000,
  }
  return tspec


def _array_metadata_from_tensorstore(
    t: Any, info: ParamInfo, sharding: Optional[jax.sharding.Sharding] = None
) -> ArrayMetadata:
  return ArrayMetadata(
      name=info.name,
      directory=info.parent_dir,
      shape=t.shape,
      dtype=jnp.dtype(t.dtype.name),
      sharding=sharding,
  )


def _dump_debug_data(key, infos):
  ts_metrics = ts.experimental_collect_matching_metrics('/tensorstore/')
  ts_metrics += [
      {'key': key},
      {'infos': [f'{info.name}' for info in infos]},
  ]

  return json.dumps(ts_metrics)


class NumpyHandler(TypeHandler):
  """Provides an implementation of TypeHandler for replicated numpy arrays."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
    """
    self._metadata_key = metadata_key

  def _get_json_tspec(
      self,
      info: ParamInfo,
      use_ocdbt: bool,
      process_index: Optional[int] = None,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    directory = os.fspath(info.parent_dir)
    tspec: Dict[str, Any] = get_tensorstore_spec(
        directory,
        name=info.name,
        use_ocdbt=use_ocdbt,
        process_id=process_index,
    )
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    return tspec

  def _get_json_tspec_write(
      self,
      info: ParamInfo,
      value: np.ndarray,
      use_ocdbt: bool,
      process_index: Optional[int] = None,
      arg: Optional[SaveArgs] = None,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for writing."""
    tspec = self._get_json_tspec(
        info, use_ocdbt=use_ocdbt, process_index=process_index
    )
    tspec['metadata'] = {
        'shape': value.shape,
    }
    tspec['metadata'].update(
        _build_ts_zarr_shard_and_chunk_metadata(
            global_shape=value.shape,
            shard_shape=value.shape,
            dtype=value.dtype,
            use_zarr3=info.use_zarr3,
            write_chunk_shape=arg.write_chunk_shape if arg else None,
            read_chunk_shape=arg.read_chunk_shape if arg else None,
            chunk_byte_size=arg.chunk_byte_size if arg else None,
        )
    )
    if use_ocdbt:
      tspec = _add_write_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_read(
      self,
      info: ParamInfo,
      use_ocdbt: bool,
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
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      open_ops.append(
          ts.open(ts.Spec(tspec), open=True, context=get_ts_context(use_ocdbt))
      )

    tensorstores = await asyncio.gather(*open_ops)
    return [
        _array_metadata_from_tensorstore(t, info, sharding=None)
        for t, info in zip(tensorstores, infos)
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
      tspec = self._get_json_tspec_write(
          info,
          value,
          use_ocdbt=info.is_ocdbt_checkpoint,
          process_index=get_process_index_for_subdir(
              info.is_ocdbt_checkpoint, info.ocdbt_merge
          ),
          arg=arg,
      )
      tspec = _get_cast_tspec_serialize(tspec, value, arg)
      if logging.level_debug():
        logging.debug('tspec = %s', tspec)
        logging.debug('infos = %s', info)
        logging.debug('args = %s', arg)
      if jax.process_index() == 0:
        ts_context = get_ts_context(info.is_ocdbt_checkpoint, info.ocdbt_merge)
        # Open once to create metadata and allow the operation to happen
        # asynchronously.
        open_future = ts.open(
            ts.Spec(tspec), create=True, open=True, context=ts_context
        )
        # Open again (no disk I/O) to get the write location.
        t = await ts.open(
            ts.Spec(tspec),
            open=True,
            assume_metadata=True,
            context=ts_context,
        )
        write_future = t.write(value)
        copy_ops += [write_future.copy]
        futures += [open_future, write_future.commit]
    await asyncio.gather(*copy_ops)

    if logging.level_debug():
      logging.debug(
          'ts_metrics: %s', _dump_debug_data(self._metadata_key, infos)
      )
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
        await _assert_parameter_files_exist(
            info.path, self._metadata_key, info.use_zarr3
        )
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = _get_cast_tspec_deserialize(tspec, arg)

      if logging.level_debug():
        logging.debug('tspec = %s', tspec)
        logging.debug('infos = %s', infos)
        logging.debug('args = %s', args)
      open_futures += [
          ts.open(ts.Spec(tspec), open=True, context=get_ts_context(use_ocdbt))
      ]
    tensorstores = await asyncio.gather(*open_futures)
    read_ops = [t.read() for t in tensorstores]
    ret = await asyncio.gather(*read_ops)

    if logging.level_debug():
      for a in ret:
        logging.debug(
            'restored ndarray.shape = %s, array.dtype = %s', a.shape, a.dtype
        )
      logging.debug(
          'ts_metrics: %s', _dump_debug_data(self._metadata_key, infos)
      )

    return ret


class ScalarHandler(NumpyHandler):
  """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.)."""

  def typestr(self) -> str:
    return 'scalar'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[ScalarMetadata]:
    metadatas = await super().metadata(infos)
    return [
        ScalarMetadata(name=m.name, directory=m.directory, dtype=m.dtype)
        for m in metadatas
    ]

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


def get_sharding_tensorstore_spec(
    directory: str, param_name: str
) -> Dict[str, Any]:
  kvstore = get_tensorstore_spec(directory, name=_SHARDING, use_ocdbt=False)[
      'kvstore'
  ]
  return {
      'driver': 'json',
      'kvstore': kvstore,
      'json_pointer': '/' + base64.urlsafe_b64encode(
          param_name.encode()
      ).decode('utf-8'),
  }


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

  mesh: Optional[jax.sharding.Mesh] = None
  mesh_axes: Optional[jax.sharding.PartitionSpec] = None
  sharding: Optional[jax.sharding.Sharding] = None
  global_shape: Optional[Tuple[int, ...]] = None


class ArrayHandler(TypeHandler):
  """An implementation of TypeHandler for jax.Array."""

  def __init__(
      self,
      metadata_key: Optional[str] = None,
  ):
    """Constructor.

    Args:
      metadata_key: name to give to Tensorstore metadata files.
    """
    self._metadata_key = metadata_key

  def _get_json_tspec(
      self,
      info: ParamInfo,
      use_ocdbt: bool,
      process_index: Optional[int] = None,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec in JSON format."""
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    directory = os.fspath(info.parent_dir)
    tspec: Dict[str, Any] = get_tensorstore_spec(
        directory,
        name=info.name,
        use_ocdbt=use_ocdbt,
        process_id=process_index,
        use_zarr3=info.use_zarr3,
    )
    if self._metadata_key is not None:
      tspec['metadata_key'] = self._metadata_key
    return tspec

  def _get_json_tspec_write(
      self,
      info: ParamInfo,
      value: jax.Array,
      use_ocdbt: bool,
      process_index: Optional[int] = None,
      arg: Optional[SaveArgs] = None,
  ) -> Dict[str, Any]:
    """Gets Tensorstore spec for writing."""
    tspec = self._get_json_tspec(
        info, use_ocdbt=use_ocdbt, process_index=process_index
    )
    tspec['metadata'] = _get_metadata(
        value,
        info.use_zarr3,
        arg.write_chunk_shape if arg else None,
        arg.read_chunk_shape if arg else None,
        arg.chunk_byte_size if arg else None,
    )
    del tspec['metadata']['dtype']
    if use_ocdbt:
      tspec = _add_write_tspec_ocdbt_options(tspec)
    return tspec

  def _get_json_tspec_read(
      self,
      info: ParamInfo,
      use_ocdbt: bool,
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
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      ts_context = get_ts_context(use_ocdbt)
      open_ops.append(ts.open(ts.Spec(tspec), open=True, context=ts_context))

      sharding_op = None
      if info.name:
        tspec_sharding = get_sharding_tensorstore_spec(
            os.fspath(info.parent_dir), info.name
        )
        if sharding_file_exists:
          sharding_op = ts.open(
              tspec_sharding,
              open=True,
              read=True,
              # OCDBT is not used for sharding metadata.
              context=get_ts_context(use_ocdbt=False),
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
        _array_metadata_from_tensorstore(t, info, sharding)
        for (t, info, sharding) in zip(tensorstores, infos, shardings)
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
    txn = ts.Transaction()
    for value, info, arg in zip(values, infos, args):
      tspec = self._get_json_tspec_write(
          info,
          value,
          use_ocdbt=info.is_ocdbt_checkpoint,
          process_index=get_process_index_for_subdir(
              info.is_ocdbt_checkpoint, info.ocdbt_merge
          ),
          arg=arg,
      )
      tspec = _get_cast_tspec_serialize(tspec, value, arg)
      if logging.level_debug():
        logging.debug(
            'sharding=%s, addressable_shards=%s, global_shards=%s',
            value.sharding,
            value.addressable_shards,
            value.global_shards,
        )
        logging.debug('tspec = %s', tspec)
        logging.debug('infos = %s', info)
        logging.debug('args = %s', arg)
      ts_context = get_ts_context(info.is_ocdbt_checkpoint, info.ocdbt_merge)
      synchronous_ops += [
          serialization.async_serialize(
              value,
              tspec,
              commit_future=futures,
              context=ts_context,
          )
      ]

      if value.sharding is not None:
        if info.parent_dir is None:
          raise ValueError('parent_dir cannot be None')
        tspec_sharding = get_sharding_tensorstore_spec(
            os.fspath(info.parent_dir), info.name
        )
        if jax.process_index() == 0:
          # OCDBT is not used for sharding metadata.
          sharding_ts_context = get_ts_context(use_ocdbt=False)
          t = await ts.open(
              tspec_sharding,
              open=True,
              context=sharding_ts_context,
          )
          serialized_sharding = _serialize_sharding(value.sharding)
          if serialized_sharding is not None:
            write_future = t.with_transaction(txn).write(serialized_sharding)
            synchronous_ops += [write_future.copy]
    await asyncio.gather(*synchronous_ops)

    if logging.level_debug():
      logging.debug(
          'ts_metrics: %s', _dump_debug_data(self._metadata_key, infos)
      )
    futures.append(txn.commit_async())
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
        sharding = None
        if info.name:
          tspec_sharding = get_sharding_tensorstore_spec(
              os.fspath(info.parent_dir), info.name
          )
          t = await ts.open(
              tspec_sharding,
              # OCDBT is not used for sharding metadata.
              context=get_ts_context(use_ocdbt=False),
              open=True,
              read=True,
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
        await _assert_parameter_files_exist(
            info.path,
            self._metadata_key,
            info.use_zarr3,
        )
      # Use OCDBT flag from the existing checkpoint.
      use_ocdbt = info.is_ocdbt_checkpoint
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = _get_cast_tspec_deserialize(tspec, arg)
      if logging.level_debug():
        logging.debug('tspec = %s', tspec)
        logging.debug('infos = %s', infos)
        logging.debug('args = %s', args)
      deserialize_ops += [
          serialization.async_deserialize(
              sharding,
              tspec,
              global_shape=arg.global_shape
              if hasattr(arg, 'global_shape')
              else None,
              byte_limiter=info.byte_limiter,
              context=get_ts_context(use_ocdbt),
          )
      ]
    ret = await asyncio.gather(*deserialize_ops)

    if logging.level_debug():
      for a in ret:
        logging.debug(
            'restored jax.Array.shape = %s, jax.array.dtype = %s',
            a.shape,
            a.dtype,
        )
      logging.debug(
          'ts_metrics: %s', _dump_debug_data(self._metadata_key, infos)
      )

    return ret


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
    directory = os.fspath(info.parent_dir / self._filename)
    tspec: Dict[str, Any] = get_tensorstore_spec(directory, use_ocdbt=False)
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
    return [
        StringMetadata(name=info.name, directory=info.parent_dir)
        for info in infos
    ]

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
    txn = ts.Transaction()
    for (
        info,
        value,
    ) in zip(infos, values):
      tspec = self._get_json_tspec(info)
      if jax.process_index() == 0:
        t = await ts.open(
            tspec,
            open=True,
            context=self._ts_context,
        )
        write_future = t.with_transaction(txn).write(value)
        synchronous_ops += [write_future.copy]
    await asyncio.gather(*synchronous_ops)
    futures.append(txn.commit_async())
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
  # TODO(b/314258967): clean those up.
  del kwargs['use_ocdbt'], kwargs['ts_context']
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
