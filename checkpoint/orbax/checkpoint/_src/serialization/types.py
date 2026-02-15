# Copyright 2026 The Orbax Authors.
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

"""Types and constructs for PyTree metadata and serialization."""

from __future__ import annotations

import abc
import asyncio
import dataclasses
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, Union

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.serialization import limits
import tensorstore as ts

PyTreeMetadataOptions = pytree_metadata_options_lib.PyTreeMetadataOptions


def is_supported_type(
    value: Any,
    pytree_metadata_options: PyTreeMetadataOptions = (
        pytree_metadata_options_lib.PYTREE_METADATA_OPTIONS
    ),
) -> bool:
  """Determines if the `value` is supported without custom TypeHandler."""
  return isinstance(
      value,
      (str, int, float, np.number, np.ndarray, bytes, jax.Array),
  ) or empty_values.is_supported_empty_value(value, pytree_metadata_options)


def check_input_arguments(*args):
  l = None
  for arg in args:
    if l == 0:
      raise ValueError('Cannot pass TypeHandler input of length 0.')
    if l is None:
      l = len(arg)
    elif len(arg) != l:
      raise ValueError('Found input args with mismatched lengths.')


@dataclasses.dataclass(kw_only=True)
class ParamInfo:
  """Information describing a parameter in a PyTree.

  Note that ParamInfo is distinct from SaveArgs and RestoreArgs in that in
  represents information not provided by a user, and should be computed
  internally.

  name:
    Name of the parameter.
  parent_dir:
    A path providing location where all files under the same checkpoint should
    be saved under. All `ParamInfo` provided to a given TypeHandler should have
    the same `parent_dir`. The parent_dir is assumed to be a directory.
  path:
    Do not provide directly. Automatically set to `parent_dir / name`.
  skip_deserialize:
    If specified, skips deserialization of the given parameter using the
    TypeHandler. This may be for multiple different reasons, including that the
    parameter may have been aggregated, or it will be unneeded after
    transformations. Note: this parameter is handled by PyTreeCheckpointHandler,
    so it is unnecessary for TypeHandler implementations to deal with it.
  byte_limiter:
    Object to limit the number of bytes that can be read or written in
    parallel.
  device_host_byte_limiter:
    Object to limit the number of bytes that can be transferred from device to
    host memory in parallel.
  is_ocdbt_checkpoint:
    Indicates whether the checkpoint path uses OCDBT format
    or not. Only used for restoration.
  use_compression:
    When True, turn on zstd compression. Default is True.
  use_zarr3:
    If True, use Zarr ver3 otherwise ver2.
  ocdbt_target_data_file_size:
    Specifies the target size (in bytes) of each OCDBT data file. If set to 0,
    data file size is not limited. If omitted (None), the TensorStore default
    is used.
  ts_context:
    Tensorstore context to use for reading/writing.
  value_typestr: stores the original value's typestr (from TypeHandler).
    Only required when saving.
  enable_pinned_host_transfer:
    True by default. If False, disables transfer to pinned host when copying
    from device to host, regardless of the presence of pinned host memory.
  raise_array_data_missing_error:
    Only used for restoring. See documentation in `tensorstore_utils.py`. Comes
    from tree metadata and should be the same across all parameters.
  write_shape:
    Shape of the array shard. Used in the subchunking context.
  is_prioritized_key_fn: See `IsPrioritizedKeyFn` definition.
  """

  name: str
  parent_dir: Union[epath.Path, 'asyncio.Future[epath.Path]']
  path: Optional[epath.Path] = None
  keypath: Optional[Tuple[Any, ...]] = None
  skip_deserialize: Optional[bool] = None
  byte_limiter: Optional[limits.ByteLimiter] = None
  device_host_byte_limiter: Optional[limits.ByteLimiter] = None
  is_ocdbt_checkpoint: Optional[bool] = None
  use_compression: bool | None = True
  use_zarr3: Optional[bool] = False
  ocdbt_target_data_file_size: Optional[int] = None
  ts_context: Optional[ts.Context] = None
  value_typestr: Optional[str] = None
  enable_pinned_host_transfer: bool = False
  raise_array_data_missing_error: bool = True
  write_shape: arrays_types.Shape | None = None
  is_prioritized_key_fn: Optional[IsPrioritizedKeyFn] = None

  def __post_init__(self):
    if self.path is None and not isinstance(self.parent_dir, asyncio.Future):
      self.path = self.parent_dir / self.name

  async def get_resolved_path(self) -> epath.Path:
    """Resolves and returns the path, awaiting parent_dir if needed."""
    if isinstance(self.parent_dir, asyncio.Future):
      resolved_dir = await self.parent_dir
      return resolved_dir / self.name
    else:
      if self.path is None:
        self.path = self.parent_dir / self.name
      return self.path

  async def get_parent_dir(self) -> epath.Path:
    """Resolves and returns the parent directory, awaiting if needed."""
    if isinstance(self.parent_dir, asyncio.Future):
      return await self.parent_dir
    return self.parent_dir


@dataclasses.dataclass
class SaveArgs:
  """Extra arguments that can be provided for saving.

  aggregate:
    Deprecated, please use custom TypeHandler
    (https://orbax.readthedocs.io/en/latest/guides/checkpoint/custom_handlers.html#typehandler)
    or contact Orbax team to migrate before August 1st, 2024. If true, saves the
    given
    parameter in an aggregated tree format rather than individually. See
    AggregateHandler.
  dtype:
    If provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  chunk_byte_size:
    This is an experimental feature that automatically chooses the largest chunk
    shape possible, while keeping the chunk byte size less than or equal to the
    specified chunk_byte_size. Both the write_chunk_shape and read_chunk_shape
    are automatically set to the chosen shape. This uses a greedy algorithm that
    prioritizes splitting the largest dimensions first.
  shard_axes: An optional list of axes that should be prioritized when
      sharding array for storage. If empty, storage sharding implementation will
      prioritize axes which are already sharded.
  """

  aggregate: bool = False
  dtype: Optional[jnp.dtype] = None
  chunk_byte_size: Optional[int] = None
  shard_axes: tuple[int, ...] = tuple()

  def __post_init__(self):
    if self.aggregate:
      jax.monitoring.record_event('/jax/orbax/deprecation/aggregate')
      logging.log_every_n_seconds(
          logging.WARNING,
          'The `aggregate` option is deprecated and will be ignored.',
          n_seconds=12 * 60 * 60,  # once every 12 hours
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
  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[value_metadata.Metadata]:
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
  ) -> Sequence[future.Future]:
    """Writes the parameter to a storage location.

    This method is responsible for copying the parameter from a remote device in
    a synchronous fashion (if applicable). It should then return a list of
    futures which can be later awaited to complete the final commit operation
    to a storage location.

    Note: Any operations writing to storage location should be done by using
    `future.CommitFutureAwaitingContractedSignals` to wait for the directories
    to be created.

    The function can be used in a multihost setting, but should not implement
    extra logic to ensure atomicity.

    Args:
      values: a sequence of parameters to save.
      infos: a sequence of ParamInfo containing relevant information for
        serialization of each value.
      args: a sequence of additional arguments for serialization, provided by
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
      infos: Sequence of ParamInfo for deserialization.
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

  def memory_size(self, values: Sequence[Any]) -> Sequence[Tuple[int, int]]:
    """For a batch of values, returns the size of each value in bytes.

    Note that the default implementation uses `sys.getsizeof`, which is not
    likely to be accurate for many types.

    The value returned is intended to be per-host.

    Args:
      values: A batch of values.

    Returns:
      A sequence of elements corresponding to `values`. Each element is a tuple
      of [write_size, read_size]. In many cases these values may be the same.

    Raises:
      NotImplementedError: Raises error by default since we will rely on a
        backup implementation.
    """
    raise NotImplementedError()


class TypeHandlerRegistry(Protocol):
  """A registry for TypeHandlers.

  This internal base class is used for the global registry which serves as a
  default for any type not found in a local registry. It is also accessed
  through the module function get/set/has_type_handler.
  """

  def add(
      self,
      ty: Any,
      handler: TypeHandler,
      func: Optional[Callable[[Any], bool]] = None,
      override: bool = False,
      ignore_warnings: bool = False,
  ):
    """Registers a type for serialization/deserialization with a given handler.

    Note that it is possible for a type to match multiple different entries in
    the registry, each with a different handler. In this case, only the first
    match is used.

    Args:
      ty: A type to register.
      handler: a TypeHandler capable of reading and writing parameters of type
        `ty`.
      func: A function that accepts a type and returns True if the type should
        be handled by the provided TypeHandler. If this parameter is not
        specified, defaults to `lambda t: issubclass(t, ty)`.
      override: if True, will override an existing mapping of type to handler.
      ignore_warnings: if True, will ignore warnings when replacing an existing
        handler.

    Raises:
      ValueError if a type is already registered and override is False.
    """
    ...

  def get(self, ty: Any) -> TypeHandler:
    """Returns the handler registered for a given type, if available.

    Args:
      ty: an object type (or string representation of the type.)

    Returns:
      The TypeHandler that is registered for the given type.

    Raises:
      ValueError if the given type has no registered handler.
    """
    ...

  def has(self, ty: Any) -> bool:
    """Checks if a type is registered.

    Args:
      ty: an object type (or string representation of the type.)

    Returns:
      A boolean indicating if ty is registered.
    """
    ...


class IsPrioritizedKeyFn(Protocol):
  """Protocol for checking if a key is prioritized.

  The function accepts a PyTree keypath (obtained
  using jax.tree.map_with_path) and returns True if the D2H transfer should be
  scheduled during the blocking part of the save (defaults to True in all places
  unless False is returned by this function).

  The D2H transfer is scheduled before returning
  to the caller, so the values will never be corrupted by a concurrent update
  or donation. Keys that are not prioritized will not
  be scheduled for transfer until all prioritized keys have been fully
  written to the checkpoint. This means that these values may be altered
  if the values are updated concurrently.

  Callers should take care to call
  `wait_until_finished` before updating array values (e.g.
  `apply_gradients`) if some keys are not prioritized. Note that any
  "prioritized" keys are assumed to be lightweight, and
  `save_device_host_concurrent_gb` will be ignored for them.
  """

  def __call__(self, keypath: Tuple[Any, ...]) -> bool:
    """Returns true if the key is prioritized."""


async def resolve_param_infos(
    infos: Sequence[ParamInfo],
) -> Sequence[ParamInfo]:
  """Resolves any future directories in ParamInfos.

  If the parent_dir in any ParamInfo is an asyncio.Future, awaits it and
  returns a new sequence of ParamInfos with resolved paths. Otherwise returns
  the input sequence unchanged.

  Args:
    infos: Sequence of ParamInfo objects, potentially with future parent_dir.

  Returns:
    Sequence of ParamInfo objects with resolved parent_dir and path.
  """
  if not infos:
    return infos

  if isinstance(infos[0].parent_dir, asyncio.Future):
    resolved_dir = await infos[0].parent_dir
    return tuple([
        dataclasses.replace(
            info, parent_dir=resolved_dir, path=resolved_dir / info.name
        )
        for info in infos
    ])
  return infos
