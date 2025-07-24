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

"""Global configuration options."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Protocol, Type

import numpy as np
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types



@dataclasses.dataclass(frozen=True, kw_only=True)
class AsyncOptions:
  """Options used to configure async behavior.

  timeout_secs:
    The timeout in seconds for the async save operation.
  post_finalization_callback:
    A function that is called after the async save operation is complete.
  create_directories_asynchronously:
    If true, create directories asynchronously in the background.
  """

  timeout_secs: int = 600  # 10 minutes.
  post_finalization_callback: Callable[[], None] | None = None
  create_directories_asynchronously: bool = True

  def v0(self) -> v0_options_lib.AsyncOptions:
    return v0_options_lib.AsyncOptions(
        timeout_secs=self.timeout_secs,
        post_finalization_callback=self.post_finalization_callback,
        create_directories_asynchronously=self.create_directories_asynchronously,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MultiprocessingOptions:
  """Options used to configure multiprocessing behavior.

  primary_host:
    The host id of the primary host.  Default to 0.  If it's set to None, then
    all hosts will be considered as primary.  It's useful in the case that all
    hosts are only working with local storage.
  active_processes:
    A set of process indices (corresponding to `multihost.process_index()`) over
    which `CheckpointManager` is expected to be called. This makes it possible
    to have a `CheckpointManager` instance that runs over a subset of processes,
    rather than all processes as it is normally expected to do. If specified,
    `primary_host` must belong to `active_processes`.
  barrier_sync_key_prefix:
    A string to be prepended to the barrier sync key used to synchronize
    processes. This is useful to avoid collisions with other barrier syncs if
    another CheckpointManager is being used concurrently.
  """

  primary_host: int | None = 0
  active_processes: set[int] | None = None
  barrier_sync_key_prefix: str | None = None

  def v0(self) -> v0_options_lib.MultiprocessingOptions:
    return v0_options_lib.MultiprocessingOptions(
        primary_host=self.primary_host,
        active_processes=self.active_processes,
        barrier_sync_key_prefix=self.barrier_sync_key_prefix,
    )


# pyformat: disable
@dataclasses.dataclass(frozen=True, kw_only=True)
class FileOptions:
  """Options used to configure checkpoint directories and files.

  Attributes:
    path_permission_mode:
      Path permission mode for step directories, user metadata files. e.g.
      0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
      path is supported. default=None.
    temporary_path_class:
      A class that is used to create and finallize temporary paths, and to
      ensure atomicity.
  """

  path_permission_mode: int | None = None
  temporary_path_class: atomicity_types.TemporaryPath | None = None

  def v0(self) -> v0_options_lib.FileOptions:
    """Converts this FileOptions to a v0 FileOptions."""
    return v0_options_lib.FileOptions(
        path_permission_mode=self.path_permission_mode,
    )


# pyformat: enable


@dataclasses.dataclass(frozen=True, kw_only=True)
class PyTreeOptions:
  """Options used to configure PyTree saving and loading.

  Attributes:
    saving: Options for saving PyTrees.
    loading: Options for loading PyTrees.
    leaf_handler_registry: Optional Leaf Handler Registry. If provided, it will
      override the default Leaf Handler Registry.
  """

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Saving:
    """Options for saving PyTrees.

    create_array_storage_options_fn:
      A function that is called in order to create
      `ArrayOptions.Saving.StorageOptions` for each leaf in a PyTree, when it is
      being saved. It is called similar to:
      `jax.tree.map_with_path(create_array_storage_options_fn, pytree_to_save)`.
      If provided, it overrides any default settings in
      `ArrayOptions.Saving.StorageOptions`.
    pytree_metadata_options: Options for managing PyTree metadata.
    partial_update: NOT IMPLEMENTED.
    """

    class CreateArrayStorageOptionsFn(Protocol):

      def __call__(
          self, key: tree_types.PyTreeKeyPath, value: Any
      ) -> ArrayOptions.Saving.StorageOptions:
        ...

    create_array_storage_options_fn: CreateArrayStorageOptionsFn | None = None
    pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
        dataclasses.field(default_factory=tree_metadata.PyTreeMetadataOptions)
    )
    partial_update: bool = False

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Loading:
    """Options for loading PyTrees.

    partial_load: If True, only restore the parameters that are specified
      in the abstract PyTree.
    """

    partial_load: bool = False

  saving: Saving = dataclasses.field(default_factory=Saving)
  loading: Loading = dataclasses.field(default_factory=Loading)
  leaf_handler_registry: serialization_types.LeafHandlerRegistry | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ArrayOptions:
  """Options used to configure array saving and loading.

  Attributes:
    saving: Options for saving arrays.
    loading: Options for loading arrays.
  """

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Saving:
    """Options for saving arrays.

    Attributes:
      concurrent_bytes: Max concurrent bytes that are allowed for writing. Can
        help to reduce the possibility of OOM's when large checkpoints are
        saved.
      storage_options: Options used to customize array storage behavior for
        individual leaves. See below.
      use_ocdbt: Enables OCDBT format.
      use_zarr3: If True, use Zarr3 format.
      ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
        OCDBT data file. It only applies when OCDBT is enabled and Zarr3 must be
        turned on. If left unspecified, default size is 2GB.  A value of 0
        indicates no maximum file size limit.  For best results, ensure
        chunk_byte_size is smaller than this value.  For more details, refer to
        https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
      enable_pinned_host_transfer: Whether to use pinned_host memory for the
        transfer from device to host memory. Passing None will enable
        pinned_host memory depending on the platform used (currently only
        enables it for the GPU backend).
      enable_post_merge_validation: If True, enables validation of the
        parameters after the finalize step.
      use_replica_parallel: Whether to parallelize saving across replicas.
      enable_write_sharding_file: whether to write sharding file, defaults to
        True.
      array_metadata_store: Store to manage per host ArrayMetadata. To disable
        ArrayMetadata persistence, set it to None.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class StorageOptions:
      """Options used to customize array storage behavior for individual leaves.

      dtype:
        If provided, casts the parameter to the given dtype before saving.
        Note that the parameter must be compatible with the given type (e.g.
        jnp.bfloat16 is not compatible with np.ndarray).
      chunk_byte_size:
        This is an experimental feature that automatically chooses the largest
        chunk shape possible, while keeping the chunk byte size less than or
        equal to the specified chunk_byte_size. Both the write_chunk_shape and
        read_chunk_shape are automatically set to the chosen shape. This uses a
        greedy algorithm that prioritizes splitting the largest dimensions
        first.
      shard_axes:
        An optional list of axes that should be prioritized when sharding array
        for storage. If empty, storage sharding implementation will prioritize
        axes which are already sharded.
      """

      dtype: np.typing.DTypeLike | None = None
      chunk_byte_size: int | None = None
      shard_axes: tuple[int, ...] = tuple()

    concurrent_bytes: int | None = None
    storage_options: StorageOptions = dataclasses.field(
        default_factory=StorageOptions
    )
    use_ocdbt: bool = True
    use_zarr3: bool = True
    ocdbt_target_data_file_size: int | None = None
    enable_pinned_host_transfer: bool | None = None
    enable_post_merge_validation: bool = True
    use_replica_parallel: bool = True
    enable_write_sharding_file: bool = True
    array_metadata_store: array_metadata_store_lib.Store | None = (
        array_metadata_store_lib.Store()
    )

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Loading:
    """Options for loading arrays.

    concurrent_bytes:
      Max concurrent bytes that are allowed for reading. Can help to reduce the
      possibility of OOM's when large checkpoints are restored.
    enable_padding_and_truncation:
      If True, restoration allows silent truncating/padding of arrays if the
      stored array shape does not match the target shape. Otherwise, raises an
      error.
    """

    concurrent_bytes: int | None = None
    enable_padding_and_truncation: bool = False
    raise_array_data_missing_error: bool = True

  saving: Saving = dataclasses.field(default_factory=Saving)
  loading: Loading = dataclasses.field(default_factory=Loading)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointablesOptions:
  """Options used to configure `checkpointables` save/load behavior.

  Primarily intended for registering custom :py:class:`.CheckpointableHandler`
  classes. You can specify a registry directly, or use `create_with_handlers`.
  For example::

    checkpointables_options = (
      ocp.options.CheckpointablesOptions.create_with_handlers(
          FooHandler(),
          bar=BarHandler(),
      )
    )
    with ocp.Context(checkpointables_options=checkpointables_options)):
      ocp.save_checkpointables(directory, dict(foo=Foo(...), bar=Bar(...)))

  In this example, `FooHandler` is registered generically, which means that any
  checkpointable that is handleable by `FooHandler` can be saved/loaded (a
  `Foo` object in this case). In contrast, `BarHandler` is explicitly tied to
  the name `bar`, which means that only a checkpointable that is both handleable
  by `BarHandler` and has the name `bar` can handled by this `BarHandler`.

  Recall that a global registry also exists, containing core handlers like
  :py:class:`.PyTreeHandler` and :py:class:`.JsonHandler`. Use
  `ocp.handlers.register_handler` to register a handler globally.

  Note that registration order matters. For example, if saving a dict containing
  only strings, both :py:class:`.JsonHandler` and :py:class:`.PyTreeHandler` are
  capable of handling this object, but :py:class:`.JsonHandler` will be selected
  first because it is registered first.

  Attributes:
    registry: A `CheckpointableHandlerRegistry` that is used to resolve
      `CheckpointableHandler` classes for each provided `checkpointable` during
      saving and loading.
  """

  registry: registration.CheckpointableHandlerRegistry = dataclasses.field(
      default_factory=lambda: registration.ReadOnlyCheckpointableHandlerRegistry(
          registration.local_registry(include_global_registry=True)
      )
  )

  @classmethod
  def create_with_handlers(
      cls,
      *handlers: Type[handler_types.CheckpointableHandler],
      **named_handlers: Type[handler_types.CheckpointableHandler],
  ) -> CheckpointablesOptions:
    registry = registration.local_registry(include_global_registry=True)
    for handler in handlers:
      registry.add(handler, None)
    for name, handler in named_handlers.items():
      registry.add(handler, name)
    return cls(registry=registry)


class CheckpointLayout(enum.Enum):
  """The layout of the checkpoint.

  By default, Orbax saves and loads checkpoints with its own layout. However,
  support for other layouts is available, as a means of supporting
  interoperatibility with other checkpointing libraries.

  Currently supported layouts are:
    ORBAX: Orbax's own layout.
    SAFETENSORS: https://huggingface.co/docs/safetensors/en/index
  """

  ORBAX = 'Orbax'
  SAFETENSORS = 'SafeTensors'
