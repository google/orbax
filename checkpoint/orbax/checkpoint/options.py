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

"""Configuration options for APIs like CheckpointManager and Checkpointer."""

import dataclasses
from typing import Callable, Optional, Set

from orbax.checkpoint._src.multihost import multihost



@dataclasses.dataclass
class AsyncOptions:
  """Options used to configure async behavior.

  See :py:class:`.AsyncCheckpointer` for details.
  """

  timeout_secs: int = 600  # 10 minutes. Same as default in `AsyncCheckpointer`.
  barrier_sync_fn: Optional[multihost.BarrierSyncFn] = None
  post_finalization_callback: Optional[Callable[[], None]] = None
  create_directories_asynchronously: bool = True


@dataclasses.dataclass
class MultiprocessingOptions:
  """Options used to configure multiprocessing behavior.

  primary_host: the host id of the primary host.  Default to 0.  If it's set
    to None, then all hosts will be considered as primary.  It's useful in
    the case that all hosts are only working with local storage.
  active_processes: A set of process indices (corresponding to
    `multihost.process_index()`) over which `CheckpointManager` is expected to
    be called. This makes it possible to have a `CheckpointManager` instance
    that runs over a subset of processes, rather than all processes as it is
    normally expected to do. If specified, `primary_host` must belong to
    `active_processes`.
  barrier_sync_key_prefix: A string to be prepended to the barrier sync key
    used to synchronize processes. This is useful to avoid collisions with
    other barrier syncs if another CheckpointManager is being used concurrently.
  """

  primary_host: Optional[int] = 0
  active_processes: Optional[Set[int]] = None
  barrier_sync_key_prefix: Optional[str] = None




@dataclasses.dataclass(frozen=True)
class FileOptions:
  """Options used to configure checkpoint directories and files.

  Attributes:
    path_permission_mode: Path permission mode for step directories, user
      metadata files. e.g. 0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
  """

  path_permission_mode: int | None = None

@dataclasses.dataclass
class PyTreeOptions:
  """Options for PyTree checkpointing behavior

  Attributes:
    enable_pinned_host_transfer: Whether to use pinned host memory for D2H transfer
    use_replica_parallel: Whether to parallelize saving across replicas
    min_slice_bytes_for_replica_parallel: Minimum bytes per replica slice
    max_replicas_for_replica_parallel: Maximum replicas for parallel saving
    enable_replica_parallel_separate_folder: Save replicated/sharded in separate folders
    save_concurrent_gb: Max concurrent GB for writing
    restore_concurrent_gb: Max concurrent GB for reading
    save_device_host_concurrent_gb: Max concurrent GB for D2H transfer
    use_ocdbt: Use Tensorstore OCDBT driver
    use_zarr3: Use Zarr version 3
    use_compression: Use compression (zstd for zarr2)
  """
  enable_pinned_host_transfer: Optional[bool] = None
  use_replica_parallel: bool = True
  min_slice_bytes_for_replica_parallel: Optional[int] = None
  max_replicas_for_replica_parallel: Optional[int] = None
  enable_replica_parallel_separate_folder: bool = False
  save_concurrent_gb: Optional[int] = None
  restore_concurrent_gb: Optional[int] = None
  save_device_host_concurrent_gb: Optional[int] = None
  use_ocdbt: bool = True
  use_zarr3: bool = False
  use_compression: bool = True