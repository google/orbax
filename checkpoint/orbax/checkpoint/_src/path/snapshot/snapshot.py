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

"""Provides abstractions for creating and managing snapshots of checkpoints."""

import abc
import asyncio
import enum
import time
import uuid

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import utils as ocp_path_utils


SNAPSHOTTING_TIME = "snapshotting_time"
PENDING_DIR_SUFFIX = ".pending_"


class SnapshotType(enum.Enum):
  IN_PLACE = "in_place"
  EMPTY = "empty"


class Snapshot(abc.ABC):
  """Represents a snapshot of a checkpoint."""

  _source: epath.Path
  _snapshot: epath.Path

  @abc.abstractmethod
  async def create_snapshot(self) -> None:
    """Creates a snapshot of the checkpoint."""
    pass

  @abc.abstractmethod
  async def release_snapshot(self) -> None:
    """Deletes a snapshot of the checkpoint."""
    pass

  @abc.abstractmethod
  async def replace_source(self) -> None:
    """Replaces the source checkpoint with the snapshot."""
    pass


class _DefaultSnapshot(Snapshot):
  """Creates a copy of the checkpoint in the snapshot folder."""

  def __init__(
      self,
      src: epath.PathLike,
      dst: epath.PathLike,
  ):
    self._source = epath.Path(src)
    self._snapshot = epath.Path(dst)

  async def create_snapshot(self) -> None:
    """Creates a deep copy of the checkpoint."""
    if not await async_path.is_absolute(self._snapshot):
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    if not await async_path.exists(self._source):
      raise ValueError(f"Snapshot source does not exist: {self._source}'.")

    t = ocp_path_utils.Timer()
    await asyncio.to_thread(
        ocp_path_utils.recursively_copy_files,
        self._source,
        self._snapshot,
    )
    logging.debug(
        "Snapshot copy: %fs",
        t.get_duration(),
    )

  async def release_snapshot(self) -> None:
    """Deletes a snapshot of the checkpoint."""
    if not await async_path.exists(self._snapshot):
      raise FileNotFoundError(f"Snapshot does not exist: {self._snapshot}")

    await async_path.rmtree(self._snapshot)

  # TODO(b/434025182): Handle recovery path upon restart.
  async def replace_source(self) -> None:
    """Replaces the source checkpoint with the snapshot."""
    if not self._snapshot.is_absolute():
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    if not self._source.is_absolute():
      raise ValueError(
          f"Snapshot source must be absolute, but was '{self._source}'."
      )

    recovery_path = (
        self._source.parent / f"{self._source.name}._recovery_{time.time()}"
    )

    await async_path.rename(self._source, recovery_path)
    await async_path.rename(self._snapshot, self._source)
    await async_path.rmtree(recovery_path)




class _EmptySnapshot(Snapshot):
  """An empty snapshot doesn't copy initially.

  Used to aggregate new files into a new separate subdirectory, which is then
  cheaply moved under the `_source` directory upon `replace_source`.
  """

  def __init__(
      self,
      src: epath.PathLike,
      dst: epath.PathLike,
  ):
    self._source = epath.Path(src)
    self._snapshot = epath.Path(dst)

  async def create_snapshot(self) -> None:
    if not await async_path.is_absolute(self._snapshot):
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    await async_path.mkdir(self._snapshot, parents=True, exist_ok=True)

  async def release_snapshot(self) -> None:
    if not await async_path.exists(self._snapshot):
      return
    await async_path.rmtree(self._snapshot)

  async def replace_source(self) -> None:
    if not self._snapshot.is_absolute():
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    if not self._source.is_absolute():
      raise ValueError(
          f"Snapshot source must be absolute, but was '{self._source}'."
      )

    if not await async_path.exists(self._snapshot):
      raise FileNotFoundError(f"Snapshot does not exist: {self._snapshot}")
    if not await async_path.exists(self._source):
      await async_path.mkdir(self._source, parents=True, exist_ok=True)
    # Move files from inside the tmp snapshot into the original source
    # directory under a pending suffix. This is to avoid potentially wiping
    # out previous files.
    # Partial saving relies on this behavior to accumulate `pending_*`
    # directories in the shared parent path before merging them sequentially
    # upon finalization.
    pending_suffix = f"{PENDING_DIR_SUFFIX}{uuid.uuid4()}"
    dst_path = self._source / f"{self._source.name}{pending_suffix}"
    await async_path.rename(self._snapshot, dst_path)


def create_instance(
    source: epath.Path,
    snapshot: epath.Path,
    *,
    set_immutable: bool | None = None,
    snapshot_type: SnapshotType = SnapshotType.IN_PLACE,
):
  """Creates a snapshot instance according to the provided options."""
  if snapshot_type == SnapshotType.EMPTY:
    return _EmptySnapshot(source, snapshot)

  return _DefaultSnapshot(source, snapshot)
