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

"""Snapshot represents operations on how to create, delete snapshots of a checkpoint."""

import abc
import asyncio
import time

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import utils as ocp_path_utils


SNAPSHOTTING_TIME = "snapshotting_time"


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
    if not await asyncio.to_thread(self._snapshot.is_absolute):
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    if not await asyncio.to_thread(self._source.exists):
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
    if not await asyncio.to_thread(self._snapshot.exists):
      raise FileNotFoundError(f"Snapshot does not exist: {self._snapshot}")

    await asyncio.to_thread(self._snapshot.rmtree)

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

    def _swap_source_and_snapshot():
      self._source.rename(recovery_path)
      self._snapshot.rename(self._source)
      recovery_path.rmtree()

    await asyncio.to_thread(_swap_source_and_snapshot)




def create_instance(
    source: epath.Path,
    snapshot: epath.Path,
    *,
    set_immutable: bool | None = None,
):
  return _DefaultSnapshot(source, snapshot)
