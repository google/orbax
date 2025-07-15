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

from typing import Protocol

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import utils as ocp_path_utils


SNAPSHOTTING_TIME = "snapshotting_time"


class Snapshot(Protocol):
  """Represents a snapshot of a checkpoint."""

  _source: epath.Path
  _snapshot: epath.Path

  def create_snapshot(self) -> None:
    """Creates a snapshot of the checkpoint."""
    pass

  def release_snapshot(self) -> bool:
    """Deletes a snapshot of the checkpoint."""
    pass




class _DefaultSnapshot(Snapshot):
  """Creates a copy of the checkpoint in the snapshot folder."""

  def __init__(self, src: epath.PathLike, dst: epath.PathLike):
    self._source = epath.Path(src)
    self._snapshot = epath.Path(dst)

  def create_snapshot(self):
    """Creates a deep copy of the checkpoint."""
    if not epath.Path(self._snapshot).is_absolute():
      raise ValueError(
          f"Snapshot destination must be absolute, but was '{self._snapshot}'."
      )
    if not epath.Path(self._source).exists():
      raise ValueError(f"Snapshot source does not exist: {self._source}'.")

    t = ocp_path_utils.Timer()
    ocp_path_utils.recursively_copy_files(self._source, self._snapshot)
    logging.debug(
        "Snapshot copy: %fs",
        t.get_duration(),
    )

  def release_snapshot(self) -> bool:
    """Deletes a snapshot of the checkpoint."""
    if not epath.Path(self._snapshot).exists():
      logging.error("Snapshot does not exist: %s", self._snapshot)
      return False

    try:
      epath.Path(self._snapshot).rmtree()
    except OSError as e:
      logging.error(e)
      return False
    else:
      return True


def create_instance(source: epath.Path, snapshot: epath.Path):
  return _DefaultSnapshot(source, snapshot)
