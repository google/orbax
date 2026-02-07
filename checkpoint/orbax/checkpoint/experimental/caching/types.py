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

"""Type definitions for the storage service."""

from __future__ import annotations

import dataclasses
import enum
from typing import NamedTuple, Protocol

from etils import epath


class StorageType(enum.Enum):
  LUSTRE = "lustre"
  GCS = "gcs"


class StorageTransfer(Protocol):
  """Function that copies from one storage to another, optionally cleaning up the source too.

  Returns True if the transfer completed, or False if it was skipped because the
  destination already existed.
  """

  def __init__(
      self,
      src: epath.Path,
      dst: epath.Path,
  ):
    ...

  def transfer(self) -> bool:
    ...

  def cleanup(self) -> None:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class StorageTier:
  """Configuration for a storage tier."""
  priority: int
  storage_type: StorageType
  path: epath.Path

  def to_dict(self):
    return {
        "priority": self.priority,
        "storage_type": self.storage_type.value,
        "path": self.path.as_posix(),
    }

  def asset_path(self, asset_id: AssetId) -> epath.Path:
    return self.path / str(asset_id.execution_id) / str(asset_id.step)


class AssetId(NamedTuple):
  execution_id: int
  step: int


@dataclasses.dataclass
class AssetMetadata:
  asset_id: AssetId
  tier_id: int
  # TODO(cpgaffney): Probably this should be allowed to be multiple paths.
  path: epath.Path
  finalized: bool = False

  def to_dict(self):
    return {
        "asset_id": {
            "execution_id": self.asset_id.execution_id,
            "step": self.asset_id.step,
        },
        "tier_id": self.tier_id,
        "path": self.path.as_posix(),
        "finalized": self.finalized,
    }
