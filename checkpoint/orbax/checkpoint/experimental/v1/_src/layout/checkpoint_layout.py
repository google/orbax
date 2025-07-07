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

"""Defines `CheckpointLayout`, a broader protocol used to save and load different checkpoint formats."""

from typing import Any, Awaitable, Protocol
from orbax.checkpoint.experimental.v1._src.path.types import Path,


class CheckpointLayout(Protocol):
  """CheckpointLayout.

  This class defines a protocol for different checkpoint formats. It is a helper
  component for `save_checkpointables` and
  `load_checkpointables`. It works the same way as CompositeHandler but supports
  alternative checkpoint formats. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """


  def validate(self, path: Path):
    # Validates the given path,
    # attempting to assume it conforms to this instance.
    # If validation fails, the checkpoint does not conform to this format.
    ...

  async def load(
      self,
      directory: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    ...