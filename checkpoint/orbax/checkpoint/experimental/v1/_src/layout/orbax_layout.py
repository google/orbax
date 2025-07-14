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

"""Defines `OrbaxLayout`, a class to handle Orbax checkpoint formats."""
from typing import Any, Awaitable
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types

CompositeHandler = composite_handler.CompositeHandler
Path = types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout


class OrbaxLayout(CheckpointLayout):
  """OrbaxLayout.

  This class defines a class to handle Orbax checkpoint formats. It inherits
  abstract methods from CheckpointLayout. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self):
    self._context = context_lib.get_context()
    self._handler_registry = registration.local_registry(
        self._context.checkpointables_options.registry,
        include_global_registry=False,
    )
    self._composite_handler = CompositeHandler(self._handler_registry)

  def validate(self, path: Path):
    if (path / composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE).exists():
      return True
    try:
      format_utils.validate_checkpoint_directory(path)
      format_utils.validate_checkpoint_metadata(path)
      return True
    except (FileNotFoundError, NotADirectoryError, ValueError):
      return False

  async def load(
      self,
      directory: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._composite_handler.load(
        directory, abstract_checkpointables
    )
    return load_awaitable
