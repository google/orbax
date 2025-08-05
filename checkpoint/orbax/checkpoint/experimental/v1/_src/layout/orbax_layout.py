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
from orbax.checkpoint.experimental.v1._src.loading import v0_compatibility
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError
CompositeHandler = composite_handler.CompositeHandler
Path = types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout

ORBAX_CHECKPOINT_INDICATOR_FILE = (
    composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE
)


_V0_ERROR_MESSAGE = (
    "If your checkpoint was saved with the Orbax V0 API, please follow the"
    " instructions at"
    " https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/orbax_v0_to_v1_migration.html"
    " to load it with the Orbax V1 API."
)
_GENERAL_ERROR_MESSAGE = (
    " Note that a valid checkpoint path should always contain a file named"
    f" '{ORBAX_CHECKPOINT_INDICATOR_FILE}' (unless it was saved with the V0"
    f" API). {_V0_ERROR_MESSAGE}"
)


class OrbaxLayout(CheckpointLayout):
  """OrbaxLayout.

  This class defines a class to handle Orbax checkpoint formats. It inherits
  abstract methods from CheckpointLayout. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self, path: Path):
    self._context = context_lib.get_context()
    self._handler_registry = registration.local_registry(
        self._context.checkpointables_options.registry,
        include_global_registry=False,
    )
    self._composite_handler = CompositeHandler(self._handler_registry)
    self._path = path

  @property
  def path(self) -> Path:
    """Returns the path of the Orbax checkpoint."""
    return self._path

  @property
  def has_indicator_file(self) -> bool:
    return (self._path / ORBAX_CHECKPOINT_INDICATOR_FILE).exists()

  async def metadata(self) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    # Uses the v0 checkpointer to get v0 StepMetadata
    checkpointer, _ = v0_compatibility.get_v0_checkpointer_and_args(
        self._path, None, context=context_lib.get_context()
    )
    step_metadata = checkpointer.metadata(self._path)

    item_metadata = {k: v for k, v in step_metadata.item_metadata.items()}
    # Exclude `metrics` if present. This is relevant only for
    # `training.Checkpointer`, and is separately added to the
    # `training.CheckpointMetadata` object.
    item_metadata.pop("metrics", None)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata=item_metadata,
        init_timestamp_nsecs=step_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=step_metadata.commit_timestamp_nsecs,
        custom_metadata=step_metadata.custom_metadata,
    )

  def validate(self):
    try:
      format_utils.validate_checkpoint_directory(self._path)
      if self.has_indicator_file:
        format_utils.validate_checkpoint_metadata(self._path)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {self._path} as an Orbax checkpoint."
          f" {_GENERAL_ERROR_MESSAGE}"
      ) from e

  def validate_pytree(self, checkpointable_name: str | None) -> None:
    """Validates the given path as a PyTree checkpoint."""
    try:
      format_utils.validate_pytree_checkpoint(
          self._path, checkpointable_name=checkpointable_name
      )
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {self._path} as an Orbax PyTree"
          f" checkpoint. {_GENERAL_ERROR_MESSAGE}"
      ) from e

  async def load(
      self,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._composite_handler.load(
        self._path, abstract_checkpointables
    )
    return load_awaitable
