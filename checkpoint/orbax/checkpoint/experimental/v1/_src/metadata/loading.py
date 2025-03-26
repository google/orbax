# Copyright 2024 The Orbax Authors.
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

"""Functions for loading metadata from a checkpoint."""

from orbax.checkpoint import handlers
from orbax.checkpoint._src.checkpointers import checkpointer as checkpointer_lib
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types

CheckpointMetadata = metadata_types.CheckpointMetadata
PyTreeMetadata = metadata_types.PyTreeMetadata
PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY


def pytree_metadata(
    path: path_types.PathLike,
) -> CheckpointMetadata[PyTreeMetadata]:
  """Loads the PyTree metadata from a checkpoint."""
  format_utils.validate_pytree_checkpoint(path)
  context = context_lib.get_context()
  registry = handlers.create_default_handler_registry(
      **{PYTREE_CHECKPOINTABLE_KEY: pytree_handler.create_v0_handler(context)}
  )
  with checkpointer_lib.Checkpointer(
      handlers.CompositeCheckpointHandler(handler_registry=registry)
  ) as ckptr:
    metadata = ckptr.metadata(path)
    return CheckpointMetadata[PyTreeMetadata](
        metadata=PyTreeMetadata(
            pytree=metadata.item_metadata[PYTREE_CHECKPOINTABLE_KEY].tree,
        ),
        init_timestamp_nsecs=metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=metadata.commit_timestamp_nsecs,
        custom_metadata=metadata.custom_metadata,
    )
