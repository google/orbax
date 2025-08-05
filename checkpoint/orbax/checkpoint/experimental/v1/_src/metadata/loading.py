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

"""Functions for loading metadata from a checkpoint."""

from typing import Any

from etils import epath
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.loading import v0_compatibility
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types


CheckpointMetadata = metadata_types.CheckpointMetadata
PyTreeMetadata = metadata_types.PyTreeMetadata
PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY


def pytree_metadata(
    path: path_types.PathLike,
) -> CheckpointMetadata[PyTreeMetadata]:
  """Loads the PyTree metadata from a checkpoint.

  This function retrieves metadata for a PyTree checkpoint, returning an
  object of type `CheckpointMetadata[PyTreeMetadata]`. Please see documentation
  on this class for further details.

  In short, the returned object contains a `metadata` attribute (among other
  attributes like timestamps), which is an instance of `PyTreeMetadata`. The
  `PyTreeMetadata` describes information specific to the PyTree itself. The most
  important such property is the PyTree structure, which is a tree structure
  matching the structure of the checkpointed PyTree, with leaf metadata objects
  describing each leaf.

  For example::

    metadata = ocp.pytree_metadata(path)  # CheckpointMetadata[PyTreeMetadata]
    metadata.metadata # PyTreeMetadata
    metadata.init_timestamp_nsecs  # Checkpoint creation timestamp.

    metadata.metadata  # PyTree structure.

  The metadata can then be used to inform checkpoint loading. For example::

    metadata = ocp.pytree_metadata(path)
    restored = ocp.load_pytree(path, metadata)

    # Load with altered properties.
    def _get_abstract_array(arr):
      # Assumes all checkpoint leaves are array types.
      new_dtype = ...
      new_sharding = ...
      return jax.ShapeDtypeStruct(arr.shape, new_dtype, sharding=new_sharding)

    metadata = dataclasses.replace(metadata,
          metadata=jax.tree.map(_get_abstract_array, metadata.metadata)
    )
    ocp.load_pytree(path, metadata)

  Args:
    path: The path to the checkpoint.

  Returns:
    A `CheckpointMetadata[PyTreeMetadata]` object.
  """
  format_utils.validate_checkpoint_directory(path)
  format_utils.validate_checkpoint_metadata(path)
  format_utils.validate_pytree_checkpoint(path)

  metadata = _checkpointables_metadata_impl(path)
  return CheckpointMetadata[PyTreeMetadata](
      metadata=metadata.metadata[PYTREE_CHECKPOINTABLE_KEY],
      init_timestamp_nsecs=metadata.init_timestamp_nsecs,
      commit_timestamp_nsecs=metadata.commit_timestamp_nsecs,
      custom_metadata=metadata.custom_metadata,
  )


def checkpointables_metadata(
    path: path_types.PathLike,
) -> CheckpointMetadata[dict[str, Any]]:
  """Loads all checkpointables metadata from a checkpoint.

  This function is a more general version of `pytree_metadata`. The same
  `CheckpointMetadata` object is returned (with properties like
  `init_timestamp_nsecs` as shown above), but the type of the core `metadata`
  property is a dictionary, mapping checkpointable names to their metadata. This
  mirrors the return value of `load_checkpointables`, which similarly returns a
  dictionary mapping checkpointable names to their loaded values.

  For example::

    ocp.save_checkpointables(path, {
        'foo': Foo(),
        'bar': Bar(),
    })
    metadata = ocp.checkpointables_metadata(path)
    metadata.metadata  # {'foo': AbstractFoo(), 'bar': AbstractBar()}

  Args:
    path: The path to the checkpoint.

  Returns:
    A `CheckpointMetadata[dict[str, Any]]` object.
  """
  format_utils.validate_checkpoint_directory(path)
  format_utils.validate_checkpoint_metadata(path)

  return _checkpointables_metadata_impl(path)


def _checkpointables_metadata_impl(
    path: path_types.PathLike,
) -> CheckpointMetadata[dict[str, Any]]:
  """Shared implementation for checkpointables_metadata."""
  path = epath.Path(path)
  checkpointer, _ = v0_compatibility.get_v0_checkpointer_and_args(
      path, None, context=context_lib.get_context()
  )
  metadata = checkpointer.metadata(path)
  item_metadata = {k: v for k, v in metadata.item_metadata.items()}
  # Exclude `metrics` if present. This is relevant only for
  # `training.Checkpointer`, and is separately added to the
  # `training.CheckpointMetadata` object.
  item_metadata.pop('metrics', None)
  return CheckpointMetadata[dict[str, Any]](
      metadata=item_metadata,
      init_timestamp_nsecs=metadata.init_timestamp_nsecs,
      commit_timestamp_nsecs=metadata.commit_timestamp_nsecs,
      custom_metadata=metadata.custom_metadata,
  )
