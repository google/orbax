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

"""Testing utils for compatibility tests."""

import os
from typing import Any
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler


_BASE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def get_checkpoint_path(
    version: str,
    metadata_present: bool,
    is_direct_checkpoint: bool,
    is_pytree: bool,
) -> epath.Path | None:
  """Returns the path to the checkpoint for each combination of parameters."""
  if version == 'v1' and is_direct_checkpoint:
    return None  # V1 does not support direct checkpoints.

  version_dir = f'{version}_checkpoints'
  type_dir = (
      'direct_checkpoint' if is_direct_checkpoint else 'composite_checkpoint'
  )
  metadata_dir = (
      'checkpoint_metadata_present'
      if metadata_present
      else 'checkpoint_metadata_missing'
  )
  pytree_dir = (
      'pytree_checkpointable_has_metadata'
      if is_pytree
      else 'pytree_checkpointable_missing_metadata'
  )

  return (
      epath.Path(_BASE_DIR)
      / version_dir
      / type_dir
      / metadata_dir
      / pytree_dir
  )


def create_value_metadata(value: Any) -> Any:
  """Creates Metadata for the given value matching Orbax's return type."""
  if isinstance(value, jax.Array):
    sharding_metadata_obj = sharding_metadata.from_jax_sharding(value.sharding)
    storage_metadata = value_metadata.StorageMetadata(
        chunk_shape=value.sharding.shard_shape(value.shape),
        write_shape=value.shape,
    )
    return array_leaf_handler.ArrayMetadata(
        shape=value.shape,
        dtype=jnp.dtype(value.dtype),
        sharding_metadata=sharding_metadata_obj,
        storage_metadata=storage_metadata,
    )
  elif isinstance(value, (int, np.integer)):
    return 0
  elif isinstance(value, (float, np.floating)):
    return 0.0
  else:
    raise TypeError(f'Unsupported type: {type(value)}')


