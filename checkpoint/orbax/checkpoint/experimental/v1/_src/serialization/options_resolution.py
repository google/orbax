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

"""Utility functions for serialization."""

from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def resolve_storage_options(
    keypath: tree_types.PyTreeKeyPath,
    value: tree_types.LeafType,
    array_saving_options: options_lib.ArrayOptions.Saving,
) -> options_lib.ArrayOptions.Saving.StorageOptions:
  """Resolves storage options using a global default and a per-leaf creator.

  When dealing with PyTrees, `scoped_storage_options_creator` is applied to
  every leaf. Its fields take precedence when merging if they are set to
  non-None or non-default values with respect to the global `storage_options`.
  If the creator returns `None`, the global `storage_options` is used for all
  fields.

  Args:
    keypath: The PyTree keypath of the array being saved.
    value: The PyTree leaf value (array) being saved.
    array_saving_options: The Orbax array saving options to use for resolution.

  Returns:
    The resolved StorageOptions containing storage options.
  """
  global_opts = array_saving_options.storage_options
  if global_opts is None:
    global_opts = options_lib.ArrayOptions.Saving.StorageOptions()

  fn = array_saving_options.scoped_storage_options_creator
  individual_opts = None
  if fn is not None:
    individual_opts = fn(keypath, value)

  if individual_opts is not None:
    resolved_dtype = (
        individual_opts.dtype
        if individual_opts.dtype is not None
        else global_opts.dtype
    )
    resolved_chunk_byte_size = (
        individual_opts.chunk_byte_size
        if individual_opts.chunk_byte_size is not None
        else global_opts.chunk_byte_size
    )
    resolved_shard_axes = (
        individual_opts.shard_axes
        if individual_opts.shard_axes is not None
        else global_opts.shard_axes
    )
  else:
    resolved_dtype = global_opts.dtype
    resolved_chunk_byte_size = global_opts.chunk_byte_size
    resolved_shard_axes = global_opts.shard_axes

  return options_lib.ArrayOptions.Saving.StorageOptions(
      dtype=resolved_dtype,
      chunk_byte_size=resolved_chunk_byte_size,
      shard_axes=resolved_shard_axes if resolved_shard_axes is not None else (),
  )

