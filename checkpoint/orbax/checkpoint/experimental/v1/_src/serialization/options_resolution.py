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

import copy

from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def resolve_storage_options(
    keypath: tree_types.PyTreeKeyPath,
    value: tree_types.Leaf,
    array_saving_options: options_lib.ArrayOptions.Saving,
) -> options_lib.ArrayOptions.Saving.StorageOptions:
  """Resolves storage options using a global default and a per-leaf creator.

  When dealing with PyTrees, `scoped_storage_options_creator` is applied to
  every leaf to mutate its fields in-place on an isolated copy of the global
  `storage_options`.

  Args:
    keypath: The PyTree keypath of the array being saved.
    value: The PyTree leaf value (array) being saved.
    array_saving_options: The Orbax array saving options to use for resolution.

  Returns:
    The resolved StorageOptions containing storage options.
  """
  global_opts = array_saving_options.storage_options
  resolved = (
      copy.copy(global_opts)
      if global_opts is not None
      else options_lib.ArrayOptions.Saving.StorageOptions()
  )

  if array_saving_options.scoped_storage_options_creator is not None:
    array_saving_options.scoped_storage_options_creator(
        keypath, value, resolved
    )

  return resolved
