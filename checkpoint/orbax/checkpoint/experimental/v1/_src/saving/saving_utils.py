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

"""Internal utilities for saving whole and partial checkpoints."""

from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types



def get_temporary_path(
    path: path_types.Path, *, context: context_lib.Context
) -> atomicity_types.TemporaryPath:
  """Gets a TemporaryPath for the given path."""
  temporary_path_class = (
      context.file_options.temporary_path_class
      or atomicity_defaults.get_default_temporary_path_class(path)
  )
  tmpdir = temporary_path_class.from_final(
      path,
      # Ensure metadata store is NOT passed, to prevent separate metadata
      # writing.
      checkpoint_metadata_store=None,
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
  )
  return tmpdir
