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

"""Test utils for work with PyTrees."""

from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost


def is_pytree_checkpoint_complete(directory):
  return (
      directory / format_utils.PYTREE_CHECKPOINTABLE_KEY / 'manifest.ocdbt'
  ).exists()


def get_d_files_mtimes(path: path_types.Path) -> list[int]:
  """Gets a list of last modified times for d files in a PyTree checkpoint.

  Assumes a structure like::

    path/
      <PYTREE_CHECKPOINTABLE_KEY>/
        ocdbt.process_0/
          d/
            <d_file_1>
            ...
            <d_file_n>
        ocdbt.process_1/
          ...

  The `path` and immediate subdirectories may optionally contain tmp suffixes.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    A list of last modified times for d files in the checkpoint.
  """
  mtimes = []
  matching_dirs = list(path.parent.glob(f'{path.name}*'))
  if not matching_dirs:
    # Temp path not created yet.
    return []
  assert (
      len(matching_dirs) == 1
  ), f'Expected exactly one matching directory, got {matching_dirs}.'
  tmpdir = matching_dirs[0]
  matching_pytree_dirs = list(
      tmpdir.glob(f'{format_utils.PYTREE_CHECKPOINTABLE_KEY}*')
  )
  if not matching_pytree_dirs:
    # Temp path not created yet.
    return []
  assert len(matching_pytree_dirs) == 1, (
      'Expected exactly one matching pytree directory, got'
      f' {matching_pytree_dirs}.'
  )
  pytree_dir = matching_pytree_dirs[0]
  for idx in range(multihost.process_count()):
    d_path = pytree_dir / f'ocdbt.process_{idx}' / 'd'
    if not d_path.exists():
      continue
    for f in d_path.iterdir():
      # TensorStore tmp files can get deleted while iterating. We try to be
      # resilient to this.
      try:
        mtimes.append(f.stat().mtime)
      except OSError:
        pass
  return mtimes
