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

"""Defines free-function interface for deletion."""

from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types


def delete(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None = None,
    missing_ok: bool = False,
) -> None:
  """Deletes a checkpoint root or one named checkpointable."""
  raise NotImplementedError('Checkpoint deletion is not yet implemented.')


def delete_async(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None = None,
    missing_ok: bool = False,
) -> async_types.AsyncResponse[None]:
  """Deletes asynchronously with the same scope and arguments as delete."""
  raise NotImplementedError(
      'Asynchronous checkpoint deletion is not yet implemented.'
  )
