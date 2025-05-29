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

"""Types for path-related constructs."""

from __future__ import annotations

import typing
from typing import Protocol

from etils import epath

Path = epath.Path
PathLike = epath.PathLike


@typing.runtime_checkable
class PathAwaitingCreation(Protocol):
  """A path that may not exist yet, but will exist after `await_creation`.

  This construct is used to represent a path in the process of being created.
  The underlying path can be accessed logically, but the actual location in
  the filesystem should not be accessed until `await_creation` is called.

  Usage::

    path: PathAwaitingCreation = ...
    # Logical accesses are OK.
    print(path.path)
    # Block until the path is known to exist.
    path = await path.await_creation()
    path.exists()  # True.
  """

  def __truediv__(
      self, other: PathAwaitingCreation | PathLike
  ) -> PathAwaitingCreation:
    ...

  @property
  def path(self) -> Path:
    ...

  async def await_creation(self) -> Path:
    ...
