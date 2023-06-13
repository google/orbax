# Copyright 2023 The Orbax Authors.
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

"""Orbax Future class used for duck typing."""

from typing import Any, Optional
from typing_extensions import Protocol


class Future(Protocol):
  """Abstracted Orbax Future class.

  This is used to represent the return value of
  AsyncCheckpointHandler.async_save. This method may return multiple related,
  but potentially distinct, future objects. Common examples may include
  tensorstore.Future or concurrent.futures.Future. Since these types are not
  strictly related to one another, we merely enforce that any returned future
  must have a `result` method which blocks until the future's operation
  completes. Importantly, calling `result` should not *start* execution of the
  future, but merely wait for an ongoing operation to complete.
  """

  def result(self, timeout: Optional[int] = None) -> Any:
    """Waits for the future to complete its operation."""
    ...


class NoopFuture:

  def result(self, timeout: Optional[int] = None) -> Any:
    del timeout
    return None
