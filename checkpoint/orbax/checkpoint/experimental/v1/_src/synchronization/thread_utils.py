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

"""Utilities for working with threads and asyncio event loops."""

from typing import Any, Callable, Coroutine, Generic, TypeVar

from orbax.checkpoint._src import asyncio_utils

T = TypeVar('T')


class BackgroundThreadRunner(Generic[T]):
  """A runner for an asyncio event loop in a background thread.

  This class expects an awaitable function that will be run in a background
  thread and in a dedicated event loop, which are managed by an AsyncRunner
  instance.
  """

  def __init__(
      self,
      target: Coroutine[Any, Any, T],
  ):
    self._runner = asyncio_utils.AsyncRunner()
    self._future = self._runner.run_coroutine(target)

  def result(self, timeout: float | None = None) -> T:
    r = self._future.result(timeout=timeout)
    if self._runner:
      try:
        self._runner.shutdown()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._runner = None
    return r

  def on_complete(self, callback: Callable[[T], None]) -> None:
    """Registers a callback to be called when the task is complete."""

    def _callback(fut):
      callback(fut.result())

    self._future.add_done_callback(_callback)

  def __del__(self):
    if self._runner:
      try:
        self._runner.shutdown()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._runner = None
