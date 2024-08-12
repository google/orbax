# Copyright 2024 The Orbax Authors.
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

import threading
import time
from typing import Any, Callable, Optional, Sequence
from absl import logging
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


class ThreadRaisingException(threading.Thread):
  """Thread that raises an exception if it encounters an error."""
  _exception: Optional[Exception] = None

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def join(self, timeout=None):
    super().join(timeout=timeout)
    if self._exception is not None:
      raise self._exception


class ChainedFuture:
  """A future representing a sequence of multiple futures."""

  def __init__(self, futures: Sequence[Future], cb: Callable[[], None]):
    self._futures = futures
    self._cb = cb

  def result(self, timeout: Optional[int] = None) -> Any:
    """Waits for all futures to complete."""
    n = len(self._futures)
    start = time.time()
    time_remaining = timeout
    for k, f in enumerate(self._futures):
      f.result(timeout=time_remaining)
      if time_remaining is not None:
        time_elapsed = time.time() - start
        time_remaining -= time_elapsed
        if time_remaining <= 0:
          raise TimeoutError(
              'ChainedFuture completed {:d}/{:d} futures but timed out after'
              ' {:.2f} seconds.'.format(k, n, time_elapsed)
          )
    time_elapsed = time.time() - start
    logging.info(
        'ChainedFuture completed %d/%d futures in %.2f seconds.',
        n,
        n,
        time_elapsed,
    )
    self._cb()
