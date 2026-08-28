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

"""Cancellation handling utilities."""

import asyncio
from collections.abc import AsyncIterator
import concurrent.futures
import contextlib
from absl import logging
import jax


@contextlib.asynccontextmanager
async def ignore_cancellation(
    operation_name: str | None = None,
) -> AsyncIterator[None]:
  """Async context manager that catches and ignores cancellation errors."""
  try:
    yield
  except (concurrent.futures.CancelledError, asyncio.CancelledError):
    if operation_name:
      logging.info(
          '[process=%s] Ignoring cancellation in %s.',
          jax.process_index(),
          operation_name,
      )
    else:
      logging.info(
          '[process=%s] Ignoring cancellation.',
          jax.process_index(),
      )
