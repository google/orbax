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

"""AsyncIoEngine module.

This module encapsulates the concurrency and execution orchestration layers for
Orbax.
Its primary responsibility is managing how work is dispatched to the Python
`asyncio` event loop and thread pools.

Scope:
* `asyncio.gather` and future management.
* Concurrency gating (e.g., `ByteLimiter`, `MemoryRegulator`).
* Top-level I/O telemetry and performance logging (e.g., throughput
calculation).

Anti-Scope (What does NOT belong here):
* Storage Backend Logic: Low-level serialization drivers, TensorStore
bindings.
* PyTree Math: Structural diffing, tree traversal, and `ParamInfo`
generation.
* Metadata Persistence: File-system JSON writes and Descriptor
management.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import sys
import threading
import time
from typing import Any, List, Sequence, Tuple, Union

from absl import logging
import humanize
import jax
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import memory_regulator
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.tree import types as tree_types

PyTree = tree_types.PyTree
TypeHandler = types.TypeHandler
ParamInfo = types.ParamInfo
SaveArgs = type_handlers.SaveArgs
RestoreArgs = type_handlers.RestoreArgs
BatchOfLeaves = Sequence[Any]
BatchOfInts = Sequence[int]
Batches = Sequence[BatchOfLeaves]
CommitFutures = Sequence[future.Future]
MemorySizes = Tuple[int, int]


def _get_memory_size(value: Any) -> int:
  """Gets memory size for a leaf value.

  The value is expected to be symmetric for save and load and represents the
  total memory allocated across all devices.

  Args:
    value: The leaf object to inspect.

  Returns:
    The estimated memory footprint in bytes.
  """
  if hasattr(value, 'nbytes'):
    return int(value.nbytes)
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    itemsize = getattr(value.dtype, 'itemsize', 1)
    return int(math.prod(value.shape) * itemsize)
  if isinstance(value, (int, float, complex)):
    return sys.getsizeof(value)
  if isinstance(value, bytes):
    return len(value)
  if isinstance(value, str):
    return len(value.encode('utf-8'))
  return sys.getsizeof(value)


def compute_memory_size(values: PyTree) -> int:
  """Computes the total memory size for a sequence of batch requests.

  Args:
    values: Pytree of leaves or values to compute size for.

  Returns:
    Total memory size in bytes.
  """
  leaves = jax.tree.leaves(values)
  return sum(_get_memory_size(v) for v in leaves)


def log_io_metrics(
    size: int,
    start_time: float,
    gbytes_per_sec_metric: str,
    gbytes_metric: str | None = None,
    *,
    primary_host: int | None,
):
  """Logs the bytes per second metric."""
  time_elapsed = time.time() - start_time
  bytes_per_sec = (
      float('nan') if time_elapsed == 0 else float(size) / time_elapsed
  )
  logging.info(
      '[process=%d] %s: %s/s (total size: %s) (time elapsed: %s s) (global)',
      multihost.process_index(),
      gbytes_per_sec_metric,
      humanize.naturalsize(bytes_per_sec, binary=True, format='%.3f'),
      humanize.naturalsize(size, binary=True),
      time_elapsed,
  )
  if primary_host is None:
    logging.warning(
        'Global object size logging disabled for `primary_host=None`.'
    )
  elif multihost.is_primary_host(primary_host):
    jax.monitoring.record_scalar(
        gbytes_per_sec_metric, value=bytes_per_sec / (1024**3)
    )
    if gbytes_metric is not None:
      jax.monitoring.record_scalar(gbytes_metric, value=size / (1024**3))


async def logging_serialize(
    handler: TypeHandler,
    serialize: asyncio.Coroutine[Any, Any, CommitFutures],
) -> CommitFutures:
  """Logs the time taken to serialize."""
  start = time.time()
  commit_futures = await serialize
  handler_name = f'{type(handler).__module__}.{type(handler).__qualname__}'
  logging.info(
      '[process=%s][thread=%s] Initiated %s.serialize. Time taken: %fs',
      multihost.process_index(),
      threading.current_thread().name,
      f'"{handler_name}"',
      time.time() - start,
  )
  return commit_futures


@dataclasses.dataclass
class BatchRequest:
  """Represents a a request for batched serialization or deserialization.

  Attributes:
    handler: Used to serialize or deserialize the parameters.
    keys: Used to identify the original tree keys so that the PyTree can be
      reconstructed.
    values: Values to serialize.
    infos: ParamInfos.
    args: List of SaveArgs or RestoreArgs.
  """

  handler: TypeHandler
  keys: List[str]
  values: List[Any]
  infos: List[ParamInfo]
  args: List[Union[SaveArgs, RestoreArgs]]

  def __post_init__(self):
    length = len(self.values)
    if not all((
        length == len(self.infos),
        length == len(self.args),
        length == len(self.keys),
    )):
      raise AssertionError('Found `_BatchRequest` with mismatched parameters.')


BatchRequests = Sequence[BatchRequest]


@contextlib.contextmanager
def memory_profiler_context():
  """Context manager for memory_regulator profiler."""
  memory_regulator.profiler_start()
  try:
    yield
  finally:
    # Explicitly stop the bg thread if an exception occurs
    memory_regulator.profiler_end()


class AsyncIoEngine:
  """Encapsulates concurrency, thread-pooling, and I/O telemetry logic."""

  async def execute_save(self, batch_requests: BatchRequests) -> CommitFutures:
    """Executes save requests asynchronously."""
    serialize_ops = []
    with memory_profiler_context():
      for request in batch_requests:
        serialize_ops.append(
            logging_serialize(
                request.handler,
                request.handler.serialize(
                    request.values, request.infos, request.args  # pyrefly: ignore[bad-argument-type]
                ),
            )
        )
      commit_futures = await asyncio.gather(*serialize_ops)

    logging.info(
        'MemoryRegulated: Peak usage: %f GiB',
        memory_regulator.profiler_peak_usage_gib(),
    )
    return commit_futures

  async def execute_restore(self, batch_requests: BatchRequests) -> Batches:
    """Executes restore requests asynchronously."""
    deserialized_batches_ops = []
    for request in batch_requests:
      deserialized_batches_ops.append(
          request.handler.deserialize(request.infos, request.args)  # pyrefly: ignore[bad-argument-type]
      )
    deserialized_batches = await asyncio.gather(*deserialized_batches_ops)
    return deserialized_batches
