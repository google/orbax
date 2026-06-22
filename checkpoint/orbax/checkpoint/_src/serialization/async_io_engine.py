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
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
import tensorstore as ts

TypeHandler = types.TypeHandler
ParamInfo = types.ParamInfo
SaveArgs = type_handlers.SaveArgs
RestoreArgs = type_handlers.RestoreArgs
BatchOfLeaves = Sequence[Any]
BatchOfInts = Sequence[int]
Batches = Sequence[BatchOfLeaves]
CommitFutures = Sequence[future.Future]
MemorySizes = Tuple[int, int]


def _default_sizeof_values(values: BatchOfInts) -> BatchOfInts:
  return [sys.getsizeof(v) for v in values]


def get_batch_memory_size(
    handler: TypeHandler, values: BatchOfLeaves
) -> MemorySizes:
  """Gets memory size for a batch of leaf values."""
  try:
    write_sizes, read_sizes = zip(*handler.memory_size(values))
  except NotImplementedError:
    logging.warning(
        '`memory_size` is not implemented for `TypeHandler` of type: %s. Using'
        ' the a default implementation to measure value memory consumption that'
        ' may result in inaccurate estimation.',
        type(handler),
    )
    write_sizes = read_sizes = _default_sizeof_values(values)
  assert len(write_sizes) == len(values)
  assert len(read_sizes) == len(values)
  return sum(write_sizes), sum(read_sizes)


def log_io_metrics(
    size: int,
    start_time: float,
    gbytes_per_sec_metric: str,
    gbytes_metric: str | None = None,
    initial_ts_metrics: Sequence[dict[str, Any]] | None = None,
):
  """Logs the bytes per second metric."""
  time_elapsed = time.time() - start_time
  bytes_per_sec = (
      float('nan') if time_elapsed == 0 else float(size) / time_elapsed
  )
  note = 'per-host'
  logging.info(
      '[process=%d] %s: %s/s (total size: %s) (time elapsed: %s s) (%s)',
      multihost.process_index(),
      gbytes_per_sec_metric,
      humanize.naturalsize(bytes_per_sec, binary=True, format='%.3f'),
      humanize.naturalsize(size, binary=True),
      time_elapsed,
      note,
  )
  jax.monitoring.record_scalar(
      gbytes_per_sec_metric, value=bytes_per_sec / (1024**3)
  )
  if gbytes_metric is not None:
    jax.monitoring.record_scalar(gbytes_metric, value=size / (1024**3))
  if initial_ts_metrics is not None:
    final_ts_metrics = ts.experimental_collect_matching_metrics('/tensorstore/')
    initial_bytes = ts_utils.get_total_bytes_from_tensorstore(
        initial_ts_metrics, types.IoDirection.WRITE
    )
    final_bytes = ts_utils.get_total_bytes_from_tensorstore(
        final_ts_metrics, types.IoDirection.WRITE
    )
    compressed_bytes = final_bytes - initial_bytes

    if compressed_bytes > 0 and size > 0:
      ratio = float(compressed_bytes) / size
      logging.info(
          '[process=%d] Compression ratio: %.3f (%s / %s)',
          multihost.process_index(),
          ratio,
          humanize.naturalsize(compressed_bytes, binary=True),
          humanize.naturalsize(size, binary=True),
      )
      jax.monitoring.record_scalar('/jax/orbax/write/compression_ratio', ratio)
      jax.monitoring.record_scalar(
          '/jax/orbax/write/compressed_gbytes', compressed_bytes / (1024**3)
      )


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


def compute_save_memory_size(batch_requests: BatchRequests) -> int:
  """Computes the total write memory size for a sequence of batch requests."""
  tree_memory_size = 0
  for request in batch_requests:
    write_size, _ = get_batch_memory_size(request.handler, request.values)
    tree_memory_size += write_size
  return tree_memory_size


def compute_restore_memory_size(
    batch_requests: BatchRequests,
    deserialized_batches: Batches,
) -> int:
  """Computes the total read memory size for deserialized batches."""
  tree_memory_size = 0
  for request, deserialized in zip(batch_requests, deserialized_batches):
    _, read_size = get_batch_memory_size(request.handler, deserialized)
    tree_memory_size += read_size
  return tree_memory_size


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
                    request.values, request.infos, request.args
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
          request.handler.deserialize(request.infos, request.args)
      )
    deserialized_batches = await asyncio.gather(*deserialized_batches_ops)
    return deserialized_batches
