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

Provides the `AsyncIoEngine` class and supporting helper functions responsible
for managing concurrent I/O execution, thread-pooling, and performance telemetry
collection during PyTree saving and restoration workflows.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sys
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

from absl import logging
import humanize
import jax
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import memory_regulator
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types

TypeHandler = types.TypeHandler
ParamInfo = types.ParamInfo
SaveArgs = type_handlers.SaveArgs
RestoreArgs = type_handlers.RestoreArgs


def _default_sizeof_values(values: Sequence[Any]) -> Sequence[int]:
  return [sys.getsizeof(v) for v in values]


def get_batch_memory_size(
    handler: TypeHandler, values: Sequence[Any]
) -> Tuple[int, int]:
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
    gbytes_metric: Optional[str] = None,
):
  """Logs the bytes per second metric."""
  time_elapsed = time.time() - start_time
  bytes_per_sec = (
      float('nan') if time_elapsed == 0 else float(size) / time_elapsed
  )
  note = 'per-host'
  logging.info(
      '[process=%d] %s: %s/s (total gbytes: %s) (time elapsed: %s s) (%s)',
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


async def logging_serialize(
    handler: TypeHandler,
    serialize: asyncio.Coroutine[Any, Any, Sequence[future.Future]],
) -> Sequence[future.Future]:
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

  async def execute_save(
      self, batch_requests: Sequence[BatchRequest]
  ) -> Tuple[List[Any], int]:
    """Executes save requests asynchronously with I/O telemetry."""
    serialize_ops = []
    tree_memory_size = 0
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
        write_size, _ = get_batch_memory_size(request.handler, request.values)
        tree_memory_size += write_size

      commit_futures = await asyncio.gather(*serialize_ops)

    logging.info(
        'MemoryRegulated: Peak usage: %f GiB',
        memory_regulator.profiler_peak_usage_gib(),
    )
    return commit_futures, tree_memory_size

  async def execute_restore(
      self, batch_requests: Sequence[BatchRequest]
  ) -> Tuple[List[Any], int]:
    """Executes restore requests asynchronously with I/O telemetry."""
    deserialized_batches_ops = []
    for request in batch_requests:
      deserialized_batches_ops.append(
          request.handler.deserialize(request.infos, request.args)
      )
    deserialized_batches = await asyncio.gather(*deserialized_batches_ops)

    tree_memory_size = 0
    for request, deserialized in zip(batch_requests, deserialized_batches):
      _, read_size = get_batch_memory_size(request.handler, deserialized)
      tree_memory_size += read_size

    return deserialized_batches, tree_memory_size
