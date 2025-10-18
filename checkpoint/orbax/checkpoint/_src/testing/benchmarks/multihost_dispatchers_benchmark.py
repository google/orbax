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

"""Benchmark for Orbax ArrayHandler."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import logging
import jax
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils



def log_pytree_fn(inp: Any, metadata: dict[str, Any]):
  """Logs information about the input pytree and metadata.

  Args:
    inp: The input pytree of jax.Arrays.
    metadata: Additional metadata to log.
  """

  def _log_fn(arr: jax.Array):
    sharding = arr.sharding
    assert isinstance(sharding, jax.sharding.NamedSharding)

    pytree_utils.log_pytree('array_in_worker', arr)
    mesh_utils.pretty_log_mesh('array mesh in worker: ', sharding.mesh)
    logging.info(
        'process=%s/%s, addressable_shards=%s, mesh_devices=%s',
        multihost.process_index(),
        multihost.process_count(),
        arr.addressable_shards,
        sharding.mesh.devices,
    )

  logging.info('metadata: %s', metadata)
  jax.tree.map(_log_fn, inp)


@dataclasses.dataclass(frozen=True)
class MultihostDispatchersBenchmarkOptions(core.BenchmarkOptions):
  use_colocated: bool | Sequence[bool] = False
  device_count: int | None | Sequence[int | None] = None


@core.benchmark_options(MultihostDispatchersBenchmarkOptions)
class MultihostDispatchersBenchmark(core.BenchmarksGenerator):
  """Benchmarks Multihost Dispatchers."""

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    metrics = metric_lib.Metrics()
    options = test_context.options
    assert isinstance(options, MultihostDispatchersBenchmarkOptions)
    if 'array' not in test_context.pytree:
      raise ValueError("Expected 'array' key in test_context.pytree")

    dispatcher = None
    if options.use_colocated:
      dispatcher = dispatchers.ColocatedPythonDispatcher()

    if dispatcher is None:
      raise ValueError(f'No dispatcher found for {options}')

    array = test_context.pytree['array']
    pytree_utils.log_pytree('array before d2h', array)
    mesh_utils.pretty_log_mesh('array mesh before d2h: ', array.sharding.mesh)
    logging.info(
        'array mesh devices before d2h: %s', array.sharding.mesh.devices
    )
    metadata = {'string': 'metadata', 'version': 1, 'array_shape': array.shape}
    with metrics.time('dispatch_arrays'):
      dispatch_arrays_future = dispatcher.dispatch_arrays(
          log_pytree_fn, [array], metadata=metadata
      )

    devices = None
    if options.device_count is not None:
      devices = jax.devices()[: options.device_count]

    logging.info('Dispatching to devices: %s', devices)
    with metrics.time('dispatch_devices'):
      dispatch_devices_future = dispatcher.dispatch_devices(
          lambda: logging.info(
              'Remote Worker Dispatching: %s', multihost.process_index()
          ),
          devices=devices,
      )
    with metrics.time('dispatch_arrays_future_result'):
      dispatch_arrays_future.result()
    with metrics.time('dispatch_devices_future_result'):
      dispatch_devices_future.result()

    return core.TestResult(metrics=metrics)
