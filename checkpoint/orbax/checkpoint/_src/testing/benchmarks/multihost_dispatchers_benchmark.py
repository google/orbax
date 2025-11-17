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
from jax.experimental import colocated_python as cp
import jax.numpy as jnp
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
  logging.info(
      'metadata sharding mesh devices in worker: %s',
      metadata['array_sharding'].mesh.devices,
  )
  jax.tree.map(_log_fn, inp)


def build_jax_array(
    array: jax.Array,
    shape: tuple[int, ...],
    result_specs: jax.ShapeDtypeStruct,
) -> jax.Array:
  """Builds a jax.Array."""
  del array
  zeros = jnp.zeros(shape, dtype=jnp.float32)
  return jax.make_array_from_callback(
      shape, result_specs.sharding, lambda idx: zeros[idx], dtype=jnp.float32
  )


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
    array = test_context.pytree['array']
    dispatcher = None
    if options.use_colocated:
      dispatcher = dispatchers.ColocatedPythonDispatcher()
      logging.info(
          'jax devices: %s, colocated_cpu_devices: %s',
          jax.devices(),
          cp.colocated_cpu_devices(jax.devices()),
      )

    if dispatcher is None:
      raise ValueError(f'No dispatcher found for {options}')

    pytree_utils.log_pytree('array before d2h', array)
    mesh_utils.pretty_log_mesh('array mesh before d2h: ', array.sharding.mesh)
    logging.info(
        'array mesh devices before d2h: %s', array.sharding.mesh.devices
    )
    metadata = {
        'string': 'metadata',
        'version': 1,
        'array_shape': array.shape,
        'array_sharding': array.sharding,
    }
    with metrics.measure('dispatch_without_result_specs'):
      dispatch_without_result_specs_result = dispatcher.dispatch(
          log_pytree_fn, input_arrays=array, func_kwargs={'metadata': metadata}
      )
    with metrics.measure('dispatch_without_result_specs_block_until_ready'):
      jax.block_until_ready(dispatch_without_result_specs_result)
      pytree_utils.assert_pytree_equal(
          dispatchers._make_dummy_result_array(array),  # pylint: disable=protected-access
          dispatch_without_result_specs_result,
      )

    if options.device_count is not None:
      devices = jax.devices()[: options.device_count]
    else:
      devices = jax.devices()

    logging.info('Dispatching to devices: %s', devices)
    with metrics.measure('dispatch_with_dummy_result_array'):
      dummy_array = dispatchers.get_dummy_input_array(devices)
      dispatch_with_dummy_result_array_result = dispatcher.dispatch(
          lambda _: logging.info(
              'Remote Worker Dispatching: %s', multihost.process_index()
          ),
          input_arrays=dummy_array,
      )
    with metrics.measure('dispatch_with_dummy_result_array_block_until_ready'):
      jax.block_until_ready(dispatch_with_dummy_result_array_result)
      pytree_utils.assert_pytree_equal(
          dispatchers._make_dummy_result_array(dummy_array),  # pylint: disable=protected-access
          dispatch_with_dummy_result_array_result,
      )

    with metrics.measure('dispatch_with_result_specs'):
      sharding = array.sharding
      result_specs = jax.ShapeDtypeStruct(
          array.shape, dtype=array.dtype, sharding=sharding
      )
      result_array = dispatcher.dispatch(
          build_jax_array,
          input_arrays=array,
          result_specs=result_specs,
          func_args=(array.shape, result_specs),
      )
      pytree_utils.log_pytree('result_array', result_array)
      mesh_utils.pretty_log_mesh(
          'result array mesh: ', result_array.sharding.mesh
      )
      logging.info(
          'result array mesh devices: %s', result_array.sharding.mesh.devices
      )

      expected_result = build_jax_array(array, array.shape, result_specs)
      pytree_utils.assert_pytree_equal(expected_result, result_array)

    return core.TestResult(metrics=metrics)
