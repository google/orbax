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

import asyncio
from collections.abc import Sequence
import dataclasses

from absl import logging
import jax
from orbax.checkpoint import type_handlers
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers as serialization_type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.testing.benchmarks.core import core
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


@dataclasses.dataclass(frozen=True)
class ArrayHandlerBenchmarkOptions(core.BenchmarkOptions):
  """Options for ArrayHandlerBenchmark.

  Attributes:
    use_ocdbt: Whether to use OCDBT.
    use_zarr3: Whether to use Zarr3 format.
  """

  use_ocdbt: bool | Sequence[bool] = True
  use_zarr3: bool | Sequence[bool] = False


@core.benchmark_options(ArrayHandlerBenchmarkOptions)
class ArrayHandlerBenchmark(core.BenchmarksGenerator):
  """Benchmarks ArrayHandler serialize and deserialize."""

  def _validate_metadata(
      self,
      metadata: value.ArrayMetadata,
      expected_array: jax.Array,
      param_name: str,
  ):
    """Validates the ArrayMetadata against the expected jax.Array.

    Args:
      metadata: The ArrayMetadata to validate.
      expected_array: The jax.Array with expected properties.
      param_name: The name of the parameter for error messages.

    Raises:
      ValueError: If shape, dtype, or sharding values do not match.
      TypeError: If metadata.sharding is not of the expected type.
    """
    if metadata.shape != expected_array.shape:
      raise ValueError(
          f'Metadata shape {metadata.shape} does not match array shape'
          f' {expected_array.shape} for {param_name}'
      )
    if metadata.dtype != expected_array.dtype:
      raise ValueError(
          f'Metadata dtype {metadata.dtype} does not match array dtype'
          f' {expected_array.dtype} for {param_name}'
      )
    if metadata.sharding is None:
      raise ValueError(f'Metadata sharding is None for {param_name}')
    if not isinstance(metadata.sharding, sharding_metadata.ShardingMetadata):
      raise TypeError(
          'Expected metadata.sharding to be ShardingMetadata, got '
          f'{type(metadata.sharding)} for {param_name}'
      )
    restored_sharding = metadata.sharding.to_jax_sharding()
    if restored_sharding != expected_array.sharding:
      raise ValueError(
          f'Metadata sharding {restored_sharding} does not match array'
          f' sharding {expected_array.sharding} for {param_name}'
      )

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    """Runs the ArrayHandler benchmark for a single configuration.

    This function simulates the save, metadata retrieval, and restore process
    using ArrayHandler directly, mimicking how BasePyTreeCheckpointHandler
    would use it.

    Args:
      test_context: Context containing the array to benchmark and test options.

    Returns:
      TestResult object containing the collected metrics.

    Raises:
      ValueError: If the input pytree does not contain the expected 'array' key.
    """
    metrics = core.Metrics()
    options = test_context.options
    assert isinstance(options, ArrayHandlerBenchmarkOptions)
    if 'array' not in test_context.pytree:
      raise ValueError("Expected 'array' key in test_context.pytree")

    handler = type_handlers.ArrayHandler()
    sharded_array = test_context.pytree['array']
    array_name = 'array'
    array_path = test_context.path / array_name

    ts_context = ts_utils.get_ts_context(use_ocdbt=options.use_ocdbt)
    value_typestr = types.get_param_typestr(
        sharded_array,
        serialization_type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
        tree_metadata.PYTREE_METADATA_OPTIONS,
    )

    param_info = type_handlers.ParamInfo(
        name=array_name,
        path=array_path,
        parent_dir=test_context.path,
        use_zarr3=options.use_zarr3,
        is_ocdbt_checkpoint=options.use_ocdbt,
        ts_context=ts_context,
        enable_pinned_host_transfer=False,
        value_typestr=value_typestr,
        raise_array_data_missing_error=True,
    )
    save_args = type_handlers.SaveArgs()

    # --- Serialization ---
    with metrics.time('serialize'):

      async def serialize_and_wait():
        serialize_futures = await handler.serialize(
            [sharded_array],
            [param_info],
            args=[save_args],
        )
        await asyncio.gather(
            *[asyncio.to_thread(f.result) for f in serialize_futures]
        )

      asyncio.run(serialize_and_wait())
      multihost.sync_global_processes('serialization complete')
    logging.info('Serialization complete for %s', param_info.name)

    if options.use_ocdbt:
      with metrics.time('merge_ocdbt'):
        asyncio.run(
            type_handlers.merge_ocdbt_per_process_files(
                test_context.path,
                ts_context=ts_context,
                use_zarr3=options.use_zarr3,
            )
        )
        multihost.sync_global_processes('merge_ocdbt complete')
        logging.info('OCDBT merge complete for %s', test_context.path)

    # --- Metadata Validation ---
    with metrics.time('metadata_validation'):
      metadata = asyncio.run(handler.metadata([param_info]))[0]
      self._validate_metadata(metadata, sharded_array, param_info.name)
      multihost.sync_global_processes('metadata validation complete')
    logging.info('Metadata validation complete for %s', param_info.name)

    # --- Deserialization ---
    restore_args = type_handlers.ArrayRestoreArgs(
        sharding=sharded_array.sharding,
        global_shape=sharded_array.shape,
        dtype=sharded_array.dtype,
    )
    with metrics.time('deserialize'):
      restored_array = asyncio.run(
          handler.deserialize([param_info], args=[restore_args])
      )[0]
      jax.block_until_ready(restored_array)
      multihost.sync_global_processes('deserialization complete')
    logging.info('Deserialization complete for %s', param_info.name)

    # --- Restored Array Validation ---
    with metrics.time('correctness_check'):
      pytree_utils.assert_pytree_equal(sharded_array, restored_array)
    logging.info('Correctness check passed for %s', param_info.name)

    return core.TestResult(metrics=metrics)
