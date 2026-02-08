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

import asyncio
from collections.abc import Sequence
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import type_handlers
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.testing.benchmarks import array_handler_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


ArrayHandlerBenchmarkOptions = (
    array_handler_benchmark.ArrayHandlerBenchmarkOptions
)
ArrayHandlerBenchmark = array_handler_benchmark.ArrayHandlerBenchmark


class ArrayHandlerBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_array_handler = mock.create_autospec(
        type_handlers.ArrayHandler, instance=True
    )
    self.enter_context(
        mock.patch.object(
            type_handlers, 'ArrayHandler', return_value=self.mock_array_handler
        )
    )
    self.mock_merge_ocdbt = self.enter_context(
        mock.patch.object(
            ocdbt_utils, 'merge_ocdbt_per_process_files', autospec=True
        )
    )
    self.mock_assert_pytree_equal = self.enter_context(
        mock.patch(
            'orbax.checkpoint._src.testing.benchmarks.core.pytree_utils.assert_pytree_equal',
            autospec=True,
        )
    )
    self.mock_array_handler.serialize.side_effect = self.mock_serialize
    self.mock_array_handler.deserialize.side_effect = self.mock_deserialize

  async def mock_serialize(
      self, values, infos, args
  ) -> Sequence[asyncio.Future]:
    del infos, args  # Unused.
    futures = []
    for _ in values:
      f = asyncio.Future()
      f.set_result(None)  # Complete the future
      futures.append(f)
    return futures

  async def mock_deserialize(self, infos, args) -> Sequence[mock.MagicMock]:
    del args  # Unused.
    return [mock.MagicMock(spec=jax.Array) for _ in infos]

  def _run_benchmark_workflow_test(self, options: ArrayHandlerBenchmarkOptions):
    benchmark = ArrayHandlerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=options,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    sharded_array = jax.device_put(np.arange(16).reshape((4, 4)), sharding)
    pytree = {'array': sharded_array}
    mock_sharding_meta = mock.create_autospec(
        sharding_metadata.ShardingMetadata, instance=True
    )
    mock_sharding_meta.to_jax_sharding.return_value = sharding

    async def metadata_side_effect(
        *args, **kwargs
    ) -> Sequence[value.ArrayMetadata]:
      del kwargs  # Unused.
      info = args[0][0]  # Get the first ParamInfo
      return [
          value.ArrayMetadata(
              name=info.name,
              directory=info.parent_dir,
              shape=(4, 4),
              dtype=sharded_array.dtype,
              sharding=mock_sharding_meta,
          )
      ]

    self.mock_array_handler.metadata.side_effect = metadata_side_effect
    test_dir = epath.Path(self.create_tempdir().full_path)
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_dir, options=options, mesh=mesh
    )

    result = benchmark.test_fn(context)

    return result

  @parameterized.named_parameters(
      dict(
          testcase_name='ocdbt_only',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=False),
      ),
      dict(
          testcase_name='zarr3_only',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=False, use_zarr3=True),
      ),
      dict(
          testcase_name='ocdbt_and_zarr3',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=True),
      ),
  )
  def test_benchmark_returns_test_result_with_basic_metrics(self, options):
    result = self._run_benchmark_workflow_test(options)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertIn('serialize_time_duration', result.metrics.results)
    self.assertIn('metadata_validation_time_duration', result.metrics.results)
    self.assertIn('deserialize_time_duration', result.metrics.results)
    self.assertIn('correctness_check_time_duration', result.metrics.results)

  @parameterized.named_parameters(
      dict(
          testcase_name='ocdbt_only',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=False),
      ),
      dict(
          testcase_name='ocdbt_and_zarr3',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=True),
      ),
  )
  def test_benchmark_ocdbt_enabled_calls_merge(self, options):
    result = self._run_benchmark_workflow_test(options)

    self.assertIn('merge_ocdbt_time_duration', result.metrics.results)
    self.mock_merge_ocdbt.assert_called_once()

  def test_benchmark_ocdbt_disabled_does_not_merge(self):
    options = ArrayHandlerBenchmarkOptions(use_ocdbt=False, use_zarr3=True)
    result = self._run_benchmark_workflow_test(options)

    self.assertNotIn('merge_ocdbt_time_duration', result.metrics.results)
    self.mock_merge_ocdbt.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='ocdbt_only',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=False),
      ),
      dict(
          testcase_name='zarr3_only',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=False, use_zarr3=True),
      ),
      dict(
          testcase_name='ocdbt_and_zarr3',
          options=ArrayHandlerBenchmarkOptions(use_ocdbt=True, use_zarr3=True),
      ),
  )
  def test_benchmark_calls_handler_methods_and_asserts(self, options):
    self._run_benchmark_workflow_test(options)

    self.mock_array_handler.serialize.assert_called_once()
    self.mock_array_handler.metadata.assert_called_once()
    self.mock_array_handler.deserialize.assert_called_once()
    self.mock_assert_pytree_equal.assert_called_once()

  def test_generate_benchmarks_creates_multiple_benchmark_configs(self):
    options = ArrayHandlerBenchmarkOptions(
        use_ocdbt=[True, False],
        use_zarr3=[True, False],
    )
    benchmark = ArrayHandlerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=options,
    )

    benchmarks = benchmark.generate()

    self.assertLen(benchmarks, 4)
    for b in benchmarks:
      self.assertIsInstance(b.options, ArrayHandlerBenchmarkOptions)


class ValidateMetadataTest(parameterized.TestCase):
  """Tests the _validate_metadata helper function."""

  def setUp(self):
    super().setUp()
    self.benchmark = ArrayHandlerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=ArrayHandlerBenchmarkOptions(),
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    self.sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    self.expected_array = jax.device_put(
        np.arange(16).reshape((4, 4)), self.sharding
    )
    self.mock_sharding_meta = mock.create_autospec(
        sharding_metadata.ShardingMetadata, instance=True
    )
    self.mock_sharding_meta.to_jax_sharding.return_value = self.sharding
    self.valid_metadata = value.ArrayMetadata(
        name='array',
        directory=epath.Path('/tmp'),
        shape=(4, 4),
        dtype=self.expected_array.dtype,
        sharding=self.mock_sharding_meta,
    )

  def test_valid_metadata_does_not_raise(self):
    self.benchmark._validate_metadata(
        self.valid_metadata, self.expected_array, 'array'
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='shape_mismatch',
          overrides={'shape': (2, 8)},
          expected_exception=ValueError,
          expected_regex='Metadata shape',
      ),
      dict(
          testcase_name='dtype_mismatch',
          overrides={'dtype': jnp.float32},
          expected_exception=ValueError,
          expected_regex='Metadata dtype',
      ),
      dict(
          testcase_name='sharding_none',
          overrides={'sharding': None},
          expected_exception=ValueError,
          expected_regex='Metadata sharding is None',
      ),
      dict(
          testcase_name='sharding_type_mismatch',
          overrides={'sharding': 'not_sharding_metadata'},
          expected_exception=TypeError,
          expected_regex='Expected metadata.sharding',
      ),
  )
  def test_invalid_metadata_raises_error(
      self, overrides, expected_exception, expected_regex
  ):
    invalid_metadata = dataclasses.replace(self.valid_metadata, **overrides)

    with self.assertRaisesRegex(expected_exception, expected_regex):
      self.benchmark._validate_metadata(
          invalid_metadata, self.expected_array, 'array'
      )

  def test_sharding_value_mismatch(self):
    wrong_sharding_mock = mock.create_autospec(
        sharding_metadata.ShardingMetadata, instance=True
    )
    wrong_sharding_mock.to_jax_sharding.return_value = (
        jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('model',)),
            jax.sharding.PartitionSpec('model'),
        )
    )
    invalid_metadata = dataclasses.replace(
        self.valid_metadata, sharding=wrong_sharding_mock
    )

    with self.assertRaisesRegex(ValueError, 'Metadata sharding'):
      self.benchmark._validate_metadata(
          invalid_metadata, self.expected_array, 'array'
      )


if __name__ == '__main__':
  absltest.main()
