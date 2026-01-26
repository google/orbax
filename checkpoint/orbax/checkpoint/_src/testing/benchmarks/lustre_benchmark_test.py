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

"""Tests for the LustreBenchmark class and its integration."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks import lustre_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint.experimental.caching import client

LustreBenchmarkOptions = lustre_benchmark.LustreBenchmarkOptions
LustreBenchmark = lustre_benchmark.LustreBenchmark


class LustreBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = self.enter_context(
        mock.patch.object(client, 'StorageServiceClient', autospec=True)
    )
    self.enter_context(
        mock.patch.object(lustre_benchmark, '_get_xid', return_value=12345)
    )

  def test_xid(self):
    # We patched _get_xid, so this test just verifies the patch works as
    # expected in this context
    xid = lustre_benchmark._get_xid()
    self.assertEqual(xid, 12345)

  def test_benchmark_test_fn(self):
    # Mock the client instance returned by the constructor
    mock_client_instance = self.mock_client.return_value

    # Create temporary directories for the test
    # Ensure all processes use the same directory
    base_dir = self.create_tempdir(name='benchmark')
    # cache_dir should not exist before save_pytree is called
    cache_dir_path = epath.Path(base_dir.full_path) / 'cache'
    work_dir = base_dir.mkdir('work')

    # Mock resolve to return the cache directory
    mock_client_instance.resolve.return_value = str(cache_dir_path)

    self.enter_context(
        mock.patch.object(
            lustre_benchmark, 'LUSTRE_PATH_PREFIX', str(cache_dir_path)
        )
    )

    # Setup the benchmark generator
    generator = LustreBenchmark(
        checkpoint_configs=[],  # Unused in test_fn but required by init
        options=LustreBenchmarkOptions(),
    )

    # Setup test context
    pytree = {'a': jnp.arange(10), 'b': jnp.array(1)}
    test_options = LustreBenchmarkOptions()
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(work_dir.full_path),
        options=test_options,
        repeat_index=0,
    )

    # Run the test function
    result = generator.test_fn(context)

    # Verify interactions
    self.mock_client.assert_called_once()
    mock_client_instance.resolve.assert_called_once_with(mock.ANY, 0)
    mock_client_instance.finalize.assert_called_once_with(mock.ANY, 0)

    # Verify result
    self.assertIsInstance(result, benchmarks_core.TestResult)
    expected_metrics = {
        'resolve_cache_time_duration',
        'save_cache_time_duration',
        'finalize_cache_time_duration',
        'save_time_duration',
    }
    self.assertContainsSubset(expected_metrics, result.metrics.results.keys())

    # Verify files were actually created since we are using real ocp
    # Check cache dir
    self.assertTrue(cache_dir_path.exists())
    # Check work dir (it should have a '0' subdirectory for step 0)
    self.assertTrue((epath.Path(work_dir.full_path) / '0').exists())

  def test_benchmark_test_fn_prefetch(self):
    # Mock the client instance returned by the constructor
    mock_client_instance = self.mock_client.return_value

    # Create temporary directories for the test
    base_dir = self.create_tempdir(name='benchmark_prefetch')
    cache_dir_path = epath.Path(base_dir.full_path) / 'cache'
    work_dir = base_dir.mkdir('work')

    # Mock dynamic path prefixes to match local paths
    self.enter_context(
        mock.patch.object(
            lustre_benchmark, 'LUSTRE_PATH_PREFIX', str(cache_dir_path)
        )
    )
    # We use a separate directory to simulate GCS
    gcs_dir_path = epath.Path(base_dir.full_path) / 'gcs'
    gcs_dir_path.mkdir()
    self.enter_context(
        mock.patch.object(
            lustre_benchmark, 'GCS_PATH_PREFIX', str(gcs_dir_path)
        )
    )

    # Mock resolve to return "GCS" path for step 0 and "Lustre" path for step 1
    def resolve_side_effect(xid, step):
      del xid
      if step == 0:
        return str(gcs_dir_path / '0')
      return str(cache_dir_path / str(step))

    mock_client_instance.resolve.side_effect = resolve_side_effect

    # Setup the benchmark generator
    generator = LustreBenchmark(
        checkpoint_configs=[],
        options=LustreBenchmarkOptions(),
    )

    # Setup test context for step 1
    pytree = {'a': jnp.arange(10), 'b': jnp.array(1)}
    test_options = LustreBenchmarkOptions()
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(work_dir.full_path),
        options=test_options,
        repeat_index=1,
    )

    # Create a checkpoint at the "GCS" location for step 0 so restore works
    ocp.save_pytree(gcs_dir_path / '0', pytree)

    # Run the test function
    result = generator.test_fn(context)

    # Verify interactions
    # resolve called for step 1 (current) and step 0 (prefetch)
    self.assertTrue(mock_client_instance.resolve.called)
    mock_client_instance.prefetch.assert_called_once_with(mock.ANY, 0)
    mock_client_instance.await_transfer.assert_called_once_with(mock.ANY, 0)
    mock_client_instance.finalize.assert_called_once_with(mock.ANY, 1)

    # Verify result
    self.assertIsInstance(result, benchmarks_core.TestResult)
    expected_metrics = {
        'prefetch_cache_time_duration',
        'wait_prefetch_cache_time_duration',
        'restore_cache_time_duration',
        'restore_time_duration',
    }
    self.assertContainsSubset(expected_metrics, result.metrics.results.keys())


if __name__ == '__main__':
  absltest.main()
