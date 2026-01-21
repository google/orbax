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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.testing.benchmarks import safetensors_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint.experimental import v1 as ocp_v1
import safetensors.numpy as safe_np

SafetensorsBenchmarkOptions = safetensors_benchmark.SafetensorsBenchmarkOptions
SafetensorsBenchmark = safetensors_benchmark.SafetensorsBenchmark


class SafetensorsBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir().full_path)
    self.checkpoint_path = self.test_dir / 'fake_checkpoint.safetensors'

    self.dummy_pytree = {
        'tensor_a': jnp.ones((32, 1024), dtype=jnp.float32),
        'scalar': jnp.ones((), dtype=jnp.float32),
        'vector': jnp.ones((1024,), dtype=jnp.float32),
    }

    save_pytree = jax.tree.map(np.array, self.dummy_pytree)
    safe_np.save_file(save_pytree, str(self.checkpoint_path))

  def test_benchmark_test_fn_sharded_load(self):
    # 1. Setup Benchmark Generator
    generator = SafetensorsBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=SafetensorsBenchmarkOptions(),
    )

    # 2. Create Test Context
    devices = np.array(jax.devices())
    if devices.size == 1:
      devices = devices.reshape(1, 1)
    else:
      devices = devices.reshape(1, devices.size)  # Keep it simple for this test
    mesh = jax.sharding.Mesh(devices, ('data', 'model'))
    options = SafetensorsBenchmarkOptions(
        checkpoint_path=str(self.checkpoint_path)
    )

    context = benchmarks_core.TestContext(
        pytree={},  # Unused in this test_fn implementation
        path=self.checkpoint_path,
        options=options,
        mesh=mesh,
    )

    # 3. Run the Benchmark Test Function
    result = generator.test_fn(context)

    # 4. Verify Benchmark Metrics
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertIn('metadata_load_time_duration', result.metrics.results)
    self.assertIn('data_load_sharded_time_duration', result.metrics.results)

    # 5. Verify Loaded Content by Reloading
    octx = ocp_v1.Context(
        checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
    )
    with octx:
      metadata = ocp_v1.pytree_metadata(self.checkpoint_path)
      abstract_state = metadata.metadata
      restored_pytree = ocp_v1.load_pytree(self.checkpoint_path, abstract_state)

    self.assertEqual(
        jax.tree_util.tree_structure(restored_pytree),
        jax.tree_util.tree_structure(self.dummy_pytree),
    )
    jax.tree.map(
        self.assertTrue,
        jax.tree.map(
            lambda a, b: np.array_equal(np.array(a), np.array(b)),
            restored_pytree,
            self.dummy_pytree,
        ),
    )
    jax.tree.map(
        self.assertEqual,
        jax.tree.map(lambda a: a.shape, restored_pytree),
        jax.tree.map(lambda a: a.shape, self.dummy_pytree),
    )
    jax.tree.map(
        self.assertEqual,
        jax.tree.map(lambda a: a.dtype, restored_pytree),
        jax.tree.map(lambda a: a.dtype, self.dummy_pytree),
    )

  def test_benchmark_test_fn_rank_aware_sharding(self):
    # 1. Setup Benchmark Generator
    generator = SafetensorsBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=SafetensorsBenchmarkOptions(),
    )

    # 2. Create Test Context
    devices = np.array(jax.devices())
    # Reshape devices to be 2D for the mesh axis names ('data', 'model')
    num_devices = devices.size
    if num_devices == 1:
      devices = devices.reshape(1, 1)
    elif num_devices == 2:
      devices = devices.reshape(1, 2)
    elif num_devices % 2 == 0:
      devices = devices.reshape(2, num_devices // 2)
    else:  # Fallback for odd numbers, should not happen in typical test envs
      devices = devices.reshape(1, num_devices)
    mesh = jax.sharding.Mesh(devices, ('data', 'model'))
    options = SafetensorsBenchmarkOptions(
        checkpoint_path=str(self.checkpoint_path)
    )

    context = benchmarks_core.TestContext(
        pytree={},  # Unused
        path=self.checkpoint_path,
        options=options,
        mesh=mesh,
    )

    # 3. Run the Benchmark Test Function
    result = generator.test_fn(context)

    # 4. Verify Benchmark Metrics
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertIn('metadata_load_time_duration', result.metrics.results)
    self.assertIn('data_load_sharded_time_duration', result.metrics.results)

    # 5. Verify Loaded Content by Reloading
    octx = ocp_v1.Context(
        checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
    )
    with octx:
      metadata = ocp_v1.pytree_metadata(self.checkpoint_path)
      abstract_state = metadata.metadata
      # Note: Sharding is not applied here, loading as is from the file.
      restored_pytree = ocp_v1.load_pytree(self.checkpoint_path, abstract_state)

    self.assertEqual(
        jax.tree_util.tree_structure(restored_pytree),
        jax.tree_util.tree_structure(self.dummy_pytree),
    )
    jax.tree.map(
        self.assertTrue,
        jax.tree.map(
            lambda a, b: np.array_equal(np.array(a), np.array(b)),
            restored_pytree,
            self.dummy_pytree,
        ),
    )
    jax.tree.map(
        self.assertEqual,
        jax.tree.map(lambda a: a.shape, restored_pytree),
        jax.tree.map(lambda a: a.shape, self.dummy_pytree),
    )
    jax.tree.map(
        self.assertEqual,
        jax.tree.map(lambda a: a.dtype, restored_pytree),
        jax.tree.map(lambda a: a.dtype, self.dummy_pytree),
    )


if __name__ == '__main__':
  absltest.main()
