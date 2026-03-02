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

"""Benchmarks for SafetensorsLayout (V1)."""

import asyncio
import dataclasses

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.arrays import sharding as sharding_utils
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint.experimental import v1 as ocp_v1


# ==============================================================================
# Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class SafetensorsBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting SafetensorsLayout.

  Attributes:
    checkpoint_config_path: The path to the checkpoint config file.
  """

  checkpoint_path: str | None = None


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(SafetensorsBenchmarkOptions)
class SafetensorsBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking SafetensorsLayout."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle using V1 API."""
    metrics = metric_lib.Metrics()
    options = context.options
    assert isinstance(options, SafetensorsBenchmarkOptions)

    load_path = epath.Path(options.checkpoint_path)
    logging.info('Benchmarking Load from: %s', load_path)
    mesh = context.mesh

    async def _load_gcs():
      octx = ocp_v1.Context(
          checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
      )
      with octx:
        # METRIC 1: Header/Index parsing (Metadata)
        with metrics.measure('metadata_load'):
          logging.info('Step 1: Parsing Safetensors metadata...')
          metadata = ocp_v1.pytree_metadata(load_path)
          abstract_state = metadata.metadata

        # METRIC 2: The actual data transfer (The sharded load)
        with metrics.measure('data_load_sharded'):
          logging.info('Step 2: Starting sharded data transfer...')

          shardings = sharding_utils.construct_maximal_shardings(
              abstract_state, list(mesh.devices.flatten())
          )
          sharded_abstract_state = jax.tree.map(
              lambda sds, sharding: jax.ShapeDtypeStruct(
                  sds.shape, sds.dtype, sharding=sharding
              ),
              abstract_state,
              shardings,
          )

          restored_pytree = ocp_v1.load_pytree(
              load_path, sharded_abstract_state
          )

          # Verify the result landed on TPU
          first_leaf = jax.tree_util.tree_leaves(restored_pytree)[0]
          logging.info(
              'SUCCESS: Load complete. First leaf shape: %s, on devices: %s',
              first_leaf.shape,
              first_leaf.devices(),
          )
          return restored_pytree

    # Safe execution for benchmark environments
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_load_gcs())

    return benchmarks_core.TestResult(metrics=metrics)
