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

"""Benchmarks for SafetensorsLayout (V1)."""

import asyncio
import dataclasses
import pprint

from absl import logging
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
import safetensors.numpy


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class SafetensorsBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting SafetensorsLayout.

  Attributes:
    checkpoint_config_path: The path to the checkpoint config file.
  """

  checkpoint_path: str | None = None
  use_gs_path: bool = False


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
    ckpt_config = self._checkpoint_configs[0]
    print("Abhisekar checkpoint_config: ", ckpt_config)
    metrics = metric_lib.Metrics(
        name=self.generate_benchmark_name(
            context.options, context.mesh, ckpt_config
        )
    )
    pytree = context.pytree
    save_path = context.path / "safetensors_ckpt"
    options = context.options
    assert isinstance(options, SafetensorsBenchmarkOptions)

    logging.info("Benchmark options: %s", pprint.pformat(options))

    layout = safetensors_layout.SafetensorsLayout()

    if options.use_gs_path:
      load_path = epath.Path(options.checkpoint_path)
      logging.info("Will load from ckpt_config.path: %s", load_path)
      should_verify = False
    else:
      load_path = save_path
      logging.info("Will save to temp location for restore: %s", save_path)
      should_verify = True
      save_path.mkdir(parents=True, exist_ok=True)
      logging.info("Created save_path: %s", save_path)
      if not isinstance(pytree, dict):
        raise ValueError("Safetensors benchmark requires dict pytree.")

      # Convert JAX arrays to Numpy for safetensors saving
      cpu_pytree = jax.tree.map(np.array, pytree)
      tensor_file_path = save_path / "checkpoint.safetensors"

      with metrics.measure("save"):
        safetensors.numpy.save_file(cpu_pytree, tensor_file_path)

    # 2. Restore (Benchmark Target)
    with metrics.measure("load_safetensor"):
      # We need to run the async load
      async def _load():
        # Layout.load returns an Awaitable that resolves to the pytree
        logging.info("loading checkpoint from path: %s", load_path)
        restore_fn = await layout.load(load_path)
        return await restore_fn

      restored_checkpointables = asyncio.run(_load())

      if "pytree" in restored_checkpointables:
        restored_pytree = restored_checkpointables["pytree"]
      else:
        restored_pytree = restored_checkpointables

    # 3. Verify
    if should_verify:
      logging.info("Verifying restored data...")
      try:
        # Simple shape/dtype check first
        jax.tree_util.tree_map(
            lambda x, y: (x.shape == y.shape) and (x.dtype == y.dtype),
            pytree,
            restored_pytree,
        )
        logging.info("Verification successful (shapes/dtypes match).")
      except Exception as e:
        logging.error("Verification failed: %s", e)
        raise
    else:
      logging.info(
          "Skipping verification as checkpoint_config.path was used for"
          " loading."
      )

    return benchmarks_core.TestResult(metrics=metrics)
