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

# Copyright 2024 The Orbax Authors.
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

"""Microbenchmarks for Host Pinned Transfer."""

from collections.abc import Sequence
import dataclasses
import time
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


@dataclasses.dataclass(frozen=True)
class HostPinnedTransferOptions(benchmarks_core.BenchmarkOptions):
  """Options for HostPinnedTransferBenchmark.

  Each attribute can be a single value or a list of values to create
  a parameter sweep.
  """

  # Shape of the array to transfer.
  shape: tuple[int, ...] | Sequence[tuple[int, ...]] = (8192, 128, 128)
  # Data type.
  dtype: str | Sequence[str] = "float32"
  # Target transfer mode: eager (host-side Python loop + block_until_ready) or
  # jit (jax.jit + jax.lax.scan).
  mode: str | Sequence[str] = "eager"
  # Number of transfer iterations to time.
  iterations: int | Sequence[int] = 20
  # Number of warmup iterations.
  warmup: int | Sequence[int] = 5

  def is_valid(self) -> bool:
    assert isinstance(self.shape, tuple)
    assert isinstance(self.dtype, str)
    assert isinstance(self.mode, str)
    assert isinstance(self.iterations, int)
    assert isinstance(self.warmup, int)
    if self.mode not in ("eager", "jit"):
      return False
    return True


@benchmarks_core.benchmark_options(HostPinnedTransferOptions)
class HostPinnedTransferBenchmark(benchmarks_core.BenchmarksGenerator):
  """Microbenchmark for raw TPU H2D and D2H transfer speeds."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    options = context.options
    assert isinstance(options, HostPinnedTransferOptions)
    mesh = context.mesh
    if mesh is None:
      devices = jax.devices()
      mesh = jax.sharding.Mesh(np.array(devices), ("y",))

    # Resolve shape and dtype
    shape = options.shape
    dtype = jnp.dtype(options.dtype)
    size_bytes = jnp.zeros(shape, dtype=dtype).nbytes
    size_gb = size_bytes / 1e9

    # Shardings
    device_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("y"), memory_kind="device"
    )
    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("y"), memory_kind="pinned_host"
    )
    unpinned_cpu_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("y"), memory_kind="unpinned_host"
    )

    # Generate source data on CPU/Host first, then put/run
    np_array = jax.random.normal(jax.random.PRNGKey(0), shape, dtype=dtype)

    if options.mode == "eager":
      # ----------------------------------------------------
      # 1. H2D (Host -> Device)
      # ----------------------------------------------------
      # Setup Host arrays (Pinned and Unpinned)
      host_pinned_array = jax.device_put(np_array, pinned_host_sharding)
      host_unpinned_array = jax.device_put(np_array, unpinned_cpu_sharding)

      # Benchmark: H2D Pinned Host -> Device
      for _ in range(options.warmup):
        jax.device_put(host_pinned_array, device_sharding).block_until_ready()

      t0 = time.perf_counter()
      for _ in range(options.iterations):
        jax.device_put(host_pinned_array, device_sharding).block_until_ready()
      h2d_pinned_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["h2d_pinned_latency_s"] = (h2d_pinned_s, "s")
      metrics.results["h2d_pinned_throughput_gbps"] = (
          size_gb / h2d_pinned_s,
          "Gbps",
      )

      # Benchmark: H2D Unpinned -> Device
      for _ in range(options.warmup):
        jax.device_put(host_unpinned_array, device_sharding).block_until_ready()

      t0 = time.perf_counter()
      for _ in range(options.iterations):
        jax.device_put(host_unpinned_array, device_sharding).block_until_ready()
      h2d_unpinned_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["h2d_unpinned_latency_s"] = (h2d_unpinned_s, "s")
      metrics.results["h2d_unpinned_throughput_gbps"] = (
          size_gb / h2d_unpinned_s,
          "Gbps",
      )

      # ----------------------------------------------------
      # 2. D2H (Device -> Host)
      # ----------------------------------------------------
      device_array = jax.device_put(np_array, device_sharding)

      # Benchmark: D2H Device -> Pinned Host
      for _ in range(options.warmup):
        jax.device_put(device_array, pinned_host_sharding).block_until_ready()

      t0 = time.perf_counter()
      for _ in range(options.iterations):
        jax.device_put(device_array, pinned_host_sharding).block_until_ready()
      d2h_pinned_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["d2h_pinned_latency_s"] = (d2h_pinned_s, "s")
      metrics.results["d2h_pinned_throughput_gbps"] = (
          size_gb / d2h_pinned_s,
          "Gbps",
      )

      # Benchmark: D2H Device -> Unpinned CPU
      for _ in range(options.warmup):
        jax.device_put(device_array, unpinned_cpu_sharding).block_until_ready()

      t0 = time.perf_counter()
      for _ in range(options.iterations):
        jax.device_put(device_array, unpinned_cpu_sharding).block_until_ready()
      d2h_unpinned_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["d2h_unpinned_latency_s"] = (d2h_unpinned_s, "s")
      metrics.results["d2h_unpinned_throughput_gbps"] = (
          size_gb / d2h_unpinned_s,
          "Gbps",
      )

    elif options.mode == "jit":
      # JIT Mode: We run compute + transfer in a JIT compiled scan loop to
      # measure compiler pipeline speed.
      device_array_init = jax.device_put(np_array, device_sharding)

      def offload_pinned_step(carry, _):
        # Dummy compute + HBM-to-Pinned-Host transfer
        res = carry * -1
        res_host = jax.device_put(res, pinned_host_sharding)
        return res, res_host

      def offload_unpinned_step(carry, _):
        # Dummy compute + HBM-to-Unpinned-CPU transfer
        res = carry * -1
        res_host = jax.device_put(res, unpinned_cpu_sharding)
        return res, res_host

      # Compile loops
      jit_pinned = jax.jit(
          lambda x: jax.lax.scan(
              offload_pinned_step, x, None, length=options.iterations
          )
      )
      jit_unpinned = jax.jit(
          lambda x: jax.lax.scan(
              offload_unpinned_step, x, None, length=options.iterations
          )
      )

      # Warmup
      _, res_hosts = jit_pinned(device_array_init)
      res_hosts.block_until_ready()
      _, res_hosts = jit_unpinned(device_array_init)
      res_hosts.block_until_ready()

      # Time Pinned JIT offload
      t0 = time.perf_counter()
      _, res_hosts = jit_pinned(device_array_init)
      res_hosts.block_until_ready()
      d2h_pinned_jit_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["d2h_pinned_jit_latency_s"] = (d2h_pinned_jit_s, "s")
      metrics.results["d2h_pinned_jit_throughput_gbps"] = (
          size_gb / d2h_pinned_jit_s,
          "Gbps",
      )

      # Time Unpinned JIT offload
      t0 = time.perf_counter()
      _, res_hosts = jit_unpinned(device_array_init)
      res_hosts.block_until_ready()
      d2h_unpinned_jit_s = (time.perf_counter() - t0) / options.iterations
      metrics.results["d2h_unpinned_jit_latency_s"] = (d2h_unpinned_jit_s, "s")
      metrics.results["d2h_unpinned_jit_throughput_gbps"] = (
          size_gb / d2h_unpinned_jit_s,
          "Gbps",
      )

    return benchmarks_core.TestResult(metrics=metrics)
