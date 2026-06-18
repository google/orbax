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

"""Load-only benchmark for native Orbax checkpoints.

Purpose-built A/B comparison target: run against `main` to capture a
baseline, then run against a branch that changes the load path to see the
per-stage / per-host / throughput diff, and verify the loaded tree is still
bit-exact. Points at a real Orbax checkpoint (e.g. a published GCS model) or
a synthetic checkpoint produced by `run_benchmarks --generate_fixture`.

Correctness is independent of the perf baseline: a load-only benchmark has no
in-memory reference, so it hashes the loaded tree per host.
`capture_digests_path` records the digests; `reference_digests_path` verifies
the loaded tree against a previously-captured set and raises on a mismatch. Both
are driven by `run_benchmarks` flags (the config stays mode-agnostic).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pprint
from typing import Any

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import pathways
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import inventory as inventory_lib
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


@dataclasses.dataclass(frozen=True)
class OrbaxLoadBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration for loading a native Orbax checkpoint.

  Self-contained: builds its own `ocp.Context()` (the default ORBAX layout)
  from the load-side tuning knobs, so a single run can sweep e.g.
  `use_load_and_broadcast` or `restore_concurrent_gb` for A/B comparison. Each
  knob may be a single value or a list to expand into a parameter sweep.

  Attributes:
    checkpoint_path: Path (local or `gs://`) to an Orbax checkpoint directory
      (the `items` dir saved by `ocp.save`).
    sharding_config_path: Optional JSON file describing the target sharding per
      tensor. If absent, every loaded tensor is fully replicated across local
      devices (useful for inner-loop smoke).
    capture_digests_path: Directory to write per-host SHA-256 digests of the
      loaded tree to (one `host_<idx>.json` per process). Set in this
      benchmark's `options`; mutually exclusive with `reference_digests_path`.
    reference_digests_path: Directory of previously-captured per-host digests to
      verify the loaded tree against; each host checks its own `host_<idx>.json`
      and a mismatch raises with per-tensor detail. Set in this benchmark's
      `options`; mutually exclusive with `capture_digests_path`.
    use_load_and_broadcast: Whether to load on one replica and broadcast to the
      others instead of every replica reading from storage.
    use_colocated_python: On the Pathways backend, dispatch the load via the
      colocated-Python handler. No-op on other backends.
    restore_concurrent_gb: Cap on concurrent read bytes (in GiB); `None` leaves
      the Orbax default.
    metric_tracemalloc_enabled: Whether to capture the tracemalloc metric
      (opt-in because its per-allocation snapshots are expensive).
  """

  checkpoint_path: str | None = None
  sharding_config_path: str | None = None
  capture_digests_path: str | None = None
  reference_digests_path: str | None = None
  use_load_and_broadcast: bool | Sequence[bool] = False
  use_colocated_python: bool | Sequence[bool] = False
  restore_concurrent_gb: int | None | Sequence[int | None] = None
  metric_tracemalloc_enabled: bool = False

  def is_valid(self) -> bool:
    if self.checkpoint_path is None:
      return False
    return super().is_valid()

  @property
  def context(self) -> ocp.Context:
    ctx = ocp.Context()
    ctx.array.loading.use_load_and_broadcast = self.use_load_and_broadcast
    ctx.memory.read_concurrent_bytes = (
        self.restore_concurrent_gb * 1024**3
        if self.restore_concurrent_gb is not None
        else None
    )
    if self.use_colocated_python and multihost.is_pathways_backend():
      ctx.pathways.checkpointing_impl = pathways.CheckpointingImpl.from_options(
          use_colocated_python=True
      )
    return ctx


def _replicated_abstract_state(metadata: Any) -> Any:
  """Builds an abstract state with every leaf replicated across local devices.

  Used when no `sharding_config_path` is supplied — convenient for inner-loop
  smoke against tiny fixtures where no real sharding decision needs to be made.
  Leaf shapes and dtypes come from the checkpoint metadata.

  Args:
    metadata: The checkpoint metadata pytree (shapes + dtypes per leaf).

  Returns:
    An abstract state pytree with every leaf replicated across local devices.
  """
  mesh = jax.sharding.Mesh(jax.devices(), ("data",))
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(
          shape=x.shape, dtype=x.dtype, sharding=replicated
      ),
      metadata,
  )


def _load_digests(path: epath.PathLike) -> dict[str, str] | None:
  """Loads a per-host digests JSON file, or None if absent/unreadable."""
  try:
    p = epath.Path(path)
    if not p.exists():
      return None
    return json.loads(p.read_text()) or None
  except (OSError, ValueError) as e:
    logging.warning("Could not load digests from %s: %s", path, e)
    return None


def _write_digests(path: epath.PathLike, digests: dict[str, str]) -> None:
  """Writes per-host digests to a JSON file, creating parent dirs."""
  p = epath.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(digests, indent=2))


def _clear_pytree(pytree: Any) -> None:
  """Frees the device arrays held by a pytree."""
  jax.tree.map(
      lambda x: x.delete() if isinstance(x, jax.Array) else None, pytree
  )


def _metrics_to_measure(options: OrbaxLoadBenchmarkOptions) -> list[str]:
  """Returns the metrics to capture, adding tracemalloc when opted in."""
  metrics = metric_lib.default_metrics()
  if options.metric_tracemalloc_enabled:
    metrics.append("tracemalloc")
  return metrics


@benchmarks_core.benchmark_options(OrbaxLoadBenchmarkOptions)
class OrbaxLoadBenchmark(benchmarks_core.BenchmarksGenerator):
  """Loads a native Orbax checkpoint from disk under the default ORBAX layout.

  Built for A/B comparison: capture a baseline against one build of orbax,
  re-run against another, compare. Optional per-host digests carry correctness
  so verification works without a reference copy in memory.
  """

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    options = context.options
    assert isinstance(options, OrbaxLoadBenchmarkOptions)
    logging.info("Benchmark options: %s", pprint.pformat(options))

    # This benchmark loads from `options.checkpoint_path`. A synthetic config
    # carries a `checkpoint_config` spec only so `--generate_fixture` can
    # materialise the fixture; the framework also generates that tree in memory
    # before `test_fn`, so free it here — it is not what we measure.
    if context.pytree is not None:
      _clear_pytree(context.pytree)

    metrics_to_measure = _metrics_to_measure(options)
    assert options.checkpoint_path is not None
    checkpoint_path = epath.Path(options.checkpoint_path)
    load_trace = context.trace_path("load")

    with ocp.Context(context=options.context):
      metadata = ocp.metadata(checkpoint_path)
      if options.sharding_config_path:
        abstract_pytree = (
            checkpoint_generation.get_abstract_state_from_sharding_config(
                epath.Path(options.sharding_config_path),
                metadata.metadata,
                devices=jax.devices(),
            )
        )
      else:
        abstract_pytree = _replicated_abstract_state(metadata.metadata)

      if load_trace is not None:
        jax.profiler.start_trace(str(load_trace))
      with metrics.measure("load", metrics_to_measure):
        restored_pytree = ocp.load(
            checkpoint_path, abstract_state=abstract_pytree
        )
      if load_trace is not None:
        jax.profiler.stop_trace()

    # Correctness digests are per-host: `digest_pytree` hashes the shards this
    # process owns, so capture and compare are per-host files in a directory.
    # Under a fixed sharding, host i owns the same shards on both runs, so each
    # host verifies its own slice — correct for sharded multi-host loads, not
    # just single-host / replicated ones. A run either captures or compares.
    host_file = f"host_{jax.process_index():05d}.json"
    if options.capture_digests_path:
      _write_digests(
          epath.Path(options.capture_digests_path) / host_file,
          pytree_utils.digest_pytree(restored_pytree),
      )
    elif options.reference_digests_path:
      reference_digests = _load_digests(
          epath.Path(options.reference_digests_path) / host_file
      )
      if reference_digests:
        pytree_utils.assert_digests_match(reference_digests, restored_pytree)
    _clear_pytree(restored_pytree)
    # Inventory the checkpoint we loaded (not the empty per-run dir the
    # framework would otherwise scan) so the card's on-disk size — and the
    # per-host bytes-read sharding check against it — are real. Primary only:
    # the result is suite-level and a parallel walk would race.
    checkpoint_inventory = None
    if jax.process_index() == 0:
      checkpoint_inventory = inventory_lib.scan_checkpoint(checkpoint_path)
    return benchmarks_core.TestResult(
        metrics=metrics, inventory=checkpoint_inventory
    )
