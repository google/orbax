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

"""JaxMonitoringMetric — subscribes to jax.monitoring during a measure() block.

Orbax-checkpoint emits ~30 named events through `jax.monitoring` covering
every save/load stage. Subscribing during a measure() block re-publishes
them as TB scalars under stable, navigable tag groups (`2_save_breakdown/`,
`3_load_breakdown/`, `4_throughput/`, `5_inventory/`) without changing any
production code.
"""

from typing import Any

from absl import logging
import jax
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


_PREFIX_FILTER = (
    "/jax/orbax/",
    "/jax/checkpoint/",
    "/jax/compilation_cache/",
)


_TAG_MAP: dict[str, tuple[str, str]] = {
    # ─── 2_save_breakdown — per-stage save timings ────────────────────────
    "/jax/orbax/write/blocking_duration_secs": (
        "2_save_breakdown/blocking_s",
        "s",
    ),
    "/jax/checkpoint/write/async/blocking_duration_secs": (
        "2_save_breakdown/blocking_async_s",
        "s",
    ),
    "/jax/orbax/write/total_duration_secs": (
        "2_save_breakdown/total_s",
        "s",
    ),
    "/jax/checkpoint/write/async/total_duration_secs": (
        "2_save_breakdown/total_async_s",
        "s",
    ),
    "/jax/orbax/write/directory_creation_secs": (
        "2_save_breakdown/directory_creation_s",
        "s",
    ),
    "/jax/orbax/write/async_directory_creation_secs": (
        "2_save_breakdown/async_directory_creation_s",
        "s",
    ),
    "/jax/checkpoint/write/async/metadata_write_duration_secs": (
        "2_save_breakdown/metadata_write_s",
        "s",
    ),
    "/jax/checkpoint/write/async/commit_duration_sec": (
        "2_save_breakdown/commit_s",
        "s",
    ),
    "/jax/checkpoint/write/async/thread_duration_sec": (
        "2_save_breakdown/thread_duration_s",
        "s",
    ),
    "/jax/checkpoint/write/async/commit_future_count": (
        "2_save_breakdown/commit_future_count",
        "count",
    ),
    "/jax/checkpoint/write/duration_since_last_checkpoint_secs": (
        "2_save_breakdown/duration_since_last_save_s",
        "s",
    ),
    "/jax/checkpoint/write/async/ocdbt_merge_duration_secs": (
        "2_save_breakdown/ocdbt_merge_s",
        "s",
    ),
    # ─── 3_load_breakdown — per-stage load timings ────────────────────────
    "/jax/orbax/read/blocking_duration_secs": (
        "3_load_breakdown/blocking_s",
        "s",
    ),
    "/jax/orbax/read/async/blocking_duration_secs": (
        "3_load_breakdown/blocking_async_s",
        "s",
    ),
    "/jax/orbax/read/total_duration_secs": (
        "3_load_breakdown/total_s",
        "s",
    ),
    "/jax/orbax/read/async/total_duration_secs": (
        "3_load_breakdown/total_async_s",
        "s",
    ),
    "/jax/orbax/read/worker/total_duration_secs": (
        "3_load_breakdown/worker_io_s",
        "s",
    ),
    "/jax/checkpoint/read/primary_replica_deserialization_duration_secs": (
        "3_load_breakdown/primary_deserialization_s",
        "s",
    ),
    "/jax/checkpoint/read/broadcast_duration_secs": (
        "3_load_breakdown/broadcast_s",
        "s",
    ),
    # ─── 4_throughput — effective bandwidth ───────────────────────────────
    "/jax/orbax/write/blocking_gbytes_per_sec": (
        "4_throughput/save_blocking_gbps",
        "GiB/s",
    ),
    "/jax/orbax/write/gbytes_per_sec": (
        "4_throughput/save_total_gbps",
        "GiB/s",
    ),
    "/jax/orbax/read/worker/io/requested/throughput/gbytes_per_sec": (
        "4_throughput/load_per_host_gbps",
        "GiB/s",
    ),
    "/jax/checkpoint/read/gbytes_per_sec": (
        "4_throughput/load_total_gbps",
        "GiB/s",
    ),
    # ─── 5_inventory — bytes / shapes ─────────────────────────────────────
    "/jax/orbax/write/gbytes": (
        "5_inventory/save_total_gb",
        "GiB",
    ),
    "/jax/orbax/write/replicated_array_gb": (
        "5_inventory/replicated_array_gb",
        "GiB",
    ),
    "/jax/orbax/write/sharded_array_gb": (
        "5_inventory/sharded_array_gb",
        "GiB",
    ),
    "/jax/orbax/read/worker/io/requested/gbytes": (
        "5_inventory/load_requested_gb",
        "GiB",
    ),
    "/jax/checkpoint/read/gbytes": (
        "5_inventory/load_total_gb",
        "GiB",
    ),
    # ─── 7_overhead — auxiliary costs (coordination, lifecycle) ───────────
    "/jax/checkpoint/sync_global_devices_duration_sec": (
        "7_overhead/sync_global_devices_s",
        "s",
    ),
    "/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/duration": (
        "7_overhead/delete_threaded_s",
        "s",
    ),
    "/jax/orbax/checkpoint_manager/standard_checkpoint_deleter/duration": (
        "7_overhead/delete_standard_s",
        "s",
    ),
}


def _default_tag(event: str) -> str:
  """Tag for an in-prefix event we don't have an explicit mapping for.

  Routes into 9_other/ so unmapped events stay visible (queryable in the
  dashboard) without polluting the curated 2_/3_/4_/5_ namespaces. The
  leading "/jax/" is redundant (every captured event has it) so it's stripped.
  """
  stripped = event.removeprefix("/jax/").lstrip("/")
  return "9_other/" + stripped.replace("/", "_")


class JaxMonitoringMetric(metric_lib.BaseMetric):
  """Captures jax.monitoring emissions during a measure() block.

  Result keys are the final, curated TB tags (`2_save_breakdown/blocking_s`,
  …). Splicing the registry key in between would just create
  `..._jax_monitoring_2_save_breakdown/...` for no benefit, so this class
  opts out via OMIT_REGISTRY_KEY_PREFIX.
  """

  OMIT_REGISTRY_KEY_PREFIX = True

  def start(self) -> None:
    super().start()
    self._records: list[tuple[str, Any, str]] = []

    def _on_scalar(event: str, value: float, **_kwargs):
      if any(event.startswith(p) for p in _PREFIX_FILTER):
        self._records.append((event, value, "scalar"))

    def _on_duration(event: str, duration: float, **_kwargs):
      if any(event.startswith(p) for p in _PREFIX_FILTER):
        self._records.append((event, duration, "duration"))

    self._scalar_cb = _on_scalar
    self._duration_cb = _on_duration
    jax.monitoring.register_scalar_listener(_on_scalar)
    jax.monitoring.register_event_duration_secs_listener(_on_duration)

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
    try:
      jax.monitoring.unregister_scalar_listener(self._scalar_cb)
      jax.monitoring.unregister_event_duration_listener(self._duration_cb)
    except (ValueError, AttributeError) as e:
      logging.warning("Failed to unregister jax.monitoring listener: %s", e)

    for event, value, kind in self._records:
      tag_unit = _TAG_MAP.get(event)
      if tag_unit is None:
        tag = _default_tag(event)
        unit = "s" if kind == "duration" else ""
      else:
        tag, unit = tag_unit
      results[tag] = (value, unit)
    return results


metric_lib.METRIC_REGISTRY["jax_monitoring"] = JaxMonitoringMetric
