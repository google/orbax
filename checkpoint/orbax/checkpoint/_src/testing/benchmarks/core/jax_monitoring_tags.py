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

"""jax.monitoring event-name → dashboard-tag mappings.

Kept separate from `metric.py` so the ~30-entry mapping table doesn't bloat the
metric module; `JaxMonitoringMetric` (in metric.py) consumes these.
"""

PREFIX_FILTER = (
    "/jax/orbax/",
    "/jax/checkpoint/",
    "/jax/compilation_cache/",
)


TAG_MAP: dict[str, tuple[str, str]] = {
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
    "/jax/orbax/write/compressed_gbytes": (
        "5_inventory/save_compressed_gb",
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
    # ─── 6_io — safetensors per-host storage reads ────────────────────────
    # The safetensors loader has no TensorStore kvstore; it self-reports its
    # per-host read accounting here (raw bytes; the card converts to GiB).
    # Two read counts: `file_reads` is coalesced ranged-read requests (~1 per
    # file when fully coalesced), `storage_reads` is the actual GETs issued,
    # which is >= file_reads because a block larger than the in-flight budget
    # is fetched in several chunks.
    "/jax/orbax/read/safetensors/bytes_read": (
        "6_io/file_bytes_read_per_host",
        "bytes",
    ),
    "/jax/orbax/read/safetensors/num_reads": (
        "6_io/file_reads_per_host_count",
        "count",
    ),
    "/jax/orbax/read/safetensors/storage_reads": (
        "6_io/storage_reads_per_host_count",
        "count",
    ),
    # ─── 8_jax — compile-cache durations ──────────────────────────────────
    "/jax/compilation_cache/cache_retrieval_time_sec": (
        "8_jax/cache_retrieval_s",
        "s",
    ),
    "/jax/compilation_cache/compile_time_saved_sec": (
        "8_jax/compile_time_saved_s",
        "s",
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


# /jax/compilation_cache/* events have no value — we count occurrences per
# event name during the measure block and emit as a scalar.
EVENT_TAG_MAP: dict[str, str] = {
    "/jax/compilation_cache/cache_hits": "8_jax/cache_hits_count",
    "/jax/compilation_cache/cache_misses": "8_jax/cache_misses_count",
    "/jax/compilation_cache/compile_requests_use_cache": (
        "8_jax/compile_requests_use_cache_count"
    ),
    "/jax/compilation_cache/tasks_using_cache": "8_jax/tasks_using_cache_count",
    "/jax/compilation_cache/task_disabled_cache": (
        "8_jax/task_disabled_cache_count"
    ),
}


def default_tag(event: str) -> str:
  """Tag for an in-prefix event we don't have an explicit mapping for.

  Routes into 9_other/ so unmapped events stay visible (queryable in the
  dashboard) without polluting the curated 2_/3_/4_/5_ namespaces. The
  leading "/jax/" is redundant (every captured event has it) so it's stripped.

  Args:
    event: The raw jax.monitoring event name (e.g. `/jax/.../duration`).

  Returns:
    The `9_other/`-prefixed dashboard tag for the event.
  """
  stripped = event.removeprefix("/jax/").lstrip("/")
  return "9_other/" + stripped.replace("/", "_")
