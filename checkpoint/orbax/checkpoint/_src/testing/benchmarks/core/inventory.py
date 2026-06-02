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

"""Post-save directory inventory: file count, total bytes, format mix.

`small_file_pct` (fraction <1 MiB) is the GCS small-object canary —
small-object dominance kills aggregate throughput on object stores and
typically signals that `chunk_byte_size` is set too low.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses

from absl import logging
from etils import epath

_SMALL_FILE_THRESHOLD_BYTES: int = 1 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class CheckpointInventory:
  total_bytes: int = 0
  file_count: int = 0
  small_file_count: int = 0
  small_file_pct: float = 0.0
  largest_file_bytes: int = 0
  smallest_file_bytes: int = 0
  format: Mapping[str, int] = dataclasses.field(default_factory=dict)


def _classify_file(name: str) -> str:
  """Buckets a leaf filename into a format category.

  Args:
    name: The leaf filename to classify.

  Returns:
    One of "ocdbt", "zarr3", "metadata", or "other".
  """
  if name.startswith("ocdbt.process_") or name == "manifest.ocdbt":
    return "ocdbt"
  if name in ("zarray", "zgroup", "zarr.json") or name.endswith(".zarr"):
    return "zarr3"
  if name in ("_METADATA", "_sharding", "_CHECKPOINT_METADATA"):
    return "metadata"
  return "other"


def _iter_file_sizes(path: epath.Path) -> Iterator[tuple[str, int]]:
  """Yields (leaf_name, size_bytes) for every file under path; logs IO errors."""
  try:
    children = list(path.iterdir())
  except OSError as e:
    logging.warning("inventory: cannot iterdir %s: %s", path, e)
    return
  for child in children:
    if child.is_dir():
      yield from _iter_file_sizes(child)
      continue
    try:
      size = child.stat().length
    except (OSError, AttributeError) as e:
      logging.warning("inventory: cannot stat %s: %s", child, e)
      continue
    yield child.name, size


def scan_checkpoint(path: epath.Path) -> CheckpointInventory:
  """Walks `path` recursively and returns file counts + size totals.

  Scan failures log a warning rather than raise, so a benchmark that completed
  successfully isn't marked failed just because the post-save walk hit an IO
  error.

  Args:
    path: The checkpoint directory to inventory.

  Returns:
    The inventory, or an empty one if the path doesn't exist.
  """
  path = epath.Path(path)
  if not path.exists():
    return CheckpointInventory()

  total_bytes = small_file_count = largest = 0
  smallest = -1
  file_count = 0
  format_counts: dict[str, int] = {}
  for name, size in _iter_file_sizes(path):
    file_count += 1
    total_bytes += size
    if size < _SMALL_FILE_THRESHOLD_BYTES:
      small_file_count += 1
    largest = max(largest, size)
    smallest = size if smallest < 0 else min(smallest, size)
    bucket = _classify_file(name)
    format_counts[bucket] = format_counts.get(bucket, 0) + 1

  return CheckpointInventory(
      total_bytes=total_bytes,
      file_count=file_count,
      small_file_count=small_file_count,
      small_file_pct=(small_file_count / file_count) if file_count else 0.0,
      largest_file_bytes=largest,
      smallest_file_bytes=max(smallest, 0),
      format=dict(format_counts),
  )
