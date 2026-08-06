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

"""Defines `SafetensorsLayout`, the handler for Safetensors checkpoints.

Loading is driven by the target sharding: each process resolves which shards of
each tensor its own devices need, maps those shards to byte ranges in the file,
and reads exactly those ranges. There is no cross-process communication and no
XLA compilation -- resharding happens purely through which bytes each process
reads. This mirrors how the v0 array deserializer
(`_src/serialization/serialization.py`) drives reads from
`jax.sharding.Sharding.devices_indices_map`, with hand-computed byte ranges
standing in for TensorStore's chunk index.

Byte runs from every tensor a process needs in one file are coalesced under a
bounded-over-read policy: a `max_over_read_ratio` cap on
`block_size / needed_bytes`. By default the cap is picked per file from the
planned runs themselves -- no over-read when the sharding already maps to few
contiguous reads, whole-span reads when the plan would otherwise fragment
into a very large number of small strided requests (a warning then reports
the over-read cost of that sharding). Each coalesced block is then split at a
fixed chunk size and the chunks are read concurrently, with total in-flight
bytes bounded by the shared restore budget
(`MemoryOptions.read_concurrent_bytes`). Peak host memory beyond the
destination shard buffers is bounded at that budget regardless of block,
shard, or file size, while a block larger than one chunk -- the single-host
whole-file case especially -- is fetched as parallel ranged reads instead of
one long single-stream read. As each file's reads complete, its tensors are
assembled into `jax.Array`s and their host buffers released while other files
are still reading.

`SafetensorsOptions.broadcast_replicated` removes the one case where
sharding-driven reads pull more from storage than the checkpoint's size:
tensors replicated across processes, whose bytes every process would
otherwise re-read. With the flag on, such tensors are read under a
de-duplicated sharding (each byte owned once, spread across the replicas) and
an identity computation with the target `out_shardings` then broadcasts the
shards over the accelerator interconnect.
"""

import asyncio
import collections
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import itertools
import json
import math
import time
from typing import Any, cast, NamedTuple

from absl import logging
import humanize
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types

CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path
Checkpointable = checkpoint_layout.Checkpointable
AbstractCheckpointable = checkpoint_layout.AbstractCheckpointable

HEADER_NUM_BYTES = 8
SAFETENSORS_SUFFIX = ".safetensors"

# Read-planning defaults. `SafetensorsOptions` overrides the over-read ratio
# (`max_over_read_ratio`) and the chunk size (`read_chunk_bytes`); the
# in-flight budget comes from `MemoryOptions.read_concurrent_bytes`.
#
# With `max_over_read_ratio` unset, each file's ratio is picked from its
# planned runs (`_auto_over_read_ratio`): 1.0 when coalescing at the gap floor
# already leaves at most this many reads -- large contiguous shards gain
# nothing from over-reading -- and the file's `span / needed` otherwise, which
# collapses a fragmented (strided inner-dim) plan into whole-span reads
# instead of an unbounded number of small requests.
_AUTO_MAX_READS_PER_FILE = 1024
# Always coalesce runs separated by less than this gap, regardless of the
# over-read ratio. Below roughly a page the per-read overhead -- a syscall
# locally, an RTT on object storage -- dwarfs the cost of reading the gap, so a
# strict ratio would otherwise shatter tiny strided reads into many microscopic
# requests. Larger gaps stay governed by `max_over_read_ratio`, where the
# read-count-vs-egress tradeoff is genuinely backend-dependent.
_DEFAULT_MIN_COALESCE_GAP = 4096  # one page
# `_DEFAULT_MAX_IN_FLIGHT_BYTES` caps total bytes the loader holds in flight
# across all concurrent reads in one load call. Chunks run concurrently up to
# the budget, so peak host memory beyond destination shard buffers is bounded
# at `max_in_flight_bytes` regardless of block, shard, or file size.
_DEFAULT_MAX_IN_FLIGHT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
# Coalesced blocks are split at this size and the pieces are read
# concurrently (subject to the in-flight budget). One ranged read runs at
# single-stream throughput -- on object storage a small fraction of what
# concurrent range reads achieve -- so a large block is fetched as parallel
# chunks rather than one long read. 128 MiB keeps per-request overhead
# negligible while the default 2 GiB budget still admits 16 concurrent
# streams.
_DEFAULT_READ_CHUNK_BYTES = 128 * 1024 * 1024  # 128 MiB

# Heavy over-read under the automatic ratio means the target sharding
# fragments the checkpoint's byte layout and whole spans were read to keep the
# request count bounded. Surface it so the extra bytes are diagnosable: the
# sharding, not the loader, is the thing to change. Over-read from gap-floor
# merges alone stays well below this.
_OVER_READ_WARN_RATIO = 1.5
# Per-device output bytes per compiled broadcast under
# `SafetensorsOptions.broadcast_replicated`. Tensors read once under a
# de-duplicated sharding are resharded to their replicated targets in batches
# no larger than this, bounding the transient device memory of each call.
_BROADCAST_BATCH_BYTES = 1024 * 1024 * 1024  # 1 GiB


def _get_dtypes() -> dict[str, Any]:
  """Returns the mapping from safetensor `dtype` strings to NumPy `dtypes`."""
  return {
      "BOOL": np.bool_,
      "I8": np.int8,
      "U8": np.uint8,
      "I16": np.int16,
      "U16": np.uint16,
      "I32": np.int32,
      "U32": np.uint32,
      "I64": np.int64,
      "U64": np.uint64,
      "F16": np.float16,
      "F32": np.float32,
      "F64": np.float64,
      "BF16": jnp.bfloat16,
      "F8_E5M2": jnp.float8_e5m2,
      "F8_E4M3": jnp.float8_e4m3fn,
  }


def _get_array_properties(info: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
  """Parses shape and dtype from one tensor's safetensors header entry."""
  try:
    dtype = _get_dtypes()[info["dtype"]]
  except KeyError as e:
    raise ValueError(f"Unsupported dtype in SafeTensors header: {e}") from e
  return tuple(info["shape"]), dtype


async def _read_header(path: Path) -> tuple[dict[str, Any], int]:
  """Reads a safetensors header, returning it and the data-section offset."""
  async with async_path.open_file(path, mode="rb") as f:
    if not (size_bytes := cast(bytes, await f.read(HEADER_NUM_BYTES))):
      raise ValueError(f"Could not read header size from {path}.")
    header_size = int.from_bytes(size_bytes, byteorder="little")
    header_bytes = cast(bytes, await f.read(header_size))
    if len(header_bytes) != header_size:
      raise ValueError(f"Could not read header content from {path}.")
    return json.loads(header_bytes), HEADER_NUM_BYTES + header_size


async def _discover_files(path: Path) -> list[Path]:
  """Returns the `.safetensors` file(s) at `path` (a file or a directory)."""
  if await async_path.is_dir(path):
    return sorted(await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}"))
  return [path]


async def _read_all_headers(
    path: Path,
) -> list[tuple[Path, dict[str, Any], int]]:
  """Reads every file's header concurrently.

  A sharded HuggingFace checkpoint can span hundreds of files; reading their
  headers one after another costs one storage round trip each, so they are
  fetched in parallel.

  Args:
    path: A `.safetensors` file, or a directory containing such files.

  Returns:
    One `(file_path, header, data_start)` tuple per file, in sorted file
    order.
  """
  files = await _discover_files(path)
  headers = await asyncio.gather(*[_read_header(f) for f in files])
  return [(f, h, ds) for f, (h, ds) in zip(files, headers)]


async def _check_file_length(
    path: Path, header: dict[str, Any], data_start: int
) -> None:
  """Fails fast if `path` is too short to hold the tensors its header declares.

  The header records each tensor's `data_offsets` relative to the start of the
  data section (`data_start` bytes in). A truncated or partially-copied file
  still parses, but the byte-range reads that follow would run past EOF and
  surface as a confusing mid-load failure on whichever host owns the missing
  bytes. One `stat` here turns that into a clear up-front error.

  Args:
    path: The `.safetensors` file to check.
    header: The file's parsed JSON header, mapping tensor name to its entry.
    data_start: Byte offset where the data section begins (`8 + header_size`).

  Raises:
    ValueError: If the file is shorter than the data its header declares.
  """
  required = data_start + max(
      (
          info["data_offsets"][1]
          for name, info in header.items()
          if name != "__metadata__"
      ),
      default=0,
  )
  actual = (await async_path.async_stat(path)).length
  if actual < required:
    raise ValueError(
        f"SafeTensors file {path} is truncated: its header declares tensor data"
        f" through byte {required}, but the file is only {actual} bytes."
    )


def _normalize_index(
    index: arrays_types.Index, global_shape: arrays_types.Shape
) -> arrays_types.IndexBounds:
  """Converts a sharding index (tuple of slices/ints) into (start, stop) pairs."""
  resolved = numpy_utils.resolve_slice(index, global_shape)
  for s in resolved:  # pyrefly: ignore[not-iterable]
    if s.step not in (None, 1):
      raise ValueError(f"Strided shard index is unsupported: {s}.")
  return tuple(
      (int(s.start), int(s.stop))
      for s in resolved  # pyrefly: ignore[not-iterable]
  )


def _byte_strides(global_shape: arrays_types.Shape, itemsize: int) -> list[int]:
  """Row-major byte stride along each dimension."""
  strides = [itemsize] * len(global_shape)
  acc = itemsize
  for i in range(len(global_shape) - 1, -1, -1):
    strides[i] = acc
    acc *= global_shape[i]
  return strides


def index_domain_to_byte_runs(
    bounds: arrays_types.IndexBounds,
    global_shape: arrays_types.Shape,
    itemsize: int,
    tensor_base: int,
) -> list[tuple[int, int]]:
  """Maps one shard's index domain to `(offset, length)` byte runs in the file.

  A safetensors tensor is a single contiguous row-major blob. The slice of it
  a device owns is contiguous in the file only along trailing dimensions, so a
  shard is a set of equally sized, equally spaced runs: exactly one run when
  the shard splits only the leading dimension (the FSDP / 1-D case), and
  several strided runs when it splits an inner dimension.

  Args:
    bounds: Per-dimension `(start, stop)` of the shard, from `_normalize_index`.
    global_shape: Shape of the whole tensor.
    itemsize: Bytes per element.
    tensor_base: Absolute byte offset of the tensor's first element in the file.

  Returns:
    A list of `(offset, length)` byte runs, sorted by offset.
  """
  if (ndim := len(global_shape)) == 0:  # Scalar.
    return [(tensor_base, itemsize)]
  strides = _byte_strides(global_shape, itemsize)
  full_trailing = 0
  for i in range(ndim - 1, -1, -1):
    if bounds[i] == (0, global_shape[i]):
      full_trailing += 1
    else:
      break
  if (boundary := ndim - full_trailing - 1) < 0:
    return [(tensor_base, strides[0] * global_shape[0])]
  run_length = (bounds[boundary][1] - bounds[boundary][0]) * strides[boundary]
  run_base = tensor_base + bounds[boundary][0] * strides[boundary]
  outer_ranges = [range(bounds[i][0], bounds[i][1]) for i in range(boundary)]
  runs = []
  for outer_index in itertools.product(*outer_ranges):
    offset = run_base
    for dim, coord in enumerate(outer_index):
      offset += coord * strides[dim]
    runs.append((offset, run_length))
  runs.sort()
  return runs


@dataclasses.dataclass(frozen=True)
class _CoalescedBlock:
  """One scheduled read covering several registered runs."""

  start: int  # Absolute byte offset of the read.
  end: int  # Exclusive end byte offset.
  members: tuple[int, ...]  # Indices into the scheduler's request list.


def _partition_runs(
    runs: Sequence[tuple[int, int]],
    max_over_read_ratio: float,
    min_coalesce_gap: int = _DEFAULT_MIN_COALESCE_GAP,
) -> list[_CoalescedBlock]:
  """Greedily partitions `(offset, length)` runs into coalesced read blocks.

  Each emitted block is split at the read chunk size into one or more ranged
  reads and demultiplexed back to its member runs (`_plan_chunk_reads`). The
  partitioner absorbs the next run into the current block when *either* of two
  conditions holds:

  - `block_size / needed_bytes <= max_over_read_ratio` -- bounding per-host
    egress amplification. With `max_over_read_ratio == 2.0`, a host pulls at
    most `2x` the bytes it actually needs from this file.
  - the gap to the next run is `<= min_coalesce_gap` -- an absolute floor. The
    ratio test is purely relative, so it would otherwise reject merging tiny
    runs across tiny gaps, shattering strided reads into many microscopic
    requests; below ~a page the per-read overhead dominates and reading the
    gap is always cheaper than a second request.

  Block size is intentionally unbounded here. Peak in-flight memory is bounded
  separately by the shared byte budget (`max_in_flight_bytes`), which also
  enforces correctness for any block too large to read in one go. This
  separation makes the partition policy purely about over-read and the
  budget purely about memory -- each knob expressing one concern.

  The single-host degenerate case -- this process needs every byte of the
  file -- has ratio `1.0` everywhere and collapses to a single block per file.
  FSDP with one contiguous shard per host per file likewise collapses to one
  block per host.

  Args:
    runs: `(offset, length)` byte runs; need not be sorted.
    max_over_read_ratio: Upper bound on `block_size / needed_bytes` per block.
      Must be `>= 1.0`; `1.0` disables any over-read.
    min_coalesce_gap: Runs separated by at most this many bytes are always
      merged, regardless of the ratio. `0` disables the floor (pure ratio).

  Returns:
    A list of `_CoalescedBlock` covering every input run, with each block's
    `members` tuple recording the indices (into the original `runs`) it
    serves. Members are in the order the partitioner encountered them after
    sorting by offset.
  """
  if not runs:
    return []
  if max_over_read_ratio < 1.0:
    raise ValueError(
        f"max_over_read_ratio must be >= 1.0, got {max_over_read_ratio}."
    )
  indexed = sorted(enumerate(runs), key=lambda x: x[1][0])
  blocks: list[_CoalescedBlock] = []
  cur_start: int | None = None
  cur_end = 0
  cur_needed = 0
  cur_members: list[int] = []
  for orig_idx, (offset, length) in indexed:
    end = offset + length
    if cur_start is None:
      cur_start, cur_end = offset, end
      cur_needed = length
      cur_members = [orig_idx]
      continue
    gap = offset - cur_end
    new_end = max(cur_end, end)
    new_size = new_end - cur_start
    new_needed = cur_needed + length
    if gap <= min_coalesce_gap or new_size <= max_over_read_ratio * new_needed:
      cur_end = new_end
      cur_needed = new_needed
      cur_members.append(orig_idx)
    else:
      blocks.append(_CoalescedBlock(cur_start, cur_end, tuple(cur_members)))
      cur_start, cur_end = offset, end
      cur_needed = length
      cur_members = [orig_idx]
  if cur_start is not None:
    blocks.append(_CoalescedBlock(cur_start, cur_end, tuple(cur_members)))
  return blocks


def _auto_over_read_ratio(runs: Sequence[tuple[int, int]]) -> float:
  """Picks one file's over-read ratio from its planned run geometry.

  No single ratio suits every sharding. A process reading a few large
  contiguous runs gains nothing from merging across gaps -- that only
  wastes bytes. One reading thousands of tiny strided runs is better off
  reading the whole span, but the ratio where the plan collapses is the
  file's `span / needed`, which shifts with sharding and topology. Values
  in between buy more requests *and* more waste, so the choice is binary
  and the runs themselves decide it.

  Args:
    runs: `(offset, length)` byte runs planned for one file; need not be sorted.
      Assumed non-overlapping, as `index_domain_to_byte_runs` produces.

  Returns:
    `1.0` when merging only sub-floor gaps already leaves at most
    `_AUTO_MAX_READS_PER_FILE` reads; otherwise the file's whole-span ratio
    `span / needed`, which bounds every block and collapses uniform strided
    plans into whole-span reads.
  """
  if not runs:
    return 1.0
  ordered = sorted(runs)
  needed = 0
  num_reads = 1
  prev_end = ordered[0][0]
  for offset, length in ordered:
    needed += length
    if offset - prev_end > _DEFAULT_MIN_COALESCE_GAP:
      num_reads += 1
    prev_end = max(prev_end, offset + length)
  if num_reads <= _AUTO_MAX_READS_PER_FILE:
    return 1.0
  span = prev_end - ordered[0][0]
  return max(1.0, span / needed)


def _replicated_sharding() -> jax.sharding.Sharding:
  """A `NamedSharding` that fully replicates an array across all devices."""
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("_safetensors_replica",))
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _spec_axes(entry: Any) -> tuple[str, ...]:
  """Returns one `PartitionSpec` entry's mesh axis names as a tuple."""
  if entry is None:
    return ()
  return entry if isinstance(entry, tuple) else (entry,)


def _dedup_read_sharding(
    sharding: jax.sharding.Sharding, global_shape: tuple[int, ...]
) -> jax.sharding.NamedSharding | None:
  """Returns a sharding under which each tensor byte is owned once, if any.

  In a `NamedSharding`, replication is exactly the mesh axes the spec does not
  use: devices that differ only along those axes own identical index domains,
  so under `broadcast_replicated` every replica would re-read the same
  bytes from storage. Assigning each unused axis to a tensor dimension splits
  those identical domains apart -- the same bytes are then owned (and read)
  exactly once, spread across all the replicas' hosts. The loader reads under
  the returned sharding and afterwards broadcasts to the true target
  (`_broadcast_to_target_shardings`).

  Each unused axis is greedily placed on the first dimension that remains
  evenly divisible (JAX rejects uneven partitions). Axes that fit nowhere
  stay unplaced: the result then still de-duplicates partially. Returns
  `None` when the sharding is not a `NamedSharding`, nothing is replicated,
  or no axis can be placed -- callers then read under the target sharding
  as usual.

  Args:
    sharding: The target sharding for the tensor.
    global_shape: The global shape of the tensor.

  Returns:
    The de-duplicated read sharding, or `None` if none is applicable.
  """
  if not global_shape or not isinstance(sharding, jax.sharding.NamedSharding):
    return None
  mesh = sharding.mesh
  entries = [
      _spec_axes(e)
      for e in tuple(sharding.spec) + (None,) * len(global_shape)
  ][: len(global_shape)]
  used = {axis for entry in entries for axis in entry}
  unused = [
      axis
      for axis in mesh.axis_names
      if axis not in used and mesh.shape[axis] > 1
  ]
  if not unused:
    return None
  partitions = [
      math.prod(mesh.shape[axis] for axis in entry) for entry in entries
  ]
  placed_any = False
  for axis in unused:
    for dim, dim_size in enumerate(global_shape):
      if dim_size % (partitions[dim] * mesh.shape[axis]) == 0:
        entries[dim] += (axis,)
        partitions[dim] *= mesh.shape[axis]
        placed_any = True
        break
  if not placed_any:
    return None
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(*(e or None for e in entries))
  )


def _broadcast_to_target_shardings(
    arrays: dict[str, jax.Array],
    targets: dict[str, jax.sharding.Sharding],
) -> None:
  """Reshards arrays read under a de-duplicated sharding to their targets.

  An array read under `_dedup_read_sharding` holds each byte on exactly one
  device; an identity computation whose `out_shardings` are the true targets
  makes XLA broadcast the shards to every replica over the accelerator
  interconnect. Tensors are batched so each compiled call's per-device output
  stays under a fixed byte bound, keeping transient device memory
  predictable regardless of checkpoint size.

  Args:
    arrays: Loaded arrays by tensor name; entries named in `targets` are
      replaced in place with their broadcast result.
    targets: Target sharding by tensor name for every array to broadcast.
  """
  batch: list[str] = []
  batch_bytes = 0

  def flush() -> None:
    if not batch:
      return
    resharded = jax.jit(
        lambda xs: xs, out_shardings=[targets[n] for n in batch]
    )([arrays[n] for n in batch])
    for n, out in zip(batch, resharded):
      arrays[n] = out
    batch.clear()

  for name in targets:
    arr = arrays[name]
    shard_shape = targets[name].shard_shape(arr.shape)
    per_device_bytes = math.prod(shard_shape) * arr.dtype.itemsize
    if batch and batch_bytes + per_device_bytes > _BROADCAST_BATCH_BYTES:
      flush()
      batch_bytes = 0
    batch.append(name)
    batch_bytes += per_device_bytes
  flush()


class _FileEntry(NamedTuple):
  """Where to find one tensor in the on-disk checkpoint."""

  path: Path  # The .safetensors file this tensor lives in.
  info: dict[str, Any]  # The tensor's entry from the file's header.
  data_start: int  # Byte offset of the file's data section.


class _Read(NamedTuple):
  """One byte run a tensor needs, with its destination buffer."""

  offset: int  # Absolute byte offset of the run in the file.
  length: int  # Length of the run in bytes.
  dst: np.ndarray  # Flat `uint8` destination view, `length` bytes long.


class _ChunkWrite(NamedTuple):
  """One precomputed chunk-to-buffer copy."""

  dst: np.ndarray  # Flat `uint8` destination view, sliced to the copy length.
  start: int  # Chunk-local start of the source bytes.
  stop: int  # Chunk-local end of the source bytes.


class _ChunkRead(NamedTuple):
  """One ranged read and the buffer copies its bytes serve."""

  offset: int  # Absolute byte offset of the read.
  length: int  # Length of the read in bytes.
  writes: tuple[_ChunkWrite, ...]


class _ReadStats(NamedTuple):
  """Per-file read accounting, aggregated for the fragmentation hint."""

  needed_bytes: int  # Bytes the planned runs actually need.
  read_bytes: int  # Bytes the coalesced blocks pull (>= needed_bytes).
  num_reads: int  # Coalesced blocks (ranged-read requests; ~1 per file ideal).
  num_gets: int  # Actual storage GETs (>= num_reads; blocks split per chunk).


def _plan_chunk_reads(
    reads: Sequence[_Read],
    max_over_read_ratio: float | None,
    chunk_bytes: int,
) -> tuple[list[_ChunkRead], _ReadStats]:
  """Plans the ranged reads serving one file's byte runs.

  The runs are coalesced into blocks under the over-read ratio cap
  (`_partition_runs`), then each block is split at `chunk_bytes` so that a
  large block becomes several concurrent ranged reads instead of one long
  single-stream read. Every chunk carries its precomputed buffer copies, so
  chunks can complete in any order and no per-run state is needed at read
  time.

  Args:
    reads: The byte runs to serve, each with its destination buffer.
    max_over_read_ratio: Upper bound on `block_size / needed_bytes` per block.
      `None` picks the ratio from this file's runs (`_auto_over_read_ratio`).
    chunk_bytes: Maximum size of one ranged read. Must be positive and at most
      the in-flight byte budget, so any single chunk can be reserved.

  Returns:
    The chunk reads covering every run, and the file's read accounting.
  """
  runs = [(r.offset, r.length) for r in reads]
  if max_over_read_ratio is None:
    max_over_read_ratio = _auto_over_read_ratio(runs)
  blocks = _partition_runs(runs, max_over_read_ratio)
  chunks: list[_ChunkRead] = []
  for block in blocks:
    members = block.members  # Sorted by run offset (see `_partition_runs`).
    first = 0
    for chunk_start in range(block.start, block.end, chunk_bytes):
      chunk_end = min(chunk_start + chunk_bytes, block.end)
      # Runs are offset-sorted, so a member fully below `chunk_start`
      # stays consumed for every later chunk of this block.
      while first < len(members):
        r = reads[members[first]]
        if r.offset + r.length > chunk_start:
          break
        first += 1
      writes = []
      for member in itertools.islice(members, first, None):
        r = reads[member]
        if r.offset >= chunk_end:
          break
        lo = max(r.offset, chunk_start)
        hi = min(r.offset + r.length, chunk_end)
        if lo < hi:
          writes.append(
              _ChunkWrite(
                  dst=r.dst[lo - r.offset : hi - r.offset],
                  start=lo - chunk_start,
                  stop=hi - chunk_start,
              )
          )
      chunks.append(
          _ChunkRead(chunk_start, chunk_end - chunk_start, tuple(writes))
      )
  return chunks, _ReadStats(
      needed_bytes=sum(r.length for r in reads),
      read_bytes=sum(b.end - b.start for b in blocks),
      num_reads=len(blocks),
      num_gets=len(chunks),
  )


async def _read_chunk(
    path: Path, chunk: _ChunkRead, byte_budget: limits.LimitInFlightBytes
) -> None:
  """Reads one chunk and copies its bytes into the destination buffers.

  Args:
    path: The file to read from. Each read opens its own handle, so chunks are
      safe to issue concurrently.
    chunk: The byte range to read and the buffer copies it serves.
    byte_budget: The shared in-flight byte limiter; the chunk's bytes stay
      reserved until they have been copied out.
  """
  start_reserve = time.time()
  async with limits.reserved_bytes(byte_budget, chunk.length):
    wait_time = time.time() - start_reserve
    start_read = time.time()
    async with async_path.open_file(path, mode="rb") as f:
      await f.seek(chunk.offset)
      # `AsyncFile.read` is declared `bytes | str`; `mode="rb"` guarantees
      # bytes at runtime but pytype can't narrow that statically.
      data = cast(bytes, await f.read(chunk.length))
    read_time = time.time() - start_read
    logging.vlog(
        1,
        "Read chunk of size %d bytes from %s at offset %d (limit wait: %.3f s,"
        " drive read: %.3f s)",
        chunk.length,
        path.name,
        chunk.offset,
        wait_time,
        read_time,
    )
    src = memoryview(data)
    for w in chunk.writes:
      w.dst[:] = np.frombuffer(src[w.start : w.stop], dtype=np.uint8)


async def _read_file(
    path: Path,
    reads: Sequence[_Read],
    byte_budget: limits.LimitInFlightBytes,
    max_over_read_ratio: float | None,
    chunk_bytes: int,
) -> _ReadStats:
  """Serves all of one file's byte runs with concurrent ranged reads.

  Args:
    path: The `.safetensors` file to read from.
    reads: The byte runs this host needs from the file.
    byte_budget: The shared in-flight byte limiter (shared across files).
    max_over_read_ratio: Upper bound on `block_size / needed_bytes` per block.
      `None` picks the ratio from this file's runs (`_auto_over_read_ratio`).
    chunk_bytes: Maximum size of one ranged read.

  Returns:
    The file's read accounting.
  """
  if not reads:
    return _ReadStats(needed_bytes=0, read_bytes=0, num_reads=0, num_gets=0)
  chunks, stats = _plan_chunk_reads(reads, max_over_read_ratio, chunk_bytes)
  await asyncio.gather(*[_read_chunk(path, c, byte_budget) for c in chunks])
  return stats


def _allocate_buffer(
    shape: tuple[int, ...], dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
  """Allocates a contiguous buffer + a flat `uint8` view over its bytes.

  The flat view is the canonical write-target for byte-level chunk copies; it
  works uniformly for scalars (`shape == ()`), 1-D, and N-D buffers, where
  `np.ndarray.view(uint8)` on a 0-D array does not.

  Args:
    shape: Shape of the typed buffer to allocate.
    dtype: Element dtype of the typed buffer.

  Returns:
    A `(buffer, flat_u8)` pair: the typed `shape`-shaped array and a flat
    `uint8` view aliasing the same memory.
  """
  num_elements = math.prod(shape) if shape else 1
  flat_u8 = np.empty(num_elements * dtype.itemsize, dtype=np.uint8)
  buf = flat_u8.view(dtype).reshape(shape)
  return buf, flat_u8


def _plan_whole_tensor(
    entry: _FileEntry,
) -> tuple[list[_Read], Callable[[], np.ndarray]]:
  """Plans one full-tensor read into a host array.

  Used when no `abstract_state` is given -- there is no target sharding to
  drive a partial read, so each tensor is materialised in full on this host.
  The chunk reads write directly into the pre-allocated array; the returned
  builder just returns it.

  Args:
    entry: The file entry describing the tensor's location and header info.

  Returns:
    The tensor's byte runs, and a builder that returns the materialised
    array once they have been read.
  """
  shape, dtype = _get_array_properties(entry.info)
  dtype = np.dtype(dtype)
  begin, end = entry.info["data_offsets"]
  buf, flat_u8 = _allocate_buffer(shape, dtype)
  return [_Read(entry.data_start + begin, end - begin, flat_u8)], lambda: buf


def _plan_sharded_tensor(
    entry: _FileEntry,
    abstract_leaf: Any,
    sharding: jax.sharding.Sharding,
) -> tuple[list[_Read], Callable[[], jax.Array]]:
  """Plans the shard reads this process's devices need for one tensor.

  For each unique index domain owned by one of this process's devices, the
  shard buffer is allocated once, and each of the shard's byte runs targets
  the prefix of that buffer it occupies. The returned builder turns the
  populated buffers into a sharded `jax.Array`.

  Args:
    entry: The file entry describing the tensor's location and header info.
    abstract_leaf: The abstract array (shape, dtype) to build.
    sharding: The sharding to read and assemble under -- the leaf's target
      sharding, or the de-duplicated read sharding when
      `broadcast_replicated` is enabled.

  Returns:
    The tensor's byte runs, and a builder that assembles the populated shard
    buffers into a `jax.Array` once they have been read.

  Raises:
    ValueError: If the abstract state's shape disagrees with the checkpoint.
  """
  file_shape, file_dtype = _get_array_properties(entry.info)
  file_dtype = np.dtype(file_dtype)
  tensor_base = entry.data_start + entry.info["data_offsets"][0]

  global_shape = tuple(abstract_leaf.shape)
  out_dtype = np.dtype(abstract_leaf.dtype)
  if global_shape != file_shape:
    raise ValueError(
        f"Shape mismatch: abstract state has {global_shape}, the checkpoint"
        f" has {file_shape}."
    )

  this_process = multihost.process_index()
  index_to_devices: dict[arrays_types.IndexBounds, list[jax.Device]] = (
      collections.defaultdict(list)
  )
  for device, index in sharding.devices_indices_map(global_shape).items():
    if device.process_index == this_process:
      index_to_devices[_normalize_index(index, global_shape)].append(device)

  reads: list[_Read] = []
  shard_specs: list[tuple[np.ndarray, list[jax.Device]]] = []
  for bounds, devices in index_to_devices.items():
    shard_shape = tuple(stop - start for start, stop in bounds)
    shard_buf, shard_flat_u8 = _allocate_buffer(shard_shape, file_dtype)
    runs = index_domain_to_byte_runs(
        bounds, global_shape, file_dtype.itemsize, tensor_base
    )
    # Runs are already sorted by offset; their in-memory destination is the
    # corresponding prefix of `shard_flat_u8` in the same order.
    write_offset = 0
    for off, length in runs:
      end_byte = write_offset + length
      reads.append(_Read(off, length, shard_flat_u8[write_offset:end_byte]))
      write_offset = end_byte
    shard_specs.append((shard_buf, devices))

  def build() -> jax.Array:
    arrays: list[jax.Array] = []
    # Consume front-to-back, releasing each shard's host buffer once its data
    # is on-device instead of holding every buffer until the load returns.
    # Popping here also frees the file_dtype original the moment `astype` makes
    # the out_dtype copy, so a conversion's transient overlaps just one shard.
    # `pop(0)` preserves order; the per-process shard count is small.
    while shard_specs:
      shard_buf, devices = shard_specs.pop(0)
      if out_dtype != file_dtype:
        shard_buf = shard_buf.astype(out_dtype)
      for d in devices:
        arrays.append(
            jax.device_put(shard_buf, jax.sharding.SingleDeviceSharding(d))
        )
      del shard_buf
    # `dtype` is required when a process contributes no shards (it cannot be
    # inferred from an empty buffer list).
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, arrays, dtype=out_dtype
    )

  return reads, build


def _record_read_stats(stats: Sequence[_ReadStats]) -> None:
  """Reports this host's read accounting via `jax.monitoring`.

  Emitted once per load with the per-host totals -- a single scalar per name,
  since a metric collector keyed by name would keep only the last of repeated
  emissions. Mirrors how the TensorStore restore path self-reports
  `/jax/orbax/read/...`, so a benchmark can capture safetensors reads through
  the standard jax.monitoring listener with no dependency on this loader.

  Two read counts are reported: `num_reads` is the coalesced blocks (~1 per
  file when fully coalesced), and `storage_reads` is the actual GETs issued,
  which is >= `num_reads` because a block larger than the read chunk size is
  fetched as several concurrent ranged reads.

  Args:
    stats: Per-file read accounting from each file's reads.
  """
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/bytes_read",
      float(sum(s.read_bytes for s in stats)),
  )
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/num_reads",
      float(sum(s.num_reads for s in stats)),
  )
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/storage_reads",
      float(sum(s.num_gets for s in stats)),
  )


def _warn_if_over_read(
    stats: Sequence[_ReadStats],
    max_over_read_ratio_is_auto: bool,
) -> None:
  """Logs a one-shot hint when the load read far more bytes than it needed.

  Under the automatic per-file ratio, heavy over-read happens only when the
  target sharding maps to many small strided runs per process and whole spans
  were read instead to keep the request count bounded. The extra bytes are
  the cost of that sharding's byte layout, not a loader inefficiency, so the
  actionable advice is a sharding whose per-process ranges are contiguous.
  Fires once, only on the primary process, and only for the automatic ratio
  (someone who set the knob has already weighed the tradeoff).

  Args:
    stats: Per-file read accounting from each file's reads.
    max_over_read_ratio_is_auto: Whether the user left `max_over_read_ratio`
      unset (the per-file automatic choice).
  """
  if not max_over_read_ratio_is_auto:
    return
  if multihost.process_index() != 0:
    return
  total_needed = sum(s.needed_bytes for s in stats)
  total_read = sum(s.read_bytes for s in stats)
  if total_needed == 0:
    return
  over_read = total_read / total_needed
  if over_read < _OVER_READ_WARN_RATIO:
    return
  logging.warning(
      "Safetensors load read %s to serve %s of needed bytes (%.1fx"
      " over-read). The target sharding needs many small strided ranges per"
      " process, so whole spans were read instead to keep the request count"
      " bounded. A sharding whose per-process ranges are contiguous (e.g."
      " partitioning the leading dimension) avoids the extra bytes.",
      humanize.naturalsize(total_read, binary=True),
      humanize.naturalsize(total_needed, binary=True),
      over_read,
  )


def _plan_reads(
    file_index: dict[str, _FileEntry],
    abstract_state: dict[str, Any] | None,
    names: Sequence[str],
    broadcast_replicated: bool,
) -> tuple[
    dict[Path, list[_Read]],
    dict[Path, list[tuple[str, Callable[[], Any]]]],
    dict[str, jax.sharding.Sharding],
]:
  """Plans every tensor's byte runs and builders, grouped by file.

  With `broadcast_replicated`, replicated bytes are read from storage
  once -- under a de-duplicated sharding spread across the replicas -- and
  the replicas are filled afterwards by an on-device broadcast.
  Cross-process duplication is the only waste it removes, so it is a no-op
  for single-process loads.

  Args:
    file_index: Tensor name -> `_FileEntry` for the whole checkpoint.
    abstract_state: The abstract leaves driving the load, or `None` to
      materialise every tensor in full (single-process only).
    names: The tensor names to load, in result order.
    broadcast_replicated: `SafetensorsOptions.broadcast_replicated`.

  Returns:
    Reads grouped by file, `(name, build)` pairs grouped by file, and the
    target shardings for every tensor read under a de-duplicated sharding
    (the tensors `_broadcast_to_target_shardings` must fix up afterwards).
  """
  dedup_replicated = broadcast_replicated and multihost.process_count() > 1
  reads_by_file: dict[Path, list[_Read]] = collections.defaultdict(list)
  builds_by_file: dict[Path, list[tuple[str, Callable[[], Any]]]] = (
      collections.defaultdict(list)
  )
  broadcast_targets: dict[str, jax.sharding.Sharding] = {}
  for name in names:
    entry = file_index[name]
    if abstract_state is None:
      reads, build = _plan_whole_tensor(entry)
    else:
      leaf = abstract_state[name]
      sharding = getattr(leaf, "sharding", None) or _replicated_sharding()
      if dedup_replicated and (
          read_sharding := _dedup_read_sharding(sharding, tuple(leaf.shape))
      ):
        broadcast_targets[name] = sharding
        sharding = read_sharding
      reads, build = _plan_sharded_tensor(entry, leaf, sharding)
    reads_by_file[entry.path].extend(reads)
    builds_by_file[entry.path].append((name, build))
  return reads_by_file, builds_by_file, broadcast_targets


class SafetensorsLayout(CheckpointLayout):
  """Handles checkpoints in the HuggingFace Safetensors format.

  Inherits the abstract methods of :py:class:`~.CheckpointLayout`. Loading is
  resharding-aware: see the module docstring. Saving is not yet supported.
  """

  def __init__(self):
    if multihost.is_pathways_backend():
      raise ValueError(
          "SafetensorsLayout is not supported on Pathways backend."
      )

  async def validate_checkpointables(self, path: Path):
    """Validates that `path` is a SafeTensors file or a directory of them."""
    if await async_path.is_file(path):
      if path.suffix == SAFETENSORS_SUFFIX:
        return
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a SafeTensors checkpoint. A"
          " SafeTensors checkpoint must be a file with the"
          f" '{SAFETENSORS_SUFFIX}' suffix."
      )
    elif await async_path.is_dir(path):
      files = list(await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}"))
      if not files:
        raise InvalidLayoutError(
            f"Directory {path} does not contain any '{SAFETENSORS_SUFFIX}'"
            " files."
        )
    else:
      raise InvalidLayoutError(
          f"Path {path} is neither a file nor a directory or does not exist."
      )

  async def get_checkpointable_names(  # pyrefly: ignore[bad-override]
      self, path: Path
  ) -> list[str]:
    del path
    return []

  async def validate(self, path: Path, checkpointable_name: str | None) -> None:
    """No-op: there is nothing PyTree-specific to validate."""
    del path, checkpointable_name

  async def checkpointables_metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, AbstractCheckpointable]]:
    """A SafeTensors checkpoint is always a flat checkpoint; use `metadata`."""
    raise NotImplementedError(
        "SafetensorsLayout does not support `.checkpointables_metadata`. Use"
        " `.metadata` instead."
    )

  async def metadata(
      self, path: Path, checkpointable_name: str | None
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    """Returns the flat checkpoint metadata for the Safetensors checkpoint."""
    del checkpointable_name  # Safetensors is always flat; name is unused.
    entries = await _read_all_headers(path)
    stats = await asyncio.gather(
        *[async_path.async_stat(file_path) for file_path, _, _ in entries]
    )
    commit_timestamp_nsecs = (
        max(int(stat.mtime) for stat in stats) if stats else None
    )
    item_metadata: dict[str, jax.ShapeDtypeStruct] = {}
    custom_metadata: dict[str, Any] = {}
    for _, header, _ in entries:
      for name, info in header.items():
        if name == "__metadata__":
          if info:
            custom_metadata.update(info)
          continue
        if name in item_metadata:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        shape, dtype = _get_array_properties(info)
        item_metadata[name] = jax.ShapeDtypeStruct(shape=shape, dtype=dtype)
    return metadata_types.CheckpointMetadata(
        path=path,
        metadata=item_metadata,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  async def load(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_state: AbstractCheckpointable | None = None,
  ) -> Awaitable[Checkpointable]:
    """Loads a flat PyTree of arrays from a SafeTensors checkpoint.

    Args:
      path: A `.safetensors` file, or a directory containing such files.
      checkpointable_name: Unused; SafeTensors checkpoints expose one
        checkpointable.
      abstract_state: An optional flat dict mapping tensor name to an abstract
        leaf (`jax.ShapeDtypeStruct`). When provided, each leaf's sharding
        drives the load: every process reads only the byte ranges its own
        devices need. When `None`, every tensor is materialised in full on this
        host (single-process only).

    Returns:
      An awaitable resolving to the loaded flat dict of arrays.

    Raises:
      ValueError: If `abstract_state` is provided but is not a flat dict.
    """
    del checkpointable_name
    if abstract_state is not None and not tree_utils.is_flat_dict(
        abstract_state
    ):
      raise ValueError("The PyTree is not a flat dictionary.")
    file_index = await self._build_file_index(path)
    return self._load(file_index, abstract_state)

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, AbstractCheckpointable] | None = None,
  ) -> Awaitable[dict[str, Checkpointable]]:
    raise NotImplementedError(
        "SafetensorsLayout does not support `.load_checkpointables`. Use"
        " `.load` instead."
    )

  async def save_checkpointables(
      self,
      path: types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Checkpointable],
  ) -> Awaitable[None]:
    """Raises: saving to Safetensors is not supported."""
    del path, checkpointables
    raise NotImplementedError(
        "Saving to Safetensors format is not supported yet."
    )

  async def _build_file_index(self, path: Path) -> dict[str, _FileEntry]:
    """Maps each tensor name to where to find it (`_FileEntry`)."""
    entries = await _read_all_headers(path)
    await asyncio.gather(
        *[_check_file_length(f, header, ds) for f, header, ds in entries]
    )
    file_index: dict[str, _FileEntry] = {}
    for file_path, header, data_start in entries:
      for name, info in header.items():
        if name == "__metadata__":
          continue
        if name in file_index:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        file_index[name] = _FileEntry(file_path, info, data_start)
    return file_index

  async def _load(
      self,
      file_index: dict[str, _FileEntry],
      abstract_state: dict[str, Any] | None,
  ) -> dict[str, Any]:
    """Background load phase: plans reads, reads files, assembles arrays."""
    context = context_lib.get_context()
    opts = context.safetensors_options
    # In-flight read bytes reuse the shared restore knob,
    # `MemoryOptions.read_concurrent_bytes` (the same limit TensorStore restore
    # uses). `None` means "no limit" there, but the chunk sizing here needs a
    # finite budget, so fall back to the loader default.
    read_concurrent_bytes = context.memory_options.read_concurrent_bytes
    max_in_flight_bytes = (
        read_concurrent_bytes
        if read_concurrent_bytes is not None
        else _DEFAULT_MAX_IN_FLIGHT_BYTES
    )
    read_chunk_bytes = (
        opts.read_chunk_bytes
        if opts.read_chunk_bytes is not None
        else _DEFAULT_READ_CHUNK_BYTES
    )
    if read_chunk_bytes <= 0:
      raise ValueError(
          f"read_chunk_bytes must be positive, got {read_chunk_bytes}."
      )
    chunk_bytes = min(read_chunk_bytes, max_in_flight_bytes)
    # One shared limiter across every file. Peak in-flight memory for the
    # entire load is bounded at `max_in_flight_bytes` regardless of how many
    # files or how large each file is. `LimitInFlightBytes` reserves strictly
    # fewer bytes than its max, so size it one over: a single chunk can then
    # reserve the full `max_in_flight_bytes`, and concurrent reservations
    # still sum to at most `max_in_flight_bytes`.
    byte_budget = limits.LimitInFlightBytes(max_in_flight_bytes + 1)

    if abstract_state is None:
      if multihost.process_count() > 1:
        raise ValueError(
            "abstract_state must be provided for multi-process loading."
        )
      names = list(file_index.keys())
    else:
      for name in abstract_state:
        if name not in file_index:
          raise KeyError(
              f"Tensor '{name}' not found in Safetensors checkpoint."
          )
      names = sorted(abstract_state)

    # Phase 1 (sync): compute every byte range this process needs, grouped by
    # file, and pre-allocate the destination buffers. Each tensor's `build()`
    # turns its (yet-unfilled) buffers into an array once its file is read.
    reads_by_file, builds_by_file, broadcast_targets = _plan_reads(
        file_index, abstract_state, names, opts.broadcast_replicated
    )

    # Phase 2 (async): every file's chunks in parallel, bounded together by
    # the shared limiter. The chunk reads fill the pre-allocated buffers in
    # place. As soon as one file completes, its tensors are assembled --
    # overlapping host-to-device transfer with the remaining reads -- and
    # their host buffers are released.
    arrays: dict[str, Any] = {}

    async def load_file(file_path: Path) -> _ReadStats:
      stats = await _read_file(
          file_path,
          reads_by_file[file_path],
          byte_budget,
          opts.max_over_read_ratio,
          chunk_bytes,
      )
      # Drop the read plan so each shard's host memory can be released as
      # `build()` moves it on-device.
      del reads_by_file[file_path]
      for name, build in builds_by_file[file_path]:
        arrays[name] = build()
      return stats

    stats = await asyncio.gather(*map(load_file, list(reads_by_file)))
    _record_read_stats(stats)
    _warn_if_over_read(stats, opts.max_over_read_ratio is None)
    if broadcast_targets:
      _broadcast_to_target_shardings(arrays, broadcast_targets)
    return {name: arrays[name] for name in names}
