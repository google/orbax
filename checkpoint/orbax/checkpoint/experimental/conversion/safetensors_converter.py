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

"""Streaming checkpoint format conversion between Safetensors and Orbax."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import itertools
import json
import math
import os
import struct
import sys
import time
from typing import Any, Literal, cast
import uuid

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.path import fs_probe
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import types as serialization_types
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import registry as layout_registry
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
import tensorstore as ts

SAFETENSORS_SUFFIX = safetensors_layout.SAFETENSORS_SUFFIX
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_SINGLE_SHARD_FILENAME = "model.safetensors"
DEFAULT_MAX_SHARD_SIZE_BYTES = 5 * 1024**3  # 5 GiB (HuggingFace convention)
DEFAULT_MAX_IN_FLIGHT_BYTES = 512 * 1024**2  # 512 MiB host RAM budget
_MAX_CHUNK_BYTE_SIZE_CAP = 64 * 1024**2  # 64 MiB Zarr chunk cap when streaming
_MAX_CONCURRENT_TENSORS = 32
_ORBAX_JSON_KEYS_META = "__orbax_json_keys__"

_ST_TO_DTYPE: dict[str, Any] = {
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
    "F8_E4M3": jnp.float8_e4m3fn,
    "F8_E5M2": jnp.float8_e5m2,
    "F8_E4M3FNUZ": jnp.float8_e4m3fnuz,
    "F8_E5M2FNUZ": jnp.float8_e5m2fnuz,
    "C64": np.complex64,
}

_DTYPE_NAME_TO_ST: dict[str, str] = {
    np.dtype(dt).name: code for code, dt in _ST_TO_DTYPE.items()
}


class UnsupportedDtypeError(ValueError):
  """Raised when a tensor uses a dtype unsupported by the target format."""


@dataclasses.dataclass(frozen=True)
class ConversionOptions:
  """Configuration options for Safetensors <-> Orbax conversion.

  Attributes:
    overwrite: Replace `dest_path` if it already exists.
    key_separator: Delimiter used to unflatten Safetensors keys into nested
      dicts (`"."` by default) and flatten nested Orbax PyTrees into Safetensors
      tensor names. If `None`, keeps/requires a 1-level flat dict.
    strict_unflatten: When True, raise `ValueError` if Safetensors keys contain
      empty segments or leaf-vs-prefix collisions under `key_separator`; when
      False, fall back to a flat dict and record a warning.
    checkpointable_name: Orbax checkpointable subdirectory name (`"state"` by
      default). In `orbax_to_safetensors`, `None` converts all PyTree
      checkpointables prefixed by `<checkpointable><sep>`.
    drop_non_tensor_leaves: When True, skip non-tensor Orbax leaves (such as
      strings or empty containers) instead of raising `ValueError`, recording
      dropped paths in `ConversionResult.dropped_non_tensor_leaves`.
    use_ocdbt: Write OCDBT storage when converting Safetensors -> Orbax.
    use_zarr3: Write Zarr v3 metadata/chunks when converting Safetensors ->
      Orbax.
    use_compression: Enable zstd chunk compression when converting Safetensors
      -> Orbax.
    max_shard_size_bytes: Maximum byte size per `.safetensors` shard when
      converting Orbax -> Safetensors (`<= 0` forces a single file).
    max_in_flight_bytes: Upper bound on in-memory tensor buffer bytes held at
      any point during conversion. Tensors larger than this budget are streamed
      in contiguous slices.
  """

  overwrite: bool = False
  key_separator: str | None = "."
  strict_unflatten: bool = False
  checkpointable_name: str | None = checkpoint_layout.STATE_CHECKPOINTABLE_KEY
  drop_non_tensor_leaves: bool = False
  use_ocdbt: bool = True
  use_zarr3: bool = False
  use_compression: bool = True
  max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES
  max_in_flight_bytes: int = DEFAULT_MAX_IN_FLIGHT_BYTES


@dataclasses.dataclass(frozen=True)
class ConversionResult:
  """Summary and performance metrics of a completed checkpoint conversion."""

  source_path: str
  dest_path: str
  source_format: str
  dest_format: str
  num_tensors: int
  total_logical_bytes: int
  total_stored_bytes: int
  num_output_shards: int
  elapsed_secs: float
  dropped_non_tensor_leaves: tuple[str, ...] = ()
  metadata_warnings: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of the conversion result."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class _SafetensorsTensorSpec:
  """Planned tensor read from a Safetensors shard."""

  flat_name: str
  keypath: tuple[str, ...]
  file_path: epath.Path
  data_offset: int
  nbytes: int
  shape: tuple[int, ...]
  dtype: np.dtype[Any]


@dataclasses.dataclass(frozen=True)
class _OrbaxTensorSpec:
  """Planned tensor read from an Orbax PyTree checkpoint."""

  flat_name: str
  item_dir: epath.Path
  ts_param_name: str
  use_ocdbt: bool
  use_zarr3: bool
  shape: tuple[int, ...]
  dtype: np.dtype[Any]
  st_dtype: str
  nbytes: int


def _clean_path(path: str | os.PathLike[str]) -> str:
  """Normalizes local and cloud URI paths."""
  raw = os.fspath(path).strip().removeprefix("file://")
  if "://" not in raw and not raw.startswith("/"):
    raw = os.path.abspath(raw)
  if len(raw) > 1 and not raw.endswith("://"):
    raw = raw.rstrip("/")
  return raw


def _validate_options(options: ConversionOptions) -> None:
  """Validates conversion option invariants."""
  if options.max_in_flight_bytes <= 0:
    raise ValueError(
        "max_in_flight_bytes must be positive, got"
        f" {options.max_in_flight_bytes}."
    )
  if options.key_separator is not None and not options.key_separator:
    raise ValueError("key_separator must be non-empty or None.")


def _parse_st_dtype(tensor_name: str, code: str) -> np.dtype[Any]:
  """Maps a Safetensors dtype code to a NumPy/JAX dtype."""
  if code not in _ST_TO_DTYPE:
    supported = ", ".join(sorted(_ST_TO_DTYPE))
    raise UnsupportedDtypeError(
        f"Tensor {tensor_name!r} has Safetensors dtype {code!r}, which is not"
        f" supported by Orbax Zarr storage. Supported dtypes: {supported}."
    )
  return np.dtype(_ST_TO_DTYPE[code])


def _to_st_dtype(tensor_name: str, dtype: Any) -> str:
  """Maps an Orbax/JAX/NumPy dtype to a Safetensors dtype code."""
  if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
    raise UnsupportedDtypeError(
        f"Tensor {tensor_name!r} uses JAX typed PRNG key dtype {dtype!r},"
        " which Safetensors does not support."
    )
  dt_name = np.dtype(dtype).name
  if dt_name not in _DTYPE_NAME_TO_ST:
    supported = ", ".join(sorted(_DTYPE_NAME_TO_ST))
    raise UnsupportedDtypeError(
        f"Tensor {tensor_name!r} has dtype {dt_name!r}, which Safetensors does"
        f" not support. Supported dtypes: {supported}."
    )
  return _DTYPE_NAME_TO_ST[dt_name]


def _encode_custom_metadata(
    custom_metadata: Mapping[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
  """Encodes Orbax JSON `custom_metadata` into Safetensors `dict[str, str]`."""
  if not custom_metadata:
    return {}, []
  encoded: dict[str, str] = {}
  json_keys: list[str] = []
  for key, value in custom_metadata.items():
    key_str = str(key)
    if isinstance(value, str):
      encoded[key_str] = value
    else:
      encoded[key_str] = json.dumps(value, sort_keys=True)
      json_keys.append(key_str)
  warnings: list[str] = []
  if json_keys:
    sorted_keys = sorted(json_keys)
    encoded[_ORBAX_JSON_KEYS_META] = json.dumps(sorted_keys)
    warnings.append(
        "JSON-stringified non-string custom_metadata keys for Safetensors"
        f" header: {sorted_keys}"
    )
  return encoded, warnings


def _decode_safetensors_metadata(raw_meta: Mapping[str, Any]) -> dict[str, Any]:
  """Restores Orbax `custom_metadata` from Safetensors `__metadata__`."""
  decoded: dict[str, Any] = {str(k): v for k, v in raw_meta.items()}
  raw_json_keys = decoded.pop(_ORBAX_JSON_KEYS_META, None)
  if isinstance(raw_json_keys, str):
    try:
      json_keys = json.loads(raw_json_keys)
      if isinstance(json_keys, list):
        for key in json_keys:
          if isinstance(key, str) and isinstance(decoded.get(key), str):
            decoded[key] = json.loads(decoded[key])
    except json.JSONDecodeError as err:
      logging.warning("Failed to decode %s: %s", _ORBAX_JSON_KEYS_META, err)
  return decoded


def _unflatten_keys(
    flat_names: Sequence[str],
    sep: str | None,
    *,
    strict: bool,
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
  """Maps flat Safetensors names to Orbax keypath tuples."""
  if sep is None:
    return {name: (name,) for name in flat_names}, []

  split_map = {name: tuple(name.split(sep)) for name in flat_names}
  empty_segment = [name for name, parts in split_map.items() if "" in parts]
  leaf_paths = set(split_map.values())
  ancestor_paths: set[tuple[str, ...]] = set()
  for parts in split_map.values():
    for idx in range(1, len(parts)):
      ancestor_paths.add(parts[:idx])
  collisions = sorted(
      sep.join(parts) for parts in (leaf_paths & ancestor_paths)
  )
  if empty_segment or collisions:
    reason = (
        f"empty key segments in {empty_segment[:3]}"
        if empty_segment
        else f"leaf/prefix collisions on {collisions[:3]}"
    )
    if strict:
      raise ValueError(
          f"Cannot unflatten Safetensors keys with separator {sep!r}: {reason}."
      )
    warning = (
        f"Kept flat PyTree dictionary because separator {sep!r} produced"
        f" {reason}."
    )
    return {name: (name,) for name in flat_names}, [warning]
  return split_map, []


def _bytes_to_array(
    data: bytes, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> np.ndarray:
  """Reinterprets little-endian Safetensors bytes as a NumPy array."""
  arr_u8 = np.frombuffer(data, dtype=np.uint8)
  if dtype.itemsize > 1 and sys.byteorder == "big":
    le_dtype = dtype.newbyteorder("<")
    return arr_u8.view(le_dtype).astype(dtype).reshape(shape)
  return arr_u8.view(dtype).reshape(shape)


def _array_to_le_bytes_view(
    arr: np.ndarray,
) -> tuple[np.ndarray, memoryview[int]]:
  """Returns `(keep_alive_array, little_endian_uint8_memoryview)`."""
  contig = np.ascontiguousarray(arr)
  byteorder = contig.dtype.byteorder
  is_big = byteorder == ">" or (byteorder == "=" and sys.byteorder == "big")
  if contig.dtype.itemsize > 1 and is_big:
    contig = contig.byteswap(inplace=False)
  u8_view = contig.reshape(-1).view(np.uint8)
  return contig, memoryview(u8_view)


def _iter_contiguous_slices(
    shape: tuple[int, ...],
    itemsize: int,
    max_bytes: int,
    preferred_chunk_dim: int | None = None,
) -> Iterator[tuple[tuple[int | slice, ...], tuple[int, ...], int, int]]:
  """Yields `(index_tuple, slice_shape, byte_offset, byte_length)` slices.

  Each yielded slice maps to a single contiguous byte range in C-order
  row-major storage and fits within `max(max_bytes, itemsize)` bytes.

  Args:
    shape: Full tensor shape.
    itemsize: Bytes per element.
    max_bytes: Target maximum bytes per slice.
    preferred_chunk_dim: Optional leading-axis Zarr chunk size to align slices
      to when slicing along axis 0.

  Yields:
    Tuples of `(ts_index, slice_shape, byte_offset, byte_length)`. An empty
    `ts_index` `()` denotes the full array.
  """
  total_bytes = math.prod(shape) * itemsize
  if total_bytes <= max_bytes or not shape:
    yield (), shape, 0, total_bytes
    return

  effective_max = max(max_bytes, itemsize)
  ndim = len(shape)
  axis = ndim - 1
  unit_bytes = itemsize
  for candidate_axis in range(ndim):
    rem_bytes = math.prod(shape[candidate_axis + 1 :]) * itemsize
    if rem_bytes <= effective_max:
      axis = candidate_axis
      unit_bytes = rem_bytes
      break

  max_span = max(1, effective_max // max(1, unit_bytes))
  if (
      axis == 0
      and preferred_chunk_dim is not None
      and 0 < preferred_chunk_dim <= max_span
  ):
    span = (max_span // preferred_chunk_dim) * preferred_chunk_dim
  else:
    span = max_span

  outer_ranges = [range(shape[k]) for k in range(axis)]
  for outer_coords in itertools.product(*outer_ranges):
    outer_linear = 0
    for dim_idx, coord in enumerate(outer_coords):
      outer_linear = outer_linear * shape[dim_idx] + coord
    for r0 in range(0, shape[axis], span):
      r1 = min(r0 + span, shape[axis])
      start_units = outer_linear * shape[axis] + r0
      byte_offset = start_units * unit_bytes
      byte_length = (r1 - r0) * unit_bytes
      idx: tuple[int | slice, ...] = (*outer_coords, slice(r0, r1))
      slice_shape = (r1 - r0, *shape[axis + 1 :])
      yield idx, slice_shape, byte_offset, byte_length


async def _prepare_destination(dest_ep: epath.Path, overwrite: bool) -> None:
  """Checks or clears `dest_ep` prior to writing."""
  if await async_path.exists(dest_ep):
    if not overwrite:
      raise FileExistsError(
          f"Destination already exists: '{dest_ep}'. Pass overwrite=True to"
          " replace it."
      )
    if await async_path.is_dir(dest_ep):
      await async_path.rmtree(dest_ep)
    else:
      await async_path.unlink(dest_ep)


async def _directory_stored_bytes(root: epath.Path) -> int:
  """Computes recursive stored file bytes under `root`."""
  root_str = os.fspath(root)
  if os.path.exists(root_str):

    def _sum_local_bytes() -> int:
      if os.path.isfile(root_str):
        return os.path.getsize(root_str)
      total = 0
      for dirpath, _, filenames in os.walk(root_str):
        for fname in filenames:
          total += os.path.getsize(os.path.join(dirpath, fname))
      return total

    return await asyncio.to_thread(_sum_local_bytes)

  async def _walk_async(curr: epath.Path) -> int:
    if await async_path.is_file(curr):
      return (await async_path.async_stat(curr)).length
    entries = list(await async_path.iterdir(curr))
    if not entries:
      return 0
    sizes = await asyncio.gather(*(_walk_async(e) for e in entries))
    return sum(sizes)

  return await _walk_async(root)


async def _plan_safetensors_source(
    src_ep: epath.Path,
    options: ConversionOptions,
) -> tuple[list[_SafetensorsTensorSpec], dict[str, Any], list[str]]:
  """Reads Safetensors headers, validates shards/dtypes, and plans writes."""
  if not await async_path.exists(src_ep):
    raise FileNotFoundError(f"Source path does not exist: '{src_ep}'")

  if await async_path.is_file(src_ep):
    if src_ep.suffix != SAFETENSORS_SUFFIX:
      raise ValueError(
          f"Expected a '{SAFETENSORS_SUFFIX}' file or directory, got '{src_ep}'"
      )
    files = [src_ep]
    index_ep = None
  else:
    files = sorted(await async_path.glob(src_ep, f"*{SAFETENSORS_SUFFIX}"))
    if not files:
      raise ValueError(
          f"No '{SAFETENSORS_SUFFIX}' files found in directory '{src_ep}'."
      )
    candidate_index = src_ep / SAFETENSORS_INDEX_FILENAME
    index_ep = (
        candidate_index if await async_path.exists(candidate_index) else None
    )

  index_weight_map: dict[str, str] | None = None
  if index_ep is not None:
    index_data = json.loads(await async_path.read_text(index_ep))
    raw_wm = index_data.get("weight_map")
    if isinstance(raw_wm, dict):
      index_weight_map = {str(k): str(v) for k, v in raw_wm.items()}
      present_names = {f.name for f in files}
      missing_shards = sorted(set(index_weight_map.values()) - present_names)
      if missing_shards:
        raise FileNotFoundError(
            f"Safetensors index '{index_ep}' references missing shard file(s):"
            f" {missing_shards}"
        )

  headers_and_offsets = await asyncio.gather(
      *(safetensors_layout._read_header(f) for f in files)  # pylint: disable=protected-access
  )
  await asyncio.gather(*(
      safetensors_layout._check_file_length(f, header, data_start)  # pylint: disable=protected-access
      for f, (header, data_start) in zip(files, headers_and_offsets)
  ))

  raw_custom_metadata: dict[str, Any] = {}
  raw_entries: dict[str, tuple[epath.Path, dict[str, Any], int]] = {}
  for file_path, (header, data_start) in zip(files, headers_and_offsets):
    for name, info in header.items():
      if name == "__metadata__":
        if isinstance(info, Mapping):
          raw_custom_metadata.update(info)
        continue
      if name in raw_entries:
        raise ValueError(
            f"Duplicate tensor {name!r} found in multiple Safetensors files."
        )
      if (
          index_weight_map is not None
          and name in index_weight_map
          and index_weight_map[name] != file_path.name
      ):
        raise ValueError(
            f"Tensor {name!r} found in '{file_path.name}', but"
            f" {SAFETENSORS_INDEX_FILENAME} maps it to"
            f" '{index_weight_map[name]}'."
        )
      raw_entries[name] = (file_path, info, data_start)

  if index_weight_map is not None:
    missing_tensors = sorted(set(index_weight_map) - set(raw_entries))
    if missing_tensors:
      raise ValueError(
          f"{SAFETENSORS_INDEX_FILENAME} lists tensor(s) absent from shard"
          f" headers: {missing_tensors[:5]}"
      )

  sorted_names = sorted(raw_entries)
  keypaths_by_name, warnings = _unflatten_keys(
      sorted_names, options.key_separator, strict=options.strict_unflatten
  )

  specs: list[_SafetensorsTensorSpec] = []
  for name in sorted_names:
    file_path, info, data_start = raw_entries[name]
    dtype = _parse_st_dtype(name, info["dtype"])
    shape = tuple(int(d) for d in info["shape"])
    begin, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
    expected_nbytes = math.prod(shape) * dtype.itemsize
    if end - begin != expected_nbytes:
      raise ValueError(
          f"Tensor {name!r} header offsets [{begin}, {end}] ({end - begin} B)"
          f" disagree with shape {shape} and dtype {dtype}"
          f" ({expected_nbytes} B)."
      )
    specs.append(
        _SafetensorsTensorSpec(
            flat_name=name,
            keypath=keypaths_by_name[name],
            file_path=file_path,
            data_offset=data_start + begin,
            nbytes=expected_nbytes,
            shape=shape,
            dtype=dtype,
        )
    )

  custom_metadata = _decode_safetensors_metadata(raw_custom_metadata)
  return specs, custom_metadata, warnings


async def _write_tensor_to_orbax(
    spec: _SafetensorsTensorSpec,
    item_dir: epath.Path,
    options: ConversionOptions,
    ts_context: ts.Context,
    byte_budget: limits.LimitInFlightBytes,
    sem: asyncio.Semaphore,
) -> ts_utils.ArrayMetadata:
  """Streams one Safetensors tensor into TensorStore (whole or sliced)."""
  ts_param_name = ".".join(spec.keypath)
  process_id = 0 if options.use_ocdbt else None
  chunk_byte_size = (
      min(options.max_in_flight_bytes, _MAX_CHUNK_BYTE_SIZE_CAP)
      if spec.nbytes > options.max_in_flight_bytes
      else None
  )
  write_spec = ts_utils.ArrayWriteSpec(
      item_dir.as_posix(),
      relative_array_filename=ts_param_name,
      global_shape=spec.shape,
      write_shape=spec.shape,
      dtype=spec.dtype,
      chunk_byte_size=chunk_byte_size,
      use_compression=options.use_compression,
      use_zarr3=options.use_zarr3,
      use_ocdbt=options.use_ocdbt,
      process_id=process_id,
  )

  async with sem:
    if spec.nbytes <= options.max_in_flight_bytes:
      reserve = max(1, spec.nbytes)
      async with limits.reserved_bytes(byte_budget, reserve):
        if spec.nbytes == 0:
          raw = b""
        else:
          async with async_path.open_file(spec.file_path, mode="rb") as f:
            await f.seek(spec.data_offset)
            raw = cast(bytes, await f.read(spec.nbytes))
        arr = _bytes_to_array(raw, spec.dtype, spec.shape)
        t = await ts.open(
            ts.Spec(write_spec.json), create=True, open=True, context=ts_context
        )
        await t.write(arr, can_reference_source_data_indefinitely=False)
        del arr, raw
      return write_spec.metadata

    # Tensor exceeds max_in_flight_bytes: stream contiguous slices along leading
    # dimension(s), aligned to Zarr chunk_shape[0] to avoid read-modify-write.
    t = await ts.open(
        ts.Spec(write_spec.json), create=True, open=True, context=ts_context
    )
    preferred_d0 = (
        write_spec.metadata.chunk_shape[0]
        if write_spec.metadata.chunk_shape
        else None
    )
    async with async_path.open_file(spec.file_path, mode="rb") as f:
      for idx, slice_shape, rel_off, length in _iter_contiguous_slices(
          spec.shape,
          spec.dtype.itemsize,
          options.max_in_flight_bytes,
          preferred_chunk_dim=preferred_d0,
      ):
        reserve = max(1, length)
        async with limits.reserved_bytes(byte_budget, reserve):
          await f.seek(spec.data_offset + rel_off)
          raw = cast(bytes, await f.read(length))
          slice_arr = _bytes_to_array(raw, spec.dtype, slice_shape)
          target_view = t[idx] if idx else t
          await target_view.write(
              slice_arr, can_reference_source_data_indefinitely=False
          )
          del slice_arr, raw
    return write_spec.metadata


def _build_param_infos_tree(
    specs: Sequence[_SafetensorsTensorSpec],
    item_dir: epath.Path,
    options: ConversionOptions,
) -> Any:
  """Builds a nested or flat dict PyTree of `ParamInfo` for `_METADATA`."""
  flat_infos: dict[tuple[str, ...], serialization_types.ParamInfo] = {}
  for spec in specs:
    jax_kp = tuple(jax.tree_util.DictKey(k) for k in spec.keypath)
    flat_infos[spec.keypath] = serialization_types.ParamInfo(
        name=".".join(spec.keypath),
        keypath=jax_kp,
        parent_dir=item_dir,
        skip_deserialize=False,
        is_ocdbt_checkpoint=options.use_ocdbt,
        use_compression=options.use_compression,
        use_zarr3=options.use_zarr3,
        value_typestr="np.ndarray",
        write_shape=spec.shape,
    )
  return tree_utils.from_flat_dict(flat_infos)


async def async_safetensors_to_orbax(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Asynchronously converts a Safetensors checkpoint to Orbax V1 format.

  Streams tensors with peak host memory bounded by
  `options.max_in_flight_bytes`. Never loads the entire checkpoint into memory.

  Args:
    source_path: Path to a `.safetensors` file or directory of sharded
      `.safetensors` files.
    dest_path: Destination directory for the Orbax V1 checkpoint.
    options: Conversion options (defaults to `ConversionOptions()`).

  Returns:
    `ConversionResult` with tensor counts, byte sizes, and elapsed time.
  """
  t0 = time.perf_counter()
  init_ns = time.time_ns()
  opts = options or ConversionOptions()
  _validate_options(opts)
  item_name = (
      opts.checkpointable_name or checkpoint_layout.STATE_CHECKPOINTABLE_KEY
  )

  clean_src = _clean_path(source_path)
  clean_dst = _clean_path(dest_path)
  if clean_src == clean_dst:
    raise ValueError("source_path and dest_path must be different.")

  src_ep = epath.Path(clean_src)
  dst_ep = epath.Path(clean_dst)

  specs, custom_metadata, warnings = await _plan_safetensors_source(
      src_ep, opts
  )
  if not specs:
    raise ValueError(
        f"Safetensors checkpoint '{clean_src}' contains 0 tensors."
    )

  await _prepare_destination(dst_ep, opts.overwrite)

  # Stage in a sibling temporary directory and rename upon completion when on
  # storage that supports atomic directory rename; on object stores (`gs://`),
  # write directly and commit indicator/checkpoint metadata files last.
  use_staging_rename = (
      "://" not in clean_dst and not gcs_utils.is_gcs_path(dst_ep)
  )
  stage_ep = (
      dst_ep.parent / f".{dst_ep.name}.orbax-convert-tmp-{uuid.uuid4().hex[:8]}"
      if use_staging_rename
      else dst_ep
  )
  item_dir = stage_ep / item_name

  try:
    await async_path.mkdir(item_dir, parents=True, exist_ok=True)
    ts_context = ts_utils.get_ts_context(use_ocdbt=opts.use_ocdbt)
    budget_cap = max(opts.max_in_flight_bytes, 64) + 1
    byte_budget = limits.LimitInFlightBytes(budget_cap)
    sem = asyncio.Semaphore(_MAX_CONCURRENT_TENSORS)

    array_metadatas = await asyncio.gather(*(
        _write_tensor_to_orbax(
            spec, item_dir, opts, ts_context, byte_budget, sem
        )
        for spec in specs
    ))

    if opts.use_ocdbt:
      await ocdbt_utils.merge_ocdbt_per_process_files(
          item_dir,
          ts_context=ts_context,
          use_zarr3=opts.use_zarr3,
          enable_validation=not opts.use_zarr3,
      )

    await array_metadata_store_lib.Store().write(
        item_dir, array_metadatas, process_index=0
    )

    param_infos_tree = _build_param_infos_tree(specs, item_dir, opts)
    pytree_meta = tree_metadata.InternalTreeMetadata.build(
        param_infos_tree,
        use_ocdbt=opts.use_ocdbt,
        use_zarr3=opts.use_zarr3,
    )
    await async_path.write_text(
        item_dir / format_utils.PYTREE_METADATA_FILE,
        json.dumps(pytree_meta.to_json()),
    )

    internal_ckpt_meta = (
        step_metadata_serialization.InternalCheckpointMetadata.create(
            handler_typestrs={
                item_name: handler_types.typestr(pytree_handler.PyTreeHandler)
            },
            init_timestamp_nsecs=init_ns,
            commit_timestamp_nsecs=time.time_ns(),
            custom_metadata=custom_metadata or None,
        )
    )
    await metadata_serialization.write(
        metadata_serialization.checkpoint_metadata_file_path(stage_ep),
        internal_ckpt_meta.serialize(),
    )
    await async_path.touch(
        stage_ep / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE, exist_ok=True
    )

    if use_staging_rename:
      await asyncio.to_thread(stage_ep.rename, dst_ep)
  except BaseException:
    if use_staging_rename and await async_path.exists(stage_ep):
      await async_path.rmtree(stage_ep)
    raise

  stored_bytes = await _directory_stored_bytes(dst_ep)
  logical_bytes = sum(s.nbytes for s in specs)
  dest_fmt = "orbax-ocdbt" if opts.use_ocdbt else "orbax-zarr"
  return ConversionResult(
      source_path=clean_src,
      dest_path=clean_dst,
      source_format="safetensors",
      dest_format=dest_fmt,
      num_tensors=len(specs),
      total_logical_bytes=logical_bytes,
      total_stored_bytes=stored_bytes,
      num_output_shards=1,
      elapsed_secs=round(time.perf_counter() - t0, 4),
      metadata_warnings=tuple(warnings),
  )


async def _resolve_orbax_items(
    src_ep: epath.Path,
    options: ConversionOptions,
) -> tuple[str, list[tuple[str | None, epath.Path]], dict[str, Any]]:
  """Resolves source Orbax format label, `(prefix, item_dir)` pairs, and metadata."""
  root_index = await fs_probe.index_directory(src_ep)
  if not root_index.exists():
    raise FileNotFoundError(f"Source path does not exist: '{src_ep}'")

  layout_enum = await layout_registry.detect_layout(
      src_ep, root_index=root_index
  )
  if layout_enum != layout_registry.CheckpointLayoutEnum.ORBAX:
    raise ValueError(
        f"Expected an Orbax checkpoint at '{src_ep}', detected"
        f" {layout_enum.value!r}."
    )
  resolver = await layout_registry.CheckpointLayoutResolver.resolve(
      src_ep,
      layout_enum,
      pytree_name=checkpoint_layout.AUTO_CHECKPOINTABLE_KEY,
  )
  layout = resolver.layout

  if isinstance(layout, orbax_v0_layout.OrbaxV0Layout) and (
      await layout._is_direct_checkpoint(src_ep)  # pylint: disable=protected-access
  ):
    ckpt_meta = await layout.metadata(src_ep, None)
    is_ocdbt = format_utils.is_ocdbt_checkpoint(src_ep)
    src_fmt = "orbax-ocdbt" if is_ocdbt else "orbax-zarr"
    custom = (
        dict(ckpt_meta.custom_metadata)
        if isinstance(ckpt_meta.custom_metadata, Mapping)
        else {}
    )
    return src_fmt, [(None, src_ep)], custom

  ckpt_meta = await layout.checkpointables_metadata(src_ep)
  valid_pytree_names = await orbax_layout.get_valid_pytree_names(src_ep)
  available = [n for n in valid_pytree_names if n in ckpt_meta.metadata]
  if not available:
    raise ValueError(
        f"No PyTree checkpointables found in Orbax checkpoint '{src_ep}'."
    )

  selected: list[tuple[str | None, epath.Path]]
  if options.checkpointable_name is not None:
    if options.checkpointable_name not in available:
      raise ValueError(
          f"Checkpointable {options.checkpointable_name!r} not found in"
          f" '{src_ep}'. Available PyTree checkpointables: {available}."
      )
    selected = [(None, src_ep / options.checkpointable_name)]
  elif len(available) == 1:
    selected = [(None, src_ep / available[0])]
  else:
    selected = [(name, src_ep / name) for name in sorted(available)]

  any_ocdbt = any(format_utils.is_ocdbt_checkpoint(d) for _, d in selected)
  src_fmt = "orbax-ocdbt" if any_ocdbt else "orbax-zarr"
  custom = (
      dict(ckpt_meta.custom_metadata)
      if isinstance(ckpt_meta.custom_metadata, Mapping)
      else {}
  )
  return src_fmt, selected, custom


async def _plan_orbax_item_tensors(
    prefix: str | None,
    item_dir: epath.Path,
    options: ConversionOptions,
) -> tuple[list[_OrbaxTensorSpec], list[str], dict[str, Any]]:
  """Reads `_METADATA` and TensorStore array headers for one PyTree directory."""
  meta_path = item_dir / format_utils.PYTREE_METADATA_FILE
  internal_meta = tree_metadata.InternalTreeMetadata.from_json(
      json.loads(await async_path.read_text(meta_path))
  )
  use_ocdbt = (
      internal_meta.use_ocdbt
      if internal_meta.use_ocdbt is not None
      else format_utils.is_ocdbt_checkpoint(item_dir)
  )
  use_zarr3 = bool(internal_meta.use_zarr3)
  item_custom = (
      dict(internal_meta.custom_metadata)
      if isinstance(internal_meta.custom_metadata, Mapping)
      else {}
  )

  candidate_entries: list[tuple[str, str]] = []
  dropped: list[str] = []
  for entry in internal_meta.tree_metadata_entries:
    jax_kp = entry.jax_keypath()
    ts_param_name = tree_utils.param_name_from_keypath(jax_kp)
    str_parts = tree_utils.str_keypath(jax_kp)
    full_parts = (prefix, *str_parts) if prefix is not None else str_parts

    if options.key_separator is None:
      if len(full_parts) != 1:
        raise ValueError(
            f"Nested Orbax keypath {full_parts!r} cannot be flattened with"
            " key_separator=None; specify e.g. key_separator='.'."
        )
      flat_name = full_parts[0]
    else:
      flat_name = options.key_separator.join(full_parts)

    val_type = entry.value_metadata.value_type
    is_non_tensor = val_type == "string" or empty_values.is_empty_typestr(
        val_type
    )
    if is_non_tensor:
      if not options.drop_non_tensor_leaves:
        raise ValueError(
            f"Leaf {flat_name!r} has non-tensor type {val_type!r}, which"
            " Safetensors cannot store. Pass drop_non_tensor_leaves=True to"
            " omit non-tensor leaves."
        )
      dropped.append(flat_name)
      continue
    candidate_entries.append((flat_name, ts_param_name))

  ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
  sem = asyncio.Semaphore(_MAX_CONCURRENT_TENSORS)

  async def _inspect_one(
      flat_name: str, ts_param_name: str
  ) -> _OrbaxTensorSpec:
    async with sem:
      read_spec = ts_utils.ArrayReadSpec(
          directory=item_dir.as_posix(),
          relative_array_filename=ts_param_name,
          use_zarr3=use_zarr3,
          use_ocdbt=use_ocdbt,
      )
      t = await ts.open(ts.Spec(read_spec.json), open=True, context=ts_context)
      shape = tuple(int(d) for d in t.shape)
      dtype = np.dtype(jnp.dtype(t.dtype.name))
      st_dtype = _to_st_dtype(flat_name, dtype)
      nbytes = math.prod(shape) * dtype.itemsize
      return _OrbaxTensorSpec(
          flat_name=flat_name,
          item_dir=item_dir,
          ts_param_name=ts_param_name,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
          shape=shape,
          dtype=dtype,
          st_dtype=st_dtype,
          nbytes=nbytes,
      )

  specs = await asyncio.gather(
      *(_inspect_one(fn, tpn) for fn, tpn in candidate_entries)
  )
  return list(specs), dropped, item_custom


def _partition_into_shards(
    specs: Sequence[_OrbaxTensorSpec],
    max_shard_size_bytes: int,
) -> list[list[_OrbaxTensorSpec]]:
  """Partitions sorted tensor specs into shards capped at `max_shard_size_bytes`."""
  ordered = sorted(specs, key=lambda s: s.flat_name)
  if max_shard_size_bytes <= 0:
    return [ordered]

  shards: list[list[_OrbaxTensorSpec]] = []
  current: list[_OrbaxTensorSpec] = []
  current_bytes = 0
  for spec in ordered:
    if current and current_bytes + spec.nbytes > max_shard_size_bytes:
      shards.append(current)
      current = [spec]
      current_bytes = spec.nbytes
    else:
      current.append(spec)
      current_bytes += spec.nbytes
  if current:
    shards.append(current)
  return shards


def _build_safetensors_header_bytes(
    shard_specs: Sequence[_OrbaxTensorSpec],
    header_metadata: Mapping[str, str],
) -> bytes:
  """Serializes an 8-byte-aligned Safetensors binary header prefix."""
  header: dict[str, Any] = {}
  if header_metadata:
    header["__metadata__"] = dict(header_metadata)
  offset = 0
  for spec in shard_specs:
    end = offset + spec.nbytes
    header[spec.flat_name] = {
        "dtype": spec.st_dtype,
        "shape": list(spec.shape),
        "data_offsets": [offset, end],
    }
    offset = end
  raw_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
  pad = (8 - (len(raw_json) % 8)) % 8
  padded_json = raw_json + (b" " * pad)
  return struct.pack("<Q", len(padded_json)) + padded_json


async def _write_safetensors_shard(
    shard_ep: epath.Path,
    shard_specs: Sequence[_OrbaxTensorSpec],
    header_metadata: Mapping[str, str],
    options: ConversionOptions,
) -> int:
  """Streams one `.safetensors` shard sequentially from TensorStore."""
  header_prefix = _build_safetensors_header_bytes(shard_specs, header_metadata)
  ts_contexts: dict[bool, ts.Context] = {}

  def _get_ctx(use_ocdbt: bool) -> ts.Context:
    if use_ocdbt not in ts_contexts:
      ts_contexts[use_ocdbt] = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    return ts_contexts[use_ocdbt]

  async def _open_ts(spec: _OrbaxTensorSpec) -> Any:
    read_spec = ts_utils.ArrayReadSpec(
        directory=spec.item_dir.as_posix(),
        relative_array_filename=spec.ts_param_name,
        use_zarr3=spec.use_zarr3,
        use_ocdbt=spec.use_ocdbt,
    )
    return await ts.open(
        ts.Spec(read_spec.json), open=True, context=_get_ctx(spec.use_ocdbt)
    )

  bytes_written = 0
  with shard_ep.open("wb") as out_f:
    await asyncio.to_thread(out_f.write, header_prefix)
    bytes_written += len(header_prefix)

    idx = 0
    num_specs = len(shard_specs)
    while idx < num_specs:
      spec = shard_specs[idx]
      if spec.nbytes > options.max_in_flight_bytes:
        t = await _open_ts(spec)
        read_chunk_d0 = (
            t.chunk_layout.read_chunk_template.shape[0]
            if spec.shape
            else None
        )
        for ts_idx, _, _, _ in _iter_contiguous_slices(
            spec.shape,
            spec.dtype.itemsize,
            options.max_in_flight_bytes,
            preferred_chunk_dim=read_chunk_d0,
        ):
          view = t[ts_idx] if ts_idx else t
          slice_arr = await view.read()
          keep_alive, mem = _array_to_le_bytes_view(slice_arr)
          await asyncio.to_thread(out_f.write, mem)
          bytes_written += len(mem)
          del mem, keep_alive, slice_arr
        idx += 1
        continue

      # Batch consecutive small tensors up to max_in_flight_bytes for parallel
      # TensorStore reads while preserving sequential shard write order.
      batch: list[_OrbaxTensorSpec] = []
      batch_bytes = 0
      while idx < num_specs and len(batch) < _MAX_CONCURRENT_TENSORS:
        cand = shard_specs[idx]
        if cand.nbytes > options.max_in_flight_bytes:
          break
        if batch and batch_bytes + cand.nbytes > options.max_in_flight_bytes:
          break
        batch.append(cand)
        batch_bytes += cand.nbytes
        idx += 1

      async def _read_whole(s: _OrbaxTensorSpec) -> np.ndarray:
        t = await _open_ts(s)
        return await t.read()

      arrays = await asyncio.gather(*(_read_whole(s) for s in batch))
      for arr in arrays:
        keep_alive, mem = _array_to_le_bytes_view(arr)
        await asyncio.to_thread(out_f.write, mem)
        bytes_written += len(mem)
        del mem, keep_alive
      del arrays

  return bytes_written


async def async_orbax_to_safetensors(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Asynchronously converts an Orbax V0/V1 checkpoint to Safetensors format.

  Streams tensors sequentially into one or more `.safetensors` shards with peak
  host memory bounded by `options.max_in_flight_bytes`.

  Args:
    source_path: Path to an Orbax V0 or V1 checkpoint step directory.
    dest_path: Output directory (or `.safetensors` file path for single-shard
      output).
    options: Conversion options (defaults to `ConversionOptions()`).

  Returns:
    `ConversionResult` with shard count, byte sizes, and elapsed time.
  """
  t0 = time.perf_counter()
  opts = options or ConversionOptions()
  _validate_options(opts)

  clean_src = _clean_path(source_path)
  clean_dst = _clean_path(dest_path)
  if clean_src == clean_dst:
    raise ValueError("source_path and dest_path must be different.")

  src_ep = epath.Path(clean_src)
  dst_ep = epath.Path(clean_dst)

  src_fmt, items, root_custom_meta = await _resolve_orbax_items(src_ep, opts)

  all_specs: list[_OrbaxTensorSpec] = []
  dropped_leaves: list[str] = []
  seen_names: dict[str, str] = {}
  merged_custom_meta = dict(root_custom_meta)

  for prefix, item_dir in items:
    specs, dropped, item_custom = await _plan_orbax_item_tensors(
        prefix, item_dir, opts
    )
    if not merged_custom_meta and item_custom:
      merged_custom_meta.update(item_custom)
    for spec in specs:
      if spec.flat_name in seen_names:
        raise ValueError(
            f"Key collision when flattening with separator"
            f" {opts.key_separator!r}: {spec.flat_name!r} produced by both"
            f" {seen_names[spec.flat_name]!r} and {spec.ts_param_name!r}."
        )
      seen_names[spec.flat_name] = spec.ts_param_name
      all_specs.append(spec)
    dropped_leaves.extend(dropped)

  if not all_specs:
    raise ValueError(f"Orbax checkpoint '{clean_src}' contains 0 tensors.")

  shards = _partition_into_shards(all_specs, opts.max_shard_size_bytes)
  is_file_dest = clean_dst.endswith(SAFETENSORS_SUFFIX)
  if is_file_dest and len(shards) > 1:
    raise ValueError(
        f"Destination '{clean_dst}' is a single '{SAFETENSORS_SUFFIX}' file,"
        f" but checkpoint requires {len(shards)} shards under"
        f" max_shard_size_bytes={opts.max_shard_size_bytes}. Provide a"
        " directory path or set max_shard_size_bytes=0."
    )

  header_meta, warnings = _encode_custom_metadata(merged_custom_meta)
  if dropped_leaves:
    warnings.append(
        f"Dropped {len(dropped_leaves)} non-tensor leaf/leaves:"
        f" {dropped_leaves[:5]}"
    )

  await _prepare_destination(dst_ep, opts.overwrite)
  if is_file_dest:
    if dst_ep.parent:
      await async_path.mkdir(dst_ep.parent, parents=True, exist_ok=True)
  else:
    await async_path.mkdir(dst_ep, parents=True, exist_ok=True)

  total_logical_bytes = sum(s.nbytes for s in all_specs)
  total_stored_bytes = 0

  if len(shards) == 1:
    out_file = (
        dst_ep if is_file_dest else dst_ep / DEFAULT_SINGLE_SHARD_FILENAME
    )
    total_stored_bytes += await _write_safetensors_shard(
        out_file, shards[0], header_meta, opts
    )
  else:
    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    for idx_zero, shard_specs in enumerate(shards):
      shard_name = f"model-{idx_zero + 1:05d}-of-{num_shards:05d}.safetensors"
      for s in shard_specs:
        weight_map[s.flat_name] = shard_name
      total_stored_bytes += await _write_safetensors_shard(
          dst_ep / shard_name, shard_specs, header_meta, opts
      )
    index_payload = {
        "metadata": {"total_size": total_logical_bytes},
        "weight_map": weight_map,
    }
    index_text = json.dumps(index_payload, indent=2, sort_keys=True)
    await async_path.write_text(dst_ep / SAFETENSORS_INDEX_FILENAME, index_text)
    total_stored_bytes += len(index_text.encode("utf-8"))

  return ConversionResult(
      source_path=clean_src,
      dest_path=clean_dst,
      source_format=src_fmt,
      dest_format="safetensors",
      num_tensors=len(all_specs),
      total_logical_bytes=total_logical_bytes,
      total_stored_bytes=total_stored_bytes,
      num_output_shards=len(shards),
      elapsed_secs=round(time.perf_counter() - t0, 4),
      dropped_non_tensor_leaves=tuple(sorted(dropped_leaves)),
      metadata_warnings=tuple(warnings),
  )


async def async_convert_checkpoint(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    target_format: Literal["orbax", "safetensors"] | None = None,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Auto-detects source format and dispatches async checkpoint conversion."""
  clean_src = _clean_path(source_path)
  src_ep = epath.Path(clean_src)
  if not await async_path.exists(src_ep):
    raise FileNotFoundError(f"Source path does not exist: '{clean_src}'")

  if await async_path.is_file(src_ep):
    detected = (
        layout_registry.CheckpointLayoutEnum.SAFETENSORS
        if src_ep.suffix == SAFETENSORS_SUFFIX
        else None
    )
  else:
    root_index = await fs_probe.index_directory(src_ep)
    try:
      detected = await layout_registry.detect_layout(
          src_ep, root_index=root_index
      )
    except checkpoint_layout.InvalidLayoutError:
      detected = None

  if detected == layout_registry.CheckpointLayoutEnum.SAFETENSORS:
    if target_format not in (None, "orbax"):
      raise ValueError(
          f"Source '{clean_src}' is already in Safetensors format; expected"
          f" target_format='orbax', got {target_format!r}."
      )
    return await async_safetensors_to_orbax(
        clean_src, dest_path, options=options
    )

  if detected == layout_registry.CheckpointLayoutEnum.ORBAX:
    if target_format not in (None, "safetensors"):
      raise ValueError(
          f"Source '{clean_src}' is already in Orbax format; expected"
          f" target_format='safetensors', got {target_format!r}."
      )
    return await async_orbax_to_safetensors(
        clean_src, dest_path, options=options
    )

  raise ValueError(
      f"Unrecognized or unsupported source checkpoint format at '{clean_src}'."
      " Expected an Orbax (V0/V1) or Safetensors checkpoint."
  )


def safetensors_to_orbax(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Converts a Safetensors checkpoint to Orbax V1 format."""
  return asyncio_utils.run_sync(
      async_safetensors_to_orbax(source_path, dest_path, options=options)
  )


def orbax_to_safetensors(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Converts an Orbax V0/V1 checkpoint to Safetensors format."""
  return asyncio_utils.run_sync(
      async_orbax_to_safetensors(source_path, dest_path, options=options)
  )


def convert_checkpoint(
    source_path: str | os.PathLike[str],
    dest_path: str | os.PathLike[str],
    *,
    target_format: Literal["orbax", "safetensors"] | None = None,
    options: ConversionOptions | None = None,
) -> ConversionResult:
  """Converts between Safetensors and Orbax checkpoint formats."""
  return asyncio_utils.run_sync(
      async_convert_checkpoint(
          source_path,
          dest_path,
          target_format=target_format,
          options=options,
      )
  )
