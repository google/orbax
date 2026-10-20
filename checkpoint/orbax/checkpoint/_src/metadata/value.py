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

"""User facing metadata describing PyTree values."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from etils import epath
import jax
from jax import numpy as jnp
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.metadata import sharding as sharding_metadata


@dataclasses.dataclass
class Metadata:
  """Metadata describing PyTree values.

  name:
    A string representing the original name of the parameter.
  directory:
    The directory where the parameter can be found, after taking `name` into
    account.
  """

  name: str
  directory: Optional[epath.Path]

  def __eq__(self, other: Metadata) -> bool:  # pyrefly: ignore[bad-override]
    return isinstance(other, Metadata)

  def __repr__(self) -> str:
    return (
        f'{self.__class__.__name__} :'
        f'  name={self.name},'
        f'  directory={self.directory},'
    )


@dataclasses.dataclass(frozen=True)
class Compression:
  """A compression codec; repr is `algorithm[:level]`, e.g. `zstd:1`."""

  algorithm: str
  level: int | None = None

  def __repr__(self) -> str:
    level = '' if self.level is None else f':{self.level}'
    return f'{self.algorithm}{level}'


def _compression(algorithm: Any, config: Any) -> Compression:
  """Builds a `Compression` from a codec name and its JSON config."""
  level = config.get('level') if isinstance(config, dict) else None
  return Compression(str(algorithm), level)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZarrMetadata:
  """Version-independent view of a Zarr array's metadata.

  Attributes:
    zarr_format: Zarr format version (2 or 3), fixed by the subclass.
    shape: Array shape.
    dtype: Zarr dtype string (v2 `"<f4"`, v3 `"float32"` / `"bfloat16"`).
    chunk_shape: Stored chunk shape (v3: the outer shard).
    compression: Compression codec, or None if uncompressed.
  """

  zarr_format: int = dataclasses.field(default=0, init=False)
  shape: tuple[int, ...]
  dtype: str
  chunk_shape: tuple[int, ...]
  compression: Compression | None

  @classmethod
  def from_json(cls, meta: dict[str, Any] | None) -> ZarrMetadata | None:
    """Parses spec JSON into the matching subclass.

    Args:
      meta: The `metadata` member of a TensorStore Zarr spec JSON, or None.

    Returns:
      A `Zarr2Metadata` / `Zarr3Metadata`, or None if `zarr_format` is not 2/3.
    """
    meta = meta or {}
    match meta.get('zarr_format'):
      case 2:
        return Zarr2Metadata.from_json(meta)
      case 3:
        return Zarr3Metadata.from_json(meta)
    return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class Zarr2Metadata(ZarrMetadata):
  """Zarr v2 (`.zarray`) metadata.

  Attributes:
    order: Memory layout of each chunk, 'C' or 'F'.
    dimension_separator: Separator in chunk keys, '.' or '/'.
  """

  zarr_format: int = dataclasses.field(default=2, init=False)
  order: str
  dimension_separator: str

  @classmethod
  def from_json(cls, meta: dict[str, Any] | None) -> Zarr2Metadata:
    """Parses Zarr v2 `.zarray` JSON."""
    meta = meta or {}
    compressor = meta.get('compressor')
    return cls(
        shape=tuple(meta.get('shape', ())),
        dtype=str(meta.get('dtype', '')),
        chunk_shape=tuple(meta.get('chunks', ())),
        compression=(
            _compression(compressor.get('id', 'unknown'), compressor)
            if isinstance(compressor, dict)
            else None
        ),
        order=meta.get('order', 'C'),
        dimension_separator=meta.get('dimension_separator', '.'),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Zarr3Metadata(ZarrMetadata):
  """Zarr v3 (`zarr.json`) metadata.

  Attributes:
    inner_chunk_shape: Read chunk inside each `sharding_indexed` shard, if any.
  """

  zarr_format: int = dataclasses.field(default=3, init=False)
  inner_chunk_shape: tuple[int, ...] | None

  @classmethod
  def from_json(cls, meta: dict[str, Any] | None) -> Zarr3Metadata:
    """Parses Zarr v3 `zarr.json`, unwrapping `sharding_indexed` codecs."""
    meta = meta or {}
    codecs, inner = meta.get('codecs') or [], None
    for codec in codecs:
      if isinstance(codec, dict) and codec.get('name') == 'sharding_indexed':
        config = codec.get('configuration') or {}
        inner = tuple(config.get('chunk_shape') or ()) or None
        codecs = config.get('codecs') or []
        break
    compressors = [
        _compression(c.get('name', ''), c.get('configuration'))
        for c in codecs
        if isinstance(c, dict)
        and c.get('name') not in ('bytes', 'crc32c', 'transpose')
    ]
    grid = (meta.get('chunk_grid') or {}).get('configuration') or {}
    return cls(
        shape=tuple(meta.get('shape', ())),
        dtype=str(meta.get('data_type', '')),
        chunk_shape=tuple(grid.get('chunk_shape', ())),
        compression=compressors[0] if compressors else None,
        inner_chunk_shape=inner,
    )


@dataclasses.dataclass(frozen=True)
class StorageMetadata:
  """Metadata describing how arrays are stored in a checkpoint."""

  chunk_shape: arrays_types.Shape | None
  write_shape: arrays_types.Shape | None = None
  zarr_metadata: ZarrMetadata | None = dataclasses.field(
      default=None, compare=False, repr=False
  )


@dataclasses.dataclass
class ArrayMetadata(Metadata):
  """Metadata describing an array.

  shape:
    Tuple of integers describing the array shape.
  sharding:
    ShardingMetadata to indicate how the array is sharded. ShardingMetadata is
    an orbax representation of `jax.sharding.Sharding` which stores the same
    properties but not require accessing real devices.
  dtype:
    Dtype of array elements.
  storage:
    Optional metadata describing how the array is stored in a checkpoint.
  """

  shape: tuple[int, ...]
  sharding: Optional[sharding_metadata.ShardingMetadata]
  dtype: Optional[jnp.dtype]
  storage: Optional[StorageMetadata] = None

  def __eq__(self, other: Metadata) -> bool:
    return (
        isinstance(other, ArrayMetadata)
        and self.shape == other.shape
        and self.sharding == other.sharding
        and self.dtype == other.dtype
        and self.storage == other.storage
    )

  @classmethod
  def from_shape_dtype_struct(
      cls,
      s: jax.ShapeDtypeStruct,
      name: Optional[str] = None,
      directory: Optional[epath.Path] = None,
  ) -> ArrayMetadata:
    return cls(
        name=name,  # pyrefly: ignore[bad-argument-type]
        directory=directory,
        shape=s.shape,
        sharding=s.sharding,
        dtype=s.dtype,
    )

  def __repr__(self) -> str:
    return super().__repr__() + (
        f'  shape={self.shape},'
        f'  sharding={self.sharding},'
        f'  dtype={self.dtype},'
        f'  storage={self.storage},'
    )


@dataclasses.dataclass
class ScalarMetadata(ArrayMetadata):
  """Metadata describing a scalar value.

  dtype:
    Scalar dtype.
  """

  shape: tuple[int, ...] = tuple([])
  sharding: Optional[sharding_metadata.ShardingMetadata] = None
  dtype: Optional[jnp.dtype] = None

  def __eq__(self, other: Metadata) -> bool:
    return isinstance(other, ScalarMetadata) and self.dtype == other.dtype


@dataclasses.dataclass
class StringMetadata(Metadata):
  """Metadata describing a string value."""

  def __eq__(self, other: Metadata) -> bool:
    return isinstance(other, StringMetadata)
