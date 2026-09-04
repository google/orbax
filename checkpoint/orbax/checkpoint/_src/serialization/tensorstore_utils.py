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

"""TensorStore serialization helper functions."""

import base64
from collections.abc import Sequence
import copy
import dataclasses
import enum

import json
import math
import os
import re
from typing import Any, TypeAlias

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import subchunking
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.metadata import array_metadata
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.serialization import ocdbt_process_spec as ocdbt_process_spec_lib
from orbax.checkpoint._src.serialization import types
import tensorstore as ts

JsonSpec: TypeAlias = dict[str, Any]
Shape: TypeAlias = arrays_types.Shape
DType: TypeAlias = arrays_types.DType
ArrayMetadata: TypeAlias = array_metadata.ArrayMetadata
ExtMetadata: TypeAlias = array_metadata.ExtMetadata

OcdbtProcessSpec: TypeAlias = ocdbt_process_spec_lib.OcdbtProcessSpec

FILE_DRIVER = 'file'
DEFAULT_DRIVER = FILE_DRIVER

PROCESS_SUBDIR_PREFIX = ocdbt_process_spec_lib.PROCESS_PREFIX
REPLICA_SUBDIR_SUFFIX = ocdbt_process_spec_lib.REPLICA_SUFFIX

# OCDBT-specific options.

_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2**31  # 2 GiB
_GCS_OCDBT_TARGET_DATA_FILE_SIZE = 400 * 2**20  # 400 MiB
# By default, OCDBT stores both values (i.e. kvstore values, which are
# .zarray metadata files and array chunk data files) and OCDBT B-tree metadata
# (version tree nodes, b-tree nodes and inlined values) in the same data files
# under 'd/' subdirectory. When store_ocdbt_metadata_and_values_separately
# is enabled in ArrayWriteSpec, B-tree metadata will be stored in separate files
# under 'ocdbt_meta/' subdirectory, while values will be stored in files under
# 'ocdbt_data/' subdirectory.
_OCDBT_SPLIT_VALUE_DATA_PREFIX = 'ocdbt_data/'
_OCDBT_SPLIT_META_DATA_PREFIX = 'ocdbt_meta/'
_OCDBT_TMP_METADATA_PREFIX = 'ocdbt_tmp_meta/'

ZARR_VER2 = 'zarr'
ZARR_VER3 = 'zarr3'

_GCS_PATH_RE = r'^gs://([^/]*)(?:/(.*))?$'

# Even if the data is equal to the fill value, we still want to write it
# to the checkpoint. This results in unnecessary writes in some edge
# cases, but it allows us to verify that data was actually written when
# later restoring.
# Must match `store_data_equal_to_fill_value` property in Orbax
# metadata.
STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE = True


# How many TS data files can be buffered at once. We may expect that the number
# of files being concurrently written to be less than this limit.
_BASE_TS_CONTEXT = {
    'file_io_concurrency': {'limit': 128},
}
_DEFAULT_OCDBT_TS_CONTEXT = {
    **_BASE_TS_CONTEXT,
    # Provide cache pool for B-tree nodes to avoid repeated reads.
    # 100MB limit.
    **{'cache_pool#ocdbt': {'total_bytes_limit': 100000000}},
}

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_REMOTE_DRIVER_VALIDATIONS = [
    {'driver': 'gcs', 'path_regex': None},
    {'driver': 'gcs_grpc', 'path_regex': None},
    {'driver': 's3', 'path_regex': None},
]



def get_ts_context(
    *,
    use_ocdbt: bool = True,
    file_io_concurrency_limit: int | None = None,
    data_copy_concurrency_limit: int | None = None,
) -> ts.Context:
  """Creates a TensorStore context object.

  For use with Orbax serialization APIs, or when directly opening a
  `TensorStore` object.

  Args:
    use_ocdbt: Whether to use OCDBT driver. Adds options specific to OCDBT if
      True.
    file_io_concurrency_limit: Optionally overrides the thread pool size for
      file I/O.
    data_copy_concurrency_limit: Optionally overrides the thread pool size for
      compressing and copying data.

  Returns:
    A TensorStore context object.
  """
  context = copy.deepcopy(
      _DEFAULT_OCDBT_TS_CONTEXT if use_ocdbt else _BASE_TS_CONTEXT
  )
  if file_io_concurrency_limit is not None:
    context.setdefault('file_io_concurrency', {})[
        'limit'
    ] = file_io_concurrency_limit
  if data_copy_concurrency_limit is not None:
    context.setdefault('data_copy_concurrency', {})[
        'limit'
    ] = data_copy_concurrency_limit
  return ts.Context(context)


### Building KvStore specs.


@enum.unique
class OcdbtWriteMode(enum.Enum):
  """OCDBT write mode.

  Allows to express whether the target OCDBT KvStore will be written to, so it
  could be configured with appropriate write options.

  Attributes:
    WRITE: Used when writing checkpoint data.
    MERGE: Used for target (parent) KvStore when merging OCDBT metadata from
      all per-process subdirectories.
    COMMIT_TEMPORARY: Used when committing metadata accumulated in a temporary
      metadata directory to its target persistent location.
  """

  WRITE = 'write'
  MERGE = 'merge'
  COMMIT_TEMPORARY = 'commit_temporary'


@dataclasses.dataclass(frozen=True)
class OcdbtKvStoreWriteOptions:
  """Options specific to OCDBT KvStore in writing modes.

  Attributes:
    mode: The OCDBT write mode. Required.
    target_data_file_size: The target data file size for OCDBT KvStore. If not
      set, a default value will be used, based on the underlying storage type.
    store_ocdbt_metadata_and_values_separately: Whether to store OCDBT metadata
      and values separately.
  """

  mode: OcdbtWriteMode
  target_data_file_size: int | None = None
  store_ocdbt_metadata_and_values_separately: bool = False


@dataclasses.dataclass(frozen=True)
class OcdbtTemporaryMetadataContext:
  """Context for handling OCDBT temporary metadata.

  OCDBT kvstore configuration supports storing per-process OCDBT metadata
  (manifest file and B-tree and version tree nodes) in a separate, local
  temporary directory (backed by in-memory file system), which should later be
  committed to the persistent metadata directory. This allows to achieve atomic
  OCDBT metadata writes - especially for manifest files - without having to rely
  on TensorStore transactions.

  Usage (within a single writer process):
    1) create a temporary directory and provide a OcdbtTemporaryMetadataContext
       pointing to it to the TensorStore spec construction APIs (ArrayWriteSpec,
       build_kvstore_tspec with WRITE mode) alongside the main persistent
       directory
    2) write all process-local data to TensorStore
    3) after writing, call `ocdbt_utils.commit_temporary_ocdbt_metadata` to
       atomically commit the metadata from the temporary directory to the
       persistent directory

  Attributes:
    path: The path to the temporary metadata directory. In-memory or local
      filesystem are recommended for performance.
  """
  path: epath.Path


def _get_kvstore_for_gcs(ckpt_path: str) -> JsonSpec:
  """Constructs a TensorStore kvstore spec for a GCS path."""
  m = re.fullmatch(_GCS_PATH_RE, ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError(
        'The ckpt_path should contain the bucket name and the '
        f'file path inside the bucket. Got: {ckpt_path}'
    )
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2) or ''
  # TODO(b/518937340): Consider enabling gcs_grpc by default.
  # TODO(b/518937340): Migrate TENSORSTORE_GCS_BACKEND flag to `Context`.
  gcs_backend = os.environ.get('TENSORSTORE_GCS_BACKEND', 'gcs')
  logging.vlog(
      1, 'Using GCS backend (TENSORSTORE_GCS_BACKEND): %s', gcs_backend
  )
  return {
      'driver': gcs_backend,
      'bucket': gcs_bucket,
      'path': path_without_bucket,
  }


def _normalize_path(path: str) -> str:
  """Normalizes a path, removing trailing slashes."""
  # In GCS case, we need to fix to add back the stripped '/' so that the path
  # remains valid.
  return os.path.normpath(path).replace('gs:/', 'gs://')


@dataclasses.dataclass(frozen=True)
class _OcdbtKvSpecParameters:
  """OCDBT KvStore spec key parameters.

  Attributes:
    base_driver_spec: The spec of the underlying (base) kvstore driver, pointing
      to the target storage path (or using kvstack driver to support separate
      storage of metadata in temporary path and values in the target path)
    manifest_spec_override: [Optional] The manifest spec override of the
      KvStore.
    metadata_prefix_override: [Optional] The metadata prefix override of the
      KvStore. If set, `btree_node_data_prefix` and
      `version_tree_node_data_prefix` will be set to this value.
    value_prefix_override: [Optional] The value prefix override of the KvStore.
      If set, `value_data_prefix` will be set to this value.
  """
  base_driver_spec: JsonSpec
  manifest_spec_override: JsonSpec | str | None = None
  metadata_prefix_override: str | None = None
  value_prefix_override: str | None = None


def _override_ocdbt_kvspec_parameters_for_temporary_metadata(
    temporary_metadata_context: OcdbtTemporaryMetadataContext | None,
    write_mode: OcdbtWriteMode | None,
    current_parameters: _OcdbtKvSpecParameters,
) -> _OcdbtKvSpecParameters:
  """Returns KvStore spec parameters with overrides for temporary metadata."""
  if temporary_metadata_context is None:
    if write_mode == OcdbtWriteMode.COMMIT_TEMPORARY:
      raise ValueError(
          'OCDBT commit mode requires temporary metadata context.'
      )
    return current_parameters

  if write_mode == OcdbtWriteMode.MERGE:
    raise ValueError(
        'OCDBT merge mode does not support temporary metadata context.'
    )

  manifest_spec = current_parameters.manifest_spec_override
  metadata_prefix = current_parameters.metadata_prefix_override

  base_tmp_dir_spec = f'{DEFAULT_DRIVER}://{temporary_metadata_context.path}/'

  # Ensure routing of metadata-related files' writes and reads to the temporary
  # directory. We achieve this by:
  #  1) using the kvstack driver
  #  2) when in writing mode, overriding the metadata prefix to match the prefix
  #     of the layer backed by the temporary directory
  #  3) overriding the manifest spec to point to the temporary metadata
  #     directory (unless in COMMIT_TEMPORARY mode)
  # Notes on COMMIT_TEMPORARY mode (used for copying metadata from temporary
  # to persistent location):
  #   1) we don't set manifest or metadata prefix overrides: this ensures that
  #      the target kvstore is correctly opened as empty initially, and any
  #      writes of metadata are now routed to the layer backed by the persistent
  #      directory
  #   2) kvstack driver's implementation of `experimental_copy_range_to` (used
  #      by `commit_temporary_ocdbt_metadata` to copy metadata to persistent
  #      location) is very strict about the base_driver_spec of the source
  #      and destination kvstores, requiring them to be identical. This defines
  #      how the base_driver_spec is constructed below, to look the same
  #      regardless of the mode used (read or commit).
  if write_mode != OcdbtWriteMode.COMMIT_TEMPORARY:
    manifest_spec = f'{base_tmp_dir_spec}{_OCDBT_TMP_METADATA_PREFIX}'
  if write_mode == OcdbtWriteMode.WRITE:
    metadata_prefix = _OCDBT_TMP_METADATA_PREFIX

  base_driver_spec = {
      'driver': 'kvstack',
      'layers': [
          # Write to the real persistent checkpoint directory by default.
          {'base': current_parameters.base_driver_spec},
          # Per-process metadata is stored in the separate local temporary
          # directory. `prefix` ensures that writes and reads of
          # metadata-related files are routed to the temporary directory.
          {
              'prefix': _OCDBT_TMP_METADATA_PREFIX,
              'base': base_tmp_dir_spec,
          },
      ],
  }

  return dataclasses.replace(
      current_parameters,
      base_driver_spec=base_driver_spec,
      manifest_spec_override=manifest_spec,
      metadata_prefix_override=metadata_prefix,
  )


def _build_ocdbt_kvstore_tspec(
    directory: str,
    name: str | None = None,
    *,
    process_spec: OcdbtProcessSpec | None = None,
    write_options: OcdbtKvStoreWriteOptions | None = None,
    temporary_metadata_context: OcdbtTemporaryMetadataContext | None = None,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore OCDBT KvStore.

  Args:
    directory: Base path (key prefix) of the KvStore, used by the underlying
      file driver.
    name: Name (filename) of the parameter.
    process_spec: OCDBT process spec (defines per-process subdirectory
      name).
    write_options: Options specific to OCDBT KvStore write modes. Should be
      provided when the kvstore will be used for writing or merging.
    temporary_metadata_context: Context for local temporary metadata directory.
      See `OcdbtTemporaryMetadataContext` for more details.

  Returns:
    A Tensorstore KvStore spec in dictionary form.
  """
  directory = _normalize_path(directory)
  is_gcs_path = directory.startswith('gs://')

  if not is_gcs_path and not os.path.isabs(directory):
    raise ValueError(f'Checkpoint path should be absolute. Got {directory}')

  if process_spec is not None:
    directory = os.path.join(directory, str(process_spec))

  # Base KVStore spec (nested within OCDBT KVStore spec).
  if is_gcs_path:
    base_driver_spec = _get_kvstore_for_gcs(directory)
  else:
    base_driver_spec = {
        'driver': DEFAULT_DRIVER,
        'path': str(directory) + '/',  # explicit slash required for kvstack
    }

  # For OCDBT on local filesystems (including GCSFuse), we can safely use
  # non-atomic writes for data files to avoid expensive renames. However,
  # the manifest file still requires atomic writes to avoid corruption.
  # We achieve this by splitting the spec into 'base' (for data files) and
  # 'manifest'.
  try:
    resolved_base_spec = ts.KvStore.Spec(base_driver_spec).to_json()
  except Exception:  # pylint: disable=broad-except
    logging.warning(
        'Failed to resolve base spec %r, falling back to default.',
        base_driver_spec,
        exc_info=True,
    )
    resolved_base_spec = base_driver_spec

  kvspec_params = _OcdbtKvSpecParameters(base_driver_spec=base_driver_spec)

  if (
      write_options is not None
      and write_options.store_ocdbt_metadata_and_values_separately
  ):
    kvspec_params = dataclasses.replace(
        kvspec_params,
        metadata_prefix_override=_OCDBT_SPLIT_META_DATA_PREFIX,
        value_prefix_override=_OCDBT_SPLIT_VALUE_DATA_PREFIX,
    )

  if (
      isinstance(resolved_base_spec, dict)
      and resolved_base_spec.get('driver') == 'file'
  ):
    kvspec_params = dataclasses.replace(
        kvspec_params,
        base_driver_spec={
            **resolved_base_spec,
            'file_io_locking': {'mode': 'non_atomic'},
        },
        manifest_spec_override=resolved_base_spec,
    )

  write_mode = None if write_options is None else write_options.mode
  kvspec_params = _override_ocdbt_kvspec_parameters_for_temporary_metadata(
      temporary_metadata_context=temporary_metadata_context,
      write_mode=write_mode,
      current_parameters=kvspec_params,
  )

  kv_spec = {'driver': 'ocdbt', 'base': kvspec_params.base_driver_spec}

  if kvspec_params.manifest_spec_override is not None:
    kv_spec['manifest'] = kvspec_params.manifest_spec_override
  if kvspec_params.metadata_prefix_override is not None:
    kv_spec['btree_node_data_prefix'] = kvspec_params.metadata_prefix_override
    kv_spec['version_tree_node_data_prefix'] = (
        kvspec_params.metadata_prefix_override
    )
  if kvspec_params.value_prefix_override is not None:
    kv_spec['value_data_prefix'] = kvspec_params.value_prefix_override

  if write_options is not None:
    _add_ocdbt_write_options(
        kv_spec,
        target_data_file_size=write_options.target_data_file_size,
    )

  if name is not None:
    kv_spec['path'] = name

  kv_spec.update({  # pytype: disable=attribute-error
      # References the cache specified in ts.Context.
      'cache_pool': 'cache_pool#ocdbt',
  })

  if is_remote_storage(kv_spec):
    kv_spec.update({  # pytype: disable=attribute-error
        # Enable read coalescing.  This feature merges adjacent read_ops into
        # one, which could reduce I/O ops by a factor of 10. This is
        # especially beneficial for unstacked models.
        'experimental_read_coalescing_threshold_bytes': 1000000,
        'experimental_read_coalescing_merged_bytes': 500000000000,
        'experimental_read_coalescing_interval': '1ms',
    })

  return kv_spec


def _build_non_ocdbt_kvstore_tspec(
    directory: str,
    name: str | None = None,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore KvStore, non-OCDBT."""
  directory = _normalize_path(directory)
  is_gcs_path = directory.startswith('gs://')

  if name is None:
    path = str(directory)
  else:
    path = os.path.join(directory, name)
  if is_gcs_path:
    kv_spec = _get_kvstore_for_gcs(path)
  else:
    kv_spec = {'driver': DEFAULT_DRIVER, 'path': path}

  return kv_spec


def build_kvstore_tspec(
    directory: str,
    name: str | None = None,
    *,
    use_ocdbt: bool = True,
    ocdbt_process_spec: OcdbtProcessSpec | None = None,
    ocdbt_write_options: OcdbtKvStoreWriteOptions | None = None,
    ocdbt_temporary_metadata_context: (
        OcdbtTemporaryMetadataContext | None
    ) = None,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore KvStore.

  Args:
    directory: Base path (key prefix) of the KvStore, used by the underlying
      file driver.
    name: Name (filename) of the parameter.
    use_ocdbt: Whether to use OCDBT driver.
    ocdbt_process_spec: OCDBT process spec (defines per-process subdirectory
      name).
    ocdbt_write_options: Options specific to OCDBT KvStore write modes. Should
      be provided when the kvstore will be used for writing or merging.
    ocdbt_temporary_metadata_context: Context for local temporary metadata
      directory. See `OcdbtTemporaryMetadataContext` for more details.

  Returns:
    A Tensorstore KvStore spec in dictionary form.
  """
  if use_ocdbt:
    return _build_ocdbt_kvstore_tspec(
        directory=directory,
        name=name,
        process_spec=ocdbt_process_spec,
        write_options=ocdbt_write_options,
        temporary_metadata_context=ocdbt_temporary_metadata_context,
    )

  return _build_non_ocdbt_kvstore_tspec(directory=directory, name=name)


def _get_backend_ocdbt_target_data_file_size(
    kvstore_spec: JsonSpec | None,
) -> int:
  """Gets OCDBT target data file size based on kvstore spec."""
  if kvstore_spec is None:
    return _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  base = kvstore_spec.get('base')

  if isinstance(base, str):
    # OCDBT base is generally a string when it's a GCS path.
    if gcs_utils.is_gcs_path(epath.Path(base)):
      return _GCS_OCDBT_TARGET_DATA_FILE_SIZE
  elif isinstance(base, dict):
    # OCDBT base can also be a dict with 'driver' and 'path' keys.
    if base.get('driver') in ('gcs', 'gcs_grpc'):
      return _GCS_OCDBT_TARGET_DATA_FILE_SIZE
    path_str = base.get('path')
    if path_str and gcs_utils.is_gcs_path(epath.Path(path_str)):
      return _GCS_OCDBT_TARGET_DATA_FILE_SIZE

  return _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE


def _add_ocdbt_write_options(
    kvstore_tspec: JsonSpec,
    target_data_file_size: int | None = None,
) -> None:
  """Adds write-specific options to a TensorStore OCDBT KVStore spec."""
  if target_data_file_size is None:
    target_data_file_size = _get_backend_ocdbt_target_data_file_size(
        kvstore_tspec
    )
  # TODO: b/354139177 - Disallow too small values, too.
  if target_data_file_size < 0:
    raise ValueError(
        'OCDBT target_data_file_size must be >= 0, where 0 means no limit'
        f'; got {target_data_file_size}'
    )
  kvstore_tspec['target_data_file_size'] = target_data_file_size

  kvstore_tspec['config'] = {
      # Store .zarray metadata inline but not large chunks.
      # If separate storage for OCDBT metadata is enabled, this will mean that
      # Zarr metadata will be stored in tree metadata files and not in the data
      # files.
      'max_inline_value_bytes': 1024,
      # Large value allows a single root node to support faster traversal.
      'max_decoded_node_bytes': 100000000,
      # There won't be any concurrent writes by multiple machines to the same
      # OCDBT database.  Therefore, we can use the simpler and more efficient
      # single-file manifest format in all cases.
      'manifest_kind': 'single',
  }
  # assume_config avoids writing an initial empty manifest to ensure a
  # consistent configuration, since Orbax never writes to the same OCDBT
  # database concurrently from multiple processes.
  kvstore_tspec.update(assume_config=True)


async def open_kv_store(
    kvstore_tspec: JsonSpec,
    ts_context: ts.Context,
) -> ts.KvStore:
  """Opens a TensorStore KvStore from a spec."""
  return await ts.KvStore.open(
      ts.KvStore.Spec(kvstore_tspec),
      context=ts_context,
  )


### Building Zarr array metadata.


def _build_zarr2_metadata(
    global_shape: Shape,
    chunk_shape: Shape,
    use_compression: bool,
) -> JsonSpec:
  """Constructs Zarr v2 metadata."""
  # Use default level 1 straight from TensorStore.
  compressor = {'id': 'zstd', 'level': 1} if use_compression else None
  return {
      'shape': global_shape,
      'chunks': chunk_shape,
      'compressor': compressor,  # pyrefly: ignore[bad-assignment]
  }


def _build_zarr3_metadata(
    global_shape: Shape,
    chunk_shape: Shape,
    use_compression: bool,
) -> JsonSpec:
  """Constructs Zarr v3 metadata."""
  codecs: list[JsonSpec] = [{
      'name': 'sharding_indexed',
      'configuration': {
          'chunk_shape': chunk_shape,
          'codecs': [
              {'name': 'bytes', 'configuration': {'endian': 'little'}},
          ],
          'index_codecs': [
              {'name': 'bytes', 'configuration': {'endian': 'little'}},
              {'name': 'crc32c'},
          ],
          'index_location': 'end',
      },
  }]
  if use_compression:
    # Use default level 3 straight from TensorStore.
    codecs[0]['configuration']['codecs'].append(
        {'name': 'zstd', 'configuration': {'level': 3}}
    )  # pyrefly: ignore[bad-index]

  return {
      'shape': global_shape,
      'chunk_grid': {  # pyrefly: ignore[bad-assignment]
          'name': 'regular',
          'configuration': {'chunk_shape': chunk_shape},
      },
      'codecs': codecs,  # pyrefly: ignore[bad-assignment]
  }


def _build_zarr_shard_and_chunk_metadata(
    *,
    global_shape: Shape,
    shard_shape: Shape,
    use_compression: bool = True,
    use_zarr3: bool,
    chunk_shape: Shape,
) -> tuple[JsonSpec, str, int | None]:
  """Constructs Zarr metadata for TensorStore array write spec."""
  # TODO: b/354139177 - Consider if using write shape equal to shard shape and
  # read shape equal to chosen chunk shape would be a better setting.
  del shard_shape  # Currently unused.
  if use_zarr3:
    metadata = _build_zarr3_metadata(global_shape, chunk_shape, use_compression)
    level = 3 if use_compression else None
  else:
    metadata = _build_zarr2_metadata(global_shape, chunk_shape, use_compression)
    level = 1 if use_compression else None
  algo = 'zstd' if use_compression else 'none'
  return metadata, algo, level


def calculate_chunk_byte_size(
    write_shape: Shape,
    dtype: DType,
    *,
    chunk_byte_size: int | None,
    ocdbt_target_data_file_size: int | None = None,
    kvstore_spec: JsonSpec | None = None,
) -> int | None:
  """Selects chunk byte size to fit both target data file and chunk sizes."""
  # Check if the chunk size would exceed ocdbt target file size.
  if ocdbt_target_data_file_size is None:
    ocdbt_target_data_file_size = _get_backend_ocdbt_target_data_file_size(
        kvstore_spec
    )

  if ocdbt_target_data_file_size == 0:
    # No limit.
    return chunk_byte_size

  if chunk_byte_size is None:
    write_nbytes = math.prod(write_shape) * dtype.itemsize
    if write_nbytes > ocdbt_target_data_file_size:
      chunk_byte_size = ocdbt_target_data_file_size
    else:
      # Let chunk_byte_size stay None.
      chunk_byte_size = None
  else:
    chunk_byte_size = min(chunk_byte_size, ocdbt_target_data_file_size)
  return chunk_byte_size


### Building TensorStore array specs.


def _maybe_add_cast_to_write_spec(
    array_tspec: JsonSpec,
    *,
    dtype: DType,
    target_dtype: DType,
) -> JsonSpec:
  """Adds cast driver to a write array TensorStore spec, if needed."""
  if target_dtype == dtype:
    array_tspec['dtype'] = jnp.dtype(dtype).name
    return array_tspec

  array_tspec = {
      'base': array_tspec,
      'driver': 'cast',
  }
  # Origin dtype.
  array_tspec['dtype'] = jnp.dtype(dtype).name
  # Destination dtype.
  array_tspec['base']['dtype'] = jnp.dtype(target_dtype).name
  return array_tspec


def _maybe_add_cast_to_read_spec(
    array_tspec: JsonSpec,
    *,
    dtype: DType,
) -> JsonSpec:
  """Adds cast driver to a read array TensorStore spec, if needed."""
  if not jax.dtypes.issubdtype(
      dtype, jax.dtypes.prng_key
  ):
    array_tspec = {
        'base': array_tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(dtype).name,
    }
  return array_tspec


class ArrayReadSpec:
  """Full TensorStore spec for reading an array."""

  def __init__(
      self,
      directory: str,
      relative_array_filename: str,
      use_zarr3: bool,
      *,
      use_ocdbt: bool,
      metadata_key: str | None = None,
      raise_array_data_missing_error: bool = True,
      target_dtype: DType | None = None,
  ):
    """Builds a TensorStore spec for reading an array."""
    kvstore_tspec = build_kvstore_tspec(
        directory,
        name=relative_array_filename,
        use_ocdbt=use_ocdbt,
    )

    tspec = {
        'driver': ZARR_VER3 if use_zarr3 else ZARR_VER2,
        'kvstore': kvstore_tspec,
        'recheck_cached_data': False,
        'recheck_cached_metadata': False,
        # Raise error if data is missing.
        'fill_missing_data_reads': not raise_array_data_missing_error,
    }
    if metadata_key is not None:
      tspec['metadata_key'] = metadata_key
    if target_dtype is not None:
      tspec = _maybe_add_cast_to_read_spec(
          tspec,
          dtype=target_dtype,
      )
    self._json_spec = tspec

  @property
  def json(self) -> JsonSpec:
    """Spec to be used to open a TensorStore for reading the array."""
    return self._json_spec


class ArrayWriteSpec:
  """Full TensorStore spec for writing an array."""

  def __init__(
      self,
      directory: str,
      relative_array_filename: str,
      *,
      global_shape: Shape,
      write_shape: Shape,
      dtype: DType,
      target_dtype: DType | None = None,
      chunk_byte_size: int | None = None,
      shard_axes: tuple[int, ...] = (),
      use_compression: bool = True,
      use_zarr3: bool = False,
      use_ocdbt: bool,
      ocdbt_target_data_file_size: int | None = None,
      process_id: int | str | None = None,
      metadata_key: str | None = None,
      replica_separate_folder: bool = False,
      ext_metadata: ExtMetadata | None = None,
      store_ocdbt_metadata_and_values_separately: bool = False,
      ocdbt_temporary_metadata_context: (
          OcdbtTemporaryMetadataContext | None
      ) = None,
  ):
    """Builds a TensorStore spec for writing an array."""
    # Construct the underlying KvStore spec.
    ocdbt_process_spec = None
    if process_id is not None:
      ocdbt_process_spec = OcdbtProcessSpec(
          process_id=str(process_id),
          use_replica_suffix=replica_separate_folder,
      )
    kvstore_tspec = build_kvstore_tspec(
        directory,
        name=relative_array_filename,
        use_ocdbt=use_ocdbt,
        ocdbt_process_spec=ocdbt_process_spec,
        ocdbt_write_options=OcdbtKvStoreWriteOptions(
            mode=OcdbtWriteMode.WRITE,
            target_data_file_size=ocdbt_target_data_file_size,
            store_ocdbt_metadata_and_values_separately=(
                store_ocdbt_metadata_and_values_separately
            ),
        ),
        ocdbt_temporary_metadata_context=ocdbt_temporary_metadata_context,
    )
    # Construct the top-level array spec.
    tspec = {
        'driver': ZARR_VER3 if use_zarr3 else ZARR_VER2,
        'kvstore': kvstore_tspec,
        'recheck_cached_data': False,
        'recheck_cached_metadata': False,
        'store_data_equal_to_fill_value': STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE,
    }
    if metadata_key is not None:
      tspec['metadata_key'] = metadata_key

    target_storage_dtype = target_dtype or dtype

    # Choose target file and chunk byte sizes.
    if use_ocdbt:
      chunk_byte_size = calculate_chunk_byte_size(
          write_shape,
          target_storage_dtype,
          chunk_byte_size=chunk_byte_size,
          ocdbt_target_data_file_size=ocdbt_target_data_file_size,
          kvstore_spec=tspec['kvstore'],
      )
    # Choose chunk shape.
    chunk_shape = subchunking.choose_chunk_shape(
        global_shape,
        write_shape,
        target_storage_dtype,
        chunk_byte_size,
        shard_axes=shard_axes,
    )
    if chunk_shape != write_shape:
      logging.info(
          'Array name: %r, global shape: %r, write shape: %r, chosen chunk'
          ' shape: %r',
          relative_array_filename,
          global_shape,
          write_shape,
          chunk_shape,
      )
    # Construct Zarr chunk metadata.
    tspec['metadata'], algo, level = _build_zarr_shard_and_chunk_metadata(
        global_shape=global_shape,
        shard_shape=write_shape,
        use_compression=use_compression,
        use_zarr3=use_zarr3,
        chunk_shape=chunk_shape,
    )

    # Keep the metadata in a separate field.
    self._metadata = ArrayMetadata(
        param_name=relative_array_filename,
        shape=global_shape,
        dtype=target_storage_dtype,
        write_shape=write_shape,
        chunk_shape=chunk_shape,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        ext_metadata=ext_metadata,
        compression_algorithm=algo,
        compression_level=level,
    )
    # Wrap spec into `cast` driver if needed, and keep it in a separate field.
    self._json_spec = _maybe_add_cast_to_write_spec(
        tspec,
        dtype=dtype,
        target_dtype=target_storage_dtype,
    )

  @property
  def json(self) -> JsonSpec:
    """Spec to be used to open a TensorStore for writing the array."""
    return self._json_spec

  @property
  def metadata(self) -> ArrayMetadata:
    """Checkpoint-relevant TensorStore metadata of the array."""
    return self._metadata


def is_remote_storage(tspec: dict[str, Any] | str) -> bool:
  """Detect if user is using remote storages.

  This can detect common defines and unable to detect some corner cases such as
  using gcsfuse.

  Args:
    tspec: Tensorstore spec.

  Returns:
    True if the spec is using remote storage.
  """
  if isinstance(tspec, str):
    # KvStoreUrl
    if re.match(rf'^({"|".join(_REMOTE_URL_PREFIXES)})', tspec):
      return True
    else:
      return False

  for key in ('base', 'kvstore'):
    if key in tspec:
      return is_remote_storage(tspec[key])

  if 'driver' in tspec:
    for rule in _REMOTE_DRIVER_VALIDATIONS:
      if tspec['driver'] == rule['driver']:
        if rule['path_regex'] is None:
          return True

        # check if path matches the regex.
        if re.match(rule['path_regex'], tspec['path']):
          return True

  return False


def get_sharding_tensorstore_spec(
    directory: str, param_name: str
) -> dict[str, Any]:
  kvstore_tspec = build_kvstore_tspec(
      directory, name='_sharding', use_ocdbt=False
  )
  param_name = base64.urlsafe_b64encode(param_name.encode()).decode('utf-8')
  return {
      'driver': 'json',
      'kvstore': kvstore_tspec,
      'json_pointer': f'/{param_name}',
  }


async def assert_parameter_files_exist(
    param_dir: epath.Path, metadata_key: str | None, use_zarr3: bool = False
):
  """Checks for existence of parameter subdir and .zarray file."""
  exists = await async_path.exists(param_dir)
  if not exists:
    raise FileNotFoundError(
        f'Individual parameter subdirectory not found at path: {param_dir}.'
    )
  if metadata_key is None:
    metadata_key = 'zarr.json' if use_zarr3 else '.zarray'
  metadata_path = param_dir / metadata_key
  exists = await async_path.exists(metadata_path)
  if not exists:
    raise FileNotFoundError(
        f'File not found: {metadata_path}. In many cases, this results from'
        ' copying a checkpoint without using the `-a` flag.'
    )


# TS functions
def _get_json_tspec(
    info: types.ParamInfo,
    use_ocdbt: bool,
    *,
    metadata_key: str | None = None,
    raise_array_data_missing_error: bool = True,
) -> dict[str, Any]:
  """Gets Tensorstore spec in JSON format."""
  return build_array_read_spec(
      info,
      use_ocdbt=use_ocdbt,
      metadata_key=metadata_key,
      raise_array_data_missing_error=raise_array_data_missing_error,
  ).json


# TODO: b/354139177 - Rename this to `build_array_tspec_read`.
# Keep the existing name for backward compatibility but mark as deprecated.
def get_json_tspec_read(
    info: types.ParamInfo,
    use_ocdbt: bool,
    metadata_key: str | None = None,
    raise_array_data_missing_error: bool = True,
) -> dict[str, Any]:
  """Gets Tensorstore spec for reading."""
  return build_array_read_spec(
      info,
      use_ocdbt=use_ocdbt,
      metadata_key=metadata_key,
      raise_array_data_missing_error=raise_array_data_missing_error,
  ).json


# TODO: b/354139177 - Replace usages of this with `build_array_tspec_write`
# and remove it.
def get_json_tspec_write(
    info: types.ParamInfo,
    use_ocdbt: bool,
    global_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    dtype: jnp.dtype | np.dtype,
    process_index: int | str | None = None,
    metadata_key: str | None = None,
    arg: types.SaveArgs | None = None,
) -> dict[str, Any]:
  """Gets Tensorstore spec for writing."""
  return build_array_write_spec(
      info,
      arg=arg,
      global_shape=global_shape,
      local_shape=local_shape,
      dtype=dtype,
      use_ocdbt=use_ocdbt,
      process_index=process_index,
      metadata_key=metadata_key,
  ).json


def build_array_read_spec(
    info: types.ParamInfo,
    *,
    use_ocdbt: bool,
    metadata_key: str | None = None,
    raise_array_data_missing_error: bool = True,
    target_dtype: DType | None = None,
) -> ArrayReadSpec:
  """Gets ArrayReadSpec for reading."""
  if info.name is None or info.parent_dir is None:
    raise ValueError('Must provide info.name and info.parent_dir.')
  return ArrayReadSpec(
      directory=info.parent_dir.as_posix(),
      relative_array_filename=info.name,
      use_zarr3=info.use_zarr3,  # pyrefly: ignore[bad-argument-type]
      use_ocdbt=use_ocdbt,
      metadata_key=metadata_key,
      raise_array_data_missing_error=raise_array_data_missing_error,
      target_dtype=target_dtype,
  )


def build_array_write_spec(
    info: types.ParamInfo,
    arg: types.SaveArgs | None = None,
    *,
    global_shape: arrays_types.Shape,
    local_shape: arrays_types.Shape,
    dtype: jnp.dtype | np.dtype,
    use_ocdbt: bool,
    process_index: int | str | None = None,
    replica_separate_folder: bool = False,
    metadata_key: str | None = None,
    ext_metadata: dict[str, Any] | None = None,
) -> ArrayWriteSpec:
  """Gets ArrayWriteSpec for writing."""
  if info.name is None or info.parent_dir is None:
    raise ValueError('Must provide info.name and info.parent_dir.')
  parent_dir = info.parent_dir
  assert parent_dir is not None
  directory = parent_dir.as_posix()

  return ArrayWriteSpec(
      directory,
      relative_array_filename=info.name,
      global_shape=global_shape,
      write_shape=local_shape,
      dtype=dtype,
      target_dtype=(arg.dtype if arg is not None else None),
      chunk_byte_size=(arg.chunk_byte_size if arg is not None else None),
      shard_axes=(arg.shard_axes if arg is not None else tuple()),
      use_compression=info.use_compression,  # pyrefly: ignore[bad-argument-type]
      use_zarr3=info.use_zarr3,  # pyrefly: ignore[bad-argument-type]
      use_ocdbt=use_ocdbt,
      process_id=process_index,
      replica_separate_folder=replica_separate_folder,
      ocdbt_target_data_file_size=info.ocdbt_target_data_file_size,
      metadata_key=metadata_key,
      ext_metadata=ext_metadata,
  )


def get_cast_tspec_serialize(
    tspec: dict[str, Any], value: Any, args: types.SaveArgs
) -> dict[str, Any]:
  """Creates a Tensorstore spec for casting a param during serialize."""
  tspec = {
      'base': tspec,
      'driver': 'cast',
  }
  # Origin dtype.
  tspec['dtype'] = jnp.dtype(value.dtype).name
  # Destination dtype.
  if args.dtype is None:
    tspec['base']['dtype'] = jnp.dtype(value.dtype).name
  else:
    tspec['base']['dtype'] = jnp.dtype(args.dtype).name
  return tspec


def get_cast_tspec_deserialize(
    tspec: dict[str, Any], args: types.RestoreArgs
) -> dict[str, Any]:
  """Creates a Tensorstore spec for casting a param during deserialize."""

  # Cast is not needed dtype is None or JAX random key type
  if args.dtype is not None and not jax.dtypes.issubdtype(
      args.dtype, jax.dtypes.prng_key
  ):
    tspec = {
        'base': tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(args.dtype).name,
    }
  return tspec


def array_metadata_from_tensorstore(
    t: Any,
    info: types.ParamInfo,
    sharding: sharding_metadata.ShardingMetadata | None = None,
) -> value_metadata.ArrayMetadata:
  return value_metadata.ArrayMetadata(
      name=info.name,
      directory=info.parent_dir,
      shape=t.shape,
      dtype=jnp.dtype(t.dtype.name),
      sharding=sharding,
      storage=value_metadata.StorageMetadata(
          chunk_shape=t.chunk_layout.read_chunk_template.shape,
          write_shape=info.write_shape,
      ),
  )


def get_total_bytes_from_tensorstore(
    metrics: Sequence[dict[str, Any]], direction: types.IoDirection
) -> int:
  """Sums bytes_read or bytes_written from all kvstore drivers in metrics."""
  total = 0
  if direction == types.IoDirection.WRITE:
    suffix = '/bytes_written'
  elif direction == types.IoDirection.READ:
    suffix = '/bytes_read'
  else:
    raise ValueError(f'Invalid direction: {direction}')

  for m in metrics:
    if not isinstance(m, dict):
      continue
    name = m.get('name', '')
    if name.startswith('/tensorstore/kvstore/') and name.endswith(suffix):
      for val in m.get('values', []):
        if isinstance(val, dict):
          total += val.get('value', 0)
  return total


def get_tensorstore_raw_bytes_delta(
    initial_metrics: Sequence[dict[str, Any]] | None,
    final_metrics: Sequence[dict[str, Any]] | None,
    direction: types.IoDirection = types.IoDirection.WRITE,
) -> int:
  """Computes transferred raw bytes delta between two metric snapshots."""
  if initial_metrics is None or final_metrics is None:
    return 0
  try:
    initial_bytes = get_total_bytes_from_tensorstore(initial_metrics, direction)
    final_bytes = get_total_bytes_from_tensorstore(final_metrics, direction)
    return max(0, final_bytes - initial_bytes)
  except Exception:  # pylint: disable=broad-except
    logging.exception('Failed to compute TensorStore raw bytes delta.')
    return 0


def collect_tensorstore_metrics() -> Sequence[dict[str, Any]] | None:
  """Safely collects TensorStore driver metrics."""
  try:
    return ts.experimental_collect_matching_metrics('/tensorstore')
  except Exception:  # pylint: disable=broad-except
    return None


def resolve_compression_settings(
    metadatas: Sequence[ArrayMetadata],
) -> tuple[str, str]:
  """Extracts (algo, level) across array metadata."""
  if not metadatas:
    return ('none', 'None')
  compression_settings = {
      (m.compression_algorithm, m.compression_level) for m in metadatas
  }
  if len(compression_settings) == 1:
    # this should be rare.
    algo, level = next(iter(compression_settings))
    return (str(algo), str(level))
  return ('mixed', 'mixed')


def print_ts_debug_data(key: str | None, infos: Sequence[types.ParamInfo]):
  """Log Tensorstore related metrics."""
  ts_metrics = ts.experimental_collect_matching_metrics('/tensorstore')
  ts_metrics += ts.experimental_collect_matching_metrics('/mallocz')
  ts_metrics += ts.experimental_collect_matching_metrics('/tcmalloc/')
  ts_metrics += [
      {'key': key},
      {'infos': [f'{info.name}' for info in infos]},
  ]

  for metrics in ts_metrics:
    logging.vlog(1, 'ts_metric: %s', metrics)

  return json.dumps(ts_metrics)
