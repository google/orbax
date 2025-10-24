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

"""TensorStore serialization helper functions."""

import base64
from collections.abc import Sequence
import copy
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
from orbax.checkpoint._src.serialization import types
import tensorstore as ts


JsonSpec: TypeAlias = dict[str, Any]
Shape: TypeAlias = arrays_types.Shape
DType: TypeAlias = arrays_types.DType
ArrayMetadata: TypeAlias = array_metadata.ArrayMetadata
ExtMetadata: TypeAlias = array_metadata.ExtMetadata

FILE_DRIVER = 'file'
DEFAULT_DRIVER = FILE_DRIVER

PROCESS_SUBDIR_PREFIX = 'ocdbt.process_'
REPLICA_SUBDIR_SUFFIX = 'replica_'
_OCDBT_PROCESS_ID_RE = r'[A-Za-z0-9]+'
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2**31  # 2 GiB

ZARR_VER2 = 'zarr'
ZARR_VER3 = 'zarr3'

_GCS_PATH_RE = r'^gs://([^/]*)/(.*)$'

# Even if the data is equal to the fill value, we still want to write it
# to the checkpoint. This results in unnecessary writes in some edge
# cases, but it allows us to verify that data was actually written when
# later restoring.
# Must match `store_data_equal_to_fill_value` property in Orbax
# metadata.
STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE = True


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


def _get_kvstore_for_gcs(ckpt_path: str) -> JsonSpec:
  m = re.fullmatch(_GCS_PATH_RE, ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError(
        'The ckpt_path should contain the bucket name and the '
        f'file path inside the bucket. Got: {ckpt_path}'
    )
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': gcs_bucket, 'path': path_without_bucket}


def build_kvstore_tspec(
    directory: str,
    name: str | None = None,
    *,
    use_ocdbt: bool = True,
    process_id: int | str | None = None,
    replica_separate_folder: bool = False,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore KvStore.

  Args:
    directory: Base path (key prefix) of the KvStore, used by the underlying
      file driver.
    name: Name (filename) of the parameter.
    use_ocdbt: Whether to use OCDBT driver.
    process_id: [only used with OCDBT driver] If provided,
      `{directory}/ocdbt.process_{process_id}` path is used as the base path. If
      a string, must conform to [A-Za-z0-9]+ pattern.
    replica_separate_folder: Whether a replica separated folder is used.

  Returns:
    A Tensorstore KvStore spec in dictionary form.
  """
  default_driver = DEFAULT_DRIVER
  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  directory = os.path.normpath(directory).replace('gs:/', 'gs://')
  is_gcs_path = directory.startswith('gs://')
  kv_spec = {}

  if use_ocdbt:
    if not is_gcs_path and not os.path.isabs(directory):
      raise ValueError(f'Checkpoint path should be absolute. Got {directory}')
    if process_id is not None:
      process_id = str(process_id)
      if re.fullmatch(_OCDBT_PROCESS_ID_RE, process_id) is None:
        raise ValueError(
            f'process_id must conform to {_OCDBT_PROCESS_ID_RE} pattern'
            f', got {process_id}'
        )

      join_paths = [directory, f'{PROCESS_SUBDIR_PREFIX}{process_id}']
      if replica_separate_folder:
        # make sure the the sub dictory is ended with '_process_id'
        join_paths = [
            directory,
            f'{PROCESS_SUBDIR_PREFIX}{REPLICA_SUBDIR_SUFFIX}{process_id}',
        ]
      directory = os.path.join(*join_paths)
    base_driver_spec = (
        directory
        if is_gcs_path
        else {'driver': default_driver, 'path': str(directory)}
    )
    kv_spec.update({
        'driver': 'ocdbt',
        'base': base_driver_spec,
    })
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
  else:
    if name is None:
      path = directory
    else:
      path = os.path.join(directory, name)
    if is_gcs_path:
      kv_spec = _get_kvstore_for_gcs(path)
    else:
      kv_spec = {'driver': default_driver, 'path': path}

  return kv_spec


def build_kvstore_tspec_for_merge(
    directory: str,
    subdir: str,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore KvStore."""

  tokens = subdir.split('_')
  process_id = tokens[-1]
  is_replica_separate_folder = REPLICA_SUBDIR_SUFFIX in subdir
  return build_kvstore_tspec(
      directory,
      use_ocdbt=True,
      process_id=process_id,
      replica_separate_folder=is_replica_separate_folder,
  )


def add_ocdbt_write_options(
    kvstore_tspec: JsonSpec,
    target_data_file_size: int | None = None,
) -> None:
  """Adds write-specific options to a TensorStore OCDBT KVStore spec."""
  if target_data_file_size is not None:
    # TODO: b/354139177 - disallow too small values, too.
    if target_data_file_size < 0:
      raise ValueError(
          'OCDBT target_data_file_size must be >= 0, where 0 means no limit'
          f'; got {target_data_file_size}'
      )
    kvstore_tspec['target_data_file_size'] = target_data_file_size

  kvstore_tspec['config'] = {
      # Store .zarray metadata inline but not large chunks.
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


def build_zarr_shard_and_chunk_metadata(
    *,
    global_shape: Shape,
    shard_shape: Shape,
    use_compression: bool = True,
    use_zarr3: bool,
    chunk_shape: Shape,
) -> JsonSpec:
  """Constructs Zarr metadata for TensorStore array write spec."""
  metadata = {'shape': global_shape}

  if not use_zarr3:
    # Zarr v2.
    metadata['chunks'] = chunk_shape
    if use_compression:
      metadata['compressor'] = {'id': 'zstd'}
    else:
      metadata['compressor'] = None
  else:
    # Zarr v3.
    metadata['chunk_grid'] = {
        'name': 'regular',
        'configuration': {
            'chunk_shape': chunk_shape,
        },
    }
    # TODO: b/354139177 - Consider if using write shape equal to shard shape and
    # read shape equal to chosen chunk shape would be a better setting.
    del shard_shape  # Currently unused.
    metadata['codecs'] = [
        {
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
        },
    ]
    if use_compression:
      # Remove zstd codec if not using compression.
      metadata['codecs'][0]['configuration']['codecs'].append({'name': 'zstd'})

  return metadata


def calculate_chunk_byte_size(
    write_shape: Shape,
    dtype: DType,
    *,
    chunk_byte_size: int | None,
    ocdbt_target_data_file_size: int | None = None,
) -> int | None:
  """Selects chunk byte size to fit both target data file and chunk sizes."""
  # Check if the chunk size would exceed ocdbt target file size.
  if ocdbt_target_data_file_size is None:
    # Set to default used by TensorStore
    # (from https://google.github.io/tensorstore/kvstore/ocdbt/index.html).
    ocdbt_target_data_file_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE

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
  ):
    """Builds a TensorStore spec for writing an array."""
    # Construct the underlying KvStore spec.
    kvstore_tspec = build_kvstore_tspec(
        directory,
        name=relative_array_filename,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
        replica_separate_folder=replica_separate_folder,
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
      add_ocdbt_write_options(
          tspec['kvstore'],
          ocdbt_target_data_file_size,
      )
      chunk_byte_size = calculate_chunk_byte_size(
          write_shape,
          target_storage_dtype,
          chunk_byte_size=chunk_byte_size,
          ocdbt_target_data_file_size=ocdbt_target_data_file_size,
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
    tspec['metadata'] = build_zarr_shard_and_chunk_metadata(
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
    process_index: int | str | None = None,
    metadata_key: str | None = None,
    raise_array_data_missing_error: bool = True,
) -> dict[str, Any]:
  """Gets Tensorstore spec in JSON format."""
  if info.name is None or info.parent_dir is None:
    raise ValueError('Must provide info.name and info.parent_dir.')
  parent_dir = info.parent_dir
  assert parent_dir is not None
  directory = parent_dir.as_posix()
  kvstore_tspec = build_kvstore_tspec(
      directory,
      name=info.name,
      use_ocdbt=use_ocdbt,
      process_id=process_index,
  )

  tspec = {
      'driver': ZARR_VER3 if info.use_zarr3 else ZARR_VER2,
      'kvstore': kvstore_tspec,
      'recheck_cached_data': False,
      'recheck_cached_metadata': False,
      # Raise error if data is missing.
      'fill_missing_data_reads': not raise_array_data_missing_error,
  }
  if metadata_key is not None:
    tspec['metadata_key'] = metadata_key
  return tspec


# TODO: b/354139177 - Rename this to `build_array_tspec_read`.
# Keep the existing name for backward compatibility but mark as deprecated.
def get_json_tspec_read(
    info: types.ParamInfo,
    use_ocdbt: bool,
    metadata_key: str | None = None,
    raise_array_data_missing_error: bool = True,
) -> dict[str, Any]:
  """Gets Tensorstore spec for reading."""
  return _get_json_tspec(
      info,
      use_ocdbt=use_ocdbt,
      metadata_key=metadata_key,
      raise_array_data_missing_error=raise_array_data_missing_error,
  )


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
  tspec = _get_json_tspec(
      info,
      use_ocdbt=use_ocdbt,
      process_index=process_index,
      metadata_key=metadata_key,
  )

  chunk_byte_size = arg.chunk_byte_size if arg else None
  if use_ocdbt:
    ocdbt_target_data_file_size = info.ocdbt_target_data_file_size
    add_ocdbt_write_options(
        tspec['kvstore'],
        ocdbt_target_data_file_size,
    )
    chunk_byte_size = calculate_chunk_byte_size(
        local_shape,
        dtype,
        chunk_byte_size=chunk_byte_size,
        ocdbt_target_data_file_size=ocdbt_target_data_file_size,
    )

  chunk_shape = subchunking.choose_chunk_shape(
      global_shape,
      local_shape,
      dtype,
      chunk_byte_size,
  )

  tspec['metadata'] = build_zarr_shard_and_chunk_metadata(
      global_shape=global_shape,
      shard_shape=local_shape,
      use_compression=info.use_compression,
      use_zarr3=info.use_zarr3,
      chunk_shape=chunk_shape,
  )

  return tspec


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
      use_compression=info.use_compression,
      use_zarr3=info.use_zarr3,
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
