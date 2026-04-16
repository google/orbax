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

"""Script for batch-converting Safetensors to native Orbax layout."""

import asyncio
import collections
import concurrent
from concurrent import futures
import functools
import json
import time
from typing import Any, Dict, Sequence, Tuple, cast

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
import ml_dtypes
import numpy as np
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental import v1 as ocp_v1
from orbax.checkpoint.experimental.v1._src.context import context
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types
import safetensors.numpy
from tensorflow.io import gfile


Path = types.Path
ThreadPoolExecutor = futures.ThreadPoolExecutor


_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Directory containing Safetensors files.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to save the converted Orbax checkpoint.'
)
_MAX_BATCH_SIZE_MB = flags.DEFINE_integer(
    'max_batch_size_mb', 10240, 'Maximum size of a single batch in MB.'
)
_WORKERS = flags.DEFINE_integer(
    'workers',
    1,
    'Number of worker threads to use for concurrent processing of batches.',
)
_THREADS = flags.DEFINE_integer(
    'threads',
    1,
    'Maximum number of threads to use for concurrent processing of batches.',
)
_USE_FILE_LOGGER_ONLY = flags.DEFINE_boolean(
    'use_file_logger_only',
    True,
    'If True, only logs from this file are printed to avoid excessive logging '
    'from other modules. If False, enables global INFO logging.',
)
_SAVING_ENABLED = flags.DEFINE_boolean(
    'saving_enabled',
    False,
    'If True, saves the converted Orbax checkpoint to the output directory.',
)
_OFFICIAL_LOAD_ENABLED = flags.DEFINE_boolean(
    'official_load_enabled',
    False,
    'If True, loads the same checkpoint using official Safetensors library.',
)


class _PathAwaitingCreation(types.PathAwaitingCreation):
  """PathAwaitingCreation for Orbax Layout."""

  def __init__(self, p):
    self._path = epath.Path(p)

  def __truediv__(self, other) -> types.PathAwaitingCreation:
    return _PathAwaitingCreation(self._path / other)

  @property
  def path(self) -> types.Path:
    return self._path

  async def await_creation(self) -> types.Path:
    await async_path.mkdir(self._path, parents=True, exist_ok=True)
    return self._path


def _log_info(msg: str, *args):
  """Designated method for internal logging with volume control."""
  if _USE_FILE_LOGGER_ONLY.value:
    print('INFO: ' + (msg % args if args else msg))
  else:
    logging.info(msg, *args)


def _log_error(msg: str, *args):
  """Designated method for internal error logging with volume control."""
  if _USE_FILE_LOGGER_ONLY.value:
    print('ERROR: ' + str(msg % args if args else msg))
  else:
    logging.error(msg, *args)


async def create_loading_plan(
    paths: Sequence[Path],
    abstract_pytree: dict[str, Any] | None = None,
) -> Tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
  """Creates a plan for loading tensors in batches based on memory limits."""

  async def _fetch_header(p):
    h, _ = await safetensors_layout._SingleFileLoader(p).read_header()  # pylint: disable=protected-access
    return p, h

  if abstract_pytree is None:
    abstract_pytree = {}
    total_tensors = 0
    for file_path in paths:
      _, header = await _fetch_header(file_path)
      for tensor in header:
        if tensor != '__metadata__':
          total_tensors += 1
          shape, dtype = safetensors_layout._get_array_properties(header[tensor])  # pylint: disable=protected-access
          abstract_pytree[tensor] = jax.ShapeDtypeStruct(
              shape=shape,
              dtype=dtype,
          )

  all_headers = await asyncio.gather(*[_fetch_header(p) for p in paths])
  batches = []
  batches_size = []
  current_batch = {}
  current_batch_size = 0
  ctx = context.get_context()
  max_bytes = ctx.memory_options.read_concurrent_bytes
  if max_bytes is None:
    max_bytes = float('inf')
  total_tensors = 0
  for _, header in all_headers:
    for tensor_name, leaf_meta in header.items():
      if tensor_name in abstract_pytree:
        total_tensors += 1
        if 'shape' in leaf_meta and 'dtype' in leaf_meta:
          shape, dtype = safetensors_layout._get_array_properties(leaf_meta)  # pylint: disable=protected-access
          dtype_size = np.dtype(dtype).itemsize
          size = np.prod(shape) * dtype_size
          if current_batch_size + size > max_bytes:
            logging.info(
                'Batch size: %.2f MB', current_batch_size / (1024 * 1024)
            )
            total_tensors = 0
            batches.append(current_batch)
            batches_size.append(current_batch_size / (1024 * 1024))
            current_batch = {}
            current_batch_size = 0
          current_batch[tensor_name] = jax.ShapeDtypeStruct(
              shape=shape,
              dtype=dtype,
          )
          current_batch_size += size
  if current_batch:
    logging.info('Batch size: %.2f MB', current_batch_size / (1024 * 1024))
    batches.append(current_batch)
    batches_size.append(current_batch_size / (1024 * 1024))
  return batches, batches_size, abstract_pytree


async def get_tensor_to_path_indexing(paths: Sequence[Path]) -> dict[str, Path]:
  """Returns a mapping from tensor name to safetensors file."""

  file_to_path = {}
  for file_ in paths:
    file_to_path[str(Path(file_).name)] = file_

  path_ = Path(str(paths[0].parent) + '/model.safetensors.index.json')

  tensor_to_path = {}
  if not await async_path.exists(path_):
    for path in paths:
      header, _ = await safetensors_layout._SingleFileLoader(path).read_header()  # pylint: disable=protected-access
      for name in header:
        if name == '__metadata__':
          continue
        if name in tensor_to_path:
          raise ValueError(f'Duplicate tensor {name} found in multiple files.')
        tensor_to_path[name] = path
    return tensor_to_path
  async with async_path.open_file(path_, mode='rb') as f:
    raw_data = await f.read()
    index_data = json.loads(raw_data)

  for name, path in index_data['weight_map'].items():
    if name in tensor_to_path:
      raise ValueError(f'Duplicate tensor {name} found in multiple files.')
    tensor_to_path[name] = file_to_path[str(path)]
  return tensor_to_path


def analyze_model_structure(metadata_tree: Any) -> None:
  """Logs detailed statistics about the model structure and expected size."""
  flat_metadata, _ = jax.tree_util.tree_flatten_with_path(metadata_tree)

  total_params = 0
  total_bytes = 0
  dtype_counts = {}
  total_tensors = 0

  for _, leaf in flat_metadata:
    # Get path string (e.g., "model/layers/0/self_attn/q_proj")
    if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype'):
      # Calculate size for this specific tensor
      shape = leaf.shape
      dtype = leaf.dtype
      param_count = np.prod(shape)
      byte_size = param_count * np.dtype(dtype).itemsize

      # Update totals
      total_params += param_count
      total_bytes += byte_size
      total_tensors += 1

      # Track dtype distribution
      dtype_str = str(dtype)
      dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

  total_gb = total_bytes / (1024**3)
  total_mb = total_bytes / (1024**2)

  _log_info('Total Tensors count: %d', total_tensors)
  _log_info('Total Parameters: %s', f'{total_params:,}')
  _log_info(
      'Expected Raw Size (Uncompressed): %.2f MB (%.4f GB)',
      total_mb,
      total_gb,
  )
  _log_info('Dtype Distribution: %s', dtype_counts)


def benchmark_official_safetensors(input_path: epath.Path):
  """Benchmarks loading checkpoints from GCS using the standard Safetensors library.

  It simulates a real-world scenario: Download -> RAM -> Parse.

  Args:
    input_path: The GCS path to the Safetensors files.
  """
  safetensor_blobs = list(input_path.glob('*.safetensors'))

  if not safetensor_blobs:
    _log_error('❌ No .safetensors files found in %s', input_path)
    return

  _log_info('Found %d files. Starting load...', len(safetensor_blobs))

  # --- THE WORKER FUNCTION ---
  def _load_single_file(file_path: str):
    """Uses safe_open for memory-mapped loading (Standard for local files).

    Args:
      file_path: The path to the safetensors file.

    Returns:
      The number of tensors in the file.
    """
    file_path_str = str(file_path)
    gcs_file_path_str = None
    if file_path_str.startswith('gs://'):
      gcs_file_path_str = '/gcs/' + file_path_str[5:]
    elif file_path_str.startswith('/big' + 'store/'):
      gcs_file_path_str = '/gcs/' + file_path_str[10:]

    start_time = time.time()
    _log_info('Loading path %s', gcs_file_path_str)
    try:
      if 'BF16' not in safetensors.numpy._TYPES:  # pylint: disable=protected-access
        safetensors.numpy._TYPES['BF16'] = ml_dtypes.bfloat16  # pylint: disable=protected-access

      with open(gcs_file_path_str, 'rb') as f:
        content = f.read()
      tensors = safetensors.numpy.load(content)
      del content
      end_time = time.time()
      _log_info(
          'Loaded %d tensors from %s in %.2f seconds',
          len(tensors),
          gcs_file_path_str,
          end_time - start_time,
      )
      return len(tensors), tensors
    except (OSError, safetensors.SafetensorError) as e:
      print(f'Failed to load {gcs_file_path_str}: {e}')
      return 0, {}

  # --- START BENCHMARK ---
  start_time = time.time()
  total_tensors = 0
  total_size = 0.0
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_THREADS.value
  ) as pool:
    results = pool.map(_load_single_file, safetensor_blobs)
    for i, r in enumerate(results, 1):
      count, tensors = r
      total_tensors += count
      total_size += sum(t.nbytes for t in tensors.values()) / (1024 * 1024)
      del tensors
      if i % 10 == 0:
        _log_info('- Loaded %d/%d files...', i, len(safetensor_blobs))

  end_time = time.time()
  total_time = end_time - start_time

  # --- REPORT ---
  _log_info('-' * 60)
  _log_info(
      '✅ Official Load Time:%s seconds for %d files and %d tensors | Size:'
      ' %.2f MB | Throughput: %.2f MB/s',
      total_time,
      len(safetensor_blobs),
      total_tensors,
      total_size,
      total_size / total_time,
  )
  _log_info('-' * 60)


async def _execute_batch_async(
    layout: safetensors_layout.SafetensorsLayout,
    file_abstract_trees: dict[str, Any],
    batch_index: int,
    num_batches: int,
    batch_size: float,
) -> Tuple[dict[str, Any], int]:
  """Loads and saves a single batch."""
  load_start_time = time.time()
  _log_info(
      '\033[1m[VM %d] Processing batch %d/%d...\033[0m',
      jax.process_index(),
      batch_index + 1,
      num_batches,
  )
  _log_info(
      '[VM %d] [Batch %d] Loading into Host RAM from Safetensors, starting at'
      ' %.2f seconds, total files: %d | Batch Size: %.2f MB',
      jax.process_index(),
      batch_index + 1,
      load_start_time,
      len(file_abstract_trees),
      batch_size,
  )
  loaded_tensors = {}
  for path, abstract_pytree in file_abstract_trees.items():
    _log_info(
        '[VM %d] [Batch %d] Loading path %s with %d tensors',
        jax.process_index(),
        batch_index + 1,
        path,
        len(abstract_pytree),
    )
    current_load_start_time = time.time()
    current_loaded_tensors = await layout.load_pytree(
        path=Path(path), abstract_pytree=abstract_pytree
    )
    current_loaded_tensors = await current_loaded_tensors
    current_loaded_tensors = cast(Dict[str, Any], current_loaded_tensors)
    loaded_tensors.update(current_loaded_tensors)
    current_load_size = sum(t.nbytes for t in current_loaded_tensors.values())
    _log_info(
        '[VM %d] [Batch %d] Loaded path %s with %d tensors | Load Time: %.2f'
        ' seconds | Load Size: %.2f MB'
        ' | Load Throughput: %.2f MB/s',
        jax.process_index(),
        batch_index + 1,
        path,
        len(current_loaded_tensors),
        time.time() - current_load_start_time,
        current_load_size / (1024 * 1024),
        current_load_size
        / (1024 * 1024)
        / (time.time() - current_load_start_time),
    )
    del current_loaded_tensors
  load_end_time = time.time()
  total_loaded_size = sum(t.nbytes for t in loaded_tensors.values())
  _log_info(
      '[VM %d] [Batch %d] Loaded Batch | Load Time: %.2f seconds | Load Size:'
      ' %.2f MB | Load Throughput: %.2f MB/s',
      jax.process_index(),
      batch_index + 1,
      load_end_time - load_start_time,
      total_loaded_size / (1024 * 1024),
      total_loaded_size / (1024 * 1024) / (time.time() - load_start_time),
  )
  return loaded_tensors, batch_index


def _execute_batch(
    layout: safetensors_layout.SafetensorsLayout,
    file_abstract_trees: dict[str, Any],
    batch_index: int,
    num_batches: int,
    batch_size: float,
) -> Tuple[dict[str, Any], int]:
  """Sync wrapper for _execute_batch_async to run in ThreadPoolExecutor."""
  return asyncio.run(
      _execute_batch_async(
          layout,
          file_abstract_trees,
          batch_index,
          num_batches,
          batch_size,
      )
  )


async def _load_safetensors(
    layout: safetensors_layout.SafetensorsLayout, path: Path
) -> Dict[str, Any]:
  """Calls the correct safetensors loading function."""

  start_time = time.time()
  paths = list(path.glob('*.safetensors'))
  tensor_to_path = await get_tensor_to_path_indexing(paths)
  with ocp_v1.Context(
      checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS,
      memory_options=ocp_v1.options.MemoryOptions(
          read_concurrent_bytes=_MAX_BATCH_SIZE_MB.value * 1024**2,
      ),
  ):
    batches, batch_sizes, abstract_pytree = await create_loading_plan(paths)
    batches_to_load = []
    for batch_abstract_pytree in batches:
      file_abstract_trees = collections.defaultdict(dict)
      for tensor_name in batch_abstract_pytree:
        path = tensor_to_path[tensor_name]
        file_abstract_trees[path][tensor_name] = abstract_pytree[tensor_name]
      print(f'file_abstract_trees: {len(file_abstract_trees)} files')
      batches_to_load.append(file_abstract_trees)

  total_load_size = 0
  all_loaded_tensors = {}
  num_batches = len(batches_to_load)
  with ThreadPoolExecutor(max_workers=_THREADS.value) as pool:
    results = pool.map(
        functools.partial(_execute_batch, layout),
        batches_to_load,
        range(0, num_batches),
        [num_batches] * num_batches,
        batch_sizes,
    )
    for r in results:
      loaded_tensors, batch_index = r
      batch_size = batch_sizes[batch_index]
      all_loaded_tensors.update(loaded_tensors)
      del loaded_tensors
      total_load_size += batch_size
    _log_info(
        '\033[1mFinal load size: %.2f MB | Final load time: %.2f seconds |'
        ' Final load throughput: %.2f MB/s \033[0m',
        total_load_size,
        time.time() - start_time,
        (total_load_size / (time.time() - start_time)),
    )
    return all_loaded_tensors


async def run_cpu_batching(input_dir: str, output_dir: str):
  """Orchestrates the metadata sizing, planning, and batch execution loop."""
  # Only VM 0 cleans up the directory before the distributed run starts
  if _SAVING_ENABLED.value:
    if gfile.exists(output_dir):
      _log_info('Removing existing checkpoint directory: %s', output_dir)
      gfile.rmtree(output_dir)

  input_path = epath.Path(input_dir)

  _log_info('=' * 60)
  _log_info(
      f'🔎 STARTING CONVERSION FROM SAFETENSORS TO ORBAX for {input_path}'
  )
  _log_info('Output directory: %s', output_dir)
  _log_info('=' * 60)
  _log_info('Conversion will be done in following steps.')
  _log_info('step 0: Benchmarking official safetensors library')
  _log_info('step 1: Reading Safetensors Metadata')
  _log_info('step 2: Analyzing Model Structure')
  _log_info('step 3: Executing Loading Loop')
  _log_info('step 4: Finalizing Native Orbax Checkpoint')

  # Step 0: Benchmarking official safetensors library...
  _log_info(
      '---------- Step 0: Benchmarking official safetensors library -----------'
  )
  if _OFFICIAL_LOAD_ENABLED.value:
    benchmark_official_safetensors(input_path)

  # Step 1: Reading Safetensors Metadata...
  _log_info('---------- Step 1: Reading Safetensors Metadata -----------')
  layout = safetensors_layout.SafetensorsLayout()
  metadata_start_time = time.time()
  metadata_ckpt = await layout.metadata(input_path)
  metadata_end_time = time.time()
  _log_info(
      'Safetensors Metadata Time: %.2f seconds',
      metadata_end_time - metadata_start_time,
  )

  if 'pytree' in metadata_ckpt.metadata:
    model_metadata = metadata_ckpt.metadata['pytree']
  else:
    model_metadata = metadata_ckpt.metadata

  # Step 2: Analyzing Model Structure...
  _log_info('------------ Step 2: Analyzing Model Structure --------------')
  analyze_start_time = time.time()
  analyze_model_structure(model_metadata)
  analyze_end_time = time.time()
  _log_info(
      'Model Structure Analysis Time: %.2f seconds',
      analyze_end_time - analyze_start_time,
  )

  # Step 4: Executing Loading Loop...
  _log_info('------------- Step 3: Executing Loading --------------------')
  load_start_time = time.time()
  loaded_tensors = await _load_safetensors(layout, input_path)
  total_load_size = sum(t.nbytes for t in loaded_tensors.values())
  total_load_size = total_load_size / (1024 * 1024)
  load_end_time = time.time()
  total_save_time = 0.0
  if _SAVING_ENABLED.value:
    _log_info(
        '------------ Step 4: Saving Native Orbax Checkpoint'
        ' --------------'
    )
    save_start_time = time.time()

    save_awaitable = await orbax_layout.OrbaxLayout().save(
        path=_PathAwaitingCreation(output_dir),
        checkpointables={'pytree': loaded_tensors},
    )
    await save_awaitable
    total_save_time = time.time() - save_start_time
    _log_info(
        'Saved Native Orbax Checkpoint Time: %.2f seconds',
        total_save_time,
    )
  _log_info(
      '✅ \033[1m[VM %d] Total Conversion Time: %.2f seconds | Final Size: %.2f'
      ' MB | Total Load Throughput: %.2f MB/s | Total Save Time: %.2f seconds |'
      ' Total Save Throughput: %.2f MB/s |'
      ' Final checkpoint location: %s\033[0m',
      jax.process_index(),
      load_end_time - load_start_time,
      total_load_size,
      total_load_size / (load_end_time - load_start_time),
      total_save_time,
      total_load_size / total_save_time,
      output_dir,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not _INPUT_DIR.value or not _OUTPUT_DIR.value:
    raise app.UsageError('--input_dir and --output_dir must be provided.')

  if not _USE_FILE_LOGGER_ONLY.value:
    logging.set_stderrthreshold('INFO')

  asyncio.run(run_cpu_batching(_INPUT_DIR.value, _OUTPUT_DIR.value))


if __name__ == '__main__':
  app.run(main)
