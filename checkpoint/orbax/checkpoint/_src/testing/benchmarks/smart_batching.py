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
from concurrent import futures
import inspect
import itertools
import os
import time
from typing import Any, cast, Dict, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
from etils import epath
import huggingface_hub
import jax
import numpy as np
from orbax.checkpoint.experimental import v1 as ocp_v1
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from tensorflow.io import gfile


ThreadPoolExecutor = futures.ThreadPoolExecutor
_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Directory containing Safetensors files.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to save the converted Orbax checkpoint.'
)
_MAX_BATCH_SIZE_GB = flags.DEFINE_float(
    'max_batch_size_gb', 5.0, 'Maximum size of a single batch in GB.'
)
_WORKERS = flags.DEFINE_integer(
    'workers',
    1,
    'Number of worker threads to use for concurrent processing of batches.',
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


def benchmark_official_safetensors(repo_id: str) -> float:
  """Benchmarks loading checkpoints using the standard Safetensors library.

  It simulates a real-world scenario: Download -> RAM -> Parse.

  Args:
    repo_id: The HuggingFace repo ID to download from.

  Returns:
    The total time in seconds taken to download and process the Safetensor
    files.
  """
  # 1. Get only the safetensor filenames
  all_files = huggingface_hub.list_repo_files(repo_id)
  safetensor_files = [f for f in all_files if f.endswith('.safetensors')]
  safetensor_files.sort()
  total_start_time = time.time()
  local_dir = os.path.expanduser('~/safetensor_layout_loading_test/tmp')
  os.makedirs(local_dir, exist_ok=True)

  for filename in safetensor_files:
    print(f'--- Processing {filename} ---')
    file_start = time.time()
    file_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    file_end = time.time()
    _log_info('Finished %s in %.2f seconds', filename, file_end - file_start)
    if os.path.exists(file_path):
      _log_info('Removing file: %s', file_path)
      os.remove(file_path)

  return time.time() - total_start_time


async def verify_local_integrity(
    input_dir: str,
    local_output_dir: str,
    model_metadata: Any,
):
  """Verifies integrity by comparing source and destination tensors.

  Loads entire Source (Safetensors) and Destination (Local Orbax) and compares
  them tensor-by-tensor for exact matches.

  Args:
    input_dir: Directory containing Safetensors files.
    local_output_dir: Directory to save the converted Orbax checkpoint.
    model_metadata: Model metadata.

  Raises:
    RuntimeError: If any tensor does not match exactly between the source and
    destination.
  """
  _log_info(f'🔎 STARTING DIRECT INTEGRITY CHECK on {local_output_dir}')

  input_path = epath.Path(input_dir)
  output_path = epath.Path(local_output_dir)

  with ocp_v1.Context(
      checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
  ):
    _log_info('Loading entire Safetensors checkpoint into memory...')
    load_start_time = time.time()
    st_data = ocp_v1.load_pytree(
        path=input_path, abstract_pytree=model_metadata
    )
    load_end_time = time.time()
    _log_info(
        '✅ Safetensors Load Time: %s seconds',
        load_end_time - load_start_time,
    )

    if inspect.iscoroutine(st_data):
      st_data = await st_data
    st_data = cast(Dict[str, Any], st_data)

    if 'pytree' in st_data and len(st_data) == 1:
      st_data = st_data['pytree']

  with ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.ORBAX):
    _log_info('Loading entire Local Orbax checkpoint into memory...')
    load_start_time = time.time()
    try:
      orbax_data = ocp_v1.load_pytree(output_path, {'pytree': model_metadata})
    except Exception as e:
      _log_error(
          '❌ FATAL: Could not read Local Orbax checkpoint from %s', output_path
      )
      _log_error('Error details: %s', e)
      if output_path.exists():
        _log_info('📂 Directory contents of %s:', output_path)
        for f in output_path.iterdir():
          _log_info('   - %s', f.name)
      raise e

    if isinstance(orbax_data, dict) and 'pytree' in orbax_data:
      orbax_data = orbax_data['pytree']

    load_end_time = time.time()
    _log_info(
        '✅ Orbax Load Time: %s seconds',
        load_end_time - load_start_time,
    )

  _log_info('Comparing checkpoints...')

  flat_meta, _ = jax.tree_util.tree_flatten_with_path(model_metadata)

  def get_val(d: Any, keys: Sequence[str]) -> Any:
    try:
      for k in keys:
        d = d[k]
      return d
    except KeyError:
      return None

  total_tensors = 0
  mismatches = 0

  for path, _ in flat_meta:
    key_tuple = tuple(cast(Any, p).key for p in path)
    tensor_name = '.'.join(key_tuple)
    total_tensors += 1

    val_src = get_val(st_data, key_tuple)
    val_dst = get_val(orbax_data, key_tuple)

    if val_dst is None:
      _log_error('❌ MISSING TENSOR: %s not found in Orbax load.', tensor_name)
      mismatches += 1
      continue

    if val_src.shape != val_dst.shape:
      _log_error(
          '❌ SHAPE MISMATCH: %s | Src %s != Dst %s',
          tensor_name,
          val_src.shape,
          val_dst.shape,
      )
      mismatches += 1
      continue

    if not np.allclose(val_src, val_dst, equal_nan=True, atol=1e-6):
      diff = np.abs(val_src - val_dst)
      _log_error(
          '❌ VALUE MISMATCH: %s | Max Diff: %s', tensor_name, np.max(diff)
      )
      mismatches += 1
    # else:
    #   # Optional: Log success for very large tensors just to be sure
    #   val_src_size = val_src.nbytes / (1024 * 1024 * 1024)
    #   val_dst_size = val_dst.nbytes / (1024 * 1024 * 1024)
    # _log_info(
    #     '✅ Verified Large Tensor: %s | Src Size: %s GB | Dst Size: %s GB',
    #     tensor_name,
    #     val_src_size,
    #     val_dst_size,
    # )

  del st_data
  del orbax_data

  if mismatches == 0:
    _log_info(
        '🎉 SUCCESS: All %s tensors verified matching exactly!', total_tensors
    )
  else:
    _log_error(
        '💀 FAILURE: Found %s mismatches during verification.', mismatches
    )
    raise RuntimeError('Sanity check failed! Stopping upload.')


# --- NEW HELPER FUNCTION FOR SIZE TRACKING ---
def get_dir_size_mb(start_path: str) -> float:
  """Calculates size of the checkpoint, including hidden partial folders."""
  files_to_stat = []
  # Orbax writes to a folder with a suffix like '.partial_save'
  # We check both the main folder and the partial folder to be sure.
  paths_to_check = [start_path, start_path + '.partial_save']

  for p in paths_to_check:
    if gfile.exists(p):
      try:
        for dirpath, _, filenames in gfile.Walk(p):
          for f in filenames:
            fp = os.path.join(dirpath, f)
            files_to_stat.append(fp)
      except gfile.GOSError as e:
        _log_error('Failed to walk files in %s: %s', p, e)
        raise e

  if not files_to_stat:
    return 0.0

  try:
    stats = gfile.BulkStatWithException(files_to_stat)
    total_size = sum(s.length for s in stats if s.length > -1)
  except gfile.GOSError as e:
    _log_error('Failed to stat files in %s: %s', start_path, e)
    raise e

  return total_size / (1024 * 1024)


def analyze_model_structure(metadata_tree: Any) -> None:
  """Logs detailed statistics about the model structure and expected size."""
  flat_metadata, _ = jax.tree_util.tree_flatten_with_path(metadata_tree)

  total_params = 0
  total_bytes = 0
  dtype_counts = {}

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

      # Track dtype distribution
      dtype_str = str(dtype)
      dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

  total_gb = total_bytes / (1024**3)
  total_mb = total_bytes / (1024**2)

  _log_info('Total Tensors: %d', len(flat_metadata))
  _log_info('Total Parameters: %s', f'{total_params:,}')
  _log_info(
      'Expected Raw Size (Uncompressed): %.2f MB (%.4f GB)',
      total_mb,
      total_gb,
  )
  _log_info('Dtype Distribution: %s', dtype_counts)


def get_param_size_bytes(param_info: jax.ShapeDtypeStruct) -> int:
  """Calculates size in bytes for a single parameter from metadata."""
  dtype_size = np.dtype(param_info.dtype).itemsize
  return np.prod(param_info.shape) * dtype_size


async def _execute_batch_async(
    layout: safetensors_layout.SafetensorsLayout,
    input_path: epath.Path,
    output_dir: str,
    plan: Any,
    flat_map: Dict[str, Any],
    batch_index: int,
    total_batches: int,
) -> Tuple[float, float, int]:
  """Loads and saves a single batch."""
  _log_info(
      '\033[1mProcessing batch %s/%s...\033[0m', batch_index, total_batches
  )
  batch_abstract_pytree = {}
  for i, batch_keys in enumerate(plan, 1):
    if i == batch_index:
      for k in batch_keys:
        batch_abstract_pytree[k] = flat_map[k]
      break
  load_start_time = time.time()
  _log_info(
      'Loading batch into Host RAM from Safetensors, starting at %s',
      load_start_time,
  )
  tensors = await layout.load_pytree(
      path=input_path, abstract_pytree=batch_abstract_pytree
  )
  load_end_time = time.time()
  _log_info(
      '✅ Safetensors Load Time: %.2f seconds',
      load_end_time - load_start_time,
  )
  tensors = cast(Dict[str, Any], tensors)

  total_save_time = 0.0
  if _SAVING_ENABLED.value:
    save_start_time = time.time()

    tensors_to_save = {'pytree': tensors}
    ocp_v1.partial.save_pytree(output_dir, tensors_to_save)
    save_end_time = time.time()
    _log_info(
        '✅ Orbax Save Time: %.2f seconds in directory %s',
        save_end_time - save_start_time,
        output_dir,
    )
    total_save_time = save_end_time - save_start_time
  return load_end_time - load_start_time, total_save_time, batch_index - 1


def _execute_batch(
    layout: safetensors_layout.SafetensorsLayout,
    input_path: epath.Path,
    output_dir: str,
    plan: Any,
    flat_map: Dict[str, Any],
    batch_index: int,
    total_batches: int,
) -> Tuple[float, float, int]:
  """Sync wrapper for _execute_batch_async to run in ThreadPoolExecutor."""
  return asyncio.run(
      _execute_batch_async(
          layout,
          input_path,
          output_dir,
          plan,
          flat_map,
          batch_index,
          total_batches,
      )
  )


async def run_cpu_batching(
    input_dir: str, output_dir: str, max_batch_size_gb: float
):
  """Orchestrates the metadata sizing, planning, and batch execution loop."""
  if gfile.exists(output_dir):
    _log_info('Removing existing checkpoint directory: %s', output_dir)
    gfile.DeleteRecursively(output_dir)
  partial_save_dir = output_dir + '.partial_save'
  if gfile.exists(partial_save_dir):
    _log_info('Removing existing partial save directory: %s', partial_save_dir)
    gfile.DeleteRecursively(partial_save_dir)
  start_time = time.time()
  input_path = epath.Path(input_dir)
  layout = safetensors_layout.SafetensorsLayout()

  _log_info('=' * 60)
  _log_info(f'🔎 STARTING CONVERSION FROM SAFETENSORS TO ORBAX for {input_path}')
  _log_info('Output directory: %s', output_dir)
  _log_info('=' * 60)
  _log_info('Conversion will be done in following steps.')
  _log_info('step 1: Reading Safetensors Metadata')
  _log_info('step 2: Analyzing Model Structure')
  _log_info('step 3: Creating Loading Plan')
  _log_info('step 4: Executing Loading Loop')
  _log_info('step 5: Finalizing Native Orbax Checkpoint')
  _log_info('step 6: Time to load using Safetensors Flax API')
  _log_info('step 7: Verifying Local Integrity')

  # Step 1: Reading Safetensors Metadata...
  _log_info('---------- Step 1: Reading Safetensors Metadata -----------')
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

  # Step 3: Creating Loading Plan...
  _log_info('-------------- Step 3: Creating Loading Plan ----------------')
  plan_start_time = time.time()
  plan, batch_sizes = await layout.create_loading_plan(
      input_path, max_batch_size_gb
  )
  _log_info(
      'Created a plan containing %s batches in %.2f seconds',
      len(plan),
      time.time() - plan_start_time,
  )

  # Step 4: Executing Loading Loop...
  _log_info('------------- Step 4: Executing Loading Loop ----------------')
  flat_metadata, _ = jax.tree_util.tree_flatten_with_path(model_metadata)
  flat_map = {path[0].key: leaf for path, leaf in flat_metadata}

  global_start = time.time()
  total_load_size = 0
  num_batches = len(plan)
  with ThreadPoolExecutor(max_workers=_WORKERS.value) as pool:
    results = pool.map(
        _execute_batch,
        itertools.repeat(layout),
        itertools.repeat(input_path),
        itertools.repeat(output_dir),
        itertools.repeat(plan),
        itertools.repeat(flat_map),
        range(1, num_batches + 1),
        itertools.repeat(num_batches),
    )
    for r in results:
      load_time, _, batch_index = r
      batch_size = batch_sizes[batch_index]
      total_load_size += batch_size
      print(
          f'Read Batch {batch_index} with {batch_size} MB in {load_time} sec '
          f'(throughput: {batch_size / load_time} MB/sec)'
      )
  global_end = time.time()
  _log_info(
      'Total Loading Time: %.2f seconds',
      global_end - global_start,
  )
  # total_load_time = sum(r[0] for r in results)
  total_save_time = 0.0

  official_load_time = 0.0
  if _SAVING_ENABLED.value:
    # Step 5: Finalizing Native Orbax Checkpoint...
    finalize_start_time = time.time()
    _log_info('------- Step 5: Finalizing Native Orbax Checkpoint ----------')
    ocp_v1.partial.finalize(output_dir)
    total_save_time += time.time() - finalize_start_time

  # final_size = get_dir_size_mb(output_dir)
  if _OFFICIAL_LOAD_ENABLED.value:
    # Step 6: Time to load using Safetensors Flax API..
    _log_info(
        '------------ Step 6:  Time to load using Safetensors Flax API'
        ' ------------\n'
    )
    _log_info('Loading entire checkpoint in RAM using official Safetensors...')
    repo_id_1gb = 'Qwen/Qwen2.5-0.5B-Instruct'
    repo_id_50gb = 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4'
    repo_id_150gb = 'Qwen/Qwen2.5-72B-Instruct'
    repo_id_1tb = 'moonshotai/Kimi-K2-Instruct'
    if input_dir == 'gs://safetensor-kimi-central/test-model-1gb':
      official_load_time = benchmark_official_safetensors(repo_id_1gb)
    elif input_dir == 'gs://safetensor-kimi-central/test-model-50gb':
      official_load_time = benchmark_official_safetensors(repo_id_50gb)
    elif input_dir == 'gs://safetensor-kimi-central/test-model-150gb':
      official_load_time = benchmark_official_safetensors(repo_id_150gb)
    elif input_dir == 'gs://safetensor-kimi-central/test-model-1tb':
      official_load_time = benchmark_official_safetensors(repo_id_1tb)
  _log_info(
      '✅ \033[1mTotal Conversion Time: %.2f seconds | Final Size: %.2f MB |'
      ' Total Load Throughput: %.2f MB/s | Total Save Time: %.2f seconds |'
      ' Official Load Time: %.2f seconds | Final checkpoint location:'
      ' %s\033[0m',
      time.time() - start_time,
      total_load_size / (1024 * 1024),
      total_load_size / (1024 * 1024) / (time.time() - start_time),
      total_save_time,
      official_load_time,
      output_dir,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not _INPUT_DIR.value or not _OUTPUT_DIR.value:
    raise app.UsageError('--input_dir and --output_dir must be prov ided.')

  if not _USE_FILE_LOGGER_ONLY.value:
    logging.set_stderrthreshold('INFO')

  asyncio.run(
      run_cpu_batching(
          _INPUT_DIR.value, _OUTPUT_DIR.value, _MAX_BATCH_SIZE_GB.value
      )
  )


if __name__ == '__main__':
  app.run(main)
